defmodule BstsNx.Forecaster do
  @moduledoc """
  Pure time series forecasting using Bayesian Structural Time Series.

  This module provides a `fit/predict` interface for forecasting future
  values with full uncertainty quantification via posterior credible
  intervals.  Unlike `CausalImpact`, which measures the effect of a
  past intervention, `Forecaster` is concerned with predicting what
  will happen next.

  ## Why BSTS for forecasting?

  Compared to ARIMA or Prophet, BSTS forecasting provides:
  - **Proper Bayesian uncertainty**: credible intervals from the full
    posterior, not just point estimates with approximate confidence bands
  - **Component decomposition**: separate trend, seasonal, and regression
    contributions with per-component uncertainty
  - **External regressors**: incorporate causal drivers (promotions,
    weather, events) directly in the state-space model
  - **Missing data handling**: gracefully handles gaps in the training data

  ## Workflow

      # 1. Fit model on historical data
      fit_result = Forecaster.fit(training_data,
        seasonality: 7,
        num_samples: 300
      )

      # 2. Forecast future periods
      forecast = Forecaster.predict(fit_result, horizon: 14)

      forecast.mean       # point forecasts
      forecast.lower      # lower credible interval bound
      forecast.upper      # upper credible interval bound
      forecast.components # per-component decomposition (if structured model)

      # Or one-shot:
      forecast = Forecaster.fit_predict(training_data, 14,
        seasonality: 7
      )
  """

  alias BstsNx.GibbsSampler
  alias BstsNx.ModelBuilder
  alias BstsNx.ModelSpec

  @type fit_result :: %{
          posterior_samples: list(),
          spec: ModelSpec.t(),
          training_length: non_neg_integer(),
          method: :structured | :scalar,
          n_regression_dims: non_neg_integer()
        }

  @type forecast_result :: %{
          mean: [float()],
          lower: [float()],
          upper: [float()],
          sd: [float()],
          horizon: non_neg_integer(),
          alpha: float()
        }

  @doc """
  Fits a BSTS model to training data.

  ## Parameters

    * `observations` - list of numeric training observations

  ## Options

    * `:model_spec` - a `%BstsNx.ModelSpec{}` for full control. When
      provided, `:seasonality` and `:regressors` are ignored.
    * `:seasonality` - number of seasonal periods (e.g., 7 for weekly
      data with daily observations). Composes local linear trend + seasonal.
    * `:regressors` - a `{T, p}` Nx tensor of training-period regressor values.
      Produces a composed model with trend + optional seasonal + regression.
      When used, pass `:future_regressors` to `predict/2` for forecasts.
    * `:num_samples` - posterior MCMC draws (default: 200)
    * `:burn_in` - burn-in iterations (default: num_samples / 2)
    * `:seed` - integer PRNG seed for reproducibility

  ## Examples

      iex> fit = BstsNx.Forecaster.fit(Enum.to_list(1..20), num_samples: 10, seed: 42)
      iex> fit.method
      :scalar
      iex> fit.training_length
      20
  """
  @spec fit([number()], keyword()) :: fit_result()
  def fit(observations, opts \\ []) do
    if observations == [] do
      raise ArgumentError, "observations must be non-empty"
    end

    num_samples = Keyword.get(opts, :num_samples, 200)

    if not is_integer(num_samples) or num_samples <= 0 do
      raise ArgumentError, "num_samples must be a positive integer, got: #{num_samples}"
    end

    obs_list = ModelBuilder.coerce_obs(observations)
    {spec, method} = ModelBuilder.build_spec(obs_list, opts)
    burn_in = Keyword.get(opts, :burn_in, div(num_samples, 2))

    sampler_opts =
      opts
      |> Keyword.take([:seed, :key, :thin])
      |> Keyword.put(:burn_in, burn_in)

    samples =
      case method do
        :structured ->
          GibbsSampler.sample_structured(obs_list, spec, num_samples, sampler_opts)

        :scalar ->
          first = List.first(obs_list) || 0.0

          GibbsSampler.sample(
            obs_list,
            num_samples,
            first,
            1.0,
            1.0,
            1.0,
            sampler_opts
          )
      end

    n_reg = infer_regression_dims(spec, opts, method)

    %{
      posterior_samples: samples,
      spec: spec,
      training_length: length(obs_list),
      method: method,
      n_regression_dims: n_reg
    }
  end

  @doc """
  Generates forecasts from a fitted model.

  ## Options

    * `:horizon` - number of future periods to forecast (required)
    * `:alpha` - significance level for credible intervals (default: 0.05)
    * `:seed` - PRNG seed for forecast simulation
    * `:future_regressors` - a `{horizon, p}` Nx tensor of future regressor values,
      where `p` must match the number of regressors used during `fit/2`. When provided,
      builds per-step observation matrices for regressor-aware forecasts.
      Without this option, the last training H is used for all forecast steps
      (appropriate for trend/seasonal-only models).
  """
  @spec predict(fit_result(), keyword()) :: forecast_result()
  def predict(%{method: method} = fit_result, opts) do
    horizon = Keyword.fetch!(opts, :horizon)

    if not is_integer(horizon) or horizon <= 0 do
      raise ArgumentError, "horizon must be a positive integer, got: #{inspect(horizon)}"
    end

    alpha = Keyword.get(opts, :alpha, 0.05)

    if not is_number(alpha) or alpha <= 0.0 or alpha >= 1.0 do
      raise ArgumentError, "alpha must be between 0 and 1 (exclusive), got: #{inspect(alpha)}"
    end

    seed = Keyword.get(opts, :seed, System.os_time())
    key = Keyword.get(opts, :key, Nx.Random.key(seed))

    future_regressors = Keyword.get(opts, :future_regressors)
    n_reg = Map.get(fit_result, :n_regression_dims, 0)

    # Validate future_regressors dimensions
    case future_regressors do
      %Nx.Tensor{} = fr ->
        if Nx.axis_size(fr, 0) != horizon do
          raise ArgumentError,
                "future_regressors rows (#{Nx.axis_size(fr, 0)}) must match horizon (#{horizon})"
        end

        if Nx.axis_size(fr, 1) != n_reg do
          raise ArgumentError,
                "future_regressors columns (#{Nx.axis_size(fr, 1)}) must match training regressors (#{n_reg})"
        end

      nil ->
        :ok
    end

    trajectories =
      case method do
        :structured -> predict_structured(fit_result, horizon, key, future_regressors)
        :scalar -> predict_scalar(fit_result, horizon, key)
      end

    aggregate_trajectories(trajectories, horizon, alpha)
  end

  @doc """
  Fits a model and generates forecasts in one call.

  ## Examples

      forecast = Forecaster.fit_predict(training_data, 14,
        seasonality: 7,
        num_samples: 200,
        seed: 42
      )

      iex> forecast = BstsNx.Forecaster.fit_predict(Enum.to_list(1..20), 5, num_samples: 10, seed: 42)
      iex> length(forecast.mean)
      5
      iex> forecast.horizon
      5
  """
  @spec fit_predict([number()], non_neg_integer(), keyword()) :: forecast_result()
  def fit_predict(observations, horizon, opts \\ []) do
    fit_result = fit(observations, opts)
    predict_opts = Keyword.put(opts, :horizon, horizon)
    predict(fit_result, predict_opts)
  end

  @doc """
  Extracts the posterior decomposition of the training data into
  trend, seasonal, and regression components.

  Only available for structured models. Returns `nil` for scalar models.
  """
  @spec decompose(fit_result()) :: map() | nil
  def decompose(%{method: :scalar}), do: nil

  def decompose(%{posterior_samples: samples, spec: spec, training_length: t}) do
    h = spec.h

    # Extract per-sample state trajectories and project through H
    n_samples = length(samples)

    # Convert each sample's states list to :array for O(1) random access
    sample_state_arrays = Enum.map(samples, fn sample -> :array.from_list(sample.states) end)

    # Compute mean state at each time step across all samples
    mean_states =
      Enum.map(0..(t - 1), fn step ->
        step_states =
          Enum.map(sample_state_arrays, fn arr ->
            :array.get(step, arr) |> Nx.to_flat_list()
          end)

        # Average across samples for each state dimension
        Enum.zip_with(step_states, fn vals ->
          Enum.sum(vals) / n_samples
        end)
      end)

    # Project mean states through H to get fitted values
    # Convert H to :array when time-varying for O(1) access
    h_arr = if is_list(h), do: :array.from_list(h), else: nil

    fitted =
      Enum.with_index(mean_states, fn state_list, idx ->
        state = Nx.tensor(state_list)
        h_t = if h_arr, do: :array.get(idx, h_arr), else: h
        h_row = if Nx.rank(h_t) == 2, do: Nx.squeeze(h_t, axes: [0]), else: Nx.flatten(h_t)
        Nx.to_number(Nx.dot(h_row, state))
      end)

    %{
      fitted: fitted,
      states: mean_states
    }
  end

  # -- Private implementation --

  # Model building delegated to ModelBuilder.build_spec/2

  defp predict_structured(
         %{posterior_samples: samples, spec: spec, n_regression_dims: n_reg},
         horizon,
         base_key,
         future_regressors
       ) do
    keys = Nx.Random.split(base_key, parts: length(samples))
    n_state = Nx.axis_size(spec.f, 0)

    # Build per-step H: if future_regressors provided, use build_future_h;
    # otherwise use the last training H for all steps
    h_list =
      case future_regressors do
        %Nx.Tensor{} = fr ->
          ModelBuilder.build_future_h(spec, fr, n_reg)

        _ ->
          static_h =
            case spec.h do
              list when is_list(list) -> List.last(list)
              %Nx.Tensor{} = t -> t
            end

          List.duplicate(static_h, horizon)
      end

    Enum.with_index(samples, fn sample, idx ->
      final_state = List.last(sample.states)
      q_matrix = sample.q_matrix
      obs_var = Nx.to_number(sample.obs_var)
      obs_sd = :math.sqrt(max(obs_var, 0.0))

      q_diag = Nx.take_diagonal(q_matrix)
      q_sds = Nx.sqrt(Nx.max(q_diag, Nx.tensor(0.0)))

      sample_key = split_key_at(keys, idx)
      step_keys = Nx.Random.split(sample_key, parts: horizon)

      # Iterate h_list directly to avoid O(n²) Enum.at access
      {_, trajectory} =
        h_list
        |> Enum.with_index()
        |> Enum.reduce({Nx.flatten(final_state), []}, fn {h_t, step}, {state, acc} ->
          step_key = split_key_at(step_keys, step)
          sk = Nx.Random.split(step_key, parts: 2)
          key_state = split_key_at(sk, 0)
          key_obs = split_key_at(sk, 1)

          # Process noise
          {z_state, _} = Nx.Random.normal(key_state, 0.0, 1.0, shape: {n_state})
          noise = Nx.multiply(z_state, q_sds)
          next_state = Nx.add(Nx.dot(spec.f, state), noise)

          # Observation with per-step H
          h_row = if Nx.rank(h_t) == 2, do: Nx.squeeze(h_t, axes: [0]), else: Nx.flatten(h_t)
          y_mean = Nx.to_number(Nx.dot(h_row, next_state))
          {z_obs, _} = Nx.Random.normal(key_obs, 0.0, 1.0)
          y = y_mean + Nx.to_number(z_obs) * obs_sd

          {next_state, [y | acc]}
        end)

      Enum.reverse(trajectory)
    end)
  end

  defp predict_scalar(%{posterior_samples: samples}, horizon, base_key) do
    keys = Nx.Random.split(base_key, parts: length(samples))

    Enum.with_index(samples, fn sample, idx ->
      final_state = Nx.to_number(List.last(sample.states))
      q = Nx.to_number(sample.process_var)
      r = Nx.to_number(sample.obs_var)
      sd_q = :math.sqrt(max(q, 0.0))
      sd_r = :math.sqrt(max(r, 0.0))

      sample_key = split_key_at(keys, idx)
      sk = Nx.Random.split(sample_key, parts: 2)
      key_process = split_key_at(sk, 0)
      key_obs = split_key_at(sk, 1)
      {proc_noise, _} = Nx.Random.normal(key_process, 0.0, 1.0, shape: {horizon})
      {obs_noise, _} = Nx.Random.normal(key_obs, 0.0, 1.0, shape: {horizon})

      proc_list = Nx.to_flat_list(proc_noise)
      obs_list = Nx.to_flat_list(obs_noise)

      {_, trajectory} =
        Enum.zip(proc_list, obs_list)
        |> Enum.reduce({final_state, []}, fn {pn, on}, {state, acc} ->
          next = state + pn * sd_q
          y = next + on * sd_r
          {next, [y | acc]}
        end)

      Enum.reverse(trajectory)
    end)
  end

  defp aggregate_trajectories(trajectories, horizon, alpha) do
    n = length(trajectories)

    # Transpose: from list-of-trajectories to list-of-timesteps.
    # Enum.zip_with/2 is O(horizon * n) vs O(horizon² * n) with Enum.at.
    per_step_values =
      if n == 0,
        do: List.duplicate([], horizon),
        else: Enum.zip_with(trajectories, & &1)

    per_step =
      Enum.map(per_step_values, fn vals ->
        vals = Enum.sort(vals)

        mean = Enum.sum(vals) / n

        sd =
          if n < 2 do
            0.0
          else
            ss = Enum.reduce(vals, 0.0, fn v, acc -> acc + (v - mean) * (v - mean) end)
            :math.sqrt(ss / (n - 1))
          end

        {lower, upper} = BstsNx.Utils.percentile_interval(vals, n, alpha)
        {mean, sd, lower, upper}
      end)

    %{
      mean: Enum.map(per_step, &elem(&1, 0)),
      sd: Enum.map(per_step, &elem(&1, 1)),
      lower: Enum.map(per_step, &elem(&1, 2)),
      upper: Enum.map(per_step, &elem(&1, 3)),
      horizon: horizon,
      alpha: alpha
    }
  end

  defp infer_regression_dims(spec, opts, :structured) do
    case Keyword.get(opts, :regressors) do
      %Nx.Tensor{} = t ->
        Nx.axis_size(t, 1)

      regressors when is_list(regressors) ->
        regressors
        |> Nx.tensor()
        |> Nx.axis_size(1)

      _ ->
        infer_regression_dims_from_spec(spec)
    end
  end

  defp infer_regression_dims(_spec, _opts, _method), do: 0

  defp infer_regression_dims_from_spec(%ModelSpec{h: h_list})
       when is_list(h_list) and h_list != [] do
    rows = Enum.map(h_list, &h_to_row/1)
    row_len = rows |> hd() |> length()

    varying_flags =
      Enum.map(0..(row_len - 1), fn idx ->
        first_val = rows |> hd() |> Enum.at(idx)
        Enum.any?(rows, fn row -> abs(Enum.at(row, idx) - first_val) > 1.0e-12 end)
      end)

    varying_flags
    |> Enum.reverse()
    |> Enum.take_while(& &1)
    |> length()
  end

  defp infer_regression_dims_from_spec(_), do: 0

  defp h_to_row(%Nx.Tensor{} = h_t) do
    h_t
    |> Nx.flatten()
    |> Nx.to_flat_list()
    |> Enum.map(&(&1 + 0.0))
  end

  defp split_key_at(keys, idx) do
    Nx.slice_along_axis(keys, idx, 1, axis: 0)
    |> Nx.squeeze(axes: [0])
  end

  # Observation coercion delegated to ModelBuilder.coerce_obs/1
end
