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

    sample_state_tensors = Enum.map(samples, fn sample -> Nx.stack(sample.states) end)
    mean_states_t = sample_state_tensors |> Nx.stack() |> Nx.mean(axes: [0])
    {_t_steps, state_dim} = Nx.shape(mean_states_t)

    mean_states =
      mean_states_t
      |> Nx.to_flat_list()
      |> Enum.chunk_every(state_dim)
      |> Enum.take(t)

    fitted =
      cond do
        is_list(h) ->
          h_rows = h |> Enum.map(&h_to_row_tensor/1) |> Nx.stack()

          h_rows
          |> Nx.multiply(mean_states_t)
          |> Nx.sum(axes: [1])
          |> Nx.to_flat_list()
          |> Enum.take(t)

        true ->
          h_row = h_to_row_tensor(h)
          mean_states_t |> Nx.dot(h_row) |> Nx.to_flat_list() |> Enum.take(t)
      end

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
    sample_key_rows = Nx.Random.split(base_key, parts: length(samples)) |> Nx.to_list()
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

    h_rows = Enum.map(h_list, &h_to_row_tensor/1)

    Enum.zip(samples, sample_key_rows)
    |> Enum.map(fn {sample, sample_key_row} ->
      final_state = List.last(sample.states)
      q_matrix = sample.q_matrix
      obs_var = Nx.to_number(sample.obs_var)
      obs_sd = :math.sqrt(max(obs_var, 0.0))

      q_diag = Nx.take_diagonal(q_matrix)
      q_sds = Nx.sqrt(Nx.max(q_diag, Nx.tensor(0.0)))
      sample_key = Nx.tensor(sample_key_row, type: Nx.type(base_key))
      [key_state_row, key_obs_row] = Nx.Random.split(sample_key, parts: 2) |> Nx.to_list()
      key_state = Nx.tensor(key_state_row, type: Nx.type(base_key))
      key_obs = Nx.tensor(key_obs_row, type: Nx.type(base_key))
      {proc_noise, _} = Nx.Random.normal(key_state, 0.0, 1.0, shape: {horizon, n_state})
      {obs_noise, _} = Nx.Random.normal(key_obs, 0.0, 1.0, shape: {horizon})
      proc_rows = proc_noise |> Nx.multiply(q_sds) |> Nx.to_list()
      obs_vals = Nx.to_flat_list(obs_noise)

      # Iterate h_list directly to avoid O(n²) Enum.at access
      {_, trajectory} =
        Enum.zip(h_rows, Enum.zip(proc_rows, obs_vals))
        |> Enum.reduce({Nx.flatten(final_state), []}, fn {h_row, {proc_row, z_obs}}, {state, acc} ->
          next_state = Nx.add(compat_dot(spec.f, state), Nx.tensor(proc_row, type: Nx.type(state)))
          y_mean = Nx.to_number(compat_dot(h_row, next_state))
          y = y_mean + z_obs * obs_sd
          {next_state, [y | acc]}
        end)

      Enum.reverse(trajectory)
    end)
  end

  defp predict_scalar(%{posterior_samples: samples}, horizon, base_key) do
    sample_key_rows = Nx.Random.split(base_key, parts: length(samples)) |> Nx.to_list()

    Enum.zip(samples, sample_key_rows)
    |> Enum.map(fn {sample, sample_key_row} ->
      final_state = Nx.to_number(List.last(sample.states))
      q = Nx.to_number(sample.process_var)
      r = Nx.to_number(sample.obs_var)
      sd_q = :math.sqrt(max(q, 0.0))
      sd_r = :math.sqrt(max(r, 0.0))

      sample_key = Nx.tensor(sample_key_row, type: Nx.type(base_key))
      [key_process_row, key_obs_row] = Nx.Random.split(sample_key, parts: 2) |> Nx.to_list()
      key_process = Nx.tensor(key_process_row, type: Nx.type(base_key))
      key_obs = Nx.tensor(key_obs_row, type: Nx.type(base_key))
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

    if n == 0 do
      %{
        mean: List.duplicate(0.0, horizon),
        sd: List.duplicate(0.0, horizon),
        lower: List.duplicate(0.0, horizon),
        upper: List.duplicate(0.0, horizon),
        horizon: horizon,
        alpha: alpha
      }
    else
      traj_t = Nx.tensor(trajectories, type: {:f, 64})
      means = Nx.mean(traj_t, axes: [0]) |> Nx.to_flat_list()

      sds =
        if n < 2 do
          List.duplicate(0.0, horizon)
        else
          Nx.standard_deviation(traj_t, axes: [0], ddof: 1) |> Nx.to_flat_list()
        end

      sorted = Nx.sort(traj_t, axis: 0)
      last = n - 1
      lower_idx = trunc(Float.floor(alpha / 2.0 * last))
      upper_idx = trunc(Float.ceil((1.0 - alpha / 2.0) * last))

      lower =
        sorted
        |> Nx.slice([max(lower_idx, 0), 0], [1, horizon])
        |> Nx.squeeze(axes: [0])
        |> Nx.to_flat_list()

      upper =
        sorted
        |> Nx.slice([min(upper_idx, last), 0], [1, horizon])
        |> Nx.squeeze(axes: [0])
        |> Nx.to_flat_list()

      %{
        mean: means,
        sd: sds,
        lower: lower,
        upper: upper,
        horizon: horizon,
        alpha: alpha
      }
    end
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

  defp h_to_row_tensor(%Nx.Tensor{} = h_t) do
    if Nx.rank(h_t) == 2, do: Nx.squeeze(h_t, axes: [0]), else: Nx.flatten(h_t)
  end

  # Nx.dot handles all rank combinations used here; keep scalar/scalar explicit.
  defp compat_dot(a, b) do
    case {Nx.rank(a), Nx.rank(b)} do
      {0, 0} ->
        Nx.multiply(a, b)

      _ ->
        Nx.dot(a, b)
    end
  end

  # Observation coercion delegated to ModelBuilder.coerce_obs/1
end
