defmodule BstsNx.Forecaster do
  @moduledoc """
  Pure time series forecasting using Bayesian Structural Time Series.

  This module provides a `fit/predict` interface for forecasting future
  values with full uncertainty quantification via posterior credible
  intervals. Unlike `CausalImpact`, which measures the effect of a past
  intervention, `Forecaster` is concerned with predicting what will happen
  next.

  ## Why BSTS for forecasting?

  Compared to ARIMA or Prophet, BSTS forecasting provides:

  - **Proper Bayesian uncertainty**: credible intervals from the full
    posterior, not just point estimates with approximate confidence bands
  - **Component decomposition**: separate trend, seasonal, and regression
    contributions with per-component uncertainty
  - **External regressors**: incorporate causal drivers (promotions,
    weather, events) directly in the state-space model
  - **Missing data handling**: gracefully handles gaps in the training data
  - **Known measurement reliability**: relative observation-variance weights
    can downweight less reliable periods while retaining a learned common
    variance scale

  ## Workflow

      # 1. Fit model on historical data
      fit_result = Forecaster.fit(training_data,
        seasonality: 7,
        num_samples: 300
      )

      # 2. Forecast future periods
      forecast = Forecaster.predict(fit_result, horizon: 14)

      forecast.mean
      forecast.lower
      forecast.upper

      # Retain joint posterior paths when downstream decisions depend on
      # sums, thresholds, or nonlinear combinations of forecasts.
      forecast_with_draws =
        Forecaster.predict(fit_result,
          horizon: 14,
          return: :both,
          format: :tensors
        )

      forecast_with_draws.draws
  """

  alias BstsNx.Components
  alias BstsNx.Forward
  alias BstsNx.GibbsSampler
  alias BstsNx.ModelBuilder
  alias BstsNx.ModelSpec
  alias BstsNx.ObservationWeights
  alias BstsNx.Validation

  @type observation_variance_mode :: :scalar | :scaled_known

  @type fit_result :: %{
          posterior_samples: list(),
          spec: ModelSpec.t() | nil,
          training_length: non_neg_integer(),
          method: :structured | :scalar,
          n_regression_dims: non_neg_integer(),
          observation_variance_mode: observation_variance_mode()
        }

  @type forecast_summary :: %{
          mean: [float()] | Nx.t(),
          lower: [float()] | Nx.t(),
          upper: [float()] | Nx.t(),
          sd: [float()] | Nx.t(),
          horizon: non_neg_integer(),
          alpha: float()
        }

  @type forecast_draws :: %{
          draws: [[float()]] | Nx.t(),
          horizon: non_neg_integer(),
          num_draws: non_neg_integer()
        }

  @type forecast_result :: forecast_summary() | forecast_draws() | map()

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
    * `:observation_variance_weights` - optional positive list or rank-1 tensor
      of length `T`. The observation model becomes
      `R_t = sigma_r^2 * weight_t`; larger values downweight less reliable
      observations. The common scale `sigma_r^2` remains Bayesian and learned.
    * `:num_samples` - posterior MCMC draws (default: 200)
    * `:burn_in` - burn-in iterations (default: num_samples / 2)
    * `:seed` - integer PRNG seed for reproducibility

  Measurement weighting uses exact Gaussian prewhitening. Supplying weights to
  an otherwise scalar local-level model promotes that model to the structured
  sampler so the time-varying observation rows can be represented.

  ## Examples

      iex> fit = BstsNx.Forecaster.fit(Enum.to_list(1..20), num_samples: 10, seed: 42)
      iex> fit.method
      :scalar
      iex> fit.training_length
      20
      iex> fit.observation_variance_mode
      :scalar
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

    variance_weights =
      opts
      |> Keyword.get(:observation_variance_weights)
      |> ObservationWeights.normalize!(length(obs_list), :observation_variance_weights)

    {base_spec, base_method} = ModelBuilder.build_spec(obs_list, opts)

    {spec, method} =
      ensure_weightable_model(base_spec, base_method, obs_list, variance_weights, opts)

    {fit_observations, sampling_spec} =
      case variance_weights do
        nil -> {obs_list, spec}
        weights -> ObservationWeights.prewhiten(obs_list, spec, weights)
      end

    burn_in = Keyword.get(opts, :burn_in, div(num_samples, 2))

    sampler_opts =
      opts
      |> Keyword.take([:seed, :key, :thin])
      |> Keyword.put(:burn_in, burn_in)

    samples =
      case method do
        :structured ->
          GibbsSampler.sample_structured(
            fit_observations,
            sampling_spec,
            num_samples,
            sampler_opts
          )

        :scalar ->
          first = ModelBuilder.first_obs(fit_observations)

          GibbsSampler.sample(
            fit_observations,
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
      n_regression_dims: n_reg,
      observation_variance_mode: observation_variance_mode(variance_weights)
    }
  end

  @doc """
  Generates forecasts from a fitted model.

  ## Options

    * `:horizon` - number of future periods to forecast (required)
    * `:alpha` - significance level for credible intervals (default: 0.05)
    * `:seed` - PRNG seed for forecast simulation
    * `:future_regressors` - a `{horizon, p}` Nx tensor or list of future
      regressor values, where `p` must match the number of regressors used during
      `fit/2`. When provided, builds per-step observation matrices for
      regressor-aware forecasts. Without this option, the last training H is
      used for all forecast steps.
    * `:future_observation_variance_weights` - optional positive list or rank-1
      tensor of length `horizon`. Each value multiplies the learned observation
      variance for that future period. When training used
      `:observation_variance_weights` and this option is omitted, reference
      weights of `1.0` are used.
    * `:return` - `:summary` (default), `:draws`, or `:both`. Draws preserve
      dependence across forecast steps for sums, threshold probabilities, and
      nonlinear downstream calculations.
    * `:format` - `:lists` (default) or `:tensors` for summary vectors and draws.

  The default `:summary` + `:lists` result is backward compatible.
  """
  @spec predict(fit_result(), keyword()) :: forecast_result()
  def predict(%{method: method} = fit_result, opts) do
    horizon = Keyword.fetch!(opts, :horizon)

    if not is_integer(horizon) or horizon <= 0 do
      raise ArgumentError, "horizon must be a positive integer, got: #{inspect(horizon)}"
    end

    alpha = Keyword.get(opts, :alpha, 0.05)
    Validation.validate_alpha!(alpha, "alpha must be between 0 and 1 (exclusive)")

    return = normalize_return!(Keyword.get(opts, :return, :summary))
    format = normalize_format!(Keyword.get(opts, :format, :lists))
    seed = Keyword.get(opts, :seed, System.os_time())
    key = Keyword.get(opts, :key, Nx.Random.key(seed))
    n_reg = Map.get(fit_result, :n_regression_dims, 0)

    future_regressors =
      opts
      |> Keyword.get(:future_regressors)
      |> normalize_future_regressors!(horizon, n_reg)

    future_variance_weights =
      opts
      |> Keyword.get(:future_observation_variance_weights)
      |> ObservationWeights.normalize!(
        horizon,
        :future_observation_variance_weights
      )
      |> default_future_variance_weights(fit_result, horizon)

    trajectories =
      case method do
        :structured ->
          predict_structured(
            fit_result,
            horizon,
            key,
            future_regressors,
            future_variance_weights
          )

        :scalar ->
          predict_scalar(fit_result, horizon, key, future_variance_weights)
      end

    build_forecast_result(trajectories, horizon, alpha, return, format)
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
    {fit_opts, predict_opts_base} = split_fit_predict_prng_opts(opts)
    fit_result = fit(observations, fit_opts)

    predict_opts =
      predict_opts_base
      |> Keyword.put(:horizon, horizon)

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
          h_rows = h |> Enum.map(&BstsNx.Utils.h_to_row_tensor/1) |> Nx.stack()

          h_rows
          |> Nx.multiply(mean_states_t)
          |> Nx.sum(axes: [1])
          |> Nx.to_flat_list()
          |> Enum.take(t)

        true ->
          h_row = BstsNx.Utils.h_to_row_tensor(h)
          mean_states_t |> Nx.dot(h_row) |> Nx.to_flat_list() |> Enum.take(t)
      end

    %{
      fitted: fitted,
      states: mean_states
    }
  end

  # -- Private implementation --

  defp observation_variance_mode(nil), do: :scalar
  defp observation_variance_mode(_weights), do: :scaled_known

  defp ensure_weightable_model(spec, method, _observations, nil, _opts), do: {spec, method}

  defp ensure_weightable_model(nil, :scalar, observations, _weights, opts) do
    prior_shape = Keyword.get(opts, :prior_shape, 1.0)
    prior_scale = Keyword.get(opts, :prior_scale, 1.0)

    spec =
      Components.local_level_spec(
        initial_state: ModelBuilder.first_obs(observations),
        initial_cov: Keyword.get(opts, :initial_cov, 1.0),
        process_var: Keyword.get(opts, :process_var, 1.0),
        obs_var: Keyword.get(opts, :obs_var, 1.0),
        prior_shape: prior_shape,
        prior_scale: prior_scale
      )

    {spec, :structured}
  end

  defp ensure_weightable_model(spec, :structured, _observations, _weights, _opts),
    do: {spec, :structured}

  defp predict_structured(
         %{posterior_samples: samples, spec: spec, n_regression_dims: n_reg},
         horizon,
         base_key,
         future_regressors,
         future_variance_weights
       ) do
    h_list = future_h(spec, horizon, future_regressors, n_reg)

    forecast_h =
      case future_variance_weights do
        nil -> h_list
        weights -> ObservationWeights.prewhiten_h(h_list, weights)
      end

    trajectories =
      Forward.posterior_structured_trajectories(samples, spec, forecast_h, base_key)

    case future_variance_weights do
      nil -> trajectories
      weights -> ObservationWeights.restore_trajectories(trajectories, weights)
    end
  end

  defp future_h(spec, _horizon, %Nx.Tensor{} = future_regressors, n_reg) do
    ModelBuilder.build_future_h(spec, future_regressors, n_reg)
  end

  defp future_h(spec, horizon, nil, _n_reg) do
    static_h =
      case spec.h do
        list when is_list(list) -> List.last(list)
        %Nx.Tensor{} = tensor -> tensor
      end

    List.duplicate(static_h, horizon)
  end

  defp predict_scalar(%{posterior_samples: samples}, horizon, base_key, nil) do
    Forward.posterior_scalar_trajectories(samples, horizon, base_key)
  end

  defp predict_scalar(%{posterior_samples: samples}, horizon, base_key, weights) do
    spec = Components.local_level_spec()

    structured_samples =
      Enum.map(samples, fn sample ->
        %{
          states: sample.states,
          q_matrix:
            sample.process_var
            |> BstsNx.Utils.to_tensor()
            |> Nx.reshape({1, 1}),
          obs_var: sample.obs_var
        }
      end)

    forecast_h = ObservationWeights.prewhiten_h(spec.h, weights)

    structured_samples
    |> Forward.posterior_structured_trajectories(spec, forecast_h, base_key)
    |> ObservationWeights.restore_trajectories(weights)
    |> Enum.map(&Enum.take(&1, horizon))
  end

  defp normalize_future_regressors!(nil, _horizon, _n_reg), do: nil

  defp normalize_future_regressors!(future_regressors, horizon, n_reg) do
    future_t =
      try do
        ModelBuilder.ensure_tensor(future_regressors)
      rescue
        error in ArgumentError ->
          raise ArgumentError,
                "future_regressors must be an Nx tensor or rank-2 list, got: #{Exception.message(error)}"
      end

    if Nx.rank(future_t) != 2 do
      raise ArgumentError,
            "future_regressors must be rank-2, got shape #{inspect(Nx.shape(future_t))}"
    end

    if Nx.axis_size(future_t, 0) != horizon do
      raise ArgumentError,
            "future_regressors rows (#{Nx.axis_size(future_t, 0)}) must match horizon (#{horizon})"
    end

    if Nx.axis_size(future_t, 1) != n_reg do
      raise ArgumentError,
            "future_regressors columns (#{Nx.axis_size(future_t, 1)}) must match training regressors (#{n_reg})"
    end

    future_t
  end

  defp default_future_variance_weights(
         nil,
         %{observation_variance_mode: :scaled_known},
         horizon
       ) do
    List.duplicate(1.0, horizon)
  end

  defp default_future_variance_weights(weights, _fit_result, _horizon), do: weights

  defp split_fit_predict_prng_opts(opts) do
    case Keyword.get(opts, :key) do
      %Nx.Tensor{} = key ->
        split_keys = Nx.Random.split(key, parts: 2)
        fit_key = BstsNx.Utils.split_key_at(split_keys, 0)
        predict_key = BstsNx.Utils.split_key_at(split_keys, 1)

        {
          opts |> Keyword.put(:key, fit_key) |> Keyword.delete(:seed),
          opts |> Keyword.put(:key, predict_key) |> Keyword.delete(:seed)
        }

      _ ->
        case Keyword.get(opts, :seed) do
          seed when is_integer(seed) ->
            split_keys = Nx.Random.split(Nx.Random.key(seed), parts: 2)

            {
              opts
              |> Keyword.put(:key, BstsNx.Utils.split_key_at(split_keys, 0))
              |> Keyword.delete(:seed),
              opts
              |> Keyword.put(:key, BstsNx.Utils.split_key_at(split_keys, 1))
              |> Keyword.delete(:seed)
            }

          _ ->
            {opts, opts}
        end
    end
  end

  defp build_forecast_result(trajectories, horizon, alpha, return, format) do
    summary = aggregate_trajectories(trajectories, horizon, alpha, format)
    draws = format_draws(trajectories, horizon, format)
    num_draws = length(trajectories)

    case return do
      :summary ->
        summary

      :draws ->
        %{draws: draws, horizon: horizon, num_draws: num_draws}

      :both ->
        summary
        |> Map.put(:draws, draws)
        |> Map.put(:num_draws, num_draws)
    end
  end

  defp aggregate_trajectories(trajectories, horizon, alpha, format) do
    n = length(trajectories)

    summary =
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

    format_summary(summary, format)
  end

  defp format_summary(summary, :lists), do: summary

  defp format_summary(summary, :tensors) do
    summary
    |> Map.update!(:mean, &Nx.tensor(&1, type: {:f, 64}))
    |> Map.update!(:sd, &Nx.tensor(&1, type: {:f, 64}))
    |> Map.update!(:lower, &Nx.tensor(&1, type: {:f, 64}))
    |> Map.update!(:upper, &Nx.tensor(&1, type: {:f, 64}))
  end

  defp format_draws(trajectories, _horizon, :lists), do: trajectories

  defp format_draws([], horizon, :tensors) do
    Nx.broadcast(Nx.tensor(0.0, type: {:f, 64}), {0, horizon})
  end

  defp format_draws(trajectories, _horizon, :tensors) do
    Nx.tensor(trajectories, type: {:f, 64})
  end

  defp normalize_return!(value) when value in [:summary, :draws, :both], do: value

  defp normalize_return!(value) do
    raise ArgumentError,
          "return must be :summary, :draws, or :both, got: #{inspect(value)}"
  end

  defp normalize_format!(value) when value in [:lists, :tensors], do: value

  defp normalize_format!(value) do
    raise ArgumentError,
          "format must be :lists or :tensors, got: #{inspect(value)}"
  end

  defp infer_regression_dims(spec, opts, :structured) do
    case Keyword.get(opts, :regressors) do
      %Nx.Tensor{} = tensor ->
        Nx.axis_size(tensor, 1)

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
    |> BstsNx.Utils.h_to_row_tensor()
    |> Nx.to_flat_list()
  end
end
