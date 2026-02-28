defmodule BstsNx.Applications.DemandForecaster do
  @moduledoc """
  Demand forecasting with causal drivers using Bayesian Structural Time Series.

  Forecasts future demand while incorporating external regressors (promotions,
  weather, events, pricing) and providing proper Bayesian credible intervals
  for inventory planning, capacity planning, and supply chain optimization.

  ## Why BSTS for demand forecasting?

  - **Uncertainty quantification**: Credible intervals for safety stock and
    capacity planning — critical for operations, not available from most
    ML forecasters
  - **Causal drivers**: Promotions, weather, events directly modeled as
    regression covariates with time-varying coefficients
  - **Component decomposition**: Separate trend, seasonal, and regressor
    contributions for business interpretability
  - **Robust to missing data**: Handles gaps in the training data gracefully

  ## Applications

  | Domain          | Regressors                          | Target            |
  |-----------------|-------------------------------------|-------------------|
  | Retail          | Promotions, holidays, weather       | Units sold        |
  | Energy          | Temperature, day type, pricing      | Load (MW)         |
  | Transportation  | Events, weather, day of week        | Rides / passengers|
  | Hospitality     | Events, seasonality, competition    | Room bookings     |
  | E-commerce      | Marketing spend, promotions         | Orders / GMV      |

  ## Examples

      # Basic demand forecast with seasonality
      forecast = DemandForecaster.forecast(
        historical_demand,
        horizon: 14,
        seasonality: 7
      )

      forecast.mean       # point forecasts
      forecast.lower      # lower bound (e.g., 5th percentile for safety stock)
      forecast.upper      # upper bound (e.g., 95th percentile for capacity)

      # With external regressors (promotions + weather)
      forecast = DemandForecaster.forecast(
        historical_demand,
        horizon: 14,
        seasonality: 7,
        regressors: %{
          training: promo_and_weather_history,
          future: promo_and_weather_forecast
        }
      )

      # Safety stock calculation
      safety = DemandForecaster.safety_stock(forecast, service_level: 0.95)
  """

  alias BstsNx.Components
  alias BstsNx.Forecaster
  alias BstsNx.GibbsSampler
  alias BstsNx.ModelBuilder

  @type forecast_result :: %{
          mean: [float()],
          lower: [float()],
          upper: [float()],
          sd: [float()],
          horizon: non_neg_integer(),
          alpha: float(),
          components: map() | nil
        }

  @doc """
  Forecasts future demand with optional external regressors.

  ## Parameters

    * `observations` - historical demand time series (list of numbers)

  ## Options

    * `:horizon` - number of future periods to forecast (required)
    * `:seasonality` - seasonal periods (e.g., 7 for weekly, 12 for monthly,
      52 for yearly). Can be a single integer or list for multiple seasonalities.
    * `:regressors` - map with `:training` (a `{T, p}` tensor or list-of-lists
      of historical regressor values) and `:future` (a `{horizon, p}` tensor
      or list-of-lists of future regressor values)
    * `:alpha` - significance level for credible intervals (default: 0.05)
    * `:num_samples` - posterior MCMC draws (default: 200)
    * `:seed` - PRNG seed for reproducibility
  """
  @spec forecast([number()], keyword()) :: forecast_result()
  def forecast(observations, opts \\ []) do
    {forecast_result, _samples, _spec, _obs_list} = do_forecast(observations, opts)
    Map.put(forecast_result, :components, nil)
  end

  @doc """
  Fits and returns both the forecast and a component decomposition.

  The decomposition separates trend, seasonal, and regressor contributions
  for business interpretability.

  Returns `{forecast_result, decomposition}`.
  """
  @spec forecast_with_decomposition([number()], keyword()) ::
          {forecast_result(), map()}
  def forecast_with_decomposition(observations, opts) do
    {forecast_result, samples, spec, obs_list} = do_forecast(observations, opts)
    forecast_result = Map.put(forecast_result, :components, nil)

    fit_result = %{
      posterior_samples: samples,
      spec: spec,
      training_length: length(obs_list),
      method: :structured
    }

    decomp = Forecaster.decompose(fit_result) || %{fitted: [], states: []}

    {forecast_result, decomp}
  end

  @doc """
  Computes safety stock levels from forecast standard deviations.

  Safety stock per period is `z * sd`, where `z` is the one-sided
  z-quantile corresponding to the target service level.  The total
  is aggregated over the replenishment lead time.

  ## Options

    * `:service_level` - target service level (default: 0.95).
      Uses the one-sided z-quantile: `z = z_score(2 * (1 - service_level))`.
    * `:lead_time` - number of periods in the replenishment lead time
      (default: 1). Safety stock is summed over this many periods.

  ## Examples

      iex> forecast = %{mean: [100.0, 110.0, 105.0], upper: [120.0, 130.0, 125.0], lower: [80.0, 90.0, 85.0], sd: [10.0, 10.0, 10.0], horizon: 3, alpha: 0.05}
      iex> ss = BstsNx.Applications.DemandForecaster.safety_stock(forecast, lead_time: 2)
      iex> ss.service_level
      0.95
      iex> length(ss.per_period)
      3
  """
  @spec safety_stock(forecast_result(), keyword()) :: %{
          per_period: [float()],
          total: float(),
          service_level: float()
        }
  def safety_stock(forecast, opts \\ []) do
    service_level = Keyword.get(opts, :service_level, 0.95)
    lead_time = Keyword.get(opts, :lead_time, 1)

    # One-sided z-quantile for the service level
    # service_level 0.95 → alpha_two_tailed 0.10 → z ≈ 1.645
    alpha_two_tailed = 2.0 * (1.0 - service_level)
    z = BstsNx.Utils.z_score(alpha_two_tailed)

    # Safety stock per period = z * forecast sd
    per_period = Enum.map(forecast.sd, fn s -> max(z * s, 0.0) end)

    # Aggregate over lead time
    total =
      per_period
      |> Enum.take(lead_time)
      |> Enum.sum()

    %{
      per_period: per_period,
      total: total,
      service_level: service_level
    }
  end

  @doc """
  Evaluates the impact of a promotion on demand.

  Compares demand during a promotion period against the counterfactual
  baseline to estimate incremental demand from the promotion.

  Delegates to `BstsNx.InterventionAnalysis` internally.
  """
  @spec promotion_impact([number()], map(), keyword()) ::
          BstsNx.InterventionAnalysis.analysis_result()
  def promotion_impact(observations, promotion, opts \\ []) do
    config = %{
      pre_period: {promotion.baseline_start, promotion.baseline_end},
      post_period: {promotion.promo_start, promotion.promo_end}
    }

    BstsNx.InterventionAnalysis.analyze(observations, config, opts)
  end

  # -- Private implementation --

  defp do_forecast(observations, opts) do
    if observations == [] do
      raise ArgumentError, "observations must be non-empty"
    end

    horizon = Keyword.fetch!(opts, :horizon)

    if not is_integer(horizon) or horizon <= 0 do
      raise ArgumentError, "horizon must be a positive integer, got: #{horizon}"
    end

    alpha = Keyword.get(opts, :alpha, 0.05)
    num_samples = Keyword.get(opts, :num_samples, 200)

    if not is_number(alpha) or alpha <= 0.0 or alpha >= 1.0 do
      raise ArgumentError, "alpha must be between 0 and 1 (exclusive), got: #{inspect(alpha)}"
    end

    if not is_integer(num_samples) or num_samples <= 0 do
      raise ArgumentError, "num_samples must be a positive integer, got: #{inspect(num_samples)}"
    end

    seed = Keyword.get(opts, :seed, System.os_time())
    key = Keyword.get(opts, :key, Nx.Random.key(seed))

    obs_list = ModelBuilder.coerce_obs(observations)
    regressors = Keyword.get(opts, :regressors)
    seasonality = Keyword.get(opts, :seasonality)

    {spec, future_h} = build_demand_model(obs_list, horizon, seasonality, regressors)

    burn_in = Keyword.get(opts, :burn_in, div(num_samples, 2))

    sampler_opts =
      opts
      |> Keyword.take([:seed, :thin])
      |> Keyword.put(:burn_in, burn_in)

    samples = GibbsSampler.sample_structured(obs_list, spec, num_samples, sampler_opts)

    # Forward simulation with future regressors
    key_rows = Nx.Random.split(key, parts: length(samples)) |> Nx.to_list()

    trajectories =
      Enum.zip(samples, key_rows)
      |> Enum.map(fn {sample, key_row} ->
        sample_key = Nx.tensor(key_row, type: Nx.type(key))
        forward_simulate_demand(sample, spec, future_h, horizon, sample_key)
      end)

    result = aggregate(trajectories, horizon, alpha)

    {result, samples, spec, obs_list}
  end

  defp build_demand_model(obs_list, horizon, seasonality, regressors) do
    first = List.first(obs_list) || 0.0
    t = length(obs_list)

    # Base: local linear trend
    trend =
      Components.local_linear_trend_spec(
        initial_level: first,
        initial_cov_level: 10.0,
        var_level: 0.1,
        var_slope: 0.01
      )

    # Add seasonality if specified
    spec =
      case seasonality do
        nil ->
          trend

        s when is_integer(s) and s >= 2 ->
          Components.compose_specs(trend, Components.seasonal_spec(s))

        seasons when is_list(seasons) ->
          Enum.reduce(seasons, trend, fn s, acc ->
            Components.compose_specs(acc, Components.seasonal_spec(s))
          end)
      end

    # Add regressors if provided
    case regressors do
      nil ->
        {spec, nil}

      %{training: training, future: future} ->
        train_t = ensure_tensor(training)
        future_t = ensure_tensor(future)

        {_train_rows, p} = Nx.shape(train_t)

        # Validate training dimensions
        if Nx.axis_size(train_t, 0) != t do
          raise ArgumentError,
                "training regressors rows (#{Nx.axis_size(train_t, 0)}) must match observations (#{t})"
        end

        # Validate future dimensions
        if Nx.axis_size(future_t, 1) != p do
          raise ArgumentError,
                "future regressors columns (#{Nx.axis_size(future_t, 1)}) must match training regressors (#{p})"
        end

        if Nx.axis_size(future_t, 0) != horizon do
          raise ArgumentError,
                "future regressors rows (#{Nx.axis_size(future_t, 0)}) must match horizon (#{horizon})"
        end

        reg_spec = Components.regression_spec(train_t)
        combined = Components.compose_specs(spec, reg_spec)

        # Build future H using shared helper
        future_h = ModelBuilder.build_future_h(combined, future_t, p)

        {combined, future_h}
    end
  end

  defp forward_simulate_demand(sample, spec, future_h, horizon, key) do
    n_state = Nx.axis_size(spec.f, 0)
    final_state = List.last(sample.states)
    q_matrix = sample.q_matrix
    obs_var = Nx.to_number(sample.obs_var)
    obs_sd = :math.sqrt(max(obs_var, 0.0))

    q_diag = Nx.take_diagonal(q_matrix)
    q_sds = Nx.sqrt(Nx.max(q_diag, Nx.tensor(0.0)))

    subkey_rows = Nx.Random.split(key, parts: horizon * 2) |> Nx.to_list()
    state_key_rows = Enum.take_every(subkey_rows, 2)
    obs_key_rows = subkey_rows |> Enum.drop(1) |> Enum.take_every(2)

    # Resolve H for each future step
    h_list = resolve_future_h(spec, future_h, horizon, n_state)

    # Iterate h_list directly to avoid O(n²) Enum.at access
    {_, trajectory} =
      Enum.zip(h_list, Enum.zip(state_key_rows, obs_key_rows))
      |> Enum.reduce({Nx.flatten(final_state), []}, fn {h_t, {state_key_row, obs_key_row}},
                                                       {state, acc} ->
        key_state = Nx.tensor(state_key_row, type: Nx.type(key))
        key_obs = Nx.tensor(obs_key_row, type: Nx.type(key))

        {z_state, _} = Nx.Random.normal(key_state, 0.0, 1.0, shape: {n_state})
        noise = Nx.multiply(z_state, q_sds)
        next_state = Nx.add(Nx.dot(spec.f, state), noise)

        h_row = if Nx.rank(h_t) == 2, do: Nx.squeeze(h_t, axes: [0]), else: Nx.flatten(h_t)
        y_mean = Nx.to_number(Nx.dot(h_row, next_state))

        {z_obs, _} = Nx.Random.normal(key_obs, 0.0, 1.0)
        y = y_mean + Nx.to_number(z_obs) * obs_sd

        {next_state, [y | acc]}
      end)

    Enum.reverse(trajectory)
  end

  defp resolve_future_h(_spec, future_h, _horizon, _n_state) when is_list(future_h), do: future_h

  defp resolve_future_h(spec, nil, horizon, _n_state) do
    # No future regressors — use static H
    h =
      case spec.h do
        list when is_list(list) -> List.last(list)
        %Nx.Tensor{} = t -> t
      end

    List.duplicate(h, horizon)
  end

  defp aggregate(trajectories, horizon, alpha) do
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

  defp ensure_tensor(v), do: ModelBuilder.ensure_tensor(v)
end
