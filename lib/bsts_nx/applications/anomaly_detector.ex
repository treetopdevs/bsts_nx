defmodule BstsNx.Applications.AnomalyDetector do
  @moduledoc """
  Anomaly detection using Bayesian Structural Time Series baselines.

  Detects anomalous observations by comparing actual values against a
  BSTS-fitted baseline model.  Observations that fall outside the
  posterior credible interval are flagged as anomalies with calibrated
  uncertainty scores.

  ## How it works

  1. **Fit baseline**: Learn normal behavior (trend + seasonality + noise)
     from historical data using MCMC or Kalman filtering
  2. **Score observations**: Compare each observation against the predicted
     distribution; compute z-scores and tail probabilities
  3. **Detect anomalies**: Flag observations where the credible interval
     is violated, with severity levels

  ## Advantages over ML anomaly detectors

  - **Calibrated uncertainty**: Anomaly scores are proper tail probabilities,
    not arbitrary scores
  - **Interpretable**: Decomposition shows *why* something is anomalous
    (trend break? seasonal deviation? noise spike?)
  - **Streaming-friendly**: Kalman filter updates in O(1) per observation
  - **Handles seasonality natively**: Won't flag known seasonal patterns

  ## Applications

  - **Infrastructure monitoring**: Server metrics, network traffic
  - **Financial surveillance**: Trading volume, price movements
  - **IoT/sensor data**: Equipment failure prediction
  - **E-commerce**: Unusual transaction patterns
  - **Cybersecurity**: Traffic anomalies

  ## Examples

      # Fit baseline on normal behavior
      detector = AnomalyDetector.fit(normal_traffic_data,
        seasonality: 24,  # hourly data with daily pattern
        alpha: 0.01       # 99% credible interval
      )

      # Score new observations
      scores = AnomalyDetector.score(detector, new_observations)
      # => [%{value: 150, predicted: 100, z_score: 5.2, anomaly?: true, severity: :critical}, ...]

      # Streaming: score one observation at a time
      {score, updated_detector} = AnomalyDetector.score_one(detector, 150.0)
  """

  alias BstsNx.GibbsSampler
  alias BstsNx.KalmanFilter
  alias BstsNx.ModelBuilder

  @type detector :: %{
          method: :mcmc | :filter,
          alpha: float(),
          z_threshold: float(),
          # MCMC fields
          posterior_mean: [float()] | nil,
          posterior_sd: [float()] | nil,
          spec: BstsNx.ModelSpec.t() | nil,
          # Filter fields (for streaming)
          filter_state: Nx.t() | nil,
          filter_cov: Nx.t() | nil,
          f: number() | Nx.t(),
          h: number() | Nx.t(),
          q: number() | Nx.t(),
          r: number() | Nx.t()
        }

  @type anomaly_score :: %{
          value: float(),
          predicted: float(),
          predicted_sd: float(),
          residual: float(),
          z_score: float(),
          p_value: float(),
          anomaly?: boolean(),
          severity: :normal | :warning | :critical
        }

  @doc """
  Fits a baseline model on historical "normal" data.

  ## Options

    * `:method` - `:mcmc` for full Bayesian (default) or `:filter` for
      streaming-optimized Kalman filter
    * `:seasonality` - seasonal periods (e.g., 24 for hourly with daily pattern)
    * `:regressors` - a `{T, p}` Nx tensor of known covariates (e.g., promotions,
      holidays) for the training period. When provided, known effects are modeled
      in the baseline so they don't trigger false anomalies.
    * `:alpha` - significance level for anomaly threshold (default: 0.01)
    * `:num_samples` - posterior draws for MCMC method (default: 200)
    * `:seed` - PRNG seed
    * `:model_spec` - custom model spec (overrides `:seasonality` and `:regressors`)

  ## Examples

      iex> detector = BstsNx.Applications.AnomalyDetector.fit(Enum.to_list(1..30), num_samples: 10, seed: 42)
      iex> detector.method
      :mcmc
      iex> is_list(detector.posterior_mean)
      true
  """
  @spec fit([number()], keyword()) :: detector()
  def fit(observations, opts \\ []) do
    if observations == [] do
      raise ArgumentError, "observations must be non-empty"
    end

    num_samples = Keyword.get(opts, :num_samples, 200)

    if not is_integer(num_samples) or num_samples <= 0 do
      raise ArgumentError, "num_samples must be a positive integer, got: #{num_samples}"
    end

    method = Keyword.get(opts, :method, :mcmc)
    alpha = Keyword.get(opts, :alpha, 0.01)

    if not is_number(alpha) or alpha <= 0.0 or alpha >= 1.0 do
      raise ArgumentError, "alpha must be between 0 and 1 (exclusive), got: #{inspect(alpha)}"
    end

    if method not in [:mcmc, :filter] do
      raise ArgumentError, "method must be :mcmc or :filter, got: #{inspect(method)}"
    end

    # Validate regressors dimensions if provided
    case Keyword.get(opts, :regressors) do
      nil ->
        :ok

      %Nx.Tensor{} = reg ->
        n = length(observations)

        if Nx.axis_size(reg, 0) != n do
          raise ArgumentError,
                "regressors rows (#{Nx.axis_size(reg, 0)}) must match observations (#{n})"
        end

      other ->
        raise ArgumentError,
              "regressors must be an Nx tensor, got: #{inspect(other)}"
    end

    z_threshold = BstsNx.Utils.z_score(alpha)

    case method do
      :mcmc -> fit_mcmc(observations, alpha, z_threshold, opts)
      :filter -> fit_filter(observations, alpha, z_threshold, opts)
    end
  end

  @doc """
  Scores a batch of observations against the baseline.

  For MCMC-fitted detectors, uses the posterior predictive distribution.
  For filter-fitted detectors, uses one-step-ahead predictions.

  Returns a list of `anomaly_score` maps, one per observation.
  """
  @spec score(detector(), [number()]) :: [anomaly_score()]
  def score(%{method: :mcmc} = detector, observations) do
    Enum.with_index(observations, fn obs, idx ->
      # Use the last fitted prediction as the baseline
      # For scoring new data, we extrapolate the fitted model
      predicted = extrapolate_mean(detector, idx)
      predicted_sd = extrapolate_sd(detector, idx)
      build_score(obs, predicted, predicted_sd, detector.z_threshold)
    end)
  end

  def score(%{method: :filter} = detector, observations) do
    {scores, _final_detector} =
      Enum.map_reduce(observations, detector, fn obs, det ->
        {sc, updated} = score_one(det, obs)
        {sc, updated}
      end)

    scores
  end

  @doc """
  Scores a single observation and updates the detector state.

  This is the streaming API — call this for each new observation.
  Returns `{anomaly_score, updated_detector}`.

  Only available for `:filter` method detectors.
  """
  @spec score_one(detector(), number()) :: {anomaly_score(), detector()}
  def score_one(%{method: :mcmc}, _observation) do
    raise ArgumentError,
          "score_one/2 is only available for :filter method detectors; " <>
            "use score/2 for MCMC detectors"
  end

  def score_one(%{method: :filter} = detector, observation) do
    %{filter_state: x, filter_cov: p, f: f, h: h, q: q, r: r} = detector

    # One-step prediction
    x_pred = f * x
    p_pred = f * p * f + q

    # Predicted observation
    y_pred = h * x_pred
    y_var = h * p_pred * h + r
    y_sd = :math.sqrt(max(y_var, 1.0e-10))

    score = build_score(observation, y_pred, y_sd, detector.z_threshold)

    # Kalman update
    innovation = observation - y_pred
    s = y_var
    k = if s > 1.0e-15, do: p_pred * h / s, else: 0.0

    new_x = x_pred + k * innovation
    new_p = max(p_pred * (1.0 - h * k), 0.0)

    updated = %{detector | filter_state: new_x, filter_cov: new_p}
    {score, updated}
  end

  @doc """
  Detects all anomalies in a sequence, returning only flagged observations.

  ## Options

    * `:min_severity` - minimum severity to return (`:warning` or `:critical`,
      default: `:warning`)
  """
  @spec detect(detector(), [number()], keyword()) :: [anomaly_score()]
  def detect(detector, observations, opts \\ []) do
    min_severity = Keyword.get(opts, :min_severity, :warning)

    scores = score(detector, observations)

    Enum.filter(scores, fn s ->
      s.anomaly? and severity_at_least?(s.severity, min_severity)
    end)
  end

  @doc """
  Returns a summary of anomaly detection results.

  ## Examples

      iex> scores = [
      ...>   %{value: 1.0, predicted: 1.0, predicted_sd: 0.1, residual: 0.0, z_score: 0.0, p_value: 1.0, anomaly?: false, severity: :normal},
      ...>   %{value: 10.0, predicted: 1.0, predicted_sd: 0.1, residual: 9.0, z_score: 90.0, p_value: 0.0, anomaly?: true, severity: :critical}
      ...> ]
      iex> summary = BstsNx.Applications.AnomalyDetector.summary(scores)
      iex> summary.total_observations
      2
      iex> summary.anomalies
      1
  """
  @spec summary([anomaly_score()]) :: map()
  def summary(scores) do
    total = length(scores)
    anomalies = Enum.count(scores, & &1.anomaly?)
    warnings = Enum.count(scores, fn s -> s.severity == :warning end)
    criticals = Enum.count(scores, fn s -> s.severity == :critical end)

    max_z =
      case scores do
        [] -> 0.0
        _ -> scores |> Enum.map(& &1.z_score) |> Enum.map(&abs/1) |> Enum.max()
      end

    %{
      total_observations: total,
      anomalies: anomalies,
      warnings: warnings,
      criticals: criticals,
      anomaly_rate: if(total > 0, do: anomalies / total, else: 0.0),
      max_abs_z_score: max_z
    }
  end

  # -- Private implementation --

  defp fit_mcmc(observations, alpha, z_threshold, opts) do
    obs_list = ModelBuilder.coerce_obs(observations)
    t = length(obs_list)
    num_samples = Keyword.get(opts, :num_samples, 200)
    burn_in = Keyword.get(opts, :burn_in, div(num_samples, 2))

    {spec, method_type} = ModelBuilder.build_spec(obs_list, opts)

    sampler_opts =
      opts
      |> Keyword.take([:seed, :key, :thin])
      |> Keyword.put(:burn_in, burn_in)

    samples =
      case method_type do
        :structured ->
          GibbsSampler.sample_structured(obs_list, spec, num_samples, sampler_opts)

        :scalar ->
          first = List.first(obs_list) || 0.0
          GibbsSampler.sample(obs_list, num_samples, first, 1.0, 1.0, 1.0, sampler_opts)
      end

    # Compute posterior predictive mean and sd at each training point
    n = length(samples)

    {means, sds} =
      case method_type do
        :structured ->
          compute_structured_predictions(samples, spec, t, n)

        :scalar ->
          compute_scalar_predictions(samples, t, n)
      end

    %{
      method: :mcmc,
      alpha: alpha,
      z_threshold: z_threshold,
      posterior_mean: means,
      posterior_sd: sds,
      spec: spec,
      filter_state: nil,
      filter_cov: nil,
      f: 1.0,
      h: 1.0,
      q: 1.0,
      r: 1.0
    }
  end

  defp fit_filter(observations, alpha, z_threshold, opts) do
    obs_list = ModelBuilder.coerce_obs(observations)
    t = length(obs_list)

    f = Keyword.get(opts, :f, 1.0)
    h = Keyword.get(opts, :h, 1.0)
    q = Keyword.get(opts, :q, 1.0)
    r = Keyword.get(opts, :r, 1.0)
    x0 = Keyword.get(opts, :x0, List.first(obs_list) || 0.0)
    p0 = Keyword.get(opts, :p0, 1.0)

    # Run filter through training data to get final state
    obs_tensor = Nx.tensor(obs_list, type: {:f, 32})
    {xs, ps} = KalmanFilter.filter_defn(obs_tensor, f, h, q, r, x0, p0)

    final_x = take_scalar_at(xs, t - 1)
    final_p = take_scalar_at(ps, t - 1)

    %{
      method: :filter,
      alpha: alpha,
      z_threshold: z_threshold,
      posterior_mean: nil,
      posterior_sd: nil,
      spec: nil,
      filter_state: Nx.to_number(final_x),
      filter_cov: Nx.to_number(final_p),
      f: f * 1.0,
      h: h * 1.0,
      q: q * 1.0,
      r: r * 1.0
    }
  end

  # Model building delegated to ModelBuilder.build_spec/2

  defp compute_structured_predictions(samples, spec, t, n) do
    h = spec.h
    states_t = samples |> Enum.map(&Nx.stack(&1.states)) |> Nx.stack()

    preds =
      if is_list(h) do
        h_rows = h |> Enum.map(&h_to_row_tensor/1) |> Nx.stack()
        Nx.sum(Nx.multiply(states_t, Nx.reshape(h_rows, {1, t, Nx.axis_size(h_rows, 1)})), axes: [2])
      else
        Nx.dot(states_t, h_to_row_tensor(h))
      end

    means = preds |> Nx.mean(axes: [0]) |> Nx.to_flat_list()

    sds =
      if n < 2 do
        List.duplicate(1.0, t)
      else
        preds
        |> Nx.standard_deviation(axes: [0], ddof: 1)
        |> Nx.to_flat_list()
        |> Enum.map(&max(&1, 1.0e-6))
      end

    {means, sds}
  end

  defp compute_scalar_predictions(samples, t, n) do
    states_t =
      samples
      |> Enum.map(fn s -> s.states |> Enum.map(&Nx.to_number/1) end)
      |> Nx.tensor(type: {:f, 64})

    means = states_t |> Nx.mean(axes: [0]) |> Nx.to_flat_list()

    sds =
      if n < 2 do
        List.duplicate(1.0, t)
      else
        states_t
        |> Nx.standard_deviation(axes: [0], ddof: 1)
        |> Nx.to_flat_list()
        |> Enum.map(&max(&1, 1.0e-6))
      end

    {means, sds}
  end

  defp h_to_row_tensor(%Nx.Tensor{} = h_t) do
    if Nx.rank(h_t) == 2, do: Nx.squeeze(h_t, axes: [0]), else: Nx.flatten(h_t)
  end

  defp extrapolate_mean(%{posterior_mean: means}, idx) when is_list(means) do
    # For new observations beyond training, use the last fitted value
    Enum.at(means, idx) || List.last(means) || 0.0
  end

  defp extrapolate_sd(%{posterior_sd: sds}, idx) when is_list(sds) do
    # For new observations, use the last fitted sd (grows with distance)
    base_sd = Enum.at(sds, idx) || List.last(sds) || 1.0
    # Increase uncertainty for extrapolation beyond training
    if idx >= length(sds) do
      extra_steps = idx - length(sds) + 1
      base_sd * :math.sqrt(1.0 + extra_steps * 0.1)
    else
      base_sd
    end
  end

  defp build_score(observed, predicted, predicted_sd, z_threshold) do
    residual = observed - predicted
    z = if predicted_sd > 1.0e-15, do: residual / predicted_sd, else: 0.0
    abs_z = abs(z)

    # Two-tailed p-value from z-score
    p_value = 2.0 * (1.0 - normal_cdf(abs_z))

    anomaly? = abs_z > z_threshold

    severity =
      cond do
        abs_z > z_threshold * 2.0 -> :critical
        abs_z > z_threshold -> :warning
        true -> :normal
      end

    %{
      value: observed,
      predicted: predicted,
      predicted_sd: predicted_sd,
      residual: residual,
      z_score: z,
      p_value: p_value,
      anomaly?: anomaly?,
      severity: severity
    }
  end

  defp severity_at_least?(:critical, _), do: true
  defp severity_at_least?(:warning, :warning), do: true
  defp severity_at_least?(:warning, :critical), do: false
  defp severity_at_least?(_, _), do: false

  defp normal_cdf(z) do
    0.5 * (1.0 + :math.erf(z / :math.sqrt(2.0)))
  end

  defp take_scalar_at(vec, idx) do
    Nx.slice(vec, [idx], [1]) |> Nx.squeeze()
  end

  # Observation coercion delegated to ModelBuilder.coerce_obs/1
end
