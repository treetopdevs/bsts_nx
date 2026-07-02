defmodule BstsSite.Demos.Anomaly do
  @moduledoc """
  The ops-monitoring demo: is this spike an anomaly?

  120 hours of normal server latency train a streaming Kalman baseline
  (`AnomalyDetector.fit(method: :filter)`). A 36-hour monitoring window is
  then scored one observation at a time with one-step-ahead predictions.
  The visitor plants spikes into the window and picks an alpha; the
  detector refits and rescores in a few milliseconds, so everything runs
  inline on every click.

  The latency series is a slow random-walk level plus Gaussian noise —
  deliberately inside the scalar filter's model class, so the detection
  demo is honest: no hidden model mismatch inflating or deflating scores.
  """

  alias BstsNx.Applications.AnomalyDetector
  alias BstsSite.Demos.Scenarios

  @history_hours 120
  @window_hours 36
  @total @history_hours + @window_hours

  @level 120.0
  @drift_sd 0.3
  @noise_sd 4.0

  # Slider/select bounds are the server-side contract: values outside clamp.
  @hour_range 0..(@window_hours - 1)
  @magnitude_range 0..60
  @alphas [0.01, 0.05, 0.10]
  @default_alpha 0.05
  @max_anomalies 6

  def hour_range, do: @hour_range
  def magnitude_range, do: @magnitude_range
  def alphas, do: @alphas
  def default_alpha, do: @default_alpha
  def history_hours, do: @history_hours
  def window_hours, do: @window_hours
  def max_anomalies, do: @max_anomalies

  @doc "The spikes planted on page load: one obvious, one borderline."
  def default_anomalies do
    [%{hour: 8, magnitude: 30}, %{hour: 23, magnitude: 9}]
  end

  @doc """
  Fits the detector on the clean 120-hour history and scores the 36-hour
  monitoring window with the visitor's planted spikes injected. Returns
  everything the page needs, including measured execution metadata.
  """
  def run(anomalies, alpha) do
    alpha = clamp_alpha(alpha)
    anomalies = normalize(anomalies)

    observations = inject(base_series(), anomalies)
    {history, window} = Enum.split(observations, @history_hours)

    t0 = System.monotonic_time(:microsecond)

    detector =
      AnomalyDetector.fit(history,
        method: :filter,
        alpha: alpha,
        q: @drift_sd * @drift_sd,
        r: @noise_sd * @noise_sd
      )

    t1 = System.monotonic_time(:microsecond)
    scores = AnomalyDetector.score(detector, window)
    t2 = System.monotonic_time(:microsecond)

    summary = AnomalyDetector.summary(scores)
    threshold = detector.z_threshold

    flagged_hours = for {s, h} <- Enum.with_index(scores), s.anomaly?, do: h
    planted_hours = Enum.map(anomalies, & &1.hour)
    hits = Enum.filter(planted_hours, &(&1 in flagged_hours))
    misses = planted_hours -- hits
    false_alarms = flagged_hours -- planted_hours

    steady_sd = List.last(scores).predicted_sd

    %{
      alpha: alpha,
      threshold: threshold,
      anomalies: anomalies,
      observations: observations,
      window_start: @history_hours,
      predicted: Enum.map(scores, & &1.predicted),
      band_lower: Enum.map(scores, &(&1.predicted - threshold * &1.predicted_sd)),
      band_upper: Enum.map(scores, &(&1.predicted + threshold * &1.predicted_sd)),
      z_scores: Enum.map(scores, & &1.z_score),
      scores: scores,
      flagged_hours: flagged_hours,
      hits: hits,
      misses: misses,
      false_alarms: false_alarms,
      tolerance_ms: threshold * steady_sd,
      expected_false_alarms: alpha * @window_hours,
      summary: summary,
      execution: %{
        method_used: detector.method,
        elapsed_ms: Float.round((t2 - t0) / 1000, 2),
        fit_ms: Float.round((t1 - t0) / 1000, 2),
        score_ms: Float.round((t2 - t1) / 1000, 2)
      }
    }
  end

  @doc """
  The deterministic base series: a slow random-walk level around
  #{@level} ms plus observation noise. Fixed seeds — every visitor
  sees the same 156 hours.
  """
  def base_series do
    drift = Scenarios.noise(@total, @drift_sd, 3992)
    noise = Scenarios.noise(@total, @noise_sd, 4005)

    {series, _level} =
      Enum.zip(drift, noise)
      |> Enum.map_reduce(@level, fn {d, e}, level ->
        level = level + d
        {level + e, level}
      end)

    series
  end

  defp inject(base, anomalies) do
    bump = Map.new(anomalies, fn a -> {@history_hours + a.hour, a.magnitude * 1.0} end)
    Enum.with_index(base, fn v, t -> v + Map.get(bump, t, 0.0) end)
  end

  @doc "Adds a spike, replacing any at the same hour; keeps the newest #{@max_anomalies}."
  def add_anomaly(list, hour, magnitude) do
    hour = clamp_hour(hour)
    magnitude = clamp_magnitude(magnitude)
    rest = Enum.reject(list, &(&1.hour == hour))
    Enum.take(rest ++ [%{hour: hour, magnitude: magnitude}], -@max_anomalies)
  end

  def remove_anomaly(list, hour) do
    hour = clamp_hour(hour)
    Enum.reject(list, &(&1.hour == hour))
  end

  defp normalize(anomalies) do
    anomalies
    |> Enum.map(fn a ->
      %{hour: clamp_hour(a.hour), magnitude: clamp_magnitude(a.magnitude)}
    end)
    |> Enum.reverse()
    |> Enum.uniq_by(& &1.hour)
    |> Enum.sort_by(& &1.hour)
    |> Enum.take(@max_anomalies)
  end

  def clamp_hour(value), do: clamp_int(value, @hour_range)
  def clamp_magnitude(value), do: clamp_int(value, @magnitude_range)

  def clamp_alpha(alpha) when is_float(alpha) do
    if alpha in @alphas, do: alpha, else: @default_alpha
  end

  def clamp_alpha(alpha) when is_binary(alpha) do
    case Float.parse(alpha) do
      {f, _} -> clamp_alpha(f)
      :error -> @default_alpha
    end
  end

  def clamp_alpha(_), do: @default_alpha

  defp clamp_int(value, range) when is_integer(value) do
    value |> max(range.first) |> min(range.last)
  end

  defp clamp_int(value, range) when is_binary(value) do
    case Integer.parse(value) do
      {n, _} -> clamp_int(n, range)
      :error -> range.first
    end
  end

  defp clamp_int(_value, range), do: range.first

  @doc "The code this demo runs, shown verbatim in the under-the-hood panel."
  def code_snippet do
    ~S"""
    # Fit the streaming baseline on 120 hours of normal latency.
    detector =
      BstsNx.Applications.AnomalyDetector.fit(history,
        method: :filter,   # scalar Kalman lane — no MCMC anywhere
        alpha: 0.05,       # flag when |z| > 1.96 (two-tailed)
        q: 0.09,           # how fast the true level may drift, per hour
        r: 16.0            # observation noise variance (sd 4 ms)
      )

    # Score the 36-hour monitoring window one step ahead at a time.
    scores = BstsNx.Applications.AnomalyDetector.score(detector, window)

    # Each score is the filter's one-step prediction error, scaled by
    # the sd the filter itself reports — a calibrated z-score.
    Enum.at(scores, 8)
    #=> %{value: 150.7, predicted: 120.8, predicted_sd: 4.2, residual: 29.9,
    #     z_score: 7.2, p_value: 0.0, anomaly?: true, severity: :critical}

    BstsNx.Applications.AnomalyDetector.summary(scores)
    #=> %{anomalies: 2, criticals: 1, warnings: 1, max_abs_z_score: 7.2, ...}
    """
  end
end
