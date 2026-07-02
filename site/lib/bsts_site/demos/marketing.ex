defmodule BstsSite.Demos.Marketing do
  @moduledoc """
  The marketing demo: 90 days of daily conversions with a campaign planted
  at day 61 at a visitor-controlled lift.

  Two lanes answer the same question on the same data:

    * `fast_lane/1` — the operational forecast filter, run on mount and on
      every slider drag. Milliseconds, but it is handed the noise level.
    * `fit/2` — the full Bayesian analysis via
      `BstsNx.Applications.MarketingLift.measure_lift/3`. About a second on
      this server; learns the level and both variances from the pre-period.
  """

  alias BstsNx.Applications.MarketingLift
  alias BstsNx.{Components, Operational, Utils}
  alias BstsSite.Demos.Scenarios

  # Slider bounds are the server-side contract: values outside are clamped.
  @lift_range 0..60
  @noise_range 5..25
  @default_lift 25
  @default_noise 10

  @baseline 200.0
  @total 90
  @pre_end 60
  @post_days @total - @pre_end
  @data_seed 137
  @mcmc_seed 2024
  @num_samples 80
  @burn_in 40

  def lift_range, do: @lift_range
  def noise_range, do: @noise_range
  def defaults, do: %{lift: @default_lift, noise_sd: @default_noise}
  def num_samples, do: @num_samples

  @doc """
  90 days of daily conversions around a steady #{trunc(@baseline)}/day, with a
  flat step of `lift` conversions/day planted from day #{@pre_end + 1}. The
  noise seed is fixed, so the sliders change the planted effect and the noise
  scale — never the noise realization. Deliberately within the local-level
  model class (no seasonality); the hub hero explains why.
  """
  def scenario(lift_param, noise_param) do
    lift = clamp(lift_param, @lift_range)
    noise_sd = clamp(noise_param, @noise_range)

    {observations, effects} =
      @total
      |> Scenarios.noise(noise_sd, @data_seed)
      |> Enum.with_index()
      |> Enum.map(fn {e, t} ->
        effect = if t >= @pre_end, do: lift * 1.0, else: 0.0
        {@baseline + effect + e, effect}
      end)
      |> Enum.unzip()

    %{
      lift: lift,
      noise_sd: noise_sd,
      observations: observations,
      pre_end: @pre_end,
      pre_period: {1, @pre_end},
      post_period: {@pre_end + 1, @total},
      truth: %{
        lift_per_day: lift * 1.0,
        cumulative_lift: Enum.sum(effects),
        post_days: @post_days
      }
    }
  end

  @doc """
  The instant lane: a forecast-first Kalman filter that projects the
  pre-campaign baseline forward without ever seeing post-campaign data.
  Runs in single-digit milliseconds, so it executes inline on every
  slider drag. Its observation variance tracks the noise slider — this
  lane is told how noisy the world is; the Bayesian lane must learn it.
  """
  def fast_lane(scenario) do
    spec =
      Components.local_level_spec(
        initial_state: @baseline,
        initial_cov: 25.0,
        process_var: 1.0,
        obs_var: scenario.noise_sd * scenario.noise_sd * 1.0
      )

    prepared = Operational.prepare(spec, scenario.pre_period, scenario.post_period)
    result = Operational.run(prepared, scenario.observations, [], return: :lists)

    band_z = 1.96

    {band_lower, band_upper} =
      [result.summary.baseline, result.counterfactual.variance]
      |> Enum.zip_with(fn [mean, var] ->
        sd = :math.sqrt(max(var, 0.0))
        {mean - band_z * sd, mean + band_z * sd}
      end)
      |> Enum.unzip()

    %{
      counterfactual_mean: result.summary.baseline,
      band_lower: band_lower,
      band_upper: band_upper,
      cumulative: result.summary.cumulative_effect,
      relative: result.summary.relative_effect,
      execution: result.execution,
      significant?: interval_excludes_zero?(result.summary.cumulative_effect)
    }
  end

  @doc """
  The full Bayesian lane: `MarketingLift.measure_lift/3` fits a scalar
  local-level model to the 60 pre-campaign days by Gibbs sampling
  (#{@num_samples} draws after #{@burn_in} burn-in, fixed seed), then draws
  counterfactual futures for the campaign window. Regenerates the scenario
  from the slider values so the async task depends only on plain numbers.
  """
  def fit(lift_param, noise_param) do
    scenario = scenario(lift_param, noise_param)

    campaign = %{
      name: "spring_push",
      channel: :paid_social,
      start_index: @pre_end + 1,
      end_index: @total,
      baseline_start: 1,
      baseline_end: @pre_end
    }

    result =
      MarketingLift.measure_lift(scenario.observations, campaign,
        num_samples: @num_samples,
        burn_in: @burn_in,
        seed: @mcmc_seed
      )

    {cf_mean, band_lower, band_upper} =
      posterior_band(result.analysis.impact.counterfactual)

    baseline_total = result.baseline_mean * @post_days

    %{
      scenario: scenario,
      counterfactual_mean: cf_mean,
      band_lower: band_lower,
      band_upper: band_upper,
      cumulative_draws: result.analysis.impact.cumulative_effects,
      effect: result.effect,
      relative_lower: safe_ratio(result.effect.cumulative_lower, baseline_total),
      relative_upper: safe_ratio(result.effect.cumulative_upper, baseline_total),
      significant?: result.significant?,
      num_samples: @num_samples,
      execution: result.analysis.execution
    }
  end

  # Per-timestep mean and 95% percentile band across posterior
  # counterfactual draws (a draws x post-days list of lists).
  defp posterior_band(draws) do
    draws
    |> Utils.transpose_rows()
    |> Enum.map(fn column ->
      n = length(column)
      mean = Enum.sum(column) / n
      {lower, upper} = column |> Enum.sort() |> Utils.percentile_interval(n, 0.05)
      {mean, lower, upper}
    end)
    |> Enum.reduce({[], [], []}, fn {m, l, u}, {ms, ls, us} ->
      {[m | ms], [l | ls], [u | us]}
    end)
    |> then(fn {ms, ls, us} ->
      {Enum.reverse(ms), Enum.reverse(ls), Enum.reverse(us)}
    end)
  end

  defp interval_excludes_zero?(%{lower: lower, upper: upper})
       when is_number(lower) and is_number(upper) do
    lower > 0.0 or upper < 0.0
  end

  defp interval_excludes_zero?(_), do: false

  defp safe_ratio(_num, denom) when abs(denom) < 1.0e-10, do: 0.0
  defp safe_ratio(num, denom), do: num / denom

  def clamp(value, range) when is_integer(value) do
    value |> max(range.first) |> min(range.last)
  end

  def clamp(value, range) when is_binary(value) do
    case Integer.parse(value) do
      {n, _} -> clamp(n, range)
      :error -> range.first
    end
  end

  def clamp(_value, range), do: range.first

  @doc "The code this demo runs, shown verbatim in the under-the-hood panel."
  def code_snippet do
    ~S"""
    # Lane 1 — the instant first impression (milliseconds, measured on this
    # server). A Kalman filter projects the pre-campaign baseline forward;
    # it never sees the post-campaign data, but we do hand it the noise level.
    spec =
      BstsNx.Components.local_level_spec(
        initial_state: 200.0,
        initial_cov: 25.0,
        process_var: 1.0,
        obs_var: 100.0  # tracks the noise slider: sd²
      )

    prepared = BstsNx.Operational.prepare(spec, {1, 60}, {61, 90})
    fast = BstsNx.Operational.run(prepared, daily_conversions, [], return: :lists)
    fast.execution.method_used  #=> :scalar_forecast_filter

    # Lane 2 — the full Bayesian analysis (~1.4 s measured on this server).
    # Gibbs sampling learns the level and both variances from the pre-period,
    # then draws 80 counterfactual futures for the campaign window.
    result =
      BstsNx.Applications.MarketingLift.measure_lift(
        daily_conversions,
        %{
          name: "spring_push",
          channel: :paid_social,
          start_index: 61,
          end_index: 90,
          baseline_start: 1,
          baseline_end: 60
        },
        num_samples: 80,
        burn_in: 40,
        seed: 2024
      )

    result.effect.cumulative                   #=> incremental conversions
    result.significant?                        #=> 95% CI excludes zero?
    result.analysis.impact.cumulative_effects  #=> 80 posterior draws (the histogram)
    """
  end
end
