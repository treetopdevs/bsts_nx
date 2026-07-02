defmodule BstsSite.Demos.Policy do
  @moduledoc """
  The policy demo: an interrupted time series. 48 months of monthly accident
  counts around a steady baseline; a speed-limit change lands at month 37 and
  its planted effect is a visitor-controlled *reduction* — the same estimator
  that finds marketing lift, pointed at a negative effect.

  `fit/2` runs `BstsNx.Applications.PolicyEvaluator.evaluate/3` on the scalar
  MCMC path (small sample count, fixed seed) — under a second on this server.
  """

  alias BstsNx.Applications.PolicyEvaluator
  alias BstsNx.Utils
  alias BstsSite.Demos.Scenarios

  # Slider bounds are the server-side contract: values outside are clamped.
  # Reductions are the point here, so the range is negative.
  @reduction_range -40..0
  @noise_range 3..15
  @default_reduction -18
  @default_noise 8

  @baseline 120.0
  @total 48
  @pre_end 36
  @post_months @total - @pre_end
  @data_seed 137
  @mcmc_seed 2024
  @num_samples 80
  @burn_in 40

  def reduction_range, do: @reduction_range
  def noise_range, do: @noise_range
  def defaults, do: %{reduction: @default_reduction, noise_sd: @default_noise}
  def num_samples, do: @num_samples

  @doc """
  48 months of accident counts around a steady #{trunc(@baseline)}/month, with
  a flat change of `reduction` accidents/month (zero or negative) planted from
  month #{@pre_end + 1}. The noise seed is fixed, so the sliders change the
  planted effect and the noise scale — never the noise realization. Kept
  within the local-level model class (no seasonality); the hub hero explains
  why.
  """
  def scenario(reduction_param, noise_param) do
    reduction = clamp(reduction_param, @reduction_range)
    noise_sd = clamp(noise_param, @noise_range)

    {observations, effects} =
      @total
      |> Scenarios.noise(noise_sd, @data_seed)
      |> Enum.with_index()
      |> Enum.map(fn {e, t} ->
        effect = if t >= @pre_end, do: reduction * 1.0, else: 0.0
        {@baseline + effect + e, effect}
      end)
      |> Enum.unzip()

    %{
      reduction: reduction,
      noise_sd: noise_sd,
      observations: observations,
      pre_end: @pre_end,
      pre_period: {1, @pre_end},
      post_period: {@pre_end + 1, @total},
      truth: %{
        change_per_month: reduction * 1.0,
        cumulative_change: Enum.sum(effects),
        post_months: @post_months
      }
    }
  end

  @doc """
  The interrupted time series analysis: `PolicyEvaluator.evaluate/3` fits a
  scalar local-level model to the 36 pre-change months by Gibbs sampling
  (#{@num_samples} draws after #{@burn_in} burn-in, fixed seed), projects the
  counterfactual over the 12 post-change months, and measures the gap.
  Regenerates the scenario from the slider values so the async task depends
  only on plain numbers.
  """
  def fit(reduction_param, noise_param) do
    scenario = scenario(reduction_param, noise_param)

    intervention = %{
      intervention_name: "speed_limit_change",
      intervention_date_index: @pre_end + 1,
      pre_period_start: 1,
      outcome_metric: "accidents/month"
    }

    result =
      PolicyEvaluator.evaluate(scenario.observations, intervention,
        num_samples: @num_samples,
        burn_in: @burn_in,
        seed: @mcmc_seed
      )

    {cf_mean, band_lower, band_upper} =
      posterior_band(result.analysis.impact.counterfactual)

    %{
      scenario: scenario,
      counterfactual_mean: cf_mean,
      band_lower: band_lower,
      band_upper: band_upper,
      cumulative_draws: result.analysis.impact.cumulative_effects,
      effect: result.effect,
      significant?: result.significant?,
      n_post: result.n_post,
      num_samples: @num_samples,
      execution: result.analysis.execution
    }
  end

  # Per-timestep mean and 95% percentile band across posterior
  # counterfactual draws (a draws x post-months list of lists).
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
    # Interrupted time series: fit the 36 pre-change months, project the
    # counterfactual across the 12 post-change months, measure the gap.
    # (~0.9 s measured on this server.)
    result =
      BstsNx.Applications.PolicyEvaluator.evaluate(
        monthly_accidents,
        %{
          intervention_name: "speed_limit_change",
          intervention_date_index: 37,  # the policy takes effect at month 37
          pre_period_start: 1,
          outcome_metric: "accidents/month"
        },
        num_samples: 80,
        burn_in: 40,
        seed: 2024
      )

    result.effect.cumulative   #=> negative when the policy prevented accidents
    result.effect.per_period   #=> average change per month
    result.significant?        #=> 95% CI excludes zero?
    result.report              #=> a plain-text evaluation report
    """
  end
end
