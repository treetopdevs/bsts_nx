defmodule BstsSite.Demos.Counterfactual do
  @moduledoc """
  The /engine/counterfactual demo: run the scalar MCMC estimator once,
  keep all posterior draws, and draw them as spaghetti — each thin line
  one possible what-would-have-happened.

  `BstsNx.CausalImpact.estimate/4` returns the per-draw counterfactual
  trajectories in `result.counterfactual` (one post-period-length list
  per retained Gibbs draw), so the uncertainty on screen is the raw
  posterior, not a stylized band. The alpha slider never refits:
  `CausalImpact.summary(result, alpha: a)` recomputes percentile bounds
  from the stored draws in about a millisecond.
  """

  alias BstsNx.CausalImpact
  alias BstsSite.Demos.Scenarios

  @effect_range 0..20
  @alpha_min 0.05
  @alpha_max 0.5

  @total 120
  @pre_end 90
  @n_post @total - @pre_end
  @baseline 50.0
  @noise_sd 2.5
  @noise_seed 7
  @mcmc_seed 2024
  @num_samples 60
  @burn_in 30
  @spaghetti_count 40

  def effect_range, do: @effect_range
  def alpha_min, do: @alpha_min
  def alpha_max, do: @alpha_max
  def pre_end, do: @pre_end
  def n_post, do: @n_post
  def num_samples, do: @num_samples
  def spaghetti_count, do: @spaghetti_count

  @doc """
  120 points around a flat baseline of #{@baseline}, with a clean step of
  `effect` planted from index #{@pre_end} (day #{@pre_end + 1}) onward.
  The noise realization is fixed, so the effect slider moves only the
  planted step.
  """
  def scenario(effect) do
    effect = clamp_effect(effect)
    noise = Scenarios.noise(@total, @noise_sd, @noise_seed)

    observations =
      noise
      |> Enum.with_index()
      |> Enum.map(fn {e, t} ->
        step = if t >= @pre_end, do: effect * 1.0, else: 0.0
        @baseline + step + e
      end)

    %{
      effect: effect,
      observations: observations,
      pre_period: {1, @pre_end},
      post_period: {@pre_end + 1, @total},
      truth: %{
        step: effect * 1.0,
        cumulative: effect * 1.0 * @n_post,
        post_days: @n_post
      }
    }
  end

  @doc """
  Fits the scalar MCMC estimator on the pre-period and forward-simulates
  one counterfactual trajectory per posterior draw. Short tier: run via
  `BstsSite.Demos.Limiter` inside `start_async`.
  """
  def fit(effect) do
    scenario = scenario(effect)

    t0 = System.monotonic_time(:millisecond)

    result =
      CausalImpact.estimate(
        scenario.observations,
        scenario.pre_period,
        scenario.post_period,
        num_samples: @num_samples,
        burn_in: @burn_in,
        seed: @mcmc_seed
      )

    elapsed_ms = System.monotonic_time(:millisecond) - t0

    draws = result.counterfactual

    %{
      effect: scenario.effect,
      result: result,
      spaghetti: Enum.take(draws, @spaghetti_count),
      mean_counterfactual: column_means(draws),
      cumulative_draws: result.cumulative_effects,
      num_draws: length(draws),
      execution: %{method_used: :scalar_gibbs_mcmc, elapsed_ms: elapsed_ms}
    }
  end

  @doc """
  Interval stats at significance level `alpha`, recomputed from the stored
  draws — plain percentile arithmetic, no refit.
  """
  def stats(result, alpha) do
    alpha = clamp_alpha(alpha)
    summary = CausalImpact.summary(result, alpha: alpha)
    cumulative = summary.cumulative_effect

    %{
      alpha: alpha,
      coverage_pct: round((1.0 - alpha) * 100),
      cumulative: cumulative,
      relative: summary.relative_effect,
      significant?: is_number(cumulative.lower) and cumulative.lower > 0
    }
  end

  def clamp_effect(value) when is_integer(value) do
    value |> max(@effect_range.first) |> min(@effect_range.last)
  end

  def clamp_effect(value) when is_binary(value) do
    case Integer.parse(value) do
      {n, _} -> clamp_effect(n)
      :error -> @effect_range.first
    end
  end

  def clamp_effect(_value), do: @effect_range.first

  def clamp_alpha(value) when is_number(value) do
    value |> max(@alpha_min) |> min(@alpha_max) |> Float.round(2)
  end

  def clamp_alpha(value) when is_binary(value) do
    case Float.parse(value) do
      {a, _} -> clamp_alpha(a)
      :error -> @alpha_min
    end
  end

  def clamp_alpha(_value), do: @alpha_min

  defp column_means(draws) do
    n = length(draws)

    draws
    |> Enum.zip()
    |> Enum.map(fn column ->
      Enum.sum(Tuple.to_list(column)) / n
    end)
  end

  @doc "The code this demo runs, shown verbatim in the under-the-hood panel."
  def code_snippet do
    ~S"""
    # Fit a local level model on days 1–90, then forward-simulate the
    # post-period once per posterior draw (~1.3 s on this server).
    result =
      BstsNx.CausalImpact.estimate(observations, {1, 90}, {91, 120},
        num_samples: 60,
        burn_in: 30,
        seed: 2024
      )

    # One full counterfactual trajectory per draw — this is the spaghetti.
    length(result.counterfactual)      #=> 60
    length(hd(result.counterfactual))  #=> 30

    # The alpha slider never refits: intervals are percentiles of the
    # stored draws, recomputed in about a millisecond.
    BstsNx.CausalImpact.summary(result, alpha: 0.10).cumulative_effect
    #=> %{mean: ..., sd: ..., lower: ..., upper: ...}
    """
  end
end
