defmodule BstsSite.Demos.Diagnostics do
  @moduledoc """
  One honest causal-impact fit, then five attempts to break it with the
  `BstsNx.Validation` suite: prediction error, CI coverage, Durbin-Watson,
  a placebo test, and effect stability.

  The scenario plants a +#{3.0} per-day lift from day 91 onward on a steady
  level-plus-noise series. `scenario/0` is cheap and runs at mount;
  `run/1` does the MCMC fit plus all five checks and runs behind the
  Limiter in a `start_async`. Everything is seeded.
  """

  alias BstsNx.{CausalImpact, Execution, GibbsSampler, Validation}
  alias BstsSite.Demos.Scenarios

  @total 130
  @n_pre 90
  @post_start 91
  @post_end 120
  @n_post 30
  @per_step 3.0
  @base_level 100.0
  @noise_sd 2.0
  @data_seed 11
  @fit_seed 1011
  @placebo_seed 2011
  @stability_seed 3011
  @num_samples 50
  @burn_in 25
  @aux_samples 40
  @window_delta 5
  # Chain starting points near the data scale; the inverse-gamma prior is
  # shared by both variances (that is the scalar sampler's API).
  @init_q 0.01
  @init_r @noise_sd * @noise_sd
  @prior [prior_shape: 2.0, prior_scale: 0.1]

  def n_pre, do: @n_pre
  def n_post, do: @n_post
  def post_start, do: @post_start
  def post_end, do: @post_end
  def window_delta, do: @window_delta

  @doc """
  The planted scenario: 130 days around a steady level of #{@base_level},
  noise sd #{@noise_sd}. The campaign turns on at day #{@post_start} and stays
  on through the end of the series; the measured window is days
  #{@post_start}–#{@post_end}, so the planted cumulative effect inside it is
  exactly #{@per_step} × #{@n_post}. The extra days past the window give the
  stability check room to widen it.
  """
  def scenario do
    observations =
      Scenarios.noise(@total, @noise_sd, @data_seed)
      |> Enum.with_index()
      |> Enum.map(fn {e, t} ->
        @base_level + e + if(t >= @n_pre, do: @per_step, else: 0.0)
      end)

    %{
      observations: observations,
      pre_period: {1, @n_pre},
      post_period: {@post_start, @post_end},
      truth: %{per_step: @per_step, cumulative: @per_step * @n_post, window_days: @n_post}
    }
  end

  @doc """
  The main fit plus the five validation checks.

  The headline number comes from `CausalImpact.estimate/4`. The pre-period
  baseline the first three checks need comes from a second, identically
  configured run of the same sampler — `estimate/4` keeps its posterior
  states private and derives its own PRNG stream from the seed, so this is
  an independent chain from the same posterior. The placebo and stability
  checks re-fit the model three more times through the callbacks the
  Validation API expects.
  """
  def run(scenario) do
    obs = scenario.observations
    pre = Enum.take(obs, @n_pre)

    {fit_us, fit} =
      Execution.measure(fn ->
        result =
          CausalImpact.estimate(
            obs,
            scenario.pre_period,
            scenario.post_period,
            [
              num_samples: @num_samples,
              burn_in: @burn_in,
              seed: @fit_seed,
              process_var: @init_q,
              obs_var: @init_r
            ] ++ @prior
          )

        %{result: result, summary: CausalImpact.summary(result)}
      end)

    {checks_us, checks} = Execution.measure(fn -> run_checks(obs, pre, fit.summary) end)

    {cf_mean, cf_lower, cf_upper} = counterfactual_stats(fit.result.counterfactual)
    ce = fit.summary.cumulative_effect

    %{
      counterfactual_mean: cf_mean,
      counterfactual_lower: cf_lower,
      counterfactual_upper: cf_upper,
      baseline_pre: checks.baseline_pre,
      residuals: checks.residuals,
      resid_lower: checks.resid_lower,
      resid_upper: checks.resid_upper,
      cumulative: ce,
      relative: fit.summary.relative_effect,
      significant?: ce.lower > 0,
      details: checks.details,
      verdicts: checks.verdicts,
      pass_count: Enum.count(checks.verdicts, fn {_k, v} -> v == :pass end),
      execution_fit: Execution.metadata(:bayesian, :elixir_nx, :elixir_mcmc, fit_us),
      execution_checks: Execution.metadata(:bayesian, :elixir_nx, :elixir_mcmc, checks_us)
    }
  end

  defp run_checks(obs, pre, summary) do
    samples =
      GibbsSampler.sample(
        pre,
        @num_samples,
        hd(pre),
        1.0,
        @init_q,
        @init_r,
        [burn_in: @burn_in, seed: @fit_seed] ++ @prior
      )

    states_t =
      samples
      |> Enum.map(fn s -> s.states |> Nx.stack() |> Nx.flatten() end)
      |> Nx.stack()

    baseline_t = Nx.mean(states_t, axes: [0])
    state_vars_t = Nx.variance(states_t, axes: [0], ddof: 1)
    obs_var = samples |> Enum.map(&Nx.to_number(&1.obs_var)) |> average()

    actual_t = Nx.tensor(pre, type: {:f, 64})
    pre_idx = Enum.to_list(0..(@n_pre - 1))
    on_air = Enum.to_list((@post_start - 1)..(@post_end - 1))
    main_rate = summary.cumulative_effect.mean / @n_post

    evaluation =
      Validation.evaluate(%{
        actual: actual_t,
        baseline: baseline_t,
        indices: pre_idx,
        coverage: %{state_variances: state_vars_t, obs_variance: obs_var},
        placebo: %{sessions: obs, on_air_indices: on_air, estimate_fn: &placebo_fit/2},
        effect_stability: %{
          main_lift: main_rate,
          estimate_at_window_fn: &rate_at_window(obs, &1),
          window: @n_post,
          delta: @window_delta
        }
      })

    residuals_t = evaluation.residuals

    z = Validation.z_score(0.05)
    resid_sd = Nx.sqrt(Nx.add(state_vars_t, obs_var))

    %{
      baseline_pre: Nx.to_flat_list(baseline_t),
      residuals: Nx.to_flat_list(residuals_t),
      resid_lower: Nx.to_flat_list(Nx.multiply(resid_sd, -z)),
      resid_upper: Nx.to_flat_list(Nx.multiply(resid_sd, z)),
      details: Map.put(evaluation.details, :placebo_window, placebo_window(obs, on_air)),
      verdicts: evaluation.verdicts
    }
  end

  # The estimator the placebo check re-runs: the same scalar MCMC path,
  # fit on the data before the fake window, judged on the fake window.
  defp placebo_fit(sessions, placebo_idx) do
    first = hd(placebo_idx)
    last = List.last(placebo_idx)

    result =
      CausalImpact.estimate(
        sessions,
        {1, first},
        {first + 1, last + 1},
        [
          num_samples: @aux_samples,
          burn_in: div(@aux_samples, 2),
          seed: @placebo_seed,
          process_var: @init_q,
          obs_var: @init_r
        ] ++ @prior
      )

    s = CausalImpact.summary(result)

    %{
      lift_sessions: s.cumulative_effect.mean,
      lift_pct: s.relative_effect.mean,
      lift_ci95: %{
        lower: s.cumulative_effect.lower,
        upper: s.cumulative_effect.upper,
        sd: s.cumulative_effect.sd
      }
    }
  end

  # The estimator the stability check re-runs at a nudged window width.
  # We compare the per-day rate, not the cumulative total, so a wider
  # window is not penalized for mechanically accumulating more effect.
  defp rate_at_window(obs, window) do
    result =
      CausalImpact.estimate(
        obs,
        {1, @n_pre},
        {@n_pre + 1, @n_pre + window},
        [
          num_samples: @aux_samples,
          burn_in: div(@aux_samples, 2),
          seed: @stability_seed,
          process_var: @init_q,
          obs_var: @init_r
        ] ++ @prior
      )

    CausalImpact.summary(result).cumulative_effect.mean / window
  end

  # Mirrors the placement rule inside Validation.placebo_test/3 — a block the
  # size of the real window, centered in the untreated period. Display only.
  defp placebo_window(obs, on_air) do
    on_air_set = MapSet.new(on_air)

    untreated =
      Enum.filter(0..(length(obs) - 1), fn i -> not MapSet.member?(on_air_set, i) end)

    n = length(on_air)
    start_pos = max(0, div(length(untreated), 2) - div(n, 2))
    block = Enum.slice(untreated, start_pos, n)
    {hd(block), List.last(block)}
  end

  # Column-wise mean and 95% posterior-predictive band of the sampled
  # counterfactual trajectories (they include observation noise).
  defp counterfactual_stats(counterfactual_samples) do
    t = Nx.tensor(counterfactual_samples, type: {:f, 64})
    mean = Nx.mean(t, axes: [0])
    sd = Nx.standard_deviation(t, axes: [0], ddof: 1)
    z = Validation.z_score(0.05)

    {
      Nx.to_flat_list(mean),
      Nx.to_flat_list(Nx.subtract(mean, Nx.multiply(sd, z))),
      Nx.to_flat_list(Nx.add(mean, Nx.multiply(sd, z)))
    }
  end

  defp average(values), do: Enum.sum(values) / length(values)

  @doc "The code this page runs, shown verbatim in the under-the-hood panel."
  def code_snippet do
    ~S"""
    # One honest fit...
    result =
      BstsNx.CausalImpact.estimate(observations, {1, 90}, {91, 120},
        num_samples: 50, burn_in: 25, seed: 1011,
        process_var: 0.01, obs_var: 4.0, prior_shape: 2.0, prior_scale: 0.1)

    # ...then try to break it. The pre-period baseline for the first three
    # checks comes from the same sampler (estimate/4 keeps its states private):
    samples =
      BstsNx.GibbsSampler.sample(pre_data, 50, hd(pre_data), 1.0, 0.01, 4.0,
        burn_in: 25, seed: 1011, prior_shape: 2.0, prior_scale: 0.1)

    alias BstsNx.Validation

    evaluation =
      Validation.evaluate(%{
        actual: actual_t,
        baseline: baseline_t,
        indices: pre_idx,
        coverage: %{state_variances: state_vars_t, obs_variance: obs_var},
        # both callbacks re-fit the model — a fake window, a nudged window
        placebo: %{sessions: obs, on_air_indices: on_air, estimate_fn: &placebo_fit/2},
        effect_stability: %{
          main_lift: main_rate,
          estimate_at_window_fn: &rate_at_window(obs, &1),
          window: 30,
          delta: 5
        }
      })

    evaluation.verdicts
    #=> %{prediction_error: :pass, coverage: :pass, durbin_watson: :pass,
    #     placebo: :pass, effect_stability: :pass}
    """
  end
end
