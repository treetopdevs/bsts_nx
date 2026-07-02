defmodule BstsSite.Demos.Calibration do
  @moduledoc """
  The calibration sweep: one 90-point series, five planted cumulative lifts
  (one of them zero — the placebo), five scalar MCMC fits. The page grades
  every 95% interval against the truth it was never shown.

  `series/0` is cheap and runs at mount for the data figure; `run_sweep/1`
  is the five sequential MCMC fits and runs behind the Limiter in a
  `start_async`. Everything is seeded: same page every visit.
  """

  alias BstsNx.{CausalImpact, Execution}
  alias BstsSite.Demos.Scenarios

  @lifts [0, 5, 10, 20, 40]
  @total 90
  @pre_end 70
  @n_post 20
  @base_level 100.0
  @noise_sd 1.5
  @data_seed 42
  @num_samples 40
  @burn_in 20

  def lifts, do: @lifts
  def pre_end, do: @pre_end
  def n_post, do: @n_post

  @doc """
  The five case series. They share one seeded noise realization — only the
  planted lift differs, spread uniformly over the 20 post-period points, so
  each planted cumulative effect is exactly the case's lift value.
  """
  def series do
    base = Enum.map(Scenarios.noise(@total, @noise_sd, @data_seed), &(@base_level + &1))

    cases =
      Enum.map(@lifts, fn lift ->
        per_step = lift / @n_post

        observations =
          base
          |> Enum.with_index()
          |> Enum.map(fn {v, t} -> if t >= @pre_end, do: v + per_step, else: v end)

        %{
          lift: lift,
          per_step: per_step,
          observations: observations,
          truth_mean: @base_level + per_step
        }
      end)

    %{base: base, cases: cases, noise_sd: @noise_sd, max_per_step: List.last(@lifts) / @n_post}
  end

  @doc """
  The sweep: five `CausalImpact.estimate/4` fits, sequentially, ~#{@num_samples}
  kept samples each. Coverage is judged against the 95% credible interval from
  `CausalImpact.summary/1`. Execution metadata is built by the library's own
  `BstsNx.Execution` from the measured wall time of the whole sweep.
  """
  def run_sweep(series) do
    {elapsed_us, cases} =
      Execution.measure(fn ->
        series.cases
        |> Enum.with_index()
        |> Enum.map(fn {c, i} ->
          result =
            CausalImpact.estimate(c.observations, {1, @pre_end}, {@pre_end + 1, @total},
              num_samples: @num_samples,
              burn_in: @burn_in,
              seed: 4242 + i,
              process_var: 0.01,
              obs_var: @noise_sd * @noise_sd,
              prior_shape: 2.0,
              prior_scale: 0.1
            )

          ce = CausalImpact.summary(result).cumulative_effect
          truth = c.lift * 1.0

          %{
            lift: c.lift,
            truth: truth,
            mean: ce.mean,
            lower: ce.lower,
            upper: ce.upper,
            covered?: truth >= ce.lower and truth <= ce.upper
          }
        end)
      end)

    placebo = hd(cases)

    %{
      cases: cases,
      covered_count: Enum.count(cases, & &1.covered?),
      placebo_straddles?: placebo.lower < 0.0 and placebo.upper > 0.0,
      execution: Execution.metadata(:bayesian, :elixir_nx, :elixir_mcmc, elapsed_us)
    }
  end

  @doc "The code this sweep runs, shown verbatim in the under-the-hood panel."
  def code_snippet do
    ~S"""
    # Five planted cumulative lifts: 0 (a built-in placebo), +5, +10, +20, +40.
    # Same 90-point series every time — only the planted effect changes.
    for {lift, i} <- Enum.with_index([0, 5, 10, 20, 40]) do
      result =
        BstsNx.CausalImpact.estimate(observations_with(lift), {1, 70}, {71, 90},
          num_samples: 40,
          burn_in: 20,
          seed: 4242 + i,
          # start the chain near the data scale instead of the defaults
          process_var: 0.01,
          obs_var: 2.25,
          prior_shape: 2.0,
          prior_scale: 0.1
        )

      ci = BstsNx.CausalImpact.summary(result).cumulative_effect
      covered? = lift >= ci.lower and lift <= ci.upper
    end

    # The library ships the same idea as an offline harness — structured
    # models, hundreds of samples, no latency budget:
    # BstsNx.Validation.known_lift_injection(true_lift, spec, num_samples: 200)
    """
  end
end
