defmodule BstsSite.Demos.Speed do
  @moduledoc """
  The /speed demo: race the two lanes on identical planted-lift data.

  `prepare/0` runs the instant Operational lane once for the opening figure.
  `race/0` times three paths on the same 120-point series with `:timer.tc`:
  Operational prepare+run, `CausalImpact.estimate_from_filter`, and scalar
  MCMC `CausalImpact.estimate` at 30 samples. Data is seeded, so the
  estimates are identical on every race — only the clock changes.
  """

  alias BstsNx.{CausalImpact, Components, Operational}
  alias BstsSite.Demos.Scenarios

  @total 120
  @pre_end 80
  @baseline 100.0
  @lift 12.0
  @noise_sd 5.0
  @process_var 0.5
  @initial_cov 25.0
  @seed 2024
  @mcmc_samples 30

  def total, do: @total
  def pre_end, do: @pre_end
  def mcmc_samples, do: @mcmc_samples

  @doc """
  120 days of orders around a level of #{@baseline}: a step of +#{@lift}
  orders/day switches on at day #{@pre_end + 1} and stays on. Deterministic —
  every visitor races on the same series.
  """
  def scenario do
    noise = Scenarios.noise(@total, @noise_sd, @seed)

    {observations, effects} =
      noise
      |> Enum.with_index()
      |> Enum.map(fn {eps, t} ->
        effect = if t >= @pre_end, do: @lift, else: 0.0
        {@baseline + effect + eps, effect}
      end)
      |> Enum.unzip()

    %{
      observations: observations,
      pre_period: {1, @pre_end},
      post_period: {@pre_end + 1, @total},
      truth: %{
        lift_per_day: @lift,
        cumulative_lift: Enum.sum(effects),
        post_days: @total - @pre_end
      }
    }
  end

  @doc """
  Runs the instant lane once for the opening figure: the forecast-first
  counterfactual the Operational engine projects without seeing a single
  post-period observation.
  """
  def prepare do
    scenario = scenario()

    result =
      Operational.run(
        prepared(),
        scenario.observations,
        [],
        return: :lists
      )

    band =
      Enum.zip_with(
        [result.summary.baseline, result.counterfactual.variance],
        fn [mean, var] ->
          sd = :math.sqrt(max(var, 0.0))
          {mean - 1.96 * sd, mean + 1.96 * sd}
        end
      )

    %{
      scenario: scenario,
      counterfactual_mean: result.summary.baseline,
      band_lower: Enum.map(band, &elem(&1, 0)),
      band_upper: Enum.map(band, &elem(&1, 1)),
      cumulative: result.summary.cumulative_effect,
      execution: result.execution
    }
  end

  @doc """
  The race: three estimators, one series, one wall clock.

  Timed with `:timer.tc` on this request — the Operational lane is timed
  end-to-end (prepare + run), the way a cold request path would pay for it.
  """
  def race do
    scenario = scenario()
    obs = scenario.observations

    {op_us, op} =
      :timer.tc(fn ->
        prepared = prepared()
        Operational.run(prepared, obs, [], return: :lists)
      end)

    {filt_us, filt} =
      :timer.tc(fn ->
        CausalImpact.estimate_from_filter(
          obs,
          Enum.to_list(@pre_end..(@total - 1)),
          q: @process_var,
          r: @noise_sd * @noise_sd,
          x0: @baseline,
          p0: @initial_cov
        )
      end)

    {mcmc_us, mcmc} =
      :timer.tc(fn ->
        obs
        |> CausalImpact.estimate(scenario.pre_period, scenario.post_period,
          num_samples: @mcmc_samples,
          burn_in: div(@mcmc_samples, 2),
          seed: @seed
        )
        |> CausalImpact.summary()
      end)

    lanes = [
      %{
        id: :operational,
        label: "Operational prepare+run",
        ms: op_us / 1000.0,
        cumulative: op.summary.cumulative_effect,
        method: op.execution.method_used
      },
      %{
        id: :filter,
        label: "estimate_from_filter",
        ms: filt_us / 1000.0,
        cumulative: filt.cumulative_effect,
        method: :scalar_smooth_filter
      },
      %{
        id: :mcmc,
        label: "MCMC estimate, #{@mcmc_samples} samples",
        ms: mcmc_us / 1000.0,
        cumulative: mcmc.cumulative_effect,
        method: :elixir_mcmc
      }
    ]

    means = Enum.map(lanes, & &1.cumulative.mean)

    %{
      lanes: lanes,
      operational_execution: op.execution,
      mean_spread: Enum.max(means) - Enum.min(means),
      speedup: mcmc_us / max(op_us, 1)
    }
  end

  @doc "How many of the lanes' 95% intervals contain the planted cumulative lift."
  def coverage(race, truth_cumulative) do
    covered =
      Enum.filter(race.lanes, fn lane ->
        lane.cumulative.lower <= truth_cumulative and truth_cumulative <= lane.cumulative.upper
      end)

    %{covered: Enum.map(covered, & &1.id), n: length(covered), of: length(race.lanes)}
  end

  defp prepared do
    spec =
      Components.local_level_spec(
        initial_state: @baseline,
        initial_cov: @initial_cov,
        process_var: @process_var,
        obs_var: @noise_sd * @noise_sd
      )

    Operational.prepare(spec, {1, @pre_end}, {@pre_end + 1, @total})
  end

  @doc "The code the race button runs, shown verbatim in the under-the-hood panel."
  def code_snippet do
    ~S"""
    # One series, three estimators, one wall clock. 120 points,
    # planted +12/day step at day 81. :timer.tc returns microseconds.

    # Lane 1 — Operational, timed end-to-end (prepare + run).
    # Forecast-first: the baseline never sees post-period outcomes.
    {op_us, op} =
      :timer.tc(fn ->
        spec =
          BstsNx.Components.local_level_spec(
            initial_state: 100.0,
            initial_cov: 25.0,
            process_var: 0.5,
            obs_var: 25.0
          )

        prepared = BstsNx.Operational.prepare(spec, {1, 80}, {81, 120})
        BstsNx.Operational.run(prepared, observations, [], return: :lists)
      end)

    # Lane 2 — compiled filter + RTS smoother. Masks the window as NaN
    # and smooths; the baseline may use data after the window.
    {filt_us, filt} =
      :timer.tc(fn ->
        BstsNx.CausalImpact.estimate_from_filter(
          observations,
          Enum.to_list(80..119),
          q: 0.5, r: 25.0, x0: 100.0, p0: 25.0
        )
      end)

    # Lane 3 — scalar Gibbs sampler: the full posterior, at a price.
    {mcmc_us, mcmc} =
      :timer.tc(fn ->
        observations
        |> BstsNx.CausalImpact.estimate({1, 80}, {81, 120},
          num_samples: 30,
          burn_in: 15,
          seed: 2024
        )
        |> BstsNx.CausalImpact.summary()
      end)

    op.summary.cumulative_effect.mean    # lane 1's answer
    filt.cumulative_effect.mean          # lane 2's answer
    mcmc.cumulative_effect.mean          # lane 3's answer
    """
  end
end
