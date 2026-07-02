defmodule BstsSite.Demos.Gibbs do
  @moduledoc """
  The MCMC theater for `/engine/gibbs`.

  A fixed seeded series — a slowly meandering level buried in observation
  noise — where BOTH variances are hidden from the sampler. `sample/1` runs
  `BstsNx.GibbsSampler.sample_chains/8` for three chains and returns the raw
  per-iteration variance draws so the LiveView can replay them tick by tick,
  exactly as they were drawn.

  Everything is deterministic: fixed data seed, fixed MCMC seed. What varies
  between visits is only the measured wall-clock time.
  """

  alias BstsSite.Demos.Scenarios

  @n 120
  @level0 50.0
  @q_true 1.0
  @r_true 9.0
  @data_seed 11
  @mcmc_seed 2026
  @num_chains 3
  @num_samples 70
  @burn_in 30
  # Deliberately wrong starting guesses — inverted relative to the truth —
  # so the chains have real work to do before they agree.
  @start_process_var 8.0
  @start_obs_var 1.0

  def num_chains, do: @num_chains
  def num_samples, do: @num_samples
  def burn_in, do: @burn_in

  @doc """
  The fixed scenario: a level that drifts as a random walk with variance
  #{@q_true} per step, observed through noise with variance #{@r_true}.
  The sampler is told neither number.
  """
  def scenario do
    eta = Scenarios.noise(@n, :math.sqrt(@q_true), @data_seed)
    eps = Scenarios.noise(@n, :math.sqrt(@r_true), @data_seed + 1000)

    {levels, _} = Enum.map_reduce(eta, @level0, fn e, lvl -> {lvl + e, lvl + e} end)
    observations = Enum.zip_with(levels, eps, &(&1 + &2))

    %{
      observations: observations,
      n: @n,
      truth: %{process_var: @q_true, obs_var: @r_true}
    }
  end

  @doc """
  Runs the real MCMC: #{@num_chains} chains × #{@num_samples} retained
  samples after #{@burn_in} burn-in iterations each. Returns per-chain
  traces of both sampled variances (plain floats, iteration order preserved)
  plus posterior summaries and the measured wall-clock time.
  """
  def sample(observations) do
    {elapsed_us, chains} =
      :timer.tc(fn ->
        BstsNx.GibbsSampler.sample_chains(
          observations,
          @num_chains,
          @num_samples,
          hd(observations),
          25.0,
          @start_process_var,
          @start_obs_var,
          burn_in: @burn_in,
          seed: @mcmc_seed
        )
      end)

    obs_var_traces = for chain <- chains, do: Enum.map(chain, &Nx.to_number(&1.obs_var))

    process_var_traces =
      for chain <- chains, do: Enum.map(chain, &Nx.to_number(&1.process_var))

    elapsed_ms = System.convert_time_unit(elapsed_us, :microsecond, :millisecond)

    %{
      obs_var_traces: obs_var_traces,
      process_var_traces: process_var_traces,
      total: @num_samples,
      elapsed_ms: elapsed_ms,
      execution: %{method_used: :sample_chains, elapsed_ms: elapsed_ms},
      obs_var_summary: summarize(List.flatten(obs_var_traces)),
      process_var_summary: summarize(List.flatten(process_var_traces)),
      ess_obs_var: ess(obs_var_traces),
      trace_domain: trace_domain(obs_var_traces)
    }
  end

  @doc """
  R-hat on the first `k` revealed iterations of each chain. Returns `:nan`
  for prefixes too short to say anything (the LiveView shows "warming up").
  """
  def r_hat_prefix(traces, k) when is_integer(k) and k >= 2 do
    BstsNx.Diagnostics.r_hat(Enum.map(traces, &Enum.take(&1, k)))
  end

  def r_hat_prefix(_traces, _k), do: :nan

  @doc "Was the planted value inside the sampler's 95% interval?"
  def recovered?(summary, truth) do
    truth >= summary.lower and truth <= summary.upper
  end

  defp summarize(draws) do
    sorted = Enum.sort(draws)
    n = length(sorted)
    {lower, upper} = BstsNx.Utils.percentile_interval(sorted, n, 0.05)
    %{mean: Enum.sum(draws) / n, lower: lower, upper: upper, n: n}
  end

  defp ess(traces) do
    case BstsNx.Diagnostics.effective_sample_size(traces) do
      :nan -> :nan
      v -> round(v)
    end
  end

  # Fixed y-domain for the trace plot so the axes hold still while the
  # replay grows — computed from the full traces, padded a little.
  defp trace_domain(traces) do
    values = List.flatten(traces)
    lo = Enum.min(values)
    hi = Enum.max(values)
    pad = (hi - lo) * 0.1
    {lo - pad, hi + pad}
  end

  @doc "The code this demo runs, shown verbatim in the under-the-hood panel."
  def code_snippet do
    ~S"""
    # Both variances are unknown to the sampler. Three independent chains,
    # 30 burn-in iterations each, 70 retained draws, deliberately wrong
    # starting guesses (process_var: 8.0, obs_var: 1.0).
    chains =
      BstsNx.GibbsSampler.sample_chains(
        observations,   # 120 points, generated with Q = 1.0, R = 9.0
        3,              # chains
        70,             # retained samples per chain
        hd(observations), 25.0,   # state prior: mean, variance
        8.0, 1.0,                 # starting guesses for Q and R
        burn_in: 30,
        seed: 2026
      )

    # Each draw: %{states: [...], state_covs: [...],
    #              process_var: tensor, obs_var: tensor}
    obs_var_traces =
      for chain <- chains, do: Enum.map(chain, &Nx.to_number(&1.obs_var))

    # Gelman-Rubin R-hat across the three chains — the page recomputes
    # this on every replay tick, on exactly the draws revealed so far.
    BstsNx.Diagnostics.r_hat(obs_var_traces)
    #=> 1.001 (values near 1.0 mean the chains agree)
    """
  end
end
