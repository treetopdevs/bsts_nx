defmodule BstsNx.Pipeline do
  @moduledoc """
  End-to-end counterfactual attribution pipeline.

  Operational mode is the default: it delegates to `BstsNx.Operational` for a
  forecast-first fixed-variance counterfactual, then passes the result to
  `SpotAttributor.attribute/4`.

  Bayesian mode remains available with `mode: :bayesian`: it chains
  `CausalImpact.estimate_structured/5` with
  `SpotAttributor.attribute_posterior/5` to propagate posterior draws through
  attribution for offline analysis.

  ## Usage

      spec = Components.compose_specs(
        Components.local_level_spec(initial_state: 50.0),
        Components.seasonal_spec(7)
      )

      spots = [
        %{id: "spot_1", window_start: 0, window_end: 5},
        %{id: "spot_2", window_start: 3, window_end: 8}
      ]

      result = Pipeline.run(observations, {1, 70}, {71, 84}, spots, spec)

      result.attributions      # SpotAttributor result with per-spot lifts
      result.summary           # operational counterfactual summary
      result.execution         # explicit method/baseline/backend metadata
  """

  alias BstsNx.CausalImpact
  alias BstsNx.Execution
  alias BstsNx.Operational
  alias BstsNx.SpotAttributor
  alias BstsNx.Validation

  @typedoc "Full pipeline result."
  @type pipeline_result :: %{
          attributions: SpotAttributor.attribution_result(),
          causal_impact: CausalImpact.impact_result() | nil,
          summary: map(),
          counterfactual: SpotAttributor.counterfactual() | nil,
          execution: Execution.t()
        }

  @doc """
  Runs the full attribution pipeline.

  ## Parameters

    * `observations` — full time series (list of numbers)
    * `pre_period` — `{start, end}` (1-based inclusive) for model fitting
    * `post_period` — `{start, end}` (1-based inclusive) for impact assessment;
      gaps after `pre_period` are allowed
    * `spots` — list of spot maps with 0-based half-open windows
      `[window_start, window_end)` relative to the post-period start
    * `spec` — `%BstsNx.ModelSpec{}` defining the state-space model
    * `opts` — keyword options (see below)

  ## Options

  Forwarded to `CausalImpact.estimate_structured/5`:

    * `:mode` — `:operational` (default filter baseline) or `:bayesian`
      (posterior MCMC propagation)
    * `:num_samples` — posterior draws (default: 200)
    * `:burn_in` — burn-in period (default: `num_samples / 2`)
    * `:thin` — thinning interval (default: 1)
    * `:seed` — integer PRNG seed
    * `:key` — `Nx.Random` key (overrides `:seed`)

  Forwarded to `SpotAttributor.attribute_posterior/5`:

    * `:alpha` — significance level for CIs (default: 0.05)
    * `:shapley_samples` — MC permutations for Shapley (default: 10_000)
    * `:decay` — diminishing-returns decay factor (default: 0.7)

  ## Examples

      iex> :rand.seed(:exsss, {100, 101, 102})
      iex> pre = Enum.map(1..30, fn _ -> 50.0 + :rand.normal() * 2 end)
      iex> post = Enum.map(1..10, fn _ -> 55.0 + :rand.normal() * 2 end)
      iex> obs = pre ++ post
      iex> spec = BstsNx.Components.local_level_spec(initial_state: 50.0, initial_cov: 10.0)
      iex> spots = [%{id: "s1", window_start: 0, window_end: 5}]
      iex> result = BstsNx.Pipeline.run(obs, {1, 30}, {31, 40}, spots, spec, num_samples: 5, burn_in: 2, seed: 42)
      iex> length(result.attributions.attributions)
      1
  """
  @spec run(
          [number()],
          {pos_integer(), pos_integer()},
          {pos_integer(), pos_integer()},
          [SpotAttributor.spot()],
          BstsNx.ModelSpec.t(),
          keyword()
        ) :: pipeline_result()
  def run(observations, pre_period, post_period, spots, spec, opts \\ []) do
    mode = Execution.resolve_mode!(opts, :operational)
    {sa_opts, ci_opts} = split_options(opts)

    case mode do
      :operational ->
        Operational.run(observations, pre_period, post_period, spots, spec, opts)

      :bayesian ->
        run_bayesian(observations, pre_period, post_period, spots, spec, sa_opts, ci_opts)
    end
  end

  # ── Private helpers ──────────────────────────────────────────────────

  defp run_bayesian(observations, pre_period, post_period, spots, spec, sa_opts, ci_opts) do
    {post_start, post_end} = post_period
    Validation.validate_spot_windows!(spots, post_end - post_start + 1)

    {elapsed_us, result} =
      Execution.measure(fn ->
        ci_result =
          CausalImpact.estimate_structured(observations, pre_period, post_period, spec, ci_opts)

        ci_summary = CausalImpact.summary(ci_result, alpha: Keyword.get(ci_opts, :alpha, 0.05))

        attribution_result =
          SpotAttributor.attribute_posterior(
            ci_result.actual,
            spots,
            ci_result.counterfactual,
            0.0,
            sa_opts
          )

        %{
          attributions: attribution_result,
          causal_impact: ci_result,
          summary: ci_summary,
          counterfactual: nil
        }
      end)

    Map.put(
      result,
      :execution,
      Execution.metadata(:bayesian, :elixir_nx, :elixir_mcmc, elapsed_us)
    )
  end

  defp split_options(opts) do
    sa_keys = [:alpha, :shapley_samples, :n_samples, :decay, :key, :value_fn_mode]
    pipeline_keys = [:mode, :method]
    ci_only_keys = [:shapley_samples, :n_samples, :decay, :value_fn_mode]
    sa_opts = opts |> Keyword.take(sa_keys) |> Validation.attribution_options()
    ci_opts = Keyword.drop(opts, ci_only_keys ++ pipeline_keys)

    {sa_opts, ci_opts}
  end
end
