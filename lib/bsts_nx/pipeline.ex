defmodule BstsNx.Pipeline do
  @moduledoc """
  End-to-end pipeline: CausalImpact estimation → SpotAttributor.

  Chains `CausalImpact.estimate_structured/5` with
  `SpotAttributor.attribute_posterior/5` so that a single call produces
  per-spot TV lift attributions with Shapley allocation for overlaps.

  The pipeline:

  1. Fits a structured state-space model on the pre-period via MCMC
  2. Generates posterior counterfactual draws for the post-period
  3. Propagates each draw through SpotAttributor to preserve correlation
  4. Aggregates per-spot lift distributions across draws

  Since each posterior counterfactual draw already includes observation
  noise from the forward simulation, `obs_variance` is set to 0.0 to
  avoid double-counting.  The posterior propagation approach is more
  accurate than collapsing to mean/variance first because it preserves
  the correlation structure across time steps.

  ## Usage

      spec = Components.compose_specs(
        Components.local_level_spec(initial_state: 50.0),
        Components.seasonal_spec(7)
      )

      spots = [
        %{id: "spot_1", window_start: 0, window_end: 5},
        %{id: "spot_2", window_start: 3, window_end: 8}
      ]

      result = Pipeline.run(observations, {1, 70}, {71, 84}, spots, spec,
        num_samples: 200, seed: 42
      )

      result.attributions      # SpotAttributor result with per-spot lifts
      result.causal_impact     # raw CausalImpact posterior samples
      result.summary           # CausalImpact summary statistics
  """

  alias BstsNx.CausalImpact
  alias BstsNx.SpotAttributor

  @typedoc "Full pipeline result."
  @type pipeline_result :: %{
          attributions: SpotAttributor.attribution_result(),
          causal_impact: CausalImpact.impact_result(),
          summary: map()
        }

  @doc """
  Runs the full attribution pipeline.

  ## Parameters

    * `observations` — full time series (list of numbers)
    * `pre_period` — `{start, end}` (1-based inclusive) for model fitting
    * `post_period` — `{start, end}` (1-based inclusive) for impact assessment;
      must immediately follow `pre_period`
    * `spots` — list of spot maps with 0-based half-open windows
      `[window_start, window_end)` relative to the post-period start
    * `spec` — `%BstsNx.ModelSpec{}` defining the state-space model
    * `opts` — keyword options (see below)

  ## Options

  Forwarded to `CausalImpact.estimate_structured/5`:

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
    {post_start, post_end} = post_period
    n_post = post_end - post_start + 1

    validate_spot_windows!(spots, n_post)

    {sa_opts, ci_opts} = split_options(opts)

    # 1. CausalImpact estimation
    ci_result =
      CausalImpact.estimate_structured(observations, pre_period, post_period, spec, ci_opts)

    ci_summary = CausalImpact.summary(ci_result)

    # 2. Propagate each posterior draw through SpotAttributor
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
      summary: ci_summary
    }
  end

  # ── Private helpers ──────────────────────────────────────────────────

  defp validate_spot_windows!(spots, n_post) do
    Enum.each(spots, fn spot ->
      if spot.window_start < 0 or spot.window_end > n_post do
        raise ArgumentError,
              "spot #{spot.id} window [#{spot.window_start}, #{spot.window_end}) " <>
                "is outside the post period (0..#{n_post})"
      end

      if spot.window_start > spot.window_end do
        raise ArgumentError,
              "spot #{spot.id} has window_start (#{spot.window_start}) > " <>
                "window_end (#{spot.window_end})"
      end
    end)
  end

  defp split_options(opts) do
    sa_keys = [:alpha, :shapley_samples, :decay]
    sa_raw = Keyword.take(opts, sa_keys)
    ci_opts = Keyword.drop(opts, sa_keys)

    # Map :shapley_samples → :n_samples for SpotAttributor
    sa_opts =
      case Keyword.pop(sa_raw, :shapley_samples) do
        {nil, rest} -> rest
        {val, rest} -> Keyword.put(rest, :n_samples, val)
      end

    {sa_opts, ci_opts}
  end
end
