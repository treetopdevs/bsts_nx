defmodule BstsNx.InterventionAnalysis do
  @moduledoc """
  Domain-agnostic intervention analysis using Bayesian Structural Time Series.

  This module provides a clean, high-level API for measuring the causal effect
  of any intervention on a time series.  It wraps `BstsNx.CausalImpact` with
  a friendlier interface that uses named intervention configurations instead
  of raw index tuples.

  ## When to use this module

  Use `InterventionAnalysis` when you have:
  - A time series of observations (response metric)
  - A known intervention point (when something changed)
  - A pre-intervention period (baseline behavior)
  - A post-intervention period (observed effect)

  The module fits a state-space model to the pre-period, generates
  counterfactual predictions for what would have happened without the
  intervention, and computes the causal effect as the difference.

  ## Model selection

  The default model is a local linear trend with optional seasonality.
  For more control, pass a custom `%BstsNx.ModelSpec{}` via `:model_spec`.

  ## Examples

      # Simple intervention analysis
      observations = pre_data ++ post_data
      result = InterventionAnalysis.analyze(observations, %{
        pre_period: {1, 30},
        post_period: {31, 40}
      })

      result.summary.cumulative_effect.mean  # total causal effect
      result.summary.relative_effect.mean    # % change vs baseline
      result.significant?                    # 95% CI excludes zero?

      # With seasonality and custom model
      spec = Components.compose_specs(
        Components.local_linear_trend_spec(),
        Components.seasonal_spec(7)
      )

      result = InterventionAnalysis.analyze(observations, %{
        pre_period: {1, 90},
        post_period: {91, 104}
      }, model_spec: spec, num_samples: 500)
  """

  alias BstsNx.CausalImpact
  alias BstsNx.Components
  alias BstsNx.Execution
  alias BstsNx.ModelBuilder
  alias BstsNx.Operational

  @type intervention_config :: %{
          pre_period: {pos_integer(), pos_integer()},
          post_period: {pos_integer(), pos_integer()}
        }

  @type analysis_result :: %{
          impact: CausalImpact.impact_result(),
          summary: map(),
          significant?: boolean(),
          alpha: float(),
          model_spec: BstsNx.ModelSpec.t() | nil,
          execution: Execution.t()
        }

  @doc """
  Analyzes the causal effect of an intervention on a time series.

  ## Parameters

    * `observations` - list of numeric values covering both pre and post periods
    * `config` - map with `:pre_period` and `:post_period` as `{start, end}`
      tuples (1-based inclusive indices)
    * `opts` - keyword options (see below)

  ## Options

    * `:model_spec` - a `%BstsNx.ModelSpec{}` for the state-space model.
      If not provided, a default local linear trend model is used.
    * `:seasonality` - number of seasonal periods (e.g., 7 for weekly, 12 for
      monthly). When provided without `:model_spec`, composes a local linear
      trend with seasonal component.
    * `:control_series` - list of control time series (same length as
      observations) that are correlated with the response but unaffected
      by the intervention. These become regression covariates in a composed
      model via `ModelBuilder.build_opts_with_controls/3`.
    * `:control_selection` - optional control-series selection before model
      composition. Set to `true` for default Pearson screening, or pass a
      keyword list forwarded to `BstsNx.CovariateSelection.select/3`.
    * `:control_selection_pre_period` - optional `{start, end}` window used
      for control selection. Defaults to `pre_period`.
    * `:control_regression_mode` - `:dynamic` (default random-walk coefficients)
      or `:spike_and_slab` (static coefficients sampled in-loop with
      Zellner's g-prior).
    * `:control_regression_opts` - keyword options forwarded to
      `Components.regression_spec/2` for control regression.
    * `:alpha` - significance level for credible intervals and significance
      testing (default: 0.05)
    * `:num_samples` - number of posterior MCMC samples (default: 200)
    * `:burn_in` - burn-in iterations (default: num_samples / 2)
    * `:seed` - integer PRNG seed for reproducibility
    * `:mode` - `:operational` (default, fast filter path) or `:bayesian`
      (Elixir MCMC path)
    * `:method` - compatibility alias: `:filter` maps to `mode: :operational`,
      `:mcmc` maps to `mode: :bayesian` when `:mode` is omitted
    * `:fallback` - pass `:mcmc` to explicitly allow a slow MCMC fallback when
      a structured operational filter is unsupported by the current backend

  ## Returns

  An `analysis_result` map with:
    * `:impact` - raw posterior samples from `CausalImpact`
    * `:summary` - aggregated statistics (mean, sd, CI for point/cumulative/relative effects)
    * `:significant?` - whether the credible interval excludes zero
    * `:alpha` - significance level used
    * `:model_spec` - the model specification used

  ## Examples

      iex> :rand.seed(:exsss, {1, 2, 3})
      iex> pre = Enum.map(1..20, fn _ -> 100.0 + :rand.normal() * 3 end)
      iex> post = Enum.map(1..5, fn _ -> 115.0 + :rand.normal() * 3 end)
      iex> result = BstsNx.InterventionAnalysis.analyze(pre ++ post, %{pre_period: {1, 20}, post_period: {21, 25}}, num_samples: 10, seed: 42)
      iex> is_boolean(result.significant?)
      true
      iex> is_map(result.summary)
      true
  """
  @spec analyze([number()], intervention_config(), keyword()) :: analysis_result()
  def analyze(observations, config, opts \\ []) do
    if observations == [] do
      raise ArgumentError, "observations must be non-empty"
    end

    %{pre_period: pre_period, post_period: post_period} = config

    alpha = Keyword.get(opts, :alpha, 0.05)

    if not is_number(alpha) or alpha <= 0.0 or alpha >= 1.0 do
      raise ArgumentError, "alpha must be between 0 and 1 (exclusive), got: #{inspect(alpha)}"
    end

    {pre_start, pre_end} = pre_period
    {post_start, post_end} = post_period
    obs_count = length(observations)

    if pre_start < 1 or pre_end < pre_start do
      raise ArgumentError,
            "pre_period must satisfy 1 <= start <= end, got: {#{pre_start}, #{pre_end}}"
    end

    if pre_end > obs_count do
      raise ArgumentError,
            "pre_period end (#{pre_end}) extends beyond observations (#{obs_count})"
    end

    if post_start <= pre_end or post_end < post_start do
      raise ArgumentError,
            "post_period must satisfy pre_end < start <= end, got: {#{post_start}, #{post_end}}"
    end

    if post_end > obs_count do
      raise ArgumentError,
            "post_period end (#{post_end}) extends beyond observations (#{obs_count})"
    end

    seasonality = Keyword.get(opts, :seasonality)
    has_model_spec = match?(%BstsNx.ModelSpec{}, Keyword.get(opts, :model_spec))

    if not has_model_spec and seasonality != nil and
         (not is_integer(seasonality) or seasonality < 2) do
      raise ArgumentError,
            "seasonality must be an integer >= 2, got: #{inspect(seasonality)}"
    end

    mode = Execution.resolve_mode!(opts, :operational)

    # If control_series is provided, delegate to ModelBuilder for regression composition
    controls = Keyword.get(opts, :control_series)

    # Note: control_series is now supported with method: :filter by routing
    # to estimate_structured_from_filter.

    effective_opts =
      case controls do
        nil ->
          opts

        controls ->
          opts_with_selection_window =
            Keyword.put_new(opts, :control_selection_pre_period, pre_period)

          merged =
            ModelBuilder.build_opts_with_controls(
              observations,
              controls,
              opts_with_selection_window
            )

          # Preserve method and alpha from original opts
          merged
          |> Keyword.put(:mode, mode)
          |> Keyword.put(:alpha, alpha)
      end

    case mode do
      :bayesian ->
        analyze_mcmc(observations, pre_period, post_period, alpha, effective_opts)

      :operational ->
        analyze_filter(observations, pre_period, post_period, alpha, effective_opts)
    end
  end

  @doc """
  Convenience function that analyzes with automatic period detection.

  Given a list of observations and the index where the intervention begins
  (1-based), automatically constructs the pre and post periods.

  ## Examples

      result = InterventionAnalysis.analyze_at(observations, intervention_start: 31)
  """
  @spec analyze_at([number()], keyword()) :: analysis_result()
  def analyze_at(observations, opts) do
    intervention_start = Keyword.fetch!(opts, :intervention_start)

    if intervention_start < 2 do
      raise ArgumentError,
            "intervention_start must be >= 2 (need at least 1 pre-period observation), got: #{intervention_start}"
    end

    n = length(observations)

    config = %{
      pre_period: {1, intervention_start - 1},
      post_period: {intervention_start, n}
    }

    remaining_opts = Keyword.delete(opts, :intervention_start)
    analyze(observations, config, remaining_opts)
  end

  @doc """
  Checks if the estimated effect is statistically significant.

  An effect is significant if the posterior credible interval for the
  cumulative effect excludes zero at the given alpha level.
  """
  @spec significant?(analysis_result()) :: boolean()
  def significant?(%{significant?: sig}), do: sig

  @doc """
  Returns a human-readable summary string of the analysis.

  Useful for quick inspection of results in IEx or logs.

  ## Examples

      iex> :rand.seed(:exsss, {1, 2, 3})
      iex> pre = Enum.map(1..20, fn _ -> 100.0 + :rand.normal() * 3 end)
      iex> post = Enum.map(1..5, fn _ -> 115.0 + :rand.normal() * 3 end)
      iex> result = BstsNx.InterventionAnalysis.analyze(pre ++ post, %{pre_period: {1, 20}, post_period: {21, 25}}, num_samples: 10, seed: 42)
      iex> report = BstsNx.InterventionAnalysis.report(result)
      iex> String.contains?(report, "Cumulative effect")
      true
  """
  @spec report(analysis_result()) :: String.t()
  def report(%{summary: summary, significant?: sig, alpha: alpha}) do
    cum = summary.cumulative_effect
    rel = summary.relative_effect

    sig_text = if sig, do: "YES", else: "NO"
    ci_pct = round((1.0 - alpha) * 100)

    """
    Intervention Analysis Report
    ============================
    Cumulative effect: #{format_num(cum.mean)} (SD: #{format_num(cum.sd)})
    #{ci_pct}% CI: [#{format_num(cum.lower)}, #{format_num(cum.upper)}]
    Relative effect:  #{format_pct(rel.mean)} (SD: #{format_pct(rel.sd)})
    #{ci_pct}% CI: [#{format_pct(rel.lower)}, #{format_pct(rel.upper)}]
    Significant at #{alpha} level: #{sig_text}
    """
  end

  # -- Private implementation --

  defp analyze_mcmc(observations, pre_period, post_period, alpha, opts) do
    {elapsed_us, result} =
      Execution.measure(fn ->
        model_spec = resolve_model_spec(observations, pre_period, opts)

        ci_opts =
          opts
          |> Keyword.drop([:model_spec, :seasonality, :alpha, :method, :mode, :control_series])

        impact =
          if model_spec do
            CausalImpact.estimate_structured(
              observations,
              pre_period,
              post_period,
              model_spec,
              ci_opts
            )
          else
            CausalImpact.estimate(observations, pre_period, post_period, ci_opts)
          end

        summary = CausalImpact.summary(impact)
        sig = is_significant?(summary, alpha)

        %{
          impact: impact,
          summary: summary,
          significant?: sig,
          alpha: alpha,
          model_spec: model_spec
        }
      end)

    Map.put(
      result,
      :execution,
      Execution.metadata(:bayesian, :elixir_nx, :elixir_mcmc, elapsed_us)
    )
  end

  defp analyze_filter(
         observations,
         {_pre_start, _pre_end} = pre_period,
         post_period,
         alpha,
         opts
       ) do
    model_spec = resolve_model_spec(observations, pre_period, opts)

    spec =
      model_spec ||
        Components.local_level_spec(
          initial_state: Keyword.get(opts, :x0, List.first(observations) || 0.0),
          initial_cov: Keyword.get(opts, :p0, 1.0),
          process_var: Keyword.get(opts, :q, 1.0),
          obs_var: Keyword.get(opts, :r, 1.0)
        )

    prepared =
      Operational.prepare(
        spec,
        pre_period,
        post_period,
        opts
        |> Keyword.put(:alpha, alpha)
        |> Keyword.drop([:mode, :method, :num_samples, :burn_in, :control_series])
      )

    forecast = Operational.forecast(prepared, observations, alpha: alpha)
    summary = forecast.summary

    %{
      impact: nil,
      summary: summary,
      significant?: is_significant?(summary, alpha),
      alpha: alpha,
      model_spec: model_spec,
      execution: forecast.execution
    }
  end

  defp resolve_model_spec(observations, pre_period, opts) do
    case ModelBuilder.build_intervention_spec(observations, pre_period, opts) do
      {spec, :structured} -> spec
      {nil, :scalar} -> nil
    end
  end

  defp is_significant?(summary, _alpha) do
    cum = summary.cumulative_effect

    case {cum.lower, cum.upper} do
      {:nan, _} -> false
      {_, :nan} -> false
      {lower, upper} -> lower > 0.0 or upper < 0.0
    end
  end

  defp format_num(n), do: ModelBuilder.format_num(n)
  defp format_pct(n), do: ModelBuilder.format_pct(n)
end
