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
  alias BstsNx.Validation

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

  @known_analysis_opts [
    :model_spec,
    :seasonality,
    :control_series,
    :control_selection,
    :control_selection_pre_period,
    :control_regression_mode,
    :control_regression_opts,
    :alpha,
    :num_samples,
    :burn_in,
    :thin,
    :seed,
    :key,
    :mode,
    :method,
    :fallback,
    :allow_mcmc_fallback,
    :initial_state,
    :initial_cov,
    :process_var,
    :obs_var,
    :x0,
    :p0,
    :q,
    :r,
    :baseline,
    :return
  ]

  @mcmc_ci_opts [
    :num_samples,
    :burn_in,
    :thin,
    :seed,
    :key,
    :initial_state,
    :initial_cov,
    :process_var,
    :obs_var
  ]

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
    * `:mode` - `:bayesian` (default, Elixir MCMC path) or `:operational`
      (fast filter path)
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

    validate_known_options!(opts)

    %{pre_period: pre_period, post_period: post_period} = config

    alpha = Keyword.get(opts, :alpha, 0.05)
    Validation.validate_alpha!(alpha, "alpha must be between 0 and 1 (exclusive)")

    {pre_start, pre_end} = pre_period
    {post_start, post_end} = post_period
    obs_count = length(observations)

    Validation.validate_study_periods!(obs_count, pre_start, pre_end, post_start, post_end)

    seasonality = Keyword.get(opts, :seasonality)
    has_model_spec = match?(%BstsNx.ModelSpec{}, Keyword.get(opts, :model_spec))

    if not has_model_spec and seasonality != nil and
         (not is_integer(seasonality) or seasonality < 2) do
      raise ArgumentError,
            "seasonality must be an integer >= 2, got: #{inspect(seasonality)}"
    end

    mode = Execution.resolve_mode!(opts, :bayesian)

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

        ci_opts = Keyword.take(opts, @mcmc_ci_opts)

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
    try do
      analyze_filter!(observations, pre_period, post_period, alpha, opts)
    rescue
      e ->
        stacktrace = __STACKTRACE__

        cond do
          Execution.backend_lu_missing?(e) and Execution.explicit_mcmc_fallback?(opts) ->
            observations
            |> analyze_mcmc(pre_period, post_period, alpha, Keyword.put(opts, :mode, :bayesian))
            |> mark_fallback(Exception.message(e))

          Execution.backend_lu_missing?(e) ->
            Execution.raise_structured_filter_unsupported!(e)

          true ->
            reraise e, stacktrace
        end
    end
  end

  defp analyze_filter!(observations, pre_period, post_period, alpha, opts) do
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

  defp validate_known_options!(opts) do
    unknown =
      opts
      |> Keyword.keys()
      |> Enum.reject(&(&1 in @known_analysis_opts))

    case unknown do
      [] ->
        :ok

      [one] ->
        raise ArgumentError, "unknown option #{inspect(one)}"

      many ->
        raise ArgumentError, "unknown options #{inspect(many)}"
    end
  end

  defp mark_fallback(result, reason) do
    Map.update!(result, :execution, fn execution ->
      %{execution | fallback?: true, fallback_reason: reason}
    end)
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
