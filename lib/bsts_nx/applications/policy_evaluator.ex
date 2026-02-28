defmodule BstsNx.Applications.PolicyEvaluator do
  @moduledoc """
  Interrupted time series analysis for policy and intervention evaluation.

  Evaluates the causal effect of policy interventions, regulatory changes,
  and public programs using Bayesian Structural Time Series.  This is the
  standard methodology for policy evaluation when randomized experiments
  are not possible.

  ## Methodology

  Interrupted Time Series (ITS) analysis with BSTS:
  1. Model the pre-intervention time series (trend + seasonality + covariates)
  2. Generate counterfactual predictions for the post-intervention period
  3. Compute the effect as the difference between actual and counterfactual
  4. Quantify uncertainty via posterior credible intervals

  ## Applications

  | Domain          | Intervention                      | Outcome               |
  |-----------------|-----------------------------------|-----------------------|
  | Public health   | Vaccination campaign              | Case counts           |
  | Traffic safety  | Speed limit change                | Accident rates        |
  | Education       | Curriculum reform                 | Test scores           |
  | Environmental   | Emissions regulation              | Air quality metrics   |
  | Economic        | Tax policy change                 | Consumer spending     |
  | Healthcare      | Hospital policy change            | Readmission rates     |
  | Criminal justice| Sentencing reform                 | Recidivism rates      |

  ## Key features

  - **Control series**: Include unaffected regions/populations as covariates
    to strengthen the counterfactual
  - **Multiple outcomes**: Evaluate effects on multiple outcome metrics
  - **Lagged effects**: Account for delayed policy impact
  - **Effect dynamics**: Distinguish immediate vs. gradual effects

  ## Examples

      # Evaluate a speed limit change
      result = PolicyEvaluator.evaluate(
        monthly_accidents,
        %{
          intervention_name: "speed_limit_reduction",
          intervention_date_index: 37,   # month 37 = when policy took effect
          pre_period_start: 1,
          outcome_metric: "accidents_per_100k"
        }
      )

      result.effect.cumulative         # total reduction in accidents
      result.effect.per_period         # average monthly reduction
      result.significant?              # statistically significant?
      result.report                    # human-readable summary

      # With control region
      result = PolicyEvaluator.evaluate(
        treated_region_accidents,
        %{
          intervention_name: "speed_limit_reduction",
          intervention_date_index: 37,
          pre_period_start: 1,
          control_series: [neighboring_region_accidents]
        }
      )
  """

  alias BstsNx.InterventionAnalysis
  alias BstsNx.ModelBuilder

  @type intervention :: %{
          intervention_name: String.t(),
          intervention_date_index: pos_integer(),
          pre_period_start: pos_integer(),
          outcome_metric: String.t() | nil,
          control_series: [[number()]] | nil
        }

  @type evaluation_result :: %{
          intervention: intervention(),
          effect: ModelBuilder.effect(),
          significant?: boolean(),
          alpha: float(),
          n_pre: non_neg_integer(),
          n_post: non_neg_integer(),
          report: String.t(),
          analysis: InterventionAnalysis.analysis_result()
        }

  @doc """
  Evaluates the causal effect of a policy intervention.

  ## Parameters

    * `observations` - time series of the outcome metric, covering both
      pre and post-intervention periods
    * `intervention` - map defining the intervention (see `intervention()` type)

  ## Options

    * `:post_period_end` - end index of post-period (default: end of series)
    * `:seasonality` - seasonal periods for the outcome metric
    * `:control_selection` - optional control-series selection before
      composing the regression component. Set to `true` for default Pearson
      screening, or pass options for `BstsNx.CovariateSelection.select/3`.
    * `:control_regression_mode` - `:dynamic` (default) or `:spike_and_slab`
      for in-loop Bayesian variable selection with a g-prior.
    * `:control_regression_opts` - keyword options forwarded to
      `Components.regression_spec/2` when controls are present.
    * `:alpha` - significance level (default: 0.05)
    * `:num_samples` - posterior MCMC draws (default: 200)
    * `:seed` - PRNG seed
    * `:lag` - number of periods to skip after intervention before measuring
      effects (accounts for implementation delay, default: 0)
  """
  @spec evaluate([number()], intervention(), keyword()) :: evaluation_result()
  def evaluate(observations, intervention, opts \\ []) do
    if observations == [] do
      raise ArgumentError, "observations must be non-empty"
    end

    n = length(observations)
    alpha = Keyword.get(opts, :alpha, 0.05)
    lag = Keyword.get(opts, :lag, 0)
    post_end = Keyword.get(opts, :post_period_end, n)

    if not is_number(alpha) or alpha <= 0.0 or alpha >= 1.0 do
      raise ArgumentError, "alpha must be between 0 and 1 (exclusive), got: #{inspect(alpha)}"
    end

    if not is_integer(lag) or lag < 0 do
      raise ArgumentError, "lag must be a non-negative integer, got: #{inspect(lag)}"
    end

    pre_start = intervention.pre_period_start
    effective_post_start = intervention.intervention_date_index + lag
    # The pre_period must end immediately before post_period for CausalImpact.
    # When lag > 0, we extend the pre_period to include the lag window.
    pre_end = effective_post_start - 1

    if effective_post_start > post_end do
      raise ArgumentError,
            "post_period start (#{effective_post_start}) with lag #{lag} exceeds end (#{post_end})"
    end

    config = %{
      pre_period: {pre_start, pre_end},
      post_period: {effective_post_start, post_end}
    }

    analysis_opts = build_opts(observations, intervention, opts, {pre_start, pre_end})
    analysis = InterventionAnalysis.analyze(observations, config, analysis_opts)

    n_pre = pre_end - pre_start + 1
    n_post = post_end - effective_post_start + 1

    effect = ModelBuilder.build_effect(analysis.summary, n_post)
    report = build_report(intervention, effect, analysis, n_pre, n_post, alpha)

    %{
      intervention: intervention,
      effect: effect,
      significant?: analysis.significant?,
      alpha: alpha,
      n_pre: n_pre,
      n_post: n_post,
      report: report,
      analysis: analysis
    }
  end

  @doc """
  Evaluates the same intervention across multiple outcome metrics.

  Runs `evaluate/3` for each outcome time series and returns a
  consolidated multi-outcome report.

  ## Parameters

    * `outcome_series` - map of `%{metric_name => observations}`
    * `intervention` - shared intervention definition
  """
  @spec evaluate_multi(%{String.t() => [number()]}, intervention(), keyword()) ::
          %{String.t() => evaluation_result()}
  def evaluate_multi(outcome_series, intervention, opts \\ []) do
    Map.new(outcome_series, fn {metric, observations} ->
      int_with_metric = Map.put(intervention, :outcome_metric, metric)
      {metric, evaluate(observations, int_with_metric, opts)}
    end)
  end

  @doc """
  Compares pre-intervention trends to validate the parallel trends assumption.

  For ITS analysis to be valid, the pre-intervention data should not show
  a diverging trend between treated and control series.  This function
  checks for pre-existing trends that might confound the analysis.

  Returns a diagnostic map with trend metrics.

  ## Examples

      iex> intervention = %{intervention_name: "policy", intervention_date_index: 11, pre_period_start: 1}
      iex> result = BstsNx.Applications.PolicyEvaluator.pre_trend_check(Enum.to_list(1..15), intervention)
      iex> result.valid
      true
      iex> is_float(result.slope)
      true
  """
  @spec pre_trend_check([number()], intervention(), keyword()) :: map()
  def pre_trend_check(observations, intervention, _opts \\ []) do
    pre_start = intervention.pre_period_start
    pre_end = intervention.intervention_date_index - 1

    pre_data = Enum.slice(observations, (pre_start - 1)..(pre_end - 1))
    n = length(pre_data)

    if n < 3 do
      %{valid: false, reason: "insufficient pre-period data (#{n} points, need >= 3)"}
    else
      # Simple linear trend check via OLS
      x_mean = (n - 1) / 2.0
      y_mean = Enum.sum(pre_data) / n

      {ss_xy, ss_xx} =
        pre_data
        |> Enum.with_index()
        |> Enum.reduce({0.0, 0.0}, fn {y, i}, {sxy, sxx} ->
          dx = i - x_mean
          dy = y - y_mean
          {sxy + dx * dy, sxx + dx * dx}
        end)

      slope = if ss_xx > 1.0e-10, do: ss_xy / ss_xx, else: 0.0

      # Residual variance
      intercept = y_mean - slope * x_mean

      residuals =
        pre_data
        |> Enum.with_index()
        |> Enum.map(fn {y, i} -> y - (intercept + slope * i) end)

      resid_var =
        if n > 2 do
          Enum.reduce(residuals, 0.0, fn r, acc -> acc + r * r end) / (n - 2)
        else
          0.0
        end

      slope_se = if ss_xx > 1.0e-10, do: :math.sqrt(resid_var / ss_xx), else: 0.0
      t_stat = if slope_se > 1.0e-10, do: slope / slope_se, else: 0.0

      # Check against control series if provided
      control_check =
        case Map.get(intervention, :control_series) do
          nil ->
            nil

          controls ->
            Enum.map(controls, fn control ->
              ctrl_pre = Enum.slice(control, (pre_start - 1)..(pre_end - 1))
              ctrl_mean = Enum.sum(ctrl_pre) / length(ctrl_pre)

              diff =
                Enum.zip(pre_data, ctrl_pre)
                |> Enum.map(fn {t, c} -> t - c end)

              diff_mean = Enum.sum(diff) / length(diff)
              %{control_mean: ctrl_mean, mean_difference: diff_mean}
            end)
        end

      %{
        valid: true,
        slope: slope,
        slope_se: slope_se,
        t_statistic: t_stat,
        significant_trend?: abs(t_stat) > 2.0,
        n_pre: n,
        control_diagnostics: control_check
      }
    end
  end

  # -- Private implementation --

  defp build_opts(observations, intervention, opts, pre_period) do
    controls = Map.get(intervention, :control_series)

    opts_with_selection_window =
      Keyword.put_new(opts, :control_selection_pre_period, pre_period)

    ModelBuilder.build_opts_with_controls(observations, controls, opts_with_selection_window)
  end

  defp build_report(intervention, effect, analysis, n_pre, n_post, alpha) do
    name = intervention.intervention_name || "intervention"
    metric = intervention[:outcome_metric] || "outcome"
    ci_pct = round((1.0 - alpha) * 100)
    sig_text = if analysis.significant?, do: "YES", else: "NO"

    """
    Policy Evaluation Report: #{name}
    ================================================
    Outcome metric: #{metric}
    Pre-intervention periods:  #{n_pre}
    Post-intervention periods: #{n_post}

    Cumulative effect:     #{format_num(effect.cumulative)} (SD: #{format_num(effect.cumulative_sd)})
    #{ci_pct}% CI:              [#{format_num(effect.cumulative_lower)}, #{format_num(effect.cumulative_upper)}]
    Average effect/period: #{format_num(effect.per_period)} (SD: #{format_num(effect.per_period_sd)})
    Relative change:       #{format_pct(effect.relative)} (SD: #{format_pct(effect.relative_sd)})

    Statistically significant: #{sig_text}

    Interpretation:
    #{interpret_effect(effect, analysis.significant?, name)}
    """
  end

  defp interpret_effect(effect, true, name) do
    direction = if effect.cumulative > 0, do: "increased", else: "decreased"

    "The #{name} #{direction} the outcome by #{format_pct(abs(effect.relative))} " <>
      "on average. This effect is statistically significant."
  end

  defp interpret_effect(_effect, false, name) do
    "The #{name} did not produce a statistically significant effect. " <>
      "The observed change is consistent with normal variation."
  end

  defp format_num(n), do: ModelBuilder.format_num(n)
  defp format_pct(n), do: ModelBuilder.format_pct(n)
end
