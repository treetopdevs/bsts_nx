defmodule LivebookGuideValidationTest do
  @moduledoc """
  Validates that all code cells in livebooks/bsts_nx_guide.livemd
  execute without errors. Each test corresponds to a numbered section.
  """
  use ExUnit.Case, async: false

  @moduletag :slow
  @moduletag timeout: 300_000

  alias BstsNx.{
    CausalImpact,
    Components,
    CovariateSelection,
    Diagnostics,
    Forecaster,
    GibbsSampler,
    InterventionAnalysis,
    KalmanFilter,
    Pipeline,
    RollingBaseline,
    Smoother,
    SpotAttributor,
    Validation
  }

  alias BstsNx.Applications.{
    AnomalyDetector,
    DemandForecaster,
    MarketingLift,
    PolicyEvaluator,
    TVAttribution
  }

  alias BstsNx.Synthetic.{Adstock, Generator}

  # ── Shared helpers (mirrors Guide module from the livebook) ──

  defp seed!(n) do
    :rand.seed(:exsss, {n, n + 1, n + 2})
    Nx.Random.key(n)
  end

  defp downsample(series, every) do
    series
    |> Enum.chunk_every(every, every, :discard)
    |> Enum.map(fn chunk -> Enum.sum(chunk) / length(chunk) end)
  end

  defp per_t_bands(draws, alpha) when draws != [] do
    Enum.zip_with(draws, fn vals ->
      sorted = Enum.sort(vals)

      %{
        mean: Enum.sum(vals) / length(vals),
        lower: quantile(sorted, alpha / 2),
        upper: quantile(sorted, 1 - alpha / 2)
      }
    end)
  end

  defp quantile(sorted, q) when sorted != [] do
    pos = q * (length(sorted) - 1)
    lo = trunc(:math.floor(pos))
    hi = trunc(:math.ceil(pos))

    if lo == hi,
      do: Enum.at(sorted, lo),
      else: (1 - pos + lo) * Enum.at(sorted, lo) + (pos - lo) * Enum.at(sorted, hi)
  end

  # ── Shared setup ──

  setup_all do
    Nx.global_default_backend(Nx.BinaryBackend)
    seed!(2026)

    post_start_minute = 4 * 24 * 60

    spots = [
      %{
        id: "spot_a",
        window_start: post_start_minute + 2 * 60,
        window_end: post_start_minute + 4 * 60
      },
      %{
        id: "spot_b",
        window_start: post_start_minute + 3 * 60,
        window_end: post_start_minute + 5 * 60
      },
      %{
        id: "spot_c",
        window_start: post_start_minute + 28 * 60,
        window_end: post_start_minute + 30 * 60
      },
      %{
        id: "spot_d",
        window_start: post_start_minute + 29 * 60,
        window_end: post_start_minute + 31 * 60
      }
    ]

    config = %{
      name: "Web traffic + TV campaign",
      total_minutes: 6 * 24 * 60,
      baseline: %{
        intercept: 80.0,
        trend: 0.5,
        daily_fourier: [{8.0, 3.0}, {2.0, 1.0}],
        weekly_fourier: [{2.0, 1.0}]
      },
      spots: spots,
      noise_sd: 2.0,
      effect: %{lambda: 0.65, ec: 1.0, slope: 2.0, coefficient: 12.0},
      controls: %{count: 3, correlation: 0.45},
      seed: 2026
    }

    synthetic = Generator.generate(config)

    obs = downsample(synthetic.observations, 60)
    baseline_true = downsample(synthetic.ground_truth.baseline, 60)
    tv_true = downsample(synthetic.ground_truth.tv_contribution, 60)
    controls = Enum.map(synthetic.controls, &downsample(&1, 60))

    pre_period = {1, 96}
    post_period = {97, 144}
    post_len = 48

    spots_post =
      synthetic.spots
      |> Enum.map(fn s ->
        ws = div(s.window_start - post_start_minute, 60)
        we = div(s.window_end - post_start_minute + 59, 60)
        %{id: s.id, window_start: max(ws, 0), window_end: min(we, post_len)}
      end)
      |> Enum.filter(&(&1.window_end > &1.window_start and &1.window_start < post_len))

    true_post_lift = tv_true |> Enum.slice(96..143) |> Enum.sum()

    {:ok,
     %{
       obs: obs,
       baseline_true: baseline_true,
       tv_true: tv_true,
       controls: controls,
       pre_period: pre_period,
       post_period: post_period,
       spots_post: spots_post,
       true_post_lift: true_post_lift,
       post_start_minute: post_start_minute
     }}
  end

  # ── Section 1: Random walk ──

  test "section 1 - random walk generation" do
    key = seed!(99)
    {random_steps, _} = Nx.Random.normal(key, shape: {100}, type: {:f, 32})
    random_walk = Nx.to_flat_list(Nx.cumulative_sum(random_steps))
    assert length(random_walk) == 100
    assert is_float(hd(random_walk))
  end

  # ── Section 3: Synthetic world ──

  test "section 3 - synthetic data generation", ctx do
    assert length(ctx.obs) == 144
    assert length(ctx.baseline_true) == 144
    assert length(ctx.tv_true) == 144
    assert length(ctx.spots_post) > 0
    assert is_float(ctx.true_post_lift)
  end

  # ── Section 4: Adstock + saturation ──

  test "section 4 - adstock and hill saturation" do
    impulse = [1.0] ++ List.duplicate(0.0, 47)
    adstocked = Adstock.geometric_adstock(impulse, 0.7)
    saturated = Adstock.hill_saturation(adstocked, 0.3, 3.5)

    assert length(adstocked) == 48
    assert length(saturated) == 48
    assert Enum.max(saturated) <= 1.0
    assert Enum.max(saturated) > 0.0
  end

  # ── Section 5: Kalman filter ──

  test "section 5 - kalman filter", ctx do
    kalman_obs = Enum.take(ctx.obs, 72)

    {filtered, predicted} =
      KalmanFilter.filter_with_pred(kalman_obs, 1.0, 1.0, 0.8, 4.0, List.first(kalman_obs), 10.0)

    filtered_mean = Enum.map(filtered, fn {x, _p} -> Nx.to_number(x) end)
    pred_mean = Enum.map(predicted, fn {x, _p} -> Nx.to_number(x) end)

    assert length(filtered_mean) == 72
    assert length(pred_mean) == 72

    # Compiled version
    obs_tensor = Nx.tensor(kalman_obs, type: {:f, 32})

    {xs_defn, _ps_defn} =
      KalmanFilter.filter_defn(obs_tensor, 1.0, 1.0, 0.8, 4.0, List.first(kalman_obs), 10.0)

    defn_mean = Nx.to_flat_list(xs_defn)
    assert length(defn_mean) == 72
  end

  # ── Section 6: Smoother ──

  test "section 6 - RTS smoother", ctx do
    kalman_obs = Enum.take(ctx.obs, 72)

    {filtered, predicted} =
      KalmanFilter.filter_with_pred(kalman_obs, 1.0, 1.0, 0.8, 4.0, List.first(kalman_obs), 10.0)

    smoothed = Smoother.rts(filtered, predicted, 1.0)
    smoothed_mean = Enum.map(smoothed, fn {x, _p} -> Nx.to_number(x) end)
    assert length(smoothed_mean) == 72

    # Posterior state draw
    {state_draw, _} =
      Smoother.simulate_with_key(smoothed, filtered, predicted, 1.0, key: Nx.Random.key(42))

    state_draw_vals = Enum.map(state_draw, &Nx.to_number/1)
    assert length(state_draw_vals) == 72
  end

  # ── Section 7.1: Scalar Gibbs sampler ──

  test "section 7.1 - scalar gibbs sampler", ctx do
    pre_obs = Enum.take(ctx.obs, 96)

    scalar_samples =
      GibbsSampler.sample(
        pre_obs,
        50,
        List.first(ctx.obs),
        10.0,
        0.8,
        4.0,
        burn_in: 25,
        seed: 11
      )

    q_chain = Enum.map(scalar_samples, &Nx.to_number(&1.process_var))
    r_chain = Enum.map(scalar_samples, &Nx.to_number(&1.obs_var))

    assert length(q_chain) == 50
    assert length(r_chain) == 50
    assert Enum.all?(q_chain, &(&1 > 0))
    assert Enum.all?(r_chain, &(&1 > 0))
  end

  # ── Section 7.2: Multiple chains + diagnostics ──

  test "section 7.2 - multiple chains and diagnostics", ctx do
    pre_obs = Enum.take(ctx.obs, 96)

    chains =
      GibbsSampler.sample_chains(
        pre_obs,
        3,
        30,
        List.first(ctx.obs),
        10.0,
        0.8,
        4.0,
        burn_in: 15,
        seed: 123
      )

    q_chains = Enum.map(chains, fn chain -> Enum.map(chain, &Nx.to_number(&1.process_var)) end)

    assert length(q_chains) == 3

    r_hat = Diagnostics.r_hat(q_chains)
    ess = Diagnostics.effective_sample_size(q_chains)

    assert is_float(r_hat)
    assert is_float(ess) or is_integer(ess)
  end

  # ── Section 7.3: Structured model ──

  test "section 7.3 - structured model composition", ctx do
    regressors = ctx.controls |> Nx.tensor() |> Nx.transpose()

    trend_spec =
      Components.local_linear_trend_spec(
        initial_level: List.first(ctx.obs),
        var_level: 0.2,
        var_slope: 0.02,
        obs_var: 2.0
      )

    seasonal_spec = Components.seasonal_spec(8, process_var: 0.05, obs_var: 2.0)
    regression_spec = Components.regression_spec(regressors, var_beta: 0.02, obs_var: 2.0)

    structured_spec =
      trend_spec
      |> Components.compose_specs(seasonal_spec)
      |> Components.compose_specs(regression_spec)

    assert Nx.axis_size(structured_spec.f, 0) > 1

    structured_samples =
      GibbsSampler.sample_structured(ctx.obs, structured_spec, 2, burn_in: 0, seed: 77)

    assert length(structured_samples) > 0
  end

  # ── Section 8.1: Scalar causal impact ──

  test "section 8.1 - scalar causal impact", ctx do
    impact_scalar =
      CausalImpact.estimate(ctx.obs, ctx.pre_period, ctx.post_period,
        num_samples: 20,
        burn_in: 10,
        seed: 999
      )

    summary_scalar = CausalImpact.summary(impact_scalar)
    assert is_float(summary_scalar.cumulative_effect.mean)
    assert is_list(impact_scalar.actual)
    assert is_list(impact_scalar.counterfactual)

    cf_bands = per_t_bands(impact_scalar.counterfactual, 0.05)
    assert length(cf_bands) > 0
    assert Map.has_key?(hd(cf_bands), :mean)
  end

  # ── Section 8.2: Structured causal impact ──

  test "section 8.2 - structured causal impact", ctx do
    regressors = ctx.controls |> Nx.tensor() |> Nx.transpose()

    structured_spec =
      Components.local_linear_trend_spec(
        initial_level: List.first(ctx.obs),
        var_level: 0.2,
        var_slope: 0.02,
        obs_var: 2.0
      )
      |> Components.compose_specs(Components.seasonal_spec(8, process_var: 0.05, obs_var: 2.0))
      |> Components.compose_specs(
        Components.regression_spec(regressors, var_beta: 0.02, obs_var: 2.0)
      )

    impact_structured =
      CausalImpact.estimate_structured(
        ctx.obs,
        ctx.pre_period,
        ctx.post_period,
        structured_spec,
        num_samples: 2,
        burn_in: 0,
        seed: 202
      )

    summary = CausalImpact.summary(impact_structured)
    assert is_float(summary.cumulative_effect.mean)
    assert is_list(impact_structured.counterfactual)
  end

  # ── Section 8.4: Filter-based estimate ──

  test "section 8.4 - filter-based causal impact", ctx do
    filter_estimate =
      CausalImpact.estimate_from_filter(ctx.obs, Enum.to_list(96..143),
        f: 1.0,
        h: 1.0,
        q: 0.8,
        r: 4.0,
        x0: List.first(ctx.obs),
        p0: 10.0,
        alpha: 0.05
      )

    assert is_float(filter_estimate.cumulative_effect.mean)
  end

  # ── Section 9: InterventionAnalysis ──

  test "section 9 - intervention analysis one-call API", ctx do
    analysis =
      InterventionAnalysis.analyze(
        ctx.obs,
        %{pre_period: ctx.pre_period, post_period: ctx.post_period},
        seasonality: 24,
        method: :filter,
        q: 0.8,
        r: 4.0,
        x0: List.first(ctx.obs),
        p0: 10.0,
        seed: 42
      )

    assert is_boolean(analysis.significant?)
    assert is_float(analysis.summary.cumulative_effect.mean)

    report = InterventionAnalysis.report(analysis)
    assert is_binary(report)
  end

  # ── Section 10: Attribution ──

  test "section 10 - shapley attribution", ctx do
    # Need structured impact first
    regressors = ctx.controls |> Nx.tensor() |> Nx.transpose()

    structured_spec =
      Components.local_linear_trend_spec(
        initial_level: List.first(ctx.obs),
        var_level: 0.2,
        var_slope: 0.02,
        obs_var: 2.0
      )
      |> Components.compose_specs(Components.seasonal_spec(8, process_var: 0.05, obs_var: 2.0))
      |> Components.compose_specs(
        Components.regression_spec(regressors, var_beta: 0.02, obs_var: 2.0)
      )

    impact_structured =
      CausalImpact.estimate_structured(
        ctx.obs,
        ctx.pre_period,
        ctx.post_period,
        structured_spec,
        num_samples: 2,
        burn_in: 0,
        seed: 202
      )

    post_obs = Enum.slice(ctx.obs, 96..143)

    attr_posterior =
      SpotAttributor.attribute_posterior(
        post_obs,
        ctx.spots_post,
        impact_structured.counterfactual,
        0.0,
        alpha: 0.05,
        n_samples: 500,
        decay: 0.7
      )

    assert is_float(attr_posterior.total_lift)
    assert is_list(attr_posterior.attributions)
    assert length(attr_posterior.attributions) > 0

    sum_of_spots = Enum.reduce(attr_posterior.attributions, 0.0, fn a, acc -> acc + a.lift end)
    assert abs(attr_posterior.total_lift - sum_of_spots) < 5.0
  end

  # ── Section 11: Pipeline ──

  test "section 11 - full pipeline", ctx do
    regressors = ctx.controls |> Nx.tensor() |> Nx.transpose()

    structured_spec =
      Components.local_linear_trend_spec(
        initial_level: List.first(ctx.obs),
        var_level: 0.2,
        var_slope: 0.02,
        obs_var: 2.0
      )
      |> Components.compose_specs(Components.seasonal_spec(8, process_var: 0.05, obs_var: 2.0))
      |> Components.compose_specs(
        Components.regression_spec(regressors, var_beta: 0.02, obs_var: 2.0)
      )

    pipeline_result =
      Pipeline.run(
        ctx.obs,
        ctx.pre_period,
        ctx.post_period,
        ctx.spots_post,
        structured_spec,
        num_samples: 3,
        burn_in: 1,
        seed: 404,
        alpha: 0.05,
        shapley_samples: 500,
        decay: 0.7
      )

    assert is_float(pipeline_result.attributions.total_lift)
  end

  # ── Section 12: Marketing lift ──

  test "section 12 - marketing lift case study" do
    seed!(3001)

    n_pre_days = 30
    n_post_days = 14
    total_days = n_pre_days + n_post_days
    true_campaign_effect = 25.0

    daily_baseline =
      for d <- 0..(total_days - 1) do
        200.0 + 0.8 * d + 15.0 * :math.sin(2 * :math.pi() * d / 7) + :rand.normal() * 8.0
      end

    daily_obs =
      daily_baseline
      |> Enum.with_index()
      |> Enum.map(fn {b, i} ->
        if i >= n_pre_days, do: b + true_campaign_effect + :rand.normal() * 3.0, else: b
      end)

    campaign = %{
      name: "summer_email_blast",
      channel: :email,
      start_index: n_pre_days + 1,
      end_index: total_days,
      baseline_start: 1,
      baseline_end: n_pre_days
    }

    marketing_result =
      MarketingLift.measure_lift(daily_obs, campaign,
        seasonality: 7,
        num_samples: 3,
        burn_in: 1,
        seed: 700
      )

    assert is_boolean(marketing_result.significant?)
    assert is_float(marketing_result.effect.cumulative)
  end

  # ── Section 13: Demand forecasting ──

  test "section 13 - demand forecasting case study" do
    seed!(4001)

    demand_history =
      for d <- 0..119 do
        base = 500.0 + 2.0 * d

        seasonal =
          80.0 * :math.sin(2 * :math.pi() * d / 7) + 30.0 * :math.cos(2 * :math.pi() * d / 7)

        base + seasonal + :rand.normal() * 25.0
      end

    forecast =
      DemandForecaster.forecast(demand_history,
        horizon: 14,
        seasonality: 7,
        num_samples: 3,
        burn_in: 1,
        seed: 16
      )

    assert length(forecast.mean) == 14
    assert length(forecast.lower) == 14
    assert length(forecast.upper) == 14

    safety = DemandForecaster.safety_stock(forecast, service_level: 0.95, lead_time: 7)
    assert is_float(safety.total) or is_number(safety.total)
  end

  # ── Section 14: Anomaly detection ──

  test "section 14 - anomaly detection case study", ctx do
    normal_history = Enum.take(ctx.obs, 96)
    stream = Enum.take(Enum.drop(ctx.obs, 96), 24)

    stream_with_anomalies =
      stream
      |> List.update_at(8, &(&1 + 35.0))
      |> List.update_at(16, &(&1 - 30.0))

    detector = AnomalyDetector.fit(normal_history, method: :filter, alpha: 0.01)
    scores = AnomalyDetector.score(detector, stream_with_anomalies)
    summary = AnomalyDetector.summary(scores)

    assert is_integer(summary.anomalies) or is_number(summary.anomalies)
    assert length(scores) == 24
    assert Map.has_key?(hd(scores), :z_score)
  end

  # ── Section 15: Policy evaluation ──

  test "section 15 - policy evaluation case study" do
    seed!(5001)

    n_pre_months = 36
    n_post_months = 24
    total_months = n_pre_months + n_post_months
    true_policy_effect = -8.0

    monthly_baseline =
      for m <- 0..(total_months - 1) do
        base = 45.0 - 0.1 * m
        seasonal = 5.0 * :math.sin(2 * :math.pi() * m / 12)
        base + seasonal + :rand.normal() * 3.0
      end

    monthly_obs =
      monthly_baseline
      |> Enum.with_index()
      |> Enum.map(fn {b, i} ->
        if i >= n_pre_months, do: b + true_policy_effect + :rand.normal() * 2.0, else: b
      end)

    intervention = %{
      intervention_name: "speed_limit_reduction",
      intervention_date_index: n_pre_months + 1,
      pre_period_start: 1,
      outcome_metric: "accidents per month"
    }

    policy_result =
      PolicyEvaluator.evaluate(monthly_obs, intervention,
        post_period_end: total_months,
        seasonality: 12,
        num_samples: 3,
        burn_in: 1,
        seed: 701
      )

    assert is_boolean(policy_result.significant?)
    assert is_float(policy_result.effect.cumulative)
  end

  # ── Section 16: TV attribution ──

  test "section 16 - TV attribution wrapper", ctx do
    tv_result =
      TVAttribution.attribute(
        ctx.obs,
        ctx.spots_post,
        pre_period: ctx.pre_period,
        post_period: ctx.post_period,
        seasonality: 24,
        num_samples: 3,
        burn_in: 1,
        seed: 501,
        alpha: 0.05
      )

    assert is_float(tv_result.total_lift)
    assert is_float(tv_result.total_lift_sd)
    assert is_list(tv_result.spots)
  end

  # ── Section 17: Covariate selection ──

  test "section 17 - covariate selection", ctx do
    target_pre = Enum.take(ctx.obs, 96)

    candidate_matrix =
      ctx.controls
      |> Enum.map(&Enum.take(&1, 96))
      |> Nx.tensor()
      |> Nx.transpose()

    selection =
      CovariateSelection.select(target_pre, candidate_matrix,
        threshold: 0.15,
        max_controls: 2
      )

    assert is_list(selection.correlations)
    assert is_list(selection.selected_indices)
  end

  # ── Section 18: Validation ──

  test "section 18 - validation and calibration", ctx do
    post_obs_list = Enum.slice(ctx.obs, 96..143)

    {rolling_cf, _rolling_fit} =
      RollingBaseline.fit_and_predict(
        Enum.take(ctx.obs, 96),
        48,
        num_seasons: 24,
        num_samples: 3,
        burn_in: 1,
        seed: 500
      )

    actual_t = Nx.tensor(post_obs_list)
    baseline_t = Nx.tensor(rolling_cf.mean)
    state_var_t = Nx.tensor(rolling_cf.variance)
    indices = Enum.to_list(0..(length(post_obs_list) - 1))

    evaluation =
      Validation.evaluate(%{
        actual: actual_t,
        baseline: baseline_t,
        indices: indices,
        coverage: %{state_variances: state_var_t, obs_variance: rolling_cf.obs_variance}
      })

    assert is_map(evaluation.details.prediction_error)
    assert is_map(evaluation.details.coverage)
    assert is_map(evaluation.details.durbin_watson)
    assert evaluation.verdicts.placebo == :skip
    assert evaluation.verdicts.effect_stability == :skip

    # Calibration
    calibration =
      Validation.known_lift_injection(
        40.0,
        Components.local_level_spec(),
        n_pre: 60,
        n_post: 20,
        num_samples: 3,
        burn_in: 1,
        seed: 2026,
        credible_level: 0.95
      )

    assert is_boolean(calibration.covered)
    assert is_float(calibration.estimated_mean)
  end

  # ── Section 19: Forecasting ──

  test "section 19 - forecasting", ctx do
    train = Enum.take(ctx.obs, 120)
    test_data = Enum.drop(ctx.obs, 120)

    forecast =
      Forecaster.fit_predict(train, length(test_data),
        seasonality: 24,
        num_samples: 3,
        burn_in: 1,
        seed: 15
      )

    assert length(forecast.mean) == length(test_data)
    assert length(forecast.lower) == length(test_data)
    assert length(forecast.upper) == length(test_data)

    # AR comparison
    ar_forecast =
      BstsNx.BCT.ARForecaster.fit_predict(train, length(test_data),
        order: 2,
        num_samples: 200,
        seed: 17
      )

    assert length(ar_forecast.mean) == length(test_data)
  end

  # ── Section 21: API reference ──

  test "section 21 - all modules export __info__(:functions)" do
    modules = [
      BstsNx.KalmanFilter,
      BstsNx.Smoother,
      BstsNx.GibbsSampler,
      BstsNx.Components,
      BstsNx.CausalImpact,
      BstsNx.InterventionAnalysis,
      BstsNx.Forecaster,
      BstsNx.Pipeline,
      BstsNx.Applications.AnomalyDetector,
      BstsNx.Applications.DemandForecaster,
      BstsNx.Applications.MarketingLift,
      BstsNx.Applications.PolicyEvaluator,
      BstsNx.Applications.TVAttribution,
      BstsNx.SpotAttributor,
      BstsNx.ShapleyAllocator,
      BstsNx.CovariateSelection,
      BstsNx.Validation,
      BstsNx.Diagnostics,
      BstsNx.Synthetic.Generator,
      BstsNx.Synthetic.Adstock
    ]

    for mod <- modules do
      funcs = mod.__info__(:functions)
      assert is_list(funcs), "#{inspect(mod)} should export functions"
    end
  end
end
