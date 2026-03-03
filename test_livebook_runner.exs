# Temporary script to validate all livebook code cells run without errors.
# Stubs Kino and VegaLite since we're running headless.

defmodule VegaLite do
  def new(opts \\ []), do: %{type: :vl, opts: opts}
  def data_from_values(vl, _rows), do: vl
  def mark(vl, _mark), do: vl
  def encode_field(vl, _channel, _field, _opts \\ []), do: vl
end

defmodule Kino.VegaLite do
  def new(vl), do: vl
end

defmodule Kino.DataTable do
  def new(rows), do: %{type: :data_table, n_rows: length(rows)}
end

alias VegaLite, as: Vl

# ── Section 1: Setup (skip Mix.install, already in project) ──

Nx.global_default_backend(Nx.BinaryBackend)

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
  ShapleyAllocator,
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

IO.puts("=== Section 1: Setup ✓ ===")

# ── NotebookHelpers ──

defmodule NotebookHelpers do
  def seed!(seed \\ 42) do
    :rand.seed(:exsss, {seed, seed + 1, seed + 2})
    Nx.Random.key(seed)
  end

  def backend_status do
    process_count = :erlang.system_info(:process_count)
    process_limit = :erlang.system_info(:process_limit)

    %{
      default_backend: Nx.default_backend(),
      process_count: process_count,
      process_limit: process_limit,
      headroom: process_limit - process_count
    }
  end

  def use_binary! do
    Nx.global_default_backend(Nx.BinaryBackend)
    backend_status()
  end

  def try_emlx!(opts \\ []) do
    _warn = Keyword.get(opts, :warn, true)

    if Code.ensure_loaded?(EMLX.Backend) do
      try do
        Nx.global_default_backend(EMLX.Backend)
        Nx.Defn.global_default_options(compiler: Nx.Defn.Evaluator)
        _ = Nx.backend_transfer(Nx.tensor([0.0]), EMLX.Backend)
        backend_status() |> Map.put(:emlx_status, :enabled)
      rescue
        _ ->
          Nx.global_default_backend(Nx.BinaryBackend)
          Nx.Defn.global_default_options(compiler: Nx.Defn.Evaluator)
          backend_status() |> Map.put(:emlx_status, :fallback_binary)
      catch
        :exit, _ ->
          Nx.global_default_backend(Nx.BinaryBackend)
          Nx.Defn.global_default_options(compiler: Nx.Defn.Evaluator)
          backend_status() |> Map.put(:emlx_status, :fallback_binary)
      end
    else
      backend_status() |> Map.put(:emlx_status, :unavailable)
    end
  end

  def try_exla!(opts \\ []) do
    min_headroom = Keyword.get(opts, :min_headroom, 10_000)
    status = backend_status()

    if status.headroom < min_headroom do
      Nx.global_default_backend(Nx.BinaryBackend)
      backend_status() |> Map.put(:exla_status, :skipped_low_headroom)
    else
      try do
        Nx.global_default_backend(Nx.BinaryBackend)
        Nx.Defn.global_default_options(compiler: EXLA)
        _ = Nx.backend_transfer(Nx.tensor([0.0]), EXLA.Backend)
        backend_status() |> Map.put(:exla_status, :enabled) |> Map.put(:exla_mode, :compiler_only)
      rescue
        _ ->
          Nx.global_default_backend(Nx.BinaryBackend)
          Nx.Defn.global_default_options(compiler: Nx.Defn.Evaluator)
          backend_status() |> Map.put(:exla_status, :fallback_binary)
      catch
        :exit, _ ->
          Nx.global_default_backend(Nx.BinaryBackend)
          Nx.Defn.global_default_options(compiler: Nx.Defn.Evaluator)
          backend_status() |> Map.put(:exla_status, :fallback_binary)
      end
    end
  end

  def try_compiled_backend!(opts \\ []) do
    emlx_result = try_emlx!(Keyword.put(opts, :warn, false))

    if emlx_result.emlx_status == :enabled do
      emlx_result |> Map.put(:compiled_backend, :emlx)
    else
      exla_result = try_exla!(Keyword.put(opts, :warn, false))
      compiled_backend = if exla_result.exla_status == :enabled, do: :exla, else: :binary

      exla_result
      |> Map.put(:compiled_backend, compiled_backend)
      |> Map.put(:emlx_status, emlx_result.emlx_status)
    end
  end

  def mean(list) when is_list(list) and list != [], do: Enum.sum(list) / length(list)
  def mean(_), do: 0.0

  def sd(list) when is_list(list) and length(list) > 1 do
    m = mean(list)
    ss = Enum.reduce(list, 0.0, fn x, acc -> acc + (x - m) * (x - m) end)
    :math.sqrt(ss / (length(list) - 1))
  end

  def sd(_), do: 0.0

  def mae(a, b) when is_list(a) and is_list(b) and a != [] and length(a) == length(b) do
    Enum.zip(a, b) |> Enum.map(fn {x, y} -> abs(x - y) end) |> mean()
  end

  def mae(_, _), do: 0.0

  def rmse(a, b) when is_list(a) and is_list(b) and a != [] and length(a) == length(b) do
    Enum.zip(a, b) |> Enum.map(fn {x, y} -> (x - y) * (x - y) end) |> mean() |> :math.sqrt()
  end

  def rmse(_, _), do: 0.0

  def quantile(sorted, q) when sorted != [] do
    pos = q * (length(sorted) - 1)
    lo = trunc(:math.floor(pos))
    hi = trunc(:math.ceil(pos))

    if lo == hi,
      do: Enum.at(sorted, lo),
      else: (1.0 - (pos - lo)) * Enum.at(sorted, lo) + (pos - lo) * Enum.at(sorted, hi)
  end

  def quantile(_, _), do: :nan

  def posterior_mean(draws) when is_list(draws) and draws != [] do
    n = length(draws)
    Enum.zip_with(draws, fn vals -> Enum.sum(vals) / n end)
  end

  def posterior_mean(_), do: []

  def posterior_var(draws) when is_list(draws) and draws != [] do
    draws
    |> Enum.zip_with(fn vals ->
      m = Enum.sum(vals) / length(vals)
      ss = Enum.reduce(vals, 0.0, fn x, acc -> acc + (x - m) * (x - m) end)
      ss / max(length(vals) - 1, 1)
    end)
  end

  def posterior_var(_), do: []

  def per_t_bands(draws, alpha \\ 0.05)

  def per_t_bands(draws, alpha) when is_list(draws) and draws != [] do
    Enum.zip_with(draws, fn vals ->
      sorted = Enum.sort(vals)

      %{
        mean: Enum.sum(vals) / length(vals),
        lower: quantile(sorted, alpha / 2.0),
        upper: quantile(sorted, 1.0 - alpha / 2.0)
      }
    end)
  end

  def per_t_bands(_, _), do: []

  def downsample(series, every) do
    series
    |> Enum.chunk_every(every, every, :discard)
    |> Enum.map(fn chunk -> Enum.sum(chunk) / length(chunk) end)
  end

  def to_rows(series_map) when is_map(series_map) do
    series_map
    |> Enum.flat_map(fn {name, values} ->
      Enum.with_index(values) |> Enum.map(fn {v, t} -> %{series: name, t: t, value: v} end)
    end)
  end

  def line_plot(rows, opts \\ []) do
    title = Keyword.get(opts, :title, "")
    IO.puts("  [plot] #{title} (#{length(rows)} data points)")
    :ok
  end

  def interval_plot(rows, opts \\ []) do
    title = Keyword.get(opts, :title, "")
    IO.puts("  [plot] #{title} (#{length(rows)} rows)")
    :ok
  end

  def bar_plot(rows, opts \\ []) do
    title = Keyword.get(opts, :title, "")
    IO.puts("  [plot] #{title} (#{length(rows)} bars)")
    :ok
  end
end

NotebookHelpers.seed!(2026)
_backend_setup = NotebookHelpers.try_compiled_backend!(min_headroom: 10_000, warn: false)

IO.puts("=== NotebookHelpers ✓ ===")

# ── Section 3: Synthetic world ──

minutes_per_hour = 60
hours_total = 6 * 24
total_minutes = hours_total * minutes_per_hour
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
  name: "Compact web traffic + campaign",
  total_minutes: total_minutes,
  baseline: %{
    intercept: 80.0,
    trend: 0.5,
    daily_fourier: [{8.0, 3.0}, {2.0, 1.0}],
    weekly_fourier: [{2.0, 1.0}]
  },
  spots: spots,
  noise_sd: 2.0,
  effect: %{
    lambda: 0.65,
    ec: 1.0,
    slope: 2.0,
    coefficient: 12.0
  },
  controls: %{count: 3, correlation: 0.45},
  seed: 2026
}

synthetic = Generator.generate(config)

obs_hourly = NotebookHelpers.downsample(synthetic.observations, 60)
baseline_hourly = NotebookHelpers.downsample(synthetic.ground_truth.baseline, 60)
tv_hourly = NotebookHelpers.downsample(synthetic.ground_truth.tv_contribution, 60)
controls_hourly = Enum.map(synthetic.controls, &NotebookHelpers.downsample(&1, 60))

pre_period = {1, 96}
post_period = {97, 144}
post_len = elem(post_period, 1) - elem(post_period, 0) + 1

spots_post_hours =
  synthetic.spots
  |> Enum.map(fn s ->
    ws = Integer.floor_div(s.window_start - post_start_minute, 60)
    we = div(s.window_end - post_start_minute + 59, 60)
    %{id: s.id, window_start: max(ws, 0), window_end: min(we, post_len)}
  end)
  |> Enum.filter(fn s -> s.window_end > s.window_start and s.window_start < post_len end)

reconstructed_hourly = Enum.zip(baseline_hourly, tv_hourly) |> Enum.map(fn {b, e} -> b + e end)

_decomposition_residual =
  Enum.zip(obs_hourly, reconstructed_hourly) |> Enum.map(fn {obs, recon} -> obs - recon end)

result_3 = %{
  n_hours: length(obs_hourly),
  spots_in_post: length(spots_post_hours),
  true_total_lift: synthetic.ground_truth.total_lift,
  decomposition_rmse: NotebookHelpers.rmse(obs_hourly, reconstructed_hourly)
}

IO.puts("=== Section 3: Synthetic world ✓ === #{inspect(result_3)}")

# Plot + data table (stubbed)
rows =
  NotebookHelpers.to_rows(%{
    "observed" => obs_hourly,
    "true baseline" => baseline_hourly,
    "true tv effect" => tv_hourly
  })

NotebookHelpers.line_plot(rows, title: "Synthetic hourly data")
Kino.DataTable.new(Enum.map(spots_post_hours, fn s -> %{spot_id: s.id} end))

# ── Section 4: Adstock and saturation ──

impulse = [1.0] ++ List.duplicate(0.0, 47)
adstocked = Adstock.geometric_adstock(impulse, 0.7)
saturated = Adstock.hill_saturation(adstocked, 0.3, 3.5)

rows =
  NotebookHelpers.to_rows(%{
    "raw impulse" => impulse,
    "after adstock" => adstocked,
    "after hill saturation" => saturated
  })

NotebookHelpers.line_plot(rows, title: "Carry-over and diminishing returns")

result_4 = %{
  impulse_area: Enum.sum(impulse),
  adstock_area: Enum.sum(adstocked),
  saturated_peak: Enum.max(saturated)
}

IO.puts("=== Section 4: Adstock ✓ === #{inspect(result_4)}")

# ── Section 5: Kalman filter ──

kalman_obs = Enum.take(obs_hourly, 72)

{filtered, predicted} =
  KalmanFilter.filter_with_pred(kalman_obs, 1.0, 1.0, 0.8, 4.0, List.first(kalman_obs), 10.0)

filtered_mean = Enum.map(filtered, fn {x, _p} -> Nx.to_number(x) end)
pred_mean = Enum.map(predicted, fn {x, _p} -> Nx.to_number(x) end)

obs_tensor = Nx.tensor(kalman_obs, type: {:f, 32})

{xs_defn, _ps_defn} =
  KalmanFilter.filter_defn(obs_tensor, 1.0, 1.0, 0.8, 4.0, List.first(kalman_obs), 10.0)

defn_mean = Nx.to_flat_list(xs_defn)

rows =
  NotebookHelpers.to_rows(%{
    "observed" => kalman_obs,
    "one-step prediction" => pred_mean,
    "filtered mean" => filtered_mean,
    "defn filtered mean" => defn_mean
  })

NotebookHelpers.line_plot(rows, title: "Kalman filter")

prediction_mae = NotebookHelpers.mae(kalman_obs, pred_mean)
filtered_mae = NotebookHelpers.mae(kalman_obs, filtered_mean)
defn_alignment_rmse = NotebookHelpers.rmse(filtered_mean, defn_mean)

result_5 = %{
  prediction_mae: prediction_mae,
  filtered_mae: filtered_mae,
  defn_alignment_rmse: defn_alignment_rmse
}

IO.puts("=== Section 5: Kalman filter ✓ === #{inspect(result_5)}")

# ── Section 6: RTS smoothing ──

smoothed = Smoother.rts(filtered, predicted, 1.0)
smoothed_mean = Enum.map(smoothed, fn {x, _p} -> Nx.to_number(x) end)

{state_draw, _next_key} =
  Smoother.simulate_with_key(smoothed, filtered, predicted, 1.0, key: Nx.Random.key(42))

state_draw_mean = Enum.map(state_draw, &Nx.to_number/1)

rows =
  NotebookHelpers.to_rows(%{
    "filtered mean" => filtered_mean,
    "smoothed mean" => smoothed_mean,
    "one posterior state draw" => state_draw_mean
  })

NotebookHelpers.line_plot(rows, title: "Smoothing")

true_baseline_kalman = Enum.take(baseline_hourly, length(kalman_obs))
filtered_rmse_vs_truth = NotebookHelpers.rmse(filtered_mean, true_baseline_kalman)
smoothed_rmse_vs_truth = NotebookHelpers.rmse(smoothed_mean, true_baseline_kalman)
result_6 = %{filtered_rmse: filtered_rmse_vs_truth, smoothed_rmse: smoothed_rmse_vs_truth}
IO.puts("=== Section 6: RTS smoothing ✓ === #{inspect(result_6)}")

# ── Section 7.1: Scalar Gibbs sampler ──

scalar_samples =
  GibbsSampler.sample(
    Enum.take(obs_hourly, 96),
    50,
    List.first(obs_hourly),
    10.0,
    0.8,
    4.0,
    burn_in: 12,
    seed: 11
  )

process_chain = Enum.map(scalar_samples, &Nx.to_number(&1.process_var))
obs_chain = Enum.map(scalar_samples, &Nx.to_number(&1.obs_var))

result_71 = %{
  n_samples: length(scalar_samples),
  process_var_mean: NotebookHelpers.mean(process_chain),
  obs_var_mean: NotebookHelpers.mean(obs_chain)
}

IO.puts("=== Section 7.1: Scalar Gibbs ✓ === #{inspect(result_71)}")

rows =
  NotebookHelpers.to_rows(%{
    "process variance" => process_chain,
    "observation variance" => obs_chain
  })

NotebookHelpers.line_plot(rows, title: "Scalar Gibbs chains")

# ── Section 7.2: Multiple chains + diagnostics ──

chains =
  GibbsSampler.sample_chains(
    Enum.take(obs_hourly, 96),
    3,
    12,
    List.first(obs_hourly),
    10.0,
    0.8,
    4.0,
    burn_in: 6,
    seed: 123
  )

process_chains = Enum.map(chains, fn chain -> Enum.map(chain, &Nx.to_number(&1.process_var)) end)

diagnostics = %{
  r_hat: Diagnostics.r_hat(process_chains),
  ess: Diagnostics.effective_sample_size(process_chains),
  split_r_hat_chain1: Diagnostics.split_r_hat(Enum.at(process_chains, 0)),
  ess_single_chain1: Diagnostics.ess_single(Enum.at(process_chains, 0))
}

IO.puts("=== Section 7.2: Multi-chain diagnostics ✓ === #{inspect(diagnostics)}")

rows =
  process_chains
  |> Enum.with_index(1)
  |> Map.new(fn {chain, i} -> {"chain_#{i}", chain} end)
  |> NotebookHelpers.to_rows()

NotebookHelpers.line_plot(rows, title: "Process variance across chains")

# ── Section 8: Structured models ──

regressors = controls_hourly |> Nx.tensor() |> Nx.transpose()

trend_spec =
  Components.local_linear_trend_spec(
    initial_level: List.first(obs_hourly),
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

structured_samples =
  GibbsSampler.sample_structured(obs_hourly, structured_spec, 6,
    burn_in: 3,
    seed: 77
  )

result_8 = %{
  n_structured_samples: length(structured_samples),
  q_dims_resampled: length(structured_spec.q_specs),
  state_dim: Nx.axis_size(structured_spec.f, 0),
  n_regression_controls: Nx.axis_size(regressors, 1)
}

IO.puts("=== Section 8: Structured models ✓ === #{inspect(result_8)}")

# ── Section 9.1: Scalar causal impact ──

impact_scalar =
  CausalImpact.estimate(obs_hourly, pre_period, post_period,
    num_samples: 12,
    burn_in: 6,
    seed: 999
  )

summary_scalar = CausalImpact.summary(impact_scalar)

IO.puts(
  "=== Section 9.1: Scalar CausalImpact ✓ === cumulative=#{inspect(summary_scalar.cumulative_effect)}"
)

post_actual = impact_scalar.actual
cf_bands = NotebookHelpers.per_t_bands(impact_scalar.counterfactual, 0.05)

rows =
  Enum.with_index(post_actual)
  |> Enum.map(fn {actual, t} ->
    b = Enum.at(cf_bands, t)
    %{t: t, actual: actual, mean: b.mean, lower: b.lower, upper: b.upper}
  end)

NotebookHelpers.interval_plot(rows, title: "CausalImpact (scalar)")

# ── Section 9.2: Structured causal impact ──

NotebookHelpers.try_compiled_backend!(min_headroom: 10_000)

impact_structured =
  CausalImpact.estimate_structured(
    obs_hourly,
    pre_period,
    post_period,
    structured_spec,
    num_samples: 8,
    burn_in: 4,
    seed: 202
  )

summary_structured = CausalImpact.summary(impact_structured)

true_post_lift =
  tv_hourly
  |> Enum.slice((elem(post_period, 0) - 1)..(elem(post_period, 1) - 1))
  |> Enum.sum()

result_92 = %{
  true_post_lift: true_post_lift,
  scalar_estimated: summary_scalar.cumulative_effect.mean,
  structured_estimated: summary_structured.cumulative_effect.mean
}

IO.puts("=== Section 9.2: Structured CausalImpact ✓ === #{inspect(result_92)}")

# ── Section 9.3: Filter-based estimator ──

intervention_indices = Enum.to_list(96..143)

filter_estimate =
  CausalImpact.estimate_from_filter(obs_hourly, intervention_indices,
    f: 1.0,
    h: 1.0,
    q: 0.8,
    r: 4.0,
    x0: List.first(obs_hourly),
    p0: 10.0,
    alpha: 0.05
  )

IO.puts(
  "=== Section 9.3: Filter estimator ✓ === cumulative=#{inspect(filter_estimate.cumulative_effect)}"
)

# ── Section 10: Intervention API ──

analysis_opts = [
  seasonality: 24,
  method: :filter,
  f: 1.0,
  h: 1.0,
  q: 0.8,
  r: 4.0,
  x0: List.first(obs_hourly),
  p0: 10.0,
  seed: 42
]

analysis =
  InterventionAnalysis.analyze(
    obs_hourly,
    %{pre_period: pre_period, post_period: post_period},
    analysis_opts
  )

result_10 = %{
  significant: analysis.significant?,
  cumulative_effect: analysis.summary.cumulative_effect.mean,
  relative_effect: analysis.summary.relative_effect.mean
}

IO.puts("=== Section 10: InterventionAnalysis ✓ === #{inspect(result_10)}")

report = InterventionAnalysis.report(analysis)
IO.puts("  report length: #{String.length(report)} chars")

# ── Section 11: Covariate selection ──

target_pre = Enum.take(obs_hourly, elem(pre_period, 1))

candidate_matrix_pre =
  controls_hourly
  |> Enum.map(&Enum.take(&1, elem(pre_period, 1)))
  |> Nx.tensor()
  |> Nx.transpose()

selection =
  CovariateSelection.select(target_pre, candidate_matrix_pre, threshold: 0.15, max_controls: 2)

Kino.DataTable.new(
  Enum.map(selection.correlations, fn {idx, corr} ->
    %{candidate_index: idx, pearson_corr: corr}
  end)
)

result_11 = %{
  selected_indices: selection.selected_indices,
  n_selected: length(selection.selected_indices)
}

IO.puts("=== Section 11: CovariateSelection ✓ === #{inspect(result_11)}")

# ── Section 12.1: Attribution core utilities ──

post_obs = Enum.slice(obs_hourly, (elem(post_period, 0) - 1)..(elem(post_period, 1) - 1))

counterfactual_mean = NotebookHelpers.posterior_mean(impact_structured.counterfactual)
counterfactual_var = NotebookHelpers.posterior_var(impact_structured.counterfactual)

counterfactual_summary = %{
  mean: counterfactual_mean,
  variance: counterfactual_var,
  obs_variance: 0.0
}

first_spot = hd(spots_post_hours)

window_lift =
  SpotAttributor.compute_window_lift(
    post_obs,
    counterfactual_summary,
    first_spot.window_start,
    first_spot.window_end
  )

coalition_lift =
  SpotAttributor.compute_coalition_lift(post_obs, counterfactual_summary, spots_post_hours)

overlap_groups = ShapleyAllocator.detect_overlaps(spots_post_hours)

result_121 = %{
  first_spot: first_spot.id,
  first_spot_lift: window_lift,
  coalition_lift: coalition_lift,
  overlap_groups: Enum.map(overlap_groups, &Enum.map(&1, fn s -> s.id end))
}

IO.puts("=== Section 12.1: Attribution utils ✓ === #{inspect(result_121)}")

# ── Section 12.2: Posterior propagation attribution ──

attr_posterior =
  SpotAttributor.attribute_posterior(
    post_obs,
    spots_post_hours,
    impact_structured.counterfactual,
    0.0,
    alpha: 0.05,
    n_samples: 120,
    decay: 0.7
  )

sum_of_spot_lifts = Enum.reduce(attr_posterior.attributions, 0.0, fn a, acc -> acc + a.lift end)

result_122 = %{
  total_lift: attr_posterior.total_lift,
  sum_of_spot_lifts: sum_of_spot_lifts,
  reconciliation_gap: attr_posterior.total_lift - sum_of_spot_lifts
}

IO.puts("=== Section 12.2: Posterior attribution ✓ === #{inspect(result_122)}")

rows = Enum.map(attr_posterior.attributions, fn a -> %{label: a.spot_id, value: a.lift} end)
NotebookHelpers.bar_plot(rows, title: "Posterior mean lift by spot")
Kino.DataTable.new(attr_posterior.attributions)

# ── Section 13: Pipeline ──

pipeline_result =
  Pipeline.run(
    obs_hourly,
    pre_period,
    post_period,
    spots_post_hours,
    structured_spec,
    num_samples: 4,
    burn_in: 2,
    seed: 404,
    alpha: 0.05,
    shapley_samples: 120,
    decay: 0.7
  )

result_13 = %{
  pipeline_total_lift: pipeline_result.attributions.total_lift,
  pipeline_total_sd: pipeline_result.attributions.total_lift_sd,
  true_post_lift: true_post_lift
}

IO.puts("=== Section 13: Pipeline ✓ === #{inspect(result_13)}")

# ── Section 14.1: Forecaster ──

train = Enum.take(obs_hourly, 120)
test = Enum.drop(obs_hourly, 120)

forecast =
  Forecaster.fit_predict(train, length(test),
    seasonality: 24,
    num_samples: 4,
    burn_in: 2,
    seed: 15
  )

rows =
  Enum.with_index(test)
  |> Enum.map(fn {actual, t} ->
    %{
      t: t,
      actual: actual,
      mean: Enum.at(forecast.mean, t),
      lower: Enum.at(forecast.lower, t),
      upper: Enum.at(forecast.upper, t)
    }
  end)

NotebookHelpers.interval_plot(rows, title: "Forecast horizon")

forecast_mae = NotebookHelpers.mae(test, forecast.mean)

forecast_coverage =
  Enum.with_index(test)
  |> Enum.count(fn {actual, t} ->
    actual >= Enum.at(forecast.lower, t) and actual <= Enum.at(forecast.upper, t)
  end)
  |> Kernel./(max(length(test), 1))

result_141 = %{forecast_mae: forecast_mae, coverage_pct: 100.0 * forecast_coverage}
IO.puts("=== Section 14.1: Forecaster ✓ === #{inspect(result_141)}")

# ── Section 14.2: Demand forecasting ──

full_regressors = controls_hourly |> Nx.tensor() |> Nx.transpose()
train_regressors = Nx.slice(full_regressors, [0, 0], [120, Nx.axis_size(full_regressors, 1)])

future_regressors =
  Nx.slice(full_regressors, [120, 0], [length(test), Nx.axis_size(full_regressors, 1)])

demand_forecast =
  DemandForecaster.forecast(train,
    horizon: length(test),
    seasonality: 24,
    regressors: %{training: train_regressors, future: future_regressors},
    num_samples: 4,
    burn_in: 2,
    seed: 16
  )

safety_stock = DemandForecaster.safety_stock(demand_forecast, service_level: 0.95, lead_time: 7)

result_142 = %{
  demand_first_3: Enum.take(demand_forecast.mean, 3),
  safety_stock_total: safety_stock.total
}

IO.puts("=== Section 14.2: DemandForecaster ✓ === #{inspect(result_142)}")

# ── Section 14.3: BCT AR forecaster ──

ar_forecast =
  BstsNx.BCT.ARForecaster.fit_predict(train, length(test), order: 2, num_samples: 10, seed: 17)

result_143 = %{
  backend: ar_forecast.backend,
  horizon: ar_forecast.horizon,
  mean_first_3: Enum.take(ar_forecast.mean, 3)
}

IO.puts("=== Section 14.3: AR Forecaster ✓ === #{inspect(result_143)}")

# ── Section 15: Anomaly detection ──

normal_history = Enum.take(obs_hourly, 96)
stream = Enum.take(Enum.drop(obs_hourly, 96), 24)

stream_with_anomalies =
  stream
  |> List.update_at(8, &(&1 + 35.0))
  |> List.update_at(16, &(&1 - 30.0))

detector = AnomalyDetector.fit(normal_history, method: :filter, alpha: 0.01)
scores = AnomalyDetector.score(detector, stream_with_anomalies)
summary = AnomalyDetector.summary(scores)
flagged = AnomalyDetector.detect(detector, stream_with_anomalies, min_severity: :warning)

flagged_indices =
  scores
  |> Enum.with_index()
  |> Enum.filter(fn {s, _idx} -> s.anomaly? and s.severity in [:warning, :critical] end)
  |> Enum.map(&elem(&1, 1))

result_15 = %{summary: summary, flagged_count: length(flagged), flagged_indices: flagged_indices}
IO.puts("=== Section 15: Anomaly detection ✓ === #{inspect(result_15)}")

rows =
  scores
  |> Enum.with_index()
  |> Enum.map(fn {s, t} -> %{series: "|z|", t: t, value: abs(s.z_score)} end)

threshold = BstsNx.Utils.z_score(0.01)

threshold_rows =
  Enum.with_index(scores)
  |> Enum.map(fn {_s, t} -> %{series: "threshold", t: t, value: threshold} end)

NotebookHelpers.line_plot(rows ++ threshold_rows, title: "Anomaly detector z-scores")

# ── Section 16: Rolling baseline + TV attribution ──

{rolling_cf, rolling_fit} =
  RollingBaseline.fit_and_predict(
    Enum.take(obs_hourly, 96),
    48,
    num_seasons: 24,
    num_samples: 4,
    burn_in: 2,
    seed: 500
  )

attr_from_baseline =
  TVAttribution.attribute_from_baseline(
    post_obs,
    spots_post_hours,
    rolling_cf,
    alpha: 0.05,
    n_samples: 120
  )

result_16a = %{
  rolling_converged: rolling_fit.convergence.converged,
  baseline_total_lift: attr_from_baseline.total_lift
}

IO.puts("=== Section 16a: Rolling baseline ✓ === #{inspect(result_16a)}")

tv_result =
  TVAttribution.attribute(
    obs_hourly,
    spots_post_hours,
    pre_period: pre_period,
    post_period: post_period,
    seasonality: 24,
    num_samples: 4,
    burn_in: 2,
    seed: 501,
    alpha: 0.05
  )

result_16b = %{total_lift: tv_result.total_lift, total_lift_sd: tv_result.total_lift_sd}
IO.puts("=== Section 16b: TV Attribution ✓ === #{inspect(result_16b)}")

# ── Section 17: Marketing and policy wrappers ──

campaign = %{
  name: "midnight_spike_tv",
  channel: :tv,
  start_index: 97,
  end_index: 120,
  baseline_start: 1,
  baseline_end: 96
}

marketing_result =
  MarketingLift.measure_lift(obs_hourly, campaign,
    seasonality: 24,
    method: :filter,
    f: 1.0,
    h: 1.0,
    q: 0.8,
    r: 4.0,
    x0: List.first(obs_hourly),
    p0: 10.0,
    seed: 700
  )

intervention = %{
  intervention_name: "checkout_policy_rollout",
  intervention_date_index: 97,
  pre_period_start: 1,
  outcome_metric: "sessions"
}

policy_result =
  PolicyEvaluator.evaluate(obs_hourly, intervention,
    post_period_end: 144,
    seasonality: 24,
    method: :filter,
    f: 1.0,
    h: 1.0,
    q: 0.8,
    r: 4.0,
    x0: List.first(obs_hourly),
    p0: 10.0,
    seed: 701
  )

result_17 = %{
  marketing_significant: marketing_result.significant?,
  policy_significant: policy_result.significant?
}

IO.puts("=== Section 17: Marketing + Policy ✓ === #{inspect(result_17)}")

# ── Section 18: Validation and calibration ──

actual_t = Nx.tensor(post_obs)
baseline_t = Nx.tensor(rolling_cf.mean)
state_var_t = Nx.tensor(rolling_cf.variance)
indices = Enum.to_list(0..(length(post_obs) - 1))

pred_err = Validation.prediction_error(actual_t, baseline_t, indices)

coverage =
  Validation.coverage(actual_t, baseline_t, state_var_t, rolling_cf.obs_variance, indices)

residuals = Nx.subtract(actual_t, baseline_t)
dw = Validation.durbin_watson(residuals, indices)

estimate_fn = fn sessions, idxs ->
  vals = Enum.map(idxs, &Enum.at(sessions, &1))
  lift = Enum.sum(vals) - length(vals) * (Enum.sum(sessions) / length(sessions))
  sd = max(abs(lift) * 0.1, 1.0)

  %{
    lift_sessions: lift,
    lift_pct: lift / max(Enum.sum(sessions), 1.0),
    lift_ci95: %{lower: lift - 1.96 * sd, upper: lift + 1.96 * sd, sd: sd}
  }
end

placebo = Validation.placebo_test(post_obs, Enum.to_list(5..12), estimate_fn)

stability =
  Validation.effect_stability(
    summary_structured.cumulative_effect.mean,
    fn window ->
      h = max(1, div(window, 60))
      post_slice = Enum.take(post_obs, h)
      base_slice = Enum.take(counterfactual_mean, h)
      Enum.zip(post_slice, base_slice) |> Enum.reduce(0.0, fn {a, b}, acc -> acc + (a - b) end)
    end,
    120,
    30
  )

assessment =
  Validation.assess(%{
    prediction_error: pred_err,
    coverage: coverage,
    durbin_watson: dw,
    placebo: placebo,
    effect_stability: stability
  })

release_ready? = Enum.all?(assessment, fn {_k, v} -> v in [:pass, :skip] end)
IO.puts("=== Section 18a: Validation ✓ === release_ready?=#{release_ready?}")

calibration =
  Validation.known_lift_injection(
    40.0,
    Components.local_level_spec(),
    n_pre: 60,
    n_post: 20,
    num_samples: 4,
    burn_in: 2,
    seed: 2026,
    credible_level: 0.95
  )

result_18b = %{
  covered: calibration.covered,
  relative_error_pct: 100.0 * calibration.relative_error
}

IO.puts("=== Section 18b: Calibration ✓ === #{inspect(result_18b)}")

# ── Section 19: Function map ──

modules = [
  BstsNx.Synthetic.Adstock,
  BstsNx.Synthetic.Generator,
  BstsNx.KalmanFilter,
  BstsNx.Smoother,
  BstsNx.GibbsSampler,
  BstsNx.Components,
  BstsNx.CausalImpact,
  BstsNx.InterventionAnalysis,
  BstsNx.CovariateSelection,
  BstsNx.SpotAttributor,
  BstsNx.ShapleyAllocator,
  BstsNx.Pipeline,
  BstsNx.RollingBaseline,
  BstsNx.Forecaster,
  BstsNx.Applications.DemandForecaster,
  BstsNx.Applications.AnomalyDetector,
  BstsNx.Applications.TVAttribution,
  BstsNx.Applications.MarketingLift,
  BstsNx.Applications.PolicyEvaluator,
  BstsNx.Validation,
  BstsNx.Diagnostics
]

rows =
  for mod <- modules,
      {fun, arity} <- mod.__info__(:functions),
      fun not in [:module_info, :__info__] do
    %{module: inspect(mod), function: "#{fun}/#{arity}"}
  end

Kino.DataTable.new(rows)
IO.puts("=== Section 19: Function map ✓ === #{length(rows)} public functions")

IO.puts("\n🎉 ALL LIVEBOOK SECTIONS EXECUTED SUCCESSFULLY 🎉")
