# Causal Inference and Attribution

This guide covers intervention analysis, causal effect estimation, and overlap-aware attribution.

## 1. Domain-Neutral Intervention Analysis

Use `BstsNx.InterventionAnalysis` for most causal workflows.

```elixir
analysis =
  BstsNx.InterventionAnalysis.analyze(observations, %{
    pre_period: {1, 90},
    post_period: {91, 120}
  },
    seasonality: 7,
    num_samples: 300,
    seed: 42
  )

analysis.summary
analysis.significant?
```

- supports `:mcmc` (default) and `:filter` methods,
- accepts custom model specs,
- supports control series through `ModelBuilder` composition.

## 2. Low-Level Causal Effect API

`BstsNx.CausalImpact` exposes direct estimators:

- `estimate/4`: local-level convenience API,
- `estimate_structured/5`: custom `ModelSpec` for trend/seasonal/regression/composed models,
- `estimate_from_filter/3`: non-MCMC fast path (compiled scalar filter/smoother path).

For most production workflows with covariates or richer model structure, prefer
`estimate_structured/5` (or `InterventionAnalysis` with `:model_spec`).

```elixir
impact = BstsNx.CausalImpact.estimate(observations, {1, 90}, {91, 120}, num_samples: 200)
summary = BstsNx.CausalImpact.summary(impact)
summary.cumulative_effect
```

## 3. Per-Event Attribution (`BstsNx.SpotAttributor`)

Given a counterfactual baseline, attribute lift to individual windows.

```elixir
counterfactual = %{
  mean: baseline_mean,
  variance: baseline_variance,
  obs_variance: 1.0
}

spots = [
  %{id: "a", window_start: 0, window_end: 5},
  %{id: "b", window_start: 3, window_end: 8}
]

result = BstsNx.SpotAttributor.attribute(actual_post_period, spots, counterfactual)
result.attributions
```

Use `attribute_posterior/5` when you already have posterior counterfactual draws and want better uncertainty propagation.

## 4. Overlap Handling with Shapley Values

`BstsNx.ShapleyAllocator` handles fair allocation in overlapping windows.

```elixir
value_fn = fn ids -> length(ids) * 10.0 end
alloc = BstsNx.ShapleyAllocator.allocate(spots, 50.0, value_fn)
```

- exact allocation for smaller groups,
- Monte Carlo approximation for larger groups,
- automatic normalization to coalition total.

## 5. End-to-End Attribution Pipeline

`BstsNx.Pipeline.run/6` chains structured causal impact and posterior spot attribution.

```elixir
spec = BstsNx.Components.compose_specs(
  BstsNx.Components.local_level_spec(initial_state: 100.0),
  BstsNx.Components.seasonal_spec(7)
)

result = BstsNx.Pipeline.run(observations, {1, 100}, {101, 120}, spots, spec,
  num_samples: 200,
  seed: 42
)

result.attributions
```

## 6. Rolling Production Baselines

`BstsNx.RollingBaseline` is optimized for repeated rolling-window inference.

```elixir
spec = BstsNx.RollingBaseline.build_spec(length(pre_obs), num_seasons: 96)
fit = BstsNx.RollingBaseline.fit(pre_obs, spec, num_samples: 200, seed: 42)
cf = BstsNx.RollingBaseline.counterfactual(fit, post_horizon)
```

Warm-start support (`:warm_start`) is useful for daily batch updates.

## 7. Covariate Pre-Selection

`BstsNx.CovariateSelection` selects useful controls using pre-period correlation.

```elixir
selection = BstsNx.CovariateSelection.select(target, candidate_matrix,
  threshold: 0.2,
  max_controls: 5
)

selection.selected_indices
```

## 8. Validation and Diagnostics

For quality checks of causal pipelines:

- `BstsNx.Validation` for placebo/error/coverage/autocorrelation/stability checks,
- `BstsNx.Diagnostics` for R-hat and ESS convergence metrics.

These modules are critical when calibrating production attribution systems.
