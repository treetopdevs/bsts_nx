# Documentation Overview

This guide helps you navigate `BstsNx` quickly when you are new to the codebase,
building production workflows, or prompting an LLM to generate code.

## What `BstsNx` Is

`BstsNx` is a Bayesian Structural Time Series toolkit built on `Nx`.
It provides:

- low-level state-space inference (`KalmanFilter`, `Smoother`, `GibbsSampler`),
- high-level causal inference (`CausalImpact`, `InterventionAnalysis`),
- forecasting (`Forecaster`, `BCT.ARForecaster`),
- attribution (`SpotAttributor`, `Pipeline`, application wrappers), and
- synthetic data + diagnostics modules for calibration and testing.

## Fast Module Selection

Use this map when deciding where to start:

| Goal | Start Here | Then Use |
|---|---|---|
| Learn/filter latent state | `BstsNx.KalmanFilter` | `BstsNx.Smoother`, `BstsNx.GibbsSampler` |
| Build structured models (trend/seasonal/regression) | `BstsNx.Components` | `BstsNx.ModelSpec`, `BstsNx.GibbsSampler.sample_structured/4` |
| Measure intervention effects | `BstsNx.InterventionAnalysis` | `BstsNx.CausalImpact`, `BstsNx.Validation` |
| Forecast future values | `BstsNx.Forecaster` | `BstsNx.Applications.DemandForecaster` |
| Attribute lift across overlapping events | `BstsNx.SpotAttributor` | `BstsNx.ShapleyAllocator`, `BstsNx.Pipeline` |
| TV-to-web attribution workflow | `BstsNx.Applications.TVAttribution` | `BstsNx.RollingBaseline` |
| Generate synthetic benchmark datasets | `BstsNx.Synthetic.Scenarios` | `BstsNx.Synthetic.Generator` |

## Data and Indexing Conventions

Understanding indexing upfront prevents subtle bugs:

- `pre_period` / `post_period` in `CausalImpact` and `InterventionAnalysis` are **1-based inclusive** tuples.
- spot windows in `SpotAttributor` and TV modules are **0-based half-open** ranges: `[window_start, window_end)`.
- time-varying observation matrices (`H_t`) are often represented as a list of `{1, n}` tensors (or compatible Nx tensors by module).
- missing observations are handled as:
  - `nil` in list-based APIs,
  - `NaN` in `defn` tensor APIs.

## Typical Workflow Patterns

### Pattern A: Causal effect of an intervention

```elixir
config = %{pre_period: {1, 90}, post_period: {91, 110}}

result =
  BstsNx.InterventionAnalysis.analyze(observations, config,
    seasonality: 7,
    num_samples: 300,
    seed: 42
  )

result.summary.cumulative_effect.mean
```

### Pattern B: Forecast with uncertainty bands

```elixir
forecast =
  BstsNx.Forecaster.fit_predict(observations, 14,
    seasonality: 7,
    num_samples: 300,
    seed: 42
  )

{forecast.mean, forecast.lower, forecast.upper}
```

### Pattern C: TV spot-level attribution with overlap handling

```elixir
result =
  BstsNx.Applications.TVAttribution.attribute(observations, spots,
    pre_period: {1, 1000},
    post_period: {1001, 1100},
    seasonality: 96,
    num_samples: 200,
    seed: 42
  )

result.spots
```

## Recommended Reading Order

1. [Getting Started](getting-started.html)
2. [Core Modeling](core-modeling.html)
3. [Causal Inference and Attribution](causal-inference-and-attribution.html)
4. [Forecasting and Applications](forecasting-and-applications.html)
5. [Synthetic Data and Validation](synthetic-data-and-validation.html)
6. [Module Reference](module-reference.html)
