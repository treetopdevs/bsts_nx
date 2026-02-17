# Getting Started

This guide gets you from installation to real analysis quickly.

## Installation

Add `bsts_nx` to your dependencies:

```elixir
def deps do
  [
    {:bsts_nx, "~> 0.1"}
  ]
end
```

Fetch and compile:

```bash
mix deps.get
mix compile
```

## First Working Example: Intervention Analysis

This is the fastest way to get a practical result.

```elixir
:rand.seed(:exsss, {1, 2, 3})

pre = Enum.map(1..60, fn _ -> 100.0 + :rand.normal() * 3 end)
post = Enum.map(1..20, fn _ -> 112.0 + :rand.normal() * 3 end)
obs = pre ++ post

result =
  BstsNx.InterventionAnalysis.analyze(obs, %{pre_period: {1, 60}, post_period: {61, 80}},
    seasonality: 7,
    num_samples: 200,
    burn_in: 100,
    seed: 42
  )

result.summary.cumulative_effect.mean
result.significant?
```

## Low-Level Core Flow (Filter → Smoother → Gibbs)

Use this when you want full control over state-space internals.

```elixir
obs = [1.0, 2.0, 3.0, 2.5, 3.5]

{filtered, predicted} =
  BstsNx.KalmanFilter.filter_with_pred(obs, 1.0, 1.0, 0.1, 1.0, 0.0, 1.0)

smoothed = BstsNx.Smoother.rts(filtered, predicted, 1.0)

samples =
  BstsNx.GibbsSampler.sample(obs, 100, 0.0, 1.0, 1.0, 1.0,
    burn_in: 50,
    seed: 42
  )
```

## Forecasting Example

```elixir
forecast =
  BstsNx.Forecaster.fit_predict(Enum.to_list(1..120), 14,
    seasonality: 7,
    num_samples: 200,
    seed: 42
  )

forecast.mean
forecast.lower
forecast.upper
```

## Demand Forecasting With Regressors

```elixir
history = Enum.map(1..60, &(&1 * 1.0))
train_x = Nx.iota({60, 2})
future_x = Nx.iota({14, 2})

result =
  BstsNx.Applications.DemandForecaster.forecast(history,
    horizon: 14,
    seasonality: 7,
    regressors: %{training: train_x, future: future_x},
    seed: 42
  )

result.mean
```

## TV Attribution Example

```elixir
spots = [
  %{id: "spot_a", window_start: 0, window_end: 5},
  %{id: "spot_b", window_start: 3, window_end: 8}
]

result =
  BstsNx.Applications.TVAttribution.attribute(observations, spots,
    pre_period: {1, 1000},
    post_period: {1001, 1100},
    seasonality: 96,
    num_samples: 200,
    seed: 42
  )

result.spots
result.total_lift
```

## Common Pitfalls

- `pre_period`/`post_period` are **1-based inclusive**.
- spot windows are **0-based half-open** (`[start, end)`) and usually relative to post-period start.
- use `nil` for missing values in list APIs and `NaN` for tensor `defn` APIs.
- when using regressors in forecasting, `future_regressors` must match training regressor width.

## Next Guides

- [Core Modeling](core-modeling.html)
- [Causal Inference and Attribution](causal-inference-and-attribution.html)
- [Forecasting and Applications](forecasting-and-applications.html)
- [Module Reference](module-reference.html)
