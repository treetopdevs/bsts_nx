# Forecasting and Applications

This guide explains forecasting APIs and domain-specific wrappers.

## 1. Generic Forecasting (`BstsNx.Forecaster`)

Use this for direct probabilistic forecasts with optional seasonality/regressors.

```elixir
fit = BstsNx.Forecaster.fit(training_data,
  seasonality: 7,
  num_samples: 300,
  seed: 42
)

forecast = BstsNx.Forecaster.predict(fit, horizon: 14)
{forecast.mean, forecast.lower, forecast.upper}
```

One-shot helper:

```elixir
forecast = BstsNx.Forecaster.fit_predict(training_data, 14, seasonality: 7)
```

## 2. BCT-AR Scaffold Forecaster

`BstsNx.BCT.ARForecaster` is a stable forecasting contract for a future context-tree backend.

```elixir
fit = BstsNx.BCT.ARForecaster.fit(training_data, order: 2, seed: 42)
forecast = BstsNx.BCT.ARForecaster.predict(fit, horizon: 14)
```

Use it when you want AR-style behavior with simulation intervals and a forward-compatible interface.

## 3. Demand Forecasting (`Applications.DemandForecaster`)

Adds demand-specific helpers like safety stock and promotion impact.

```elixir
result = BstsNx.Applications.DemandForecaster.forecast(demand,
  horizon: 14,
  seasonality: 7,
  num_samples: 200,
  seed: 42
)

stock = BstsNx.Applications.DemandForecaster.safety_stock(result,
  service_level: 0.95,
  lead_time: 3
)
```

With regressors:

```elixir
BstsNx.Applications.DemandForecaster.forecast(demand,
  horizon: 14,
  regressors: %{training: train_x, future: future_x}
)
```

## 4. Marketing Lift (`Applications.MarketingLift`)

Measures campaign incrementality and handles overlap across campaigns.

```elixir
campaign = %{
  name: "spring_promo",
  start_index: 31,
  end_index: 45,
  baseline_start: 1,
  baseline_end: 30
}

result = BstsNx.Applications.MarketingLift.measure_lift(observations, campaign)
result.effect
```

## 5. Policy Evaluation (`Applications.PolicyEvaluator`)

Interrupted time-series API for policy/regulatory interventions.

```elixir
intervention = %{
  intervention_name: "new_policy",
  intervention_date_index: 37,
  pre_period_start: 1
}

result = BstsNx.Applications.PolicyEvaluator.evaluate(observations, intervention)
result.report
```

## 6. Anomaly Detection (`Applications.AnomalyDetector`)

Fits baseline behavior and scores anomalies with calibrated probabilities.

```elixir
detector = BstsNx.Applications.AnomalyDetector.fit(training_data,
  method: :filter,
  alpha: 0.01
)

scores = BstsNx.Applications.AnomalyDetector.score(detector, new_data)
```

Use `score_one/2` for streaming one-point-at-a-time updates.

## 7. TV Attribution (`Applications.TVAttribution`)

Domain wrapper over pipeline + attribution + rolling baselines.

```elixir
result = BstsNx.Applications.TVAttribution.attribute(observations, spots,
  pre_period: {1, 1000},
  post_period: {1001, 1100},
  seasonality: 96,
  num_samples: 200
)

result.spots
```

If you already have a baseline, call `attribute_from_baseline/4` directly.
