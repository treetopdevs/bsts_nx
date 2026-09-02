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

## 2. Measurement-weighted forecasting

`BstsNx.Forecaster` accepts known positive relative observation-variance
weights for scalar observations whose measurement reliability changes over
time. It fits `R_t = sigma_r^2 * weight_t`, learning the common scale while
treating the weights as known reliability information.

```elixir
fit =
  BstsNx.Forecaster.fit(
    observations,
    seasonality: 7,
    regressors: historical_x,
    observation_variance_weights: historical_relative_variances,
    num_samples: 300,
    burn_in: 150,
    seed: 42
  )

forecast_result =
  BstsNx.Forecaster.predict(
    fit,
    horizon: 14,
    future_regressors: future_x,
    future_observation_variance_weights: future_relative_variances,
    return: :both,
    format: :tensors,
    seed: 43
  )
```

The returned draw tensor has shape `{draw, horizon}`. Training and prediction
use exact Gaussian prewhitening, so a weight of `4.0` represents four times the
observation variance of a reference period whose weight is `1.0`.

## 3. Joint draws, audience composition, and delivery risk

`BstsNx.Forecast` preserves dependence across future periods and provides
quantiles, weighted sums, threshold probabilities, and shortfall measures.
Convert a draw-bearing `BstsNx.Forecaster` result into this reusable form:

```elixir
put_forecast = BstsNx.Forecast.new(put_result.draws, alpha: put_result.alpha)
share_forecast = BstsNx.Forecast.new(share_result.draws, alpha: share_result.alpha)
```

For television audience forecasting, combine aligned PUT/HUT and share draws
instead of multiplying their marginal means:

```elixir
audience =
  BstsNx.Applications.AudienceForecast.combine(
    put_forecast,
    share_forecast,
    universe_by_period
  )
```

Evaluate contract-level delivery by weighting and summing each complete
trajectory:

```elixir
risk =
  BstsNx.Applications.MakegoodRisk.evaluate(
    audience,
    exposure_weights,
    guarantee,
    reserve_quantile: 0.10
  )

risk.underdelivery_probability
risk.expected_shortfall
risk.conservative_delivery
```

Forecasts being combined must share the same draw count and horizon. Pairing
draws from independently fitted models represents an independence assumption.
When shared future drivers or residual dependence matter, align draws through
explicit common scenarios or fit a joint model; matching random seeds alone is
not a substitute for a dependence model.

## 4. BCT-AR Scaffold Forecaster

`BstsNx.BCT.ARForecaster` is a stable forecasting contract for a future context-tree backend.

```elixir
fit = BstsNx.BCT.ARForecaster.fit(training_data, order: 2, seed: 42)
forecast = BstsNx.BCT.ARForecaster.predict(fit, horizon: 14)
```

Use it when you want AR-style behavior with simulation intervals and a forward-compatible interface.

## 5. Demand Forecasting (`Applications.DemandForecaster`)

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

## 6. Marketing Lift (`Applications.MarketingLift`)

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

## 7. Policy Evaluation (`Applications.PolicyEvaluator`)

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

## 8. Anomaly Detection (`Applications.AnomalyDetector`)

Fits baseline behavior and scores anomalies with calibrated probabilities.

```elixir
detector = BstsNx.Applications.AnomalyDetector.fit(training_data,
  method: :filter,
  alpha: 0.01
)

scores = BstsNx.Applications.AnomalyDetector.score(detector, new_data)
```

Use `score_one/2` for streaming one-point-at-a-time updates.

## 9. TV Attribution (`Applications.TVAttribution`)

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
