# Module Reference

This reference lists every module in `lib/`, what it is for, and a minimal usage example.

## Core Entry Point

### `BstsNx`

Use case: discover the library surface and jump to domain APIs.

```elixir
h BstsNx
```

## Modeling Primitives

### `BstsNx.KalmanFilter`

Use case: forward filtering of latent states with optional missing observations.

```elixir
BstsNx.KalmanFilter.filter([1.0, nil, 2.0], 1.0, 1.0, 0.1, 1.0, 0.0, 1.0)
```

### `BstsNx.Smoother`

Use case: backward smoothing and posterior state-path simulation.

```elixir
{filtered, predicted} = BstsNx.KalmanFilter.filter_with_pred([1.0, 2.0], 1.0, 1.0, 0.1, 1.0, 0.0, 1.0)
BstsNx.Smoother.rts(filtered, predicted, 1.0)
```

### `BstsNx.GibbsSampler`

Use case: Bayesian posterior sampling for state-space models.

```elixir
BstsNx.GibbsSampler.sample([1.0, 2.0, 3.0], 50, 0.0, 1.0, 1.0, 1.0, burn_in: 25, seed: 42)
```

### `BstsNx.StateSpace`

Use case: compose independent components into a single model.

```elixir
BstsNx.StateSpace.block_diag([Nx.tensor([[1.0]]), Nx.tensor([[2.0]])])
```

### `BstsNx.Components`

Use case: create trend/seasonal/regression components and `ModelSpec` structs.

```elixir
trend = BstsNx.Components.local_linear_trend_spec()
season = BstsNx.Components.seasonal_spec(7)
BstsNx.Components.compose_specs(trend, season)
```

### `BstsNx.ModelSpec`

Use case: typed model container for structured samplers and structured causal/forecast APIs.

```elixir
%BstsNx.ModelSpec{}
```

### `BstsNx.Distributions`

Use case: sample inverse-gamma/normal/multivariate normal random variables.

```elixir
BstsNx.Distributions.inv_gamma_sample(2.0, 1.0)
```

### `BstsNx.ModelBuilder`

Use case: centralized helper for model-option resolution and effect formatting.

```elixir
BstsNx.ModelBuilder.build_spec([1.0, 2.0, 3.0], seasonality: 7)
```

## Causal Inference and Attribution

### `BstsNx.CausalImpact`

Use case: estimate intervention effects from pre/post periods.

```elixir
BstsNx.CausalImpact.estimate(observations, {1, 90}, {91, 110}, num_samples: 200)
```

### `BstsNx.InterventionAnalysis`

Use case: higher-level causal API with significance checks and reporting.

```elixir
BstsNx.InterventionAnalysis.analyze(observations, %{pre_period: {1, 90}, post_period: {91, 110}})
```

### `BstsNx.Pipeline`

Use case: one call for structured causal impact + posterior spot attribution.

```elixir
BstsNx.Pipeline.run(observations, {1, 90}, {91, 110}, spots, spec, num_samples: 200)
```

### `BstsNx.RollingBaseline`

Use case: rolling-window baseline fitting/counterfactual generation with warm starts.

```elixir
spec = BstsNx.RollingBaseline.build_spec(length(pre_obs), num_seasons: 96)
BstsNx.RollingBaseline.fit(pre_obs, spec)
```

### `BstsNx.SpotAttributor`

Use case: allocate post-period lift to windows/spots with uncertainty.

```elixir
BstsNx.SpotAttributor.attribute(actual_post, spots, %{mean: mean, variance: var, obs_variance: 1.0})
```

### `BstsNx.ShapleyAllocator`

Use case: fair overlap allocation via exact/Monte Carlo Shapley values.

```elixir
BstsNx.ShapleyAllocator.allocate(spots, 100.0, fn ids -> length(ids) * 10.0 end)
```

### `BstsNx.CovariateSelection`

Use case: pre-select controls by pre-period correlation.

```elixir
BstsNx.CovariateSelection.select(target, candidates, threshold: 0.2, max_controls: 5)
```

### `BstsNx.Validation`

Use case: run fit/calibration/placebo/stability checks and produce pass/warn/fail assessments.

```elixir
BstsNx.Validation.assess(%{prediction_error: %{rmse: 5.0, mape: 0.1, n: 100}})
```

### `BstsNx.Diagnostics`

Use case: check MCMC convergence quality with R-hat and ESS.

```elixir
BstsNx.Diagnostics.split_r_hat(Enum.map(1..200, fn _ -> :rand.normal() end))
```

## Forecasting APIs

### `BstsNx.Forecast`

Use case: retain joint posterior trajectories and calculate quantiles, weighted totals, threshold probabilities, and expected shortfall without discarding cross-period dependence.

```elixir
forecast = BstsNx.Forecast.new([[90.0, 105.0], [100.0, 115.0]])
delivery = BstsNx.Forecast.weighted_sum(forecast, [1.0, 2.0])
```

### `BstsNx.Forecaster`

Use case: probabilistic forecasting with trend, seasonality, regressors, known relative observation-variance weights, and optional joint posterior-draw output.

```elixir
fit =
  BstsNx.Forecaster.fit(observations,
    seasonality: 7,
    observation_variance_weights: reliability_weights,
    num_samples: 200
  )

BstsNx.Forecaster.predict(fit,
  horizon: 14,
  return: :both,
  format: :tensors
)
```

### `BstsNx.ObservationWeights`

Use case: validate and apply the exact Gaussian prewhitening transformation used by measurement-weighted forecasting.

```elixir
{weighted_observations, weighted_spec} =
  BstsNx.ObservationWeights.prewhiten(observations, spec, reliability_weights)
```

### `BstsNx.BCT.ARForecaster`

Use case: AR-style forecasting interface with simulation intervals and forward-compatible metadata.

```elixir
BstsNx.BCT.ARForecaster.fit_predict(observations, 14, order: 2)
```

## Synthetic Data Modules

### `BstsNx.Synthetic.Scenarios`

Use case: load benchmark-ready synthetic configs.

```elixir
BstsNx.Synthetic.Scenarios.scenario_a()
```

### `BstsNx.Synthetic.Generator`

Use case: generate synthetic observations plus exact ground truth.

```elixir
cfg = BstsNx.Synthetic.Scenarios.scenario_a()
BstsNx.Synthetic.Generator.generate(cfg)
```

### `BstsNx.Synthetic.Adstock`

Use case: model carry-over and saturation response curves.

```elixir
BstsNx.Synthetic.Adstock.geometric_adstock([1.0, 0.0, 0.0], 0.5)
```

## Application Wrappers

### `BstsNx.Applications.TVAttribution`

Use case: TV-to-web spot attribution using pre/post periods and overlap handling.

```elixir
BstsNx.Applications.TVAttribution.attribute(observations, spots,
  pre_period: {1, 1000},
  post_period: {1001, 1100}
)
```

### `BstsNx.Applications.MarketingLift`

Use case: campaign incrementality with single or overlapping campaigns.

```elixir
BstsNx.Applications.MarketingLift.measure_lift(observations, campaign)
```

### `BstsNx.Applications.DemandForecaster`

Use case: demand forecasting and safety-stock planning with uncertainty bands.

```elixir
BstsNx.Applications.DemandForecaster.forecast(demand, horizon: 14, seasonality: 7)
```

### `BstsNx.Applications.AnomalyDetector`

Use case: calibrated anomaly scoring (batch or streaming).

```elixir
detector = BstsNx.Applications.AnomalyDetector.fit(training_data, method: :filter)
BstsNx.Applications.AnomalyDetector.score(detector, new_values)
```

### `BstsNx.Applications.PolicyEvaluator`

Use case: interrupted time-series evaluation for policy/regulatory interventions.

```elixir
BstsNx.Applications.PolicyEvaluator.evaluate(observations, intervention)
```

### `BstsNx.Applications.AudienceForecast`

Use case: multiply aligned universe, PUT/HUT, and share posterior draws so audience uncertainty is propagated coherently.

```elixir
BstsNx.Applications.AudienceForecast.combine(viewing_forecast, share_forecast, universe)
```

### `BstsNx.Applications.MakegoodRisk`

Use case: aggregate scheduled audience delivery and calculate underdelivery probability, expected shortfall, and conservative inventory from joint forecast draws.

```elixir
BstsNx.Applications.MakegoodRisk.evaluate(audience_forecast, exposure_weights, guarantee)
```

## Internal Module

### Utils (internal module)

Use case: shared numeric utility helpers in the `Utils` module; not part of the public HexDocs API contract.

```elixir
# Internal module (moduledoc false)
```
