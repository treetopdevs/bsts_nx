# BstsNx

BstsNx is a pure-Elixir library for Bayesian Structural Time Series (BSTS) models built on top of Nx.
It provides Kalman filtering and smoothing, a Gibbs sampler for a local-level model, and a
Causal Impact workflow for intervention analysis.

## Installation

Add the dependency to your `mix.exs`:

```elixir
def deps do
  [
    {:bsts_nx, "~> 0.1"}
  ]
end
```

## Quick start

### Kalman filter (local level model)

```elixir
obs = [1.0, 2.0, 3.0]
{xs, _ps} =
  BstsNx.KalmanFilter.filter(obs, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0)
  |> Enum.unzip()

Enum.map(xs, &Nx.to_number/1)
#=> [0.666..., 1.5, 2.428...]
```

### Gibbs sampler (local level model)

```elixir
obs = Enum.map(1..30, fn _ -> :rand.normal() * 2.0 + 10.0 end)

samples =
  BstsNx.GibbsSampler.sample(obs, 50, 0.0, 1.0, 1.0, 1.0,
    burn_in: 25,
    seed: 123
  )

length(samples)
#=> 50
```

### Causal Impact

```elixir
pre = Enum.map(1..60, fn _ -> 50.0 + :rand.normal() * 5 end)
post = Enum.map(1..30, fn _ -> 60.0 + :rand.normal() * 5 end)
obs = pre ++ post

result =
  BstsNx.CausalImpact.estimate(obs, {1, 60}, {61, 90},
    num_samples: 100,
    burn_in: 50,
    seed: 42
  )

summary = BstsNx.CausalImpact.summary(result)
summary.cumulative_effect.mean
```

## Missing observations

`BstsNx.KalmanFilter` supports missing observations as `nil` (or `NaN` for the compiled defn).
When used in the Gibbs sampler, missing observations are skipped when updating the observation
variance and a warning is logged.

## EXLA acceleration (optional)

Add `:exla` and configure Nx in your host application:

```elixir
config :nx, :default_backend, EXLA.Backend
```

## Documentation

- `docs/overview.md` for a map of modules and workflows
- `docs/hex-publishing-checklist.md` for release and Hex publish prep
- `docs/release-readiness-plan.md` for staged no-publish release planning
- `docs/getting-started.md` for an end-to-end starter walkthrough
- `docs/core-modeling.md` for Kalman/smoother/Gibbs/model composition details
- `docs/causal-inference-and-attribution.md` for intervention + attribution workflows
- `docs/forecasting-and-applications.md` for forecasting and domain wrappers
- `docs/synthetic-data-and-validation.md` for scenario generation and calibration
- `docs/module-reference.md` for module-by-module use cases and snippets
- API docs are generated with ExDoc (`mix docs`)

## Status

This is an early-stage implementation focused on scalar local-level models. The APIs are
intended to expand to richer component compositions and higher-dimensional state spaces.
