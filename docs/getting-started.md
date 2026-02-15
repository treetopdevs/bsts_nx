# Getting Started

This guide walks through the core workflow: filtering, smoothing, and sampling
for a scalar local-level BSTS model.

## Install

Add the dependency:

```elixir
def deps do
  [
    {:bsts_nx, "~> 0.1"}
  ]
end
```

## Kalman filtering

```elixir
obs = [1.0, 2.0, 3.0]

filtered = BstsNx.KalmanFilter.filter(obs, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0)
means = Enum.map(filtered, fn {x, _p} -> Nx.to_number(x) end)
```

## Smoothing

```elixir
{filtered, predicted} =
  BstsNx.KalmanFilter.filter_with_pred(obs, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0)

smoothed = BstsNx.Smoother.rts(filtered, predicted, 1.0)
```

## Gibbs sampling

```elixir
samples =
  BstsNx.GibbsSampler.sample(obs, 100, 0.0, 1.0, 1.0, 1.0,
    burn_in: 50,
    seed: 123
  )

process_vars = Enum.map(samples, fn s -> Nx.to_number(s.process_var) end)
```

## Missing observations

Use `nil` in the observation list. The Kalman filter skips updates when
observations are missing. In the Gibbs sampler, missing observations are
ignored when computing the observation variance, and a warning is logged.

```elixir
obs = [1.0, nil, 3.0]
BstsNx.KalmanFilter.filter(obs, 1.0, 1.0, 0.1, 0.5, 0.0, 1.0)
```

## EXLA acceleration

Enable EXLA in your host application for JIT compilation:

```elixir
config :nx, :default_backend, EXLA.Backend
```
