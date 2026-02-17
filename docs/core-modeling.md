# Core Modeling

This guide covers the low-level modeling primitives that all higher-level APIs build on.

## State-Space Model Form

`BstsNx` uses linear-Gaussian state-space models:

- state transition: `x_t = F x_{t-1} + w_t`
- observation: `y_t = H_t x_t + v_t`
- process noise: `w_t ~ N(0, Q)`
- observation noise: `v_t ~ N(0, R)`

## Kalman Filter (`BstsNx.KalmanFilter`)

Use it for forward inference of latent state means/covariances.

```elixir
obs = [1.0, 2.0, nil, 3.5]

filtered =
  BstsNx.KalmanFilter.filter(obs, 1.0, 1.0, 0.1, 0.5, 0.0, 1.0)

{filtered_states, filtered_covs} = Enum.unzip(filtered)
```

- `filter/7`: filtered posterior only
- `filter_with_pred/7`: filtered + one-step predictions (needed by RTS smoother)
- `filter_defn/7`: compiled tensor path for scalar systems (`NaN` for missing)

`filter_with_pred/7` supports scalar and multi-dimensional observations with
static or time-varying `H_t`.

## RTS and Simulation Smoothing (`BstsNx.Smoother`)

Use smoothing when you need posterior state trajectories conditioned on all observations.

```elixir
{filtered, predicted} =
  BstsNx.KalmanFilter.filter_with_pred(obs, 1.0, 1.0, 0.1, 0.5, 0.0, 1.0)

smoothed = BstsNx.Smoother.rts(filtered, predicted, 1.0)
{draw, _next_key} = BstsNx.Smoother.simulate_with_key(smoothed, filtered, predicted, 1.0, key: Nx.Random.key(42))
```

- `rts/3`: list-based smoother
- `rts_defn/4`: compiled scalar smoother
- `simulate/5`, `simulate_with_key/5`: Carter-Kohn simulation smoother

## Gibbs Sampling (`BstsNx.GibbsSampler`)

Use Gibbs sampling for posterior uncertainty over states and variances.

```elixir
samples =
  BstsNx.GibbsSampler.sample([1.0, 2.0, 3.0], 200, 0.0, 1.0, 1.0, 1.0,
    burn_in: 100,
    thin: 1,
    seed: 42
  )

hd(samples).process_var
```

Structured models:

```elixir
spec = BstsNx.Components.local_linear_trend_spec()
samples = BstsNx.GibbsSampler.sample_structured(observations, spec, 200, seed: 42)
```

Multiple chains are available via `sample_chains/8` and `sample_structured_chains/5`.

`sample_structured/4` supports higher-dimensional latent states and composed
models; it currently targets scalar observations.

## Building Models (`BstsNx.Components`, `BstsNx.StateSpace`, `BstsNx.ModelSpec`)

### Component factories

```elixir
trend = BstsNx.Components.local_linear_trend_spec()
seasonal = BstsNx.Components.seasonal_spec(7)
spec = BstsNx.Components.compose_specs(trend, seasonal)
```

### Manual composition

```elixir
level = BstsNx.Components.local_level(0.1)
season = BstsNx.Components.seasonal(7, 0.01)
composed = BstsNx.StateSpace.compose(level, season)
```

Use `ModelSpec` for structured samplers and any API that needs explicit model structure.

## Current Boundaries

Current implementation boundaries to keep in mind:

- structured Gibbs sampling assumes a scalar observation equation per time step,
- observation variance is sampled as a scalar (`R`),
- process covariance learning is diagonal (`Q` diagonal entries sampled independently),
- compiled `defn` filter/smoother paths remain scalar-oriented.

In practice, this means you can already build rich high-dimensional latent
state compositions, while full multivariate-observation Bayesian sampling
remains a roadmap item.

## Random Variables (`BstsNx.Distributions`)

`BstsNx.Distributions` provides reusable samplers used internally and for custom workflows:

- inverse-gamma (`inv_gamma_sample/3`) for variance priors/posteriors,
- normal (`normal_sample/2`),
- multivariate normal (`mv_normal_sample/3`).

## Shared Helpers (`BstsNx.ModelBuilder`)

`ModelBuilder` centralizes:

- model selection from options,
- control-series composition,
- effect formatting utilities,
- forecast helper construction for time-varying regressors.

It is especially useful when implementing your own application wrapper module.
