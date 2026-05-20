# BstsNx

BstsNx is a pure-Elixir library for Bayesian Structural Time Series (BSTS) models built on top of Nx.
It provides Kalman filtering and smoothing for scalar and multi-dimensional state-space systems,
Gibbs samplers for both local-level and structured composed models, and Causal Impact,
forecasting, and attribution workflows.

## Installation

Add the dependency to your `mix.exs`:

```elixir
def deps do
  [
    {:bsts_nx, "~> 0.1"}
  ]
end
```

`bsts_nx` targets the latest Nx stack and requires:

- Elixir `~> 1.19`
- `nx ~> 0.12.0`
- `emlx ~> 0.3.0` (optional)
- `exla ~> 0.12.0` (optional)
- `xla ~> 0.10` (optional)

If your app or `Mix.install/2` script pins `nx` to `0.6.x` (or any `< 0.12`),
dependency resolution will fail. Upgrade the full graph to Nx 0.12.

For `Mix.install/2`, keep everything in a single call and avoid pinning an older Nx:

```elixir
Mix.install([
  {:bsts_nx, "~> 0.1"}
])
```

## Quick start

### Kalman filter (local-level model)

```elixir
obs = [1.0, 2.0, 3.0]
{xs, _ps} =
  BstsNx.KalmanFilter.filter(obs, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0)
  |> Enum.unzip()

Enum.map(xs, &Nx.to_number/1)
#=> [0.666..., 1.5, 2.428...]
```

### Gibbs sampler (local-level model)

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

### Regression model spec (structured)

```elixir
regressors = Nx.tensor([
  [1.0, 0.5],
  [0.8, 1.2],
  [1.1, 0.9]
])

spec = BstsNx.Components.regression(regressors, var_beta: 0.01, obs_var: 1.0)
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

### Operational pipeline mode

High-level pipeline APIs now default to an operational Elixir/Nx lane:
forecast-first fixed-variance filters plus spot attribution. This is the path
intended for Elixir data pipelines and bounded-latency demos.

```elixir
spec = BstsNx.Components.local_level_spec(initial_state: 50.0, initial_cov: 10.0)
spots = [%{id: "spot_1", window_start: 0, window_end: 10}]

prepared = BstsNx.Operational.prepare(spec, {1, 60}, {61, 90})

result =
  BstsNx.Operational.run(prepared, obs, spots,
    return: :lists
  )

result.execution.method_used
#=> :scalar_forecast_filter

result =
  BstsNx.Pipeline.run(obs, {1, 60}, {61, 90}, spots, spec,
    mode: :operational
  )

result.execution.method_used
#=> :scalar_forecast_filter
```

Use `baseline: :smooth` only when you explicitly want the older
mask-and-smooth compatibility behavior. Use `return: :tensors` for hot paths
that should keep vector outputs in Nx tensors until presentation time.

Use `mode: :bayesian` when you want the slower Elixir MCMC posterior-draw workflow.
For offline validation and reporting, `BstsNx.RSidecar` can call R's CRAN
`CausalImpact`/`bsts` stack when `Rscript` and those packages are installed.

## Missing observations

`BstsNx.KalmanFilter` supports missing observations as `nil` or `NaN`.
When used in the Gibbs sampler, missing observations (`nil`/`NaN`) are skipped when updating
the observation variance and a warning is logged.

## Backend acceleration (optional)

Add `:exla` and configure Nx in your host application:

```elixir
config :nx, :default_backend, EXLA.Backend
```

EMLX is also supported as an optional backend dependency. For structured BSTS
workflows, EMLX GPU support is currently limited by missing linalg primitives
needed by the Kalman/smoother paths; EMLX CPU, EXLA, and BinaryBackend are the
practical fallback paths today.

To compare backend behavior on a structured workload:

```bash
mix bench.structured_backends
```

Useful environment overrides include `BSTS_NX_BENCH_BACKENDS`, `BSTS_NX_BENCH_T`,
`BSTS_NX_BENCH_SAMPLES`, `BSTS_NX_BENCH_BURN_IN`, `BSTS_NX_BENCH_DTYPE`, and
`BSTS_NX_BENCH_OUTPUT`.

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

This is an early-stage implementation with solid support for structured model composition
(trend, seasonal, regression, and composed `ModelSpec`s) and higher-dimensional latent states.

Current limitations remain in some paths:

- compiled `defn` filter/smoother paths are scalar-oriented;
- structured Gibbs samplers currently learn a diagonal `Q` and scalar observation variance `R`;
- structured MCMC workflows currently target scalar observations per time step.

The near-term direction is an Elixir-first operational pipeline with explicit
execution metadata, plus optional R-backed offline parity/reporting. The goal is
not to outpace CRAN `bsts`/`CausalImpact` immediately, but to make causal impact
and attribution usable inside Elixir systems while using R as a reference.

The roadmap includes richer component families and full multivariate-observation support
through structured MCMC and downstream APIs.
