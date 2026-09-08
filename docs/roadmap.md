# BstsNx Complete Vision and Roadmap

**Date:** 2026-09-08  
**Status:** proposed strategic roadmap  
**Current baseline:** `main` at `d6476ba8e75c6edafd69b9761ee785ad319b5742`

## Executive decision

`BstsNx` should not attempt to become another large pretrained time-series
foundation model. Its durable role is different:

> **BstsNx is an Elixir-native Bayesian evidence and decision layer that turns
> noisy time-series measurements and forecasts from one or more engines into
> calibrated, explainable, path-aware operational decisions.**

The library should continue to provide a native BSTS implementation, but native
BSTS is one forecasting and inference provider inside a larger architecture.
External predictors such as TimesFM, classical statistical models, boosted-tree
models, domain-specific simulators, and managed forecasting services should be
able to participate through stable contracts.

The initial proving ground is television audience measurement and advertising
inventory, especially Comscore-style measurement, PUT/HUT and share forecasting,
sports and specials, delivery guarantees, and makegood risk. The core contracts
must remain generic enough for demand forecasting, policy evaluation, marketing
lift, anomaly detection, and other time-series decisions.

The north-star workflow is:

```text
measured series + measurement reliability + historical covariates
                         + future schedules and scenarios
                                      |
                                      v
                         one or more forecast providers
                     (native BSTS, TimesFM, ML, classical)
                                      |
                                      v
                        normalized predictive distribution
                     (point, quantiles, joint draws, provenance)
                                      |
                                      v
                   Bayesian residual correction and calibration
                                      |
                                      v
                  scenario mixture, hierarchy, and reconciliation
                                      |
                                      v
                  business loss, risk, attribution, and decisions
                                      |
                                      v
                   backtesting, diagnostics, audit, and monitoring
```

## Product thesis

Most forecasting systems stop at a point estimate or a set of marginal
quantiles. Operational users need answers to different questions:

- How much of this movement is audience change, and how much is measurement
  change?
- What is the probability that an entire advertising flight underdelivers?
- How many makegood units should be reserved now rather than after a miss?
- How should uncertainty be propagated through `universe × PUT/HUT × share`?
- What happens if the local team advances, a playoff series ends early, or a
  live event overruns?
- Which intervention, promotion, or airing plausibly caused the observed lift?
- Is a result stable, calibrated, reproducible, and safe to act on?

That decision layer is where `BstsNx` should differentiate. Raw forecast accuracy
remains necessary, but it is not the product boundary.

## Intended users

The roadmap serves four user groups.

1. **Elixir application developers** who need forecasting, causal inference, or
   risk calculations without moving the entire application into Python.
2. **Data and forecasting teams** that want to expose an existing model through
   a stable, auditable contract and combine it with Bayesian calibration.
3. **Operators and analysts** who need decisions expressed in business units,
   probabilities, reserves, and expected loss rather than model-specific output.
4. **Library and platform maintainers** who need deterministic tests, backend
   parity, model provenance, and a clear separation between research and
   production paths.

## What BstsNx should own

The project should own the following responsibilities:

- Canonical time-series, design, forecast, scenario, and model-artifact contracts.
- Native state-space and BSTS modeling for interpretable local structure.
- Explicit measurement-error modeling, including time-varying reliability.
- Preservation of joint uncertainty across horizons, targets, and scenarios.
- Residual correction and calibration of external model forecasts.
- Rolling-origin backtesting, diagnostics, calibration checks, and model
  comparison.
- Hierarchical and compositional reconciliation.
- Causal-impact and attribution workflows with clear identification caveats.
- Decision functions such as delivery risk, expected shortfall, reserve setting,
  and eventually constrained optimization.
- Elixir-native online filtering, artifact loading, telemetry, and orchestration.

## What BstsNx should not own

The following are explicit non-goals:

- Training a general-purpose transformer at foundation-model scale.
- Reimplementing every external forecasting model in Nx before integrations can
  ship.
- Pretending marginal quantiles are a joint posterior distribution.
- Claiming that predictive covariates establish causality.
- Embedding Comscore, Nielsen, sports-league, or station-specific schemas into
  the mathematical core.
- Requiring Python, a GPU, or a network service for the base library to work.
- Silently using model weights whose license does not permit the caller's use.
- Hiding data leakage, future-data assumptions, interpolation, or measurement
  revisions behind convenient defaults.

## Current foundation

At the baseline named above, the repository already contains meaningful pieces
of the target system:

- Scalar and multidimensional Kalman filters and smoothers.
- Scalar and structured Gibbs samplers.
- Composable trend, seasonal, and regression state-space components.
- Causal Impact, intervention analysis, attribution, Shapley allocation, and
  operational forecast-first paths.
- A native `BstsNx.Forecaster` and joint-draw `BstsNx.Forecast` representation.
- Known relative observation-variance weighting using
  `R_t = sigma_R^2 × weight_t`.
- Draw-level audience composition and makegood-risk calculations.
- Synthetic-data tools, MCMC diagnostics, an integrated validation workflow,
  and an optional R sidecar for reference comparisons.
- Execution metadata and CI coverage across native and EXLA lanes.

The current limitations define the roadmap:

- The public model specification mixes static model structure with historical
  design data.
- Structured MCMC still targets one scalar observation per time step.
- Learned process noise is diagonal and learned observation variance is scalar.
- Regression metadata assumes a limited layout and does not provide a general
  named-component registry.
- Future inputs are deterministic; there is no first-class scenario mixture.
- There is no provider contract for external forecasting engines.
- There is no canonical rolling-origin forecasting benchmark or
  champion/challenger workflow.
- There is no general response-transform, known-offset, or external-baseline
  abstraction.
- Fleet-scale batched fitting, durable fit artifacts, and generic online updates
  are incomplete.
- Hierarchical market pooling, simplex-valued share forecasts, and coherent
  market/region/national reconciliation are not implemented.

## Design principles

### 1. Draws first, summaries second

Means, intervals, and quantiles are derived views. Any workflow that aggregates
periods, multiplies uncertain quantities, applies a nonlinear transform, mixes
scenarios, or evaluates a threshold must operate on complete trajectories when
available.

### 2. Never imply dependence that was not modeled

Matching shapes or random seeds does not create dependence between separately
fitted models. Draw alignment must carry explicit semantics:

- `:joint_model`
- `:shared_scenario`
- `:residual_bootstrap`
- `:copula_reconstruction`
- `:independent_pairing`

Decision APIs must reject marginal-only forecasts unless the caller chooses an
explicit joint-reconstruction policy.

### 3. Measurement is part of the model

A reported value is not identical to the latent quantity of interest. Device
footprint, sample composition, source mix, personification, calibration,
revisions, and reporting completeness belong in the observation model or its
metadata, not only in a feature matrix.

### 4. Providers are replaceable; evidence contracts are stable

TimesFM, native BSTS, a boosted-tree model, or a vendor endpoint can change. The
contracts for time, targets, uncertainty, scenarios, provenance, and decisions
should not.

### 5. Keep the core Elixir-native

The base package remains useful with Elixir and Nx alone. Python and external
services live behind adapters or sidecars. A provider failure must not corrupt
or redefine the core statistical semantics.

### 6. Separate research, validation, and production lanes

The project should preserve distinct execution modes:

- **Research:** flexible models, external checkpoints, exploratory scenarios.
- **Validation:** multi-chain inference, reference comparisons, deep backtests.
- **Production:** bounded-latency filters, loaded artifacts, controlled providers.

Execution metadata must make the chosen lane visible.

### 7. Preserve as-of truth

Every backtest and model artifact should identify the data vintage, forecast
cutoff, feature version, measurement-method version, and provider version.
Revised history must not be substituted silently for the data available when a
historical forecast would have been issued.

### 8. Domain applications depend on generic primitives

Sports, specials, audience delivery, and makegoods are important application
modules. Their reusable foundations—scenarios, joint forecasts, loss functions,
reconciliation, artifacts, and providers—belong in the generic core.

### 9. Make unsafe convenience explicit

Interpolation, independent draw pairing, quantile-to-draw conversion, clipping,
and renormalization may be reasonable in some settings. They must be named
policies with recorded metadata, never invisible defaults.

### 10. Backward compatibility is a roadmap constraint

Existing `Forecaster`, `Forecast`, `CausalImpact`, `Operational`, and application
APIs should continue to work through adapters and deprecation periods. New
contracts should compile down to the current engine before replacing it.

## Target architecture

```text
BstsNx.Data
  TimeAxis
  Series / Panel
  MeasurementMetadata
  Design / FutureDesign
        |
        v
BstsNx.ForecastProvider behaviour
  Providers.NativeBsts
  Providers.External
  optional TimesFM adapter or sidecar package
  application-owned providers
        |
        v
BstsNx.Forecast
  target and time axes
  point forecast
  marginal quantiles
  optional joint draws
  draw/scenario semantics
  provider provenance
        |
        +-------------------------+
        |                         |
        v                         v
BstsNx.Calibration           BstsNx.Scenarios
  residual BSTS                scenario trees
  bias correction              probability weights
  interval calibration         shared future drivers
  conformal overlays           mixture construction
        |                         |
        +------------+------------+
                     v
BstsNx.Reconciliation
  temporal aggregation
  hierarchy constraints
  simplex/share constraints
  draw-level coherence
                     |
                     v
BstsNx.Decision
  threshold probability
  expected shortfall / CVaR
  reserve setting
  business loss
  constrained action selection
                     |
                     v
BstsNx.Applications
  AudienceForecast
  MakegoodRisk
  TVAttribution
  DemandForecaster
  MarketingLift
  PolicyEvaluator
                     |
                     v
BstsNx.Backtest / Validation / Registry / Telemetry
```

## Canonical contracts

The names below are target API sketches, not commitments to exact spelling. The
important decision is the separation of concerns.

### `BstsNx.TimeAxis`

Represents time explicitly rather than inferring it from tensor position.

```elixir
%TimeAxis{
  timestamps: [...],
  frequency: :quarter_hour,
  timezone: "America/New_York",
  calendar: :broadcast,
  cutoff: ~U[2026-09-08 10:00:00Z],
  data_as_of: ~U[2026-09-08 12:30:00Z]
}
```

It should support regularity validation, missing-period insertion, broadcast-day
boundaries, daylight-saving transitions, and future-axis generation.

### `BstsNx.Series` and `BstsNx.Panel`

`Series` represents one target over a `TimeAxis`. `Panel` represents named target
and covariate dimensions such as market, station, demo, daypart, or program.
Neither type should impose television semantics.

### `BstsNx.Design` and `BstsNx.FutureDesign`

These objects hold observations and time-varying inputs separately from the
static model definition.

```elixir
%Design{
  targets: target_panel,
  inputs: %{
    schedule: schedule_features,
    footprint: measurement_features,
    demand: demand_features
  },
  measurement: measurement_metadata,
  offset: external_baseline,
  time_axis: historical_axis
}
```

`FutureDesign` adds known future inputs or a `ScenarioSet`. Training and future
feature schemas must be versioned and validated for identical meaning and order.

### `BstsNx.Model`

`Model` declares structure without embedding historical values:

```elixir
%Model{
  response: Transform.log1p(),
  observation: Observation.gaussian(...),
  components: [
    Components.local_linear_trend(:trend, ...),
    Components.dummy_seasonal(:weekday, period: 7),
    Components.trigonometric_seasonal(:annual, period: 365.2425, harmonics: 6),
    Components.regression(:schedule, input: :schedule, coefficients: :static),
    Components.regression(:footprint, input: :footprint, coefficients: :dynamic)
  ]
}
```

A compiler can lower this representation into the existing `ModelSpec` and
matrix engine during migration.

### `BstsNx.Forecast`

The existing draw-bearing forecast should evolve into a representation-aware
forecast contract:

```elixir
%Forecast{
  targets: target_axis,
  time_axis: future_axis,
  point: point_tensor,
  quantiles: %{0.1 => q10, 0.5 => q50, 0.9 => q90},
  draws: joint_draw_tensor_or_nil,
  draw_semantics: :shared_scenario,
  scenario_ids: scenario_ids_or_nil,
  metadata: %{
    target: :reported_observation,
    provider: :timesfm,
    provider_version: "...",
    data_as_of: "..."
  }
}
```

The current constructor from a `{draw, horizon}` tensor remains supported.
Risk functions call `Forecast.require_joint_draws!/1` and fail clearly when only
marginal quantiles are available.

### `BstsNx.ForecastProvider`

A provider normalizes native or external predictors:

```elixir
@callback capabilities() :: map()
@callback fit(Design.t(), keyword()) :: {:ok, FitArtifact.t()} | {:error, term()}
@callback forecast(FitArtifact.t() | :zero_shot, FutureDesign.t(), keyword()) ::
            {:ok, ProviderResult.t()} | {:error, term()}
```

Capabilities should declare whether a provider supports:

- zero-shot inference,
- fitting or fine-tuning,
- univariate or multivariate targets,
- past-only and known-future covariates,
- point, quantile, or joint-draw output,
- missing observations,
- online updates,
- batch inference,
- commercial production under the selected artifact license.

`ProviderResult` contains a normalized `Forecast`, execution metadata, warnings,
and provenance. Provider-specific output must not leak through the generic
contract unless explicitly attached as opaque metadata.

### `BstsNx.FitArtifact`

A durable artifact should record:

- schema version,
- model and provider identity,
- parameters or checkpoint reference,
- terminal state and covariance where applicable,
- transform and feature-scaler state,
- target and input schemas,
- training period and data vintage,
- data and code fingerprints,
- measurement-method version,
- license identifier and usage restrictions,
- diagnostics and backtest summary,
- library and backend versions.

Serialization should copy persisted tensors to a backend-neutral representation.

### `BstsNx.ScenarioSet`

A scenario set represents uncertain future inputs:

```elixir
%ScenarioSet{
  ids: [:local_team_advances, :local_team_eliminated],
  probabilities: Nx.tensor([0.42, 0.58]),
  future_inputs: %{schedule: scenario_schedule_tensor},
  shared_variables: %{bracket_draw: bracket_ids},
  metadata: %{source: :sports_simulator}
}
```

Scenario probabilities must sum to one within tolerance. Forecast outputs retain
scenario identity until the caller explicitly constructs a mixture.

## Native BSTS evolution

### Separate model structure from design data

This is the primary architectural refactor. Historical `H_t` rows, future
regressors, and model metadata should no longer be inferred from position in a
composed matrix. Named components and input schemas replace trailing-column
heuristics.

### Observation models

Implement observation variance in increasing order of complexity:

1. Learned scalar variance, preserving current behavior.
2. Fixed scalar variance.
3. Fixed time-varying variance series.
4. Scaled-known variance, preserving the current
   `R_t = sigma_R^2 × weight_t` implementation.
5. Known variance plus learned residual variance:
   `R_t = known_R_t + sigma_residual^2`.
6. Multivariate diagonal `R_t`.
7. Structured or full multivariate `R_t` where evidence and scale justify it.

Forecast APIs must distinguish the latent process from the future reported
observation:

```elixir
target: :latent_mean
target: :reported_observation
```

### Response transforms and offsets

Transforms must be fitted on training data, applied before inference, inverted
for each posterior draw, and only then summarized. Initial built-ins:

- identity,
- standardization and robust standardization,
- `log` and `log1p`,
- bounded `logit`,
- composition of transforms.

A known offset supports external baselines:

```text
y_t = external_baseline_t + H_t x_t + error_t
```

This is the key primitive for TimesFM-plus-BSTS residual forecasting, analog
forecasts for new programs, and correction of existing production models.

### Named components

Each component should have a stable name, state slice, labels, input binding,
process-noise group, and decomposition output. Required component families:

- local level and local linear trend,
- damped trend,
- dummy seasonal,
- trigonometric/Fourier seasonal,
- autoregressive residuals,
- static regression,
- dynamic regression,
- spike-and-slab regression,
- step, pulse, ramp, and slope-change interventions,
- shared-factor components for panels.

### Process-noise blocks

Replace independent diagonal `q_specs` with process-noise blocks:

- independent diagonal variances,
- one learned scale shared across a state group,
- fixed covariance blocks,
- a known covariance shape multiplied by a learned scale,
- later, carefully constrained full covariance estimation.

Forward simulation must use a covariance factor rather than dropping
off-diagonal terms.

### Inference engines

Expose inference through a common estimator contract:

- fixed-parameter Kalman filtering,
- MCMC,
- Kalman EM for initialization and fleet fitting,
- optional variational or Laplace approximations if justified,
- optional R reference sidecar.

The same `Model`, `Design`, `FitArtifact`, and `Forecast` contracts should work
across estimators.

## External forecast providers and TimesFM

### Role of TimesFM

TimesFM is a strong generic and multivariate forecast provider, particularly for
zero-shot and broad panel forecasting. It should be treated as a benchmark and
provider, not as the architecture of `BstsNx`.

At the time of this roadmap, TimesFM 3.0 provides native multivariate targets,
past-only and past-and-future covariates, and marginal quantile forecasts. The
public 3.0 checkpoint is currently licensed for non-commercial,
non-production use, while earlier weights and managed products may have
different terms. Every implementation must reverify the exact artifact and
service terms rather than relying on this roadmap's snapshot.

Reference: <https://github.com/google-research/timesfm>

### Integration boundary

The base package should not acquire PyTorch, Python, or model weights as a
runtime dependency. Use one of these boundaries:

1. An application-owned HTTP or gRPC provider.
2. A supervised local sidecar process with a stable wire protocol.
3. A separate optional adapter package, such as `bsts_nx_timesfm`.
4. A managed provider with explicit timeout, retry, and license policy.

The adapter records checkpoint identity, provider version, backend, license,
latency, and input truncation or interpolation policies.

### Hybrid progression

Implement hybrid forecasting in stages.

#### Stage A: challenger benchmark

TimesFM produces an independent point and quantile forecast. `BstsNx.Backtest`
compares it against seasonal naive, native BSTS, and current production models.

#### Stage B: fixed external baseline plus residual BSTS

Let `m_t` be the TimesFM median or another external forecast:

```text
residual_t = observed_t - m_t
residual_t = structural_state_t + measurement_error_t
final_draw_t = m_t + residual_draw_t
```

This combines pretrained pattern recognition with local structural correction,
measurement reliability, and joint residual trajectories.

#### Stage C: calibrated marginal distribution

Use historical forecast errors to calibrate TimesFM quantiles by segment and
horizon. Calibration must be evaluated out of sample.

#### Stage D: explicit joint reconstruction

When the provider returns marginal quantiles only, construct joint trajectories
only through a named policy, such as:

- block residual bootstrap,
- Gaussian or vine copula fitted on backtest residuals,
- scenario-conditioned sampling,
- a separate generative provider.

Independent sampling is allowed only when selected explicitly and recorded.

#### Stage E: ensemble and provider selection

Blend providers by backtested performance, regime, horizon, and target. Provider
weights should be learned only from data available before each backtest cutoff.

## Television audience reference architecture

The television application should remain a reference implementation of generic
primitives.

### Forecast decomposition

The primary structural model is:

```text
audience = universe × PUT_or_HUT × share
```

Forecast each uncertain factor, preserve compatible draws or scenarios, and
multiply draw by draw. Do not multiply marginal means.

For a complete station market, shares must be compositional. Independent share
forecasts should eventually use additive or isometric log-ratio coordinates and
an inverse transform that produces positive shares summing to one.

### Comscore measurement metadata

A Comscore design should be able to carry:

- active STB and ACR footprint,
- STB/ACR mix,
- provider/source mix,
- reporting completeness,
- personification or co-viewing calibration version,
- universe revision,
- historical revision magnitude,
- cell reliability tier or explicit variance estimate,
- market, demo, daypart, station, and program identifiers,
- data publication and revision timestamps.

These fields may become regressors, observation-variance inputs, hierarchy keys,
or provenance. The application layer decides their domain meaning; the core
supplies the mathematical roles.

### Forecast grain

Reference supported grains should include:

- market × demo × fixed quarter-hour or daypart, one observation per broadcast
  day,
- station × market × demo × daypart,
- program airing or episode,
- program franchise across airings,
- market or national PUT/HUT,
- station or program share.

The library should not encourage a 672-state dummy weekly seasonal for
quarter-hour data. Compact harmonic seasonality or a different series layout is
preferred.

## Sports and specials

TimesFM and native models do not understand a program title in the human sense.
Sports and specials become forecastable when their historical behavior and
future conditions are encoded numerically.

### Generic event-feature schema

The audience application should define a versioned event schema covering:

- event class: sports, awards, political, holiday, concert, franchise, local,
  news, documentary, finale,
- live versus recorded,
- first-run versus repeat,
- league, sport, tournament, or franchise,
- regular season, postseason, round, championship, elimination status,
- local-team participation and market affinity,
- national relevance and distribution reach,
- scheduled start, duration, and overrun distribution,
- lead-in and lead-out,
- promotion weight and reach,
- search, social, trailer, or external demand signals,
- competing event strength,
- analog or comp-model forecast.

Categorical values must be encoded consistently and versioned. Free-form titles
may be converted into embeddings or analog features by an external model, but
they should not enter the core as unexplained strings.

### Sports scenario trees

Important future sports conditions are uncertain. A sports simulator or
application service should generate scenarios for:

- participating teams,
- local-team advancement,
- series length and game existence,
- playoff round and elimination stakes,
- game time and network window,
- expected competitiveness,
- overrun or extra-period duration,
- downstream preemption of scheduled programs.

The forecast provider runs conditionally on each scenario. `BstsNx.ScenarioSet`
retains common scenario identity across PUT/HUT, station share, lead-in, and
other related forecasts. The decision layer then forms a probability-weighted
mixture and evaluates inventory risk.

### Recurring and one-off specials

Recurring events can use prior occurrences directly. New specials need an analog
or comp model based on event class, franchise, slot, network, promotion, demand,
and competition. The analog forecast can enter as a known offset, while BSTS
models local bias and uncertainty.

### Duration and preemption

Live events require uncertain duration. Scenario outputs should include a
schedule realization so the same draw controls both event audience and the
availability or displacement of later inventory. Treating overrun probability
as an independent afterthought would understate correlated delivery risk.

## Hierarchy, panels, and coherence

### Stage 1: empirical-Bayes pooling

Estimate typical observation, level, slope, seasonal, and coefficient variance
priors by segment such as market tier, demo, daypart, station type, and genre.
Fit cells independently with pooled priors. This yields useful shrinkage before a
joint sampler exists.

### Stage 2: shared factors

Model common national, market-tier, network, genre, or daypart factors, then fit
local residual states conditionally. Shared factor draws should be reused across
local models to preserve cross-series dependence.

### Stage 3: multivariate observations

Generalize the compiled filter, smoother, forward simulator, and structured
sampler to:

```text
y_t in R^m
H_t in R^(m × n)
R_t in R^(m × m)
```

Start with diagonal `R`; add structured full covariance only after stable tests
and benchmarks.

### Stage 4: compositional shares

Support additive and isometric log-ratio transforms for station-share vectors.
Every draw must invert to positive shares that sum to one.

### Stage 5: hierarchical reconciliation

Reconcile draws across constraints such as:

- stations sum to market viewing,
- market tiers sum to regions and national totals,
- demos sum to broader demos where definitions permit,
- quarter-hours sum to programs, dayparts, and days,
- inventory schedules sum to contracts and portfolios.

Reconciliation operates on draws or scenario paths, not only means.

## Decision and application layer

### Forecast risk primitives

The generic decision layer should provide:

- threshold probability,
- expected positive shortfall,
- conditional expected shortfall / CVaR,
- value at risk or conservative quantiles,
- expected asymmetric loss,
- marginal contribution to portfolio risk,
- scenario contribution and sensitivity,
- decision comparison under a caller-provided loss function.

### Audience and makegood applications

Expand the existing applications to support:

- decomposed audience distributions,
- scenario-conditioned delivery,
- portfolio- and contract-level aggregation,
- probability of guarantee attainment,
- expected and conditional shortfall,
- expected makegood units and cost,
- reserve recommendations,
- marginal risk of adding or moving a unit,
- risk by market, station, demo, daypart, event, and scenario.

### Optimization

Optimization should arrive after the uncertainty contract is stable. The core
can expose loss tensors and constraints while an application chooses a solver.
Reference problems include:

- reserve the minimum inventory required to keep underdelivery probability below
  a threshold,
- reallocate units while preserving dollars, impressions, length, station,
  program, network, daypart, and timing constraints,
- minimize expected makegood cost under scenario uncertainty,
- select a robust schedule against sports and breaking-event scenarios.

The solver must consume joint or scenario-aligned forecasts. Point-estimate
optimization is insufficient for the target product.

### Causal and attribution applications

Causal Impact and attribution remain first-class, but forecast-provider output is
not automatically causal. Causal workflows must continue to require explicit
pre/post periods, controls, leakage checks, placebo tests, and stability checks.
External providers may generate a counterfactual baseline only when the
identification assumptions are documented.

## Backtesting, calibration, and model selection

### Rolling-origin backtesting

Add a dedicated `BstsNx.Backtest` module that recreates historical forecast
issuance:

- train through cutoff,
- build only the future inputs known at that cutoff,
- forecast selected horizons,
- compare with the as-published target vintage,
- advance the origin and repeat.

Each fold records cutoff, data vintage, target version, feature version, provider
artifact, scenario information, latency, and forecast representation.

### Required metrics

Predictive metrics:

- MAE,
- WAPE,
- MASE or RMSSE,
- signed bias,
- pinball loss by quantile,
- CRPS when draws are available,
- interval coverage and width.

Decision metrics:

- Brier score for underdelivery thresholds,
- reliability curves,
- expected-shortfall calibration,
- reserve error,
- realized makegood cost or a documented proxy,
- regret relative to the best feasible historical decision.

Operational metrics:

- p50 and p95 latency,
- throughput by series and horizon,
- memory and artifact size,
- provider failure and fallback rate,
- cost per forecast or per thousand series.

### Benchmark registry

A standard benchmark should compare:

- seasonal naive and trailing baselines,
- native BSTS,
- current production models,
- TimesFM or other external providers,
- external baseline plus residual BSTS,
- calibrated and ensemble variants.

Results must be segmented by horizon, market tier, demo, daypart, program type,
sports/specials, measurement regime, missingness, and volatility.

### Calibration

Calibration methods may include bias correction, quantile mapping, conformal
interval overlays, and residual bootstrap. They must be trained inside each
backtest fold or on an earlier calibration window. No post-period information may
leak into calibration.

### Champion/challenger

Model selection should be policy-driven:

- one provider may win for ordinary programming,
- another for sparse series,
- another for sports,
- another for short-horizon operational updates.

Selection rules are fitted and validated like any other model. The selected
provider and fallback path are recorded on each result.

## Operations and runtime

### Offline fit, online update

Use full Bayesian or expensive provider fits offline. Persist artifacts and use
fixed-parameter filtering or provider inference online. Trigger refits on a
schedule or when drift, residual, measurement, or schedule-regime monitors fire.

### Compact posterior storage

Add retention modes that distinguish:

- terminal state and parameters for forecasting,
- state paths for decomposition,
- covariance paths for deep diagnostics,
- full research output.

The sampler should eventually avoid allocating historical covariance paths when
the caller requests forecast-only retention, rather than compacting only after
sampling.

### Batch execution

Add batch axes for structurally compatible models. Group cells by frequency,
state dimension, feature count, and history length. Use masks for padded history
where numerically safe.

### Online API

A generic online API should support:

```elixir
online = BstsNx.Online.from_artifact(artifact)
{online, one_step_forecast} = BstsNx.Online.update(online, observation, inputs)
```

Modes should include a posterior-mean filter and a retained posterior ensemble.
Checkpoint and replay are required for revised historical measurements.

### Provider orchestration

External providers require:

- timeout and cancellation,
- bounded retries,
- circuit breaking,
- concurrency and memory limits,
- deterministic request fingerprints,
- cached immutable artifacts where permitted,
- explicit fallback policies,
- telemetry for truncation, interpolation, and provider warnings.

### Telemetry

Emit events for fit, forecast, update, calibration, reconciliation, decision,
backtest fold, provider failure, fallback, and artifact load. Include dimensions
that allow latency and error analysis without leaking sensitive time-series data.

## Provenance, governance, and licensing

Every production forecast should answer:

- Which data and revision were used?
- Which features and future assumptions were used?
- Which provider, model, checkpoint, code SHA, and backend ran?
- Which transform, calibration, scenario, and reconstruction policies ran?
- Were the forecasts latent or reported-observation targets?
- Were joint draws modeled, reconstructed, or assumed independent?
- Which license governs the model artifact, and is the use allowed?
- Which backtest and diagnostics supported deployment?

TimesFM illustrates why license metadata is operational, not clerical. The core
source may be permissively licensed while a particular checkpoint is not
available for commercial production. Adapters should fail closed when a caller
requires `usage: :commercial_production` and the artifact does not declare that
capability.

`BstsNx` itself remains subject to its repository license. Packaging and
integration guidance should make those obligations clear before a 1.0 release.

## Roadmap milestones

Milestones describe dependency order, not calendar promises. Effort estimates
assume one experienced engineer working primarily on the project and include
implementation, tests, documentation, and review.

### 0.2 — Forecast contracts and evidence foundation

**Objective:** establish stable objects that can represent native and external
forecasts without weakening current APIs.

Deliverables:

- `TimeAxis`, `Series`/`Panel`, `Design`, and `FutureDesign` initial contracts.
- Representation-aware evolution of `Forecast` with target/time axes,
  provenance, and explicit joint-draw semantics.
- `ForecastProvider`, `ProviderResult`, and provider capability contracts.
- `FitArtifact` schema and backend-neutral serialization envelope.
- Response transforms and known offsets.
- Compatibility adapters for current `Forecaster` and `Forecast` calls.
- Clear deprecation policy for positional inference and ambiguous metadata.

Acceptance gates:

- Existing public examples and tests continue to pass.
- A trivial external provider can return a normalized forecast.
- Decision APIs distinguish joint draws from marginal-only output.
- Transforms invert draw by draw before summaries.
- Artifacts round-trip across BinaryBackend and EXLA hosts.

Indicative effort: 4–7 weeks.

### 0.3 — Backtesting, calibration, and hybrid forecasting

**Objective:** make provider comparison and residual correction reproducible.

Deliverables:

- Rolling-origin `Backtest` engine with as-of data and feature callbacks.
- Predictive, probabilistic, decision, and operational metrics.
- Benchmark result and model-comparison structures.
- External-baseline residual BSTS wrapper.
- Bias and interval calibration primitives.
- Champion/challenger policy interface.
- Reference provider adapter protocol and a TimesFM research adapter.

Acceptance gates:

- Seasonal naive, native BSTS, external provider, and hybrid run through one
  benchmark API.
- No fold can read data after its cutoff.
- Hybrid forecasts preserve or explicitly reconstruct joint paths.
- Provider and checkpoint license metadata appear in every result.
- A documented audience benchmark determines where TimesFM helps or hurts.

Indicative effort: 5–8 weeks.

### 0.4 — Scenario-aware audience forecasting

**Objective:** support sports, specials, and uncertain future schedules without
collapsing them into one deterministic feature path.

Deliverables:

- `ScenarioSet`, probability validation, and scenario-mixture forecasts.
- Shared scenario alignment across several target models.
- Versioned audience event-feature schema.
- Sports bracket, game-existence, stakes, and overrun reference scenarios.
- Recurring-special and new-special analog workflow.
- Scenario-aware `AudienceForecast` and `MakegoodRisk` outputs.
- Scenario contribution and sensitivity reports.

Acceptance gates:

- A playoff example preserves team advancement and game-existence dependence
  across PUT, share, and schedule availability.
- An overrun scenario shifts both event audience and displaced downstream
  inventory.
- A one-off special can use a comp forecast as an offset.
- Portfolio delivery risk is computed after scenario mixing, not from mixed
  marginal quantiles.

Indicative effort: 4–7 weeks after 0.2 and core parts of 0.3.

### 0.5 — Native model expressiveness

**Objective:** remove the most important structural constraints in the native
BSTS engine.

Deliverables:

- Static `Model` plus time-varying `Design` compiler.
- Named component registry and multiple named regression blocks.
- Trigonometric seasonality and explicit intervention components.
- Observation variance modes through known-plus-learned scalar variance.
- Process-noise blocks with shared scales and fixed covariance shapes.
- Kalman EM estimator for initialization and fleet fitting.
- Unified estimator interface across fixed, EM, and MCMC paths.

Acceptance gates:

- Current `ModelSpec` workflows compile through the new representation.
- Weekly plus annual seasonality does not require hundreds of dummy states.
- Measurement-version interventions are visible in decomposition output.
- Forward simulation respects non-diagonal fixed covariance blocks.
- EM output can initialize MCMC and improve convergence on benchmark models.

Indicative effort: 7–12 weeks.

### 0.6 — Fleet runtime and online operation

**Objective:** run thousands of production series with bounded memory and clear
operational semantics.

Deliverables:

- Tensor-native forecast-only posterior retention.
- Batched filtering and forecasting for compatible structures.
- Durable, versioned artifact store format.
- Generic online update, checkpoint, and replay APIs.
- Provider orchestration helpers and telemetry.
- Drift and calibration monitors that recommend refit or provider fallback.

Acceptance gates:

- Forecast-only MCMC memory scales with retained terminal output rather than full
  historical covariance paths.
- Batched and single-series outputs agree within declared tolerance.
- A revised observation can be replayed deterministically from a checkpoint.
- Provider failures produce explicit fallback metadata.
- Operational benchmarks publish throughput, latency, memory, and artifact size.

Indicative effort: 6–10 weeks.

### 0.7 — Multivariate and hierarchical panels

**Objective:** model cross-series structure rather than approximating every cell
as independent.

Deliverables:

- Vector-observation filter, smoother, forward simulator, and structured sampler.
- Diagonal multivariate `R`, followed by structured covariance options.
- Empirical-Bayes prior estimation by panel strata.
- Shared global/group/local factors.
- Panel-aware batch inference and diagnostics.

Acceptance gates:

- Synthetic multivariate recovery tests cover correlated targets and missing
  cells.
- A market panel demonstrates measurable out-of-sample benefit over independent
  fits.
- Shared factors preserve cross-series path dependence in downstream decisions.
- Diagnostics identify poorly mixed covariance and factor parameters.

Indicative effort: 10–16 weeks.

### 0.8 — Compositional coherence and decision optimization

**Objective:** produce forecasts and actions that satisfy business and accounting
constraints.

Deliverables:

- Additive and isometric log-ratio transforms.
- Draw-level share reconstruction on the simplex.
- Temporal and hierarchical reconciliation, beginning with MinT-style methods.
- Generic loss functions and risk-budget interfaces.
- Reference makegood reserve and schedule optimization integrations.

Acceptance gates:

- Station shares are positive and sum to one for every draw.
- Reconciled station, market, region, and national totals satisfy constraints.
- Optimization evaluates expected loss across aligned scenarios.
- Point-estimate and risk-aware schedules can be compared by historical regret.

Indicative effort: 8–14 weeks.

### 0.9 — Hardening and ecosystem

**Objective:** prepare a stable public contract and operational reference stack.

Deliverables:

- API freeze candidate and migration guide.
- Expanded backend parity and numerical-tolerance policy.
- Security and resource-exhaustion review for sidecars and large tensors.
- Hex packaging and release automation, still subject to explicit publish
  approval.
- Reference applications, examples, and benchmark datasets.
- Provider authoring guide and compatibility test suite.
- Model card and artifact provenance documentation.

Acceptance gates:

- No known unplanned breaking API change remains.
- Required providers pass a shared compatibility suite.
- Reference benchmarks are reproducible from versioned data or generators.
- Documentation clearly separates prediction, causal evidence, and business
  decisions.

### 1.0 — Production decision platform

A 1.0 release should require all of the following:

- Stable data, provider, forecast, scenario, artifact, and decision contracts.
- A useful native BSTS path and at least one supported external-provider path.
- Explicit time-varying measurement reliability.
- Correct draw- or scenario-level aggregation for business risk.
- Rolling-origin benchmark and calibration workflows.
- Durable artifacts, online update, telemetry, and reproducible execution.
- A coherent path for multivariate or hierarchical forecasts.
- Documented licensing and production-use checks.
- A clear supported-version and deprecation policy.

Full parity with every R BSTS feature or every foundation model is not a 1.0
requirement.

## Immediate execution sequence

The first implementation sequence should remain small and independently
reviewable.

### PR 1 — Roadmap and documentation alignment

- Add this roadmap.
- Link it from the README and generated documentation.
- Mark the old implementation plan as a historical record.

### PR 2 — Forecast provenance and representation semantics

- Add target/time metadata and explicit draw semantics to `Forecast`.
- Preserve existing constructors and operations.
- Add `require_joint_draws!/1` and clear failures for marginal-only data.

### PR 3 — Provider contract and external baseline

- Add `ForecastProvider`, `ProviderResult`, and capabilities.
- Add a native provider wrapper around `Forecaster`.
- Add known-offset/residual forecasting without any TimesFM dependency.

### PR 4 — Rolling-origin backtest core

- Add fold generation, cutoff-safe callbacks, core metrics, and artifact capture.
- Benchmark seasonal naive and native BSTS first.

### PR 5 — Generic sidecar provider

- Define a versioned JSON or Arrow-based request/response protocol.
- Add timeout, error, provenance, and capability handling.
- Ship a deterministic fake provider for compatibility tests.

### PR 6 — TimesFM research adapter and benchmark

- Implement the adapter outside the base runtime dependency graph.
- Gate model artifacts by declared usage rights.
- Benchmark TimesFM 2.5 where commercial-compatible and TimesFM 3.0 only where
  its then-current terms permit.
- Compare direct, residual-corrected, and calibrated variants.

### PR 7 — ScenarioSet and sports reference case

- Add generic scenario contracts and mixture semantics.
- Demonstrate local-team advancement, game existence, and overrun dependence.
- Feed the result into audience delivery and makegood risk.

## Success measures

The roadmap succeeds when `BstsNx` improves decisions, not merely when it adds
models.

### Statistical

- Better out-of-sample WAPE/MASE without unacceptable bias.
- Calibrated quantiles and intervals by horizon and segment.
- Stable underdelivery probability and expected-shortfall calibration.
- Demonstrated benefit from measurement weighting during footprint changes.

### Decision

- Lower realized or simulated makegood cost.
- Better reserve accuracy.
- Fewer surprise underdeliveries at a fixed inventory utilization level.
- Measurable value from sports/specials scenario treatment.

### Operational

- Predictable p50/p95 latency and memory.
- Efficient batch throughput.
- Deterministic artifact replay.
- Explicit provider failure and fallback rates.

### Developer

- External providers can be added without modifying core inference code.
- Current public APIs have a documented migration path.
- Examples distinguish marginal quantiles, joint draws, and scenarios correctly.
- CI, docs, and provider compatibility tests remain green.

## Principal risks and mitigations

| Risk | Mitigation |
|---|---|
| The project becomes an unfocused forecasting framework | Keep evidence, calibration, causality, and decisions as the product boundary; require each model feature to support a validated use case. |
| TimesFM or another provider dominates raw accuracy | Treat providers as replaceable inputs; compete on measurement semantics, calibration, provenance, and decisions. |
| TimesFM licensing blocks production | Record artifact rights, fail closed for incompatible usage, retain native and alternate providers, and reverify terms at implementation time. |
| Marginal quantiles are mistaken for joint risk | Require explicit joint semantics and reconstruction policies before aggregation or nonlinear operations. |
| Sports scenario counts explode | Use scenario reduction, conditional independence where validated, and probability pruning with recorded approximation error. |
| Native MCMC is too slow for fleets | Separate offline and online lanes; add EM, compact retention, batching, and provider fallback. |
| Hierarchical models become computationally intractable | Start with empirical Bayes and shared factors; avoid one giant dense state vector. |
| Causal claims outrun identification | Preserve explicit controls, leakage checks, placebo tests, stability analysis, and separate predictive from causal APIs. |
| Revised audience history invalidates backtests | Store and require as-of vintages and replayable data fingerprints. |
| Domain schemas leak into the core | Put television event and measurement mappings in application modules built on generic contracts. |
| Backend results drift | Maintain parity tests, tolerance policy, artifact backend metadata, and deterministic seeded fixtures. |

## Strategic decision log

1. **BstsNx complements foundation models; it does not attempt to out-train them.**
2. **The core remains Elixir/Nx and provider-neutral.**
3. **Measurement uncertainty is first-class.**
4. **Joint uncertainty and scenario identity are never implicit.**
5. **TimesFM enters first as a benchmark and external baseline, not a core dependency.**
6. **Sports and specials are modeled through features, analogs, and scenario trees—not program-title magic.**
7. **The television use case validates generic primitives rather than defining the entire package.**
8. **Operational decisions and historical calibration are the primary success criteria.**
9. **Multivariate and hierarchical modeling follows contracts, benchmarking, and fleet foundations rather than preceding them.**
10. **Public release remains gated by the separate release-readiness plan and explicit approval.**

## Related documents

- [Documentation overview](overview.md)
- [Current forecasting and applications guide](forecasting-and-applications.md)
- [Core modeling guide](core-modeling.md)
- [Causal inference and attribution guide](causal-inference-and-attribution.md)
- [Synthetic data and validation guide](synthetic-data-and-validation.md)
- [Historical implementation plan](implementation-plan.md)
- [Release readiness plan](release-readiness-plan.md)
- [TimesFM repository](https://github.com/google-research/timesfm)
