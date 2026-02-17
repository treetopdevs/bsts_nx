# BSTS Elixir — Implementation Plan

## Current State (as of 2026-02-10)

- **501 tests pass** (405 tests + 57 properties + 39 doctests)
- 17 library modules, 35 test files
- All 7 implementation steps complete

### Completed Components

| Component | Module | Notes |
|-----------|--------|-------|
| Kalman Filter (scalar + multidimensional) | `kalman_filter.ex` | defn JIT, missing obs |
| RTS Smoother + Carter-Kohn | `smoother.ex` | List-based + defn |
| Gibbs Sampler | `gibbs_sampler.ex` | Local-level, structured, multi-chain |
| Components (level, trend, seasonal, regression) | `components.ex` | ModelSpec factories |
| Causal Impact (MCMC + filter + structured) | `causal_impact.ex` | Local-level + ModelSpec |
| Shapley (exact + MC + time-decay) | `shapley.ex` | Overlap detection, time-weighted VF |
| SpotAttributor (point + posterior) | `spot_attributor.ex` | Per-spot lift + Shapley + posterior propagation |
| Pipeline | `pipeline.ex` | CausalImpact → SpotAttributor (posterior) |
| MCMC Diagnostics | `diagnostics.ex` | R-hat, ESS, split R-hat |
| Validation Suite + Calibration | `validation.ex` | 6 checks + assess + known_lift_injection |
| Distributions | `distributions.ex` | inv-gamma, MVN |
| Covariate Selection | `covariate_selection.ex` | Pearson correlation-based selection |

## Implementation Steps (all complete)

### Step 1: Seasonal Component ✓
Added `Components.seasonal/2` with period S using S-1 state dimensions.
Standard dummy-variable seasonal: F rotates seasonal effects, H observes current
season, Q adds innovation noise to the first seasonal state.
Also added `Components.seasonal_spec/2` for ModelSpec integration.

### Step 2: CausalImpact with Covariates ✓
Extended `CausalImpact` to accept a `ModelSpec` (or build from components).
Uses `GibbsSampler.sample_structured/4` instead of `sample/7`.
Supports trend + seasonality + regression covariates. Fixed H slicing for
time-varying observations, diagonal Q sampling for singular process covariance.

### Step 3: End-to-End Pipeline ✓
Created `BstsNx.Pipeline` chaining CausalImpact → SpotAttributor.
Takes raw observations + spot definitions, returns attributed lift per spot.
Bridges posterior draws to SpotAttributor's counterfactual format.

### Step 4: Posterior Propagation through Shapley ✓
Added `SpotAttributor.attribute_posterior/5` that accepts multiple posterior
counterfactual draws and propagates uncertainty through Shapley allocation.
Produces per-spot credible intervals from per-draw lift distributions.
Pipeline updated to use posterior propagation by default.

### Step 5: Time-Decay Value Function ✓
Added `ShapleyAllocator.time_weighted_value_function/3` for recency-weighted
coalition values using exponential half-life decay: `weight = exp(-λ(t_max - t_i))`
where `λ = ln(2) / half_life`.

### Step 6: Calibration Testing (Known Lift Injection) ✓
Added `Validation.known_lift_injection/3` for synthetic effect recovery testing.
Injects known effects, runs CausalImpact estimator, checks if credible interval
covers the true value. Returns coverage, relative error, and full CI statistics.

### Step 7: Covariate Selection ✓
Added `BstsNx.CovariateSelection` with `select/3` for pre-period correlation-based
control selection. Uses Pearson correlation with configurable threshold and
max_controls. Also exposes `pearson_correlation/2` as a public utility.

## Next Roadmap (post-Step-7)

The core structured stack is in place. Next work is mainly about broadening
model expressiveness and closing remaining scalar assumptions.

### Phase 8: Documentation and API Contract Sync

Goal: ensure docs reflect current structured capabilities and known boundaries.

- Update README and guides to clearly separate:
  - already-supported structured composition and higher-dimensional latent state spaces,
  - remaining limits (scalar-observation structured MCMC, diagonal `Q`, scalar `R`).
- Add examples that use `estimate_structured/5` and composed specs.

Estimated effort (1 engineer): 1-2 days.

### Phase 9: Structured Multivariate-Observation MCMC

Goal: support vector observations in structured Gibbs workflows.

- Extend structured residual and likelihood code paths to handle vector `y_t`.
- Add observation covariance modeling options (start with diagonal `R`, then full `R`).
- Generalize `CausalImpact.estimate_structured/5` and `Forecaster` structured paths.
- Expand tests for multivariate synthetic cases and edge conditions.

Estimated effort (1 engineer): 2-4 weeks.

### Phase 10: Richer Component Families

Goal: make model composition expressive for real production use cases.

- Add additional component factories/specs (for example AR, multiple seasonalities,
  and holiday/event effects).
- Add stronger prior controls per component.
- Extend `ModelBuilder` composition helpers for common patterns.

Estimated effort (1 engineer): 2-3 weeks.

### Phase 11: Correlated State Innovations

Goal: move beyond diagonal process-noise learning.

- Add parameterization and sampling for correlated state innovations
  (block-diagonal or full covariance approaches).
- Update structured sampler internals and diagnostics for new parameter families.

Estimated effort (1 engineer): 3-5 weeks.

### Phase 12: Performance and Defn Parity

Goal: improve throughput and reduce gap between list-based and compiled paths.

- Add compiled/optimized paths for broader structured workflows.
- Benchmark representative model sizes and tune memory/runtime hotspots.

Estimated effort (1 engineer): 2-4 weeks.
