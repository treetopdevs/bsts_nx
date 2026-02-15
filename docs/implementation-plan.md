# BSTS Elixir — Implementation Plan

## Current State (as of 2026-02-10)

- **501 tests pass** (405 tests + 57 properties + 39 doctests)
- 17 library modules, 35 test files
- All 7 implementation steps complete

### Completed Components

| Component | Module | Notes |
|-----------|--------|-------|
| Kalman Filter (scalar + multidim) | `kalman_filter.ex` | defn JIT, missing obs |
| RTS Smoother + Carter-Kohn | `smoother.ex` | List-based + defn |
| Gibbs Sampler | `gibbs_sampler.ex` | Local level, structured, multi-chain |
| Components (level, trend, seasonal, regression) | `components.ex` | ModelSpec factories |
| CausalImpact (MCMC + filter + structured) | `causal_impact.ex` | Local-level + ModelSpec |
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
