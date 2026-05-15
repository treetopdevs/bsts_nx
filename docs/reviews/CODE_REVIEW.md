# BstsNx Mathematical & Algorithmic Code Review

**Date:** 2026-04-18
**Scope:** Full codebase — mathematical correctness, algorithmic efficiency, numerical stability, code simplification
**Files reviewed:** 18 source files across core engine, model building, high-level APIs, domain applications, and synthetic data generation

---

## Executive Summary

The library implements a solid Bayesian Structural Time Series engine. The core Kalman filter and RTS smoother are mathematically correct. However, the review identified **5 critical**, **25 major**, and **20+ minor** issues spanning:

1. **Mathematical correctness bugs** — relative effect ratio bias, Shapley efficiency axiom violation in synthetic ground truth, cumulative variance inflation in filter-based causal impact
2. **Algorithmic inefficiency** — spike-and-slab variable selection uses O(p·n·k²) full OLS solves instead of rank-1 updates; pure-Elixir dot products in inner loops; triple-duplicated trajectory aggregation
3. **Numerical stability gaps** — absolute (not relative) near-zero thresholds; scale-unaware regularization; float32 draws upcast to float64
4. **Code duplication** — 6 instances of near-identical functions across modules that should be unified

---

## Critical Findings

### C1. Synthetic Ground Truth Violates Shapley Efficiency Axiom
**File:** `generator.ex:264–274`

`coalition_lift/3` filters the adstock effect to on-air indices only, discarding all carryover lift. When `lambda > 0`, post-window carryover is real lift that gets excluded from Shapley allocation but included in `total_lift = Enum.sum(tv_contribution)`. Result: `sum(spot_attributions) != total_lift`, violating the efficiency axiom.

**Fix:** Replace conditional accumulation with `Enum.sum(effect)`.

### C2. Spike-and-Slab Variable Selection is O(p·n·k²) per Gibbs Scan
**File:** `gibbs_sampler.ex:990–1004`

`resample_gamma_g_prior/7` calls `log_marginal_g_prior` twice per variable per scan. Each call reconstructs `X_active`, forms `X'X`, and solves a full `O(n·k² + k³)` system. For p=20 regressors, that's 40 full OLS solves per Gibbs iteration.

**Fix:** Use the closed-form Zellner g-prior marginal likelihood ratio with rank-1 Sherman-Morrison-Woodbury updates. Reduces per-sweep cost from O(p·n·k²) to O(p·n·k) after initial QR decomposition.

### C3. `sample_beta_g_prior` Computes Two Separate Linear Solves
**File:** `gibbs_sampler.ex:1018–1021`

`safe_solve(xtx, eye)` and `safe_solve(xtx, xty)` perform two independent solves on the same matrix. A single Cholesky factorization of `X'X` would serve both.

**Fix:** Compute `L = cholesky(X'X)`, then solve both systems via forward/back substitution.

### C4. Relative Effect Computed as Per-Sample Ratio (Biased Estimator)
**File:** `causal_impact.ex:156–160, 281–285`

Both `estimate/4` and `estimate_structured/5` compute `relative_effect_i = cumulative_i / baseline_sum_i` where the denominator is stochastic. E[X/Y] ≠ E[X]/E[Y] — this is a biased estimator with heavy tails when the baseline approaches zero.

The Brodersen et al. reference implementation defines relative effect as total effect divided by the **posterior mean** baseline.

**Fix:** `relative_effect_i = cumulative_i / mean(baseline_sums)`.

### C5. Smoother `simulate_defn_impl` and `simulate_from_filtered_defn_impl` Are 95% Identical
**File:** `smoother.ex:329–405`

Two `defn` functions share ~95% of their code. The only difference is the boundary condition (smoothed vs filtered arrays). This creates two JIT compilation artifacts and doubles maintenance burden.

**Fix:** Unify into a single `defn` parameterized by the boundary state arrays.

---

## Major Findings

### Numerical Stability

#### M1. Scale-Unaware Regularization in Matrix RTS Smoother
**File:** `smoother.ex:110–160`

Fixed `eps = 1e-6 * I` regularizer is applied unconditionally. For tightly constrained systems (P ~ 1e-6), this doubles effective predicted variance and biases smoother gains downward. Same issue at lines 444–446, 504–506.

**Fix:** Scale regularizer relative to `trace(P_pred) / n` or apply only when condition number exceeds threshold.

#### M2. Absolute Near-Zero Innovation Variance Threshold
**File:** `kalman_filter.ex:362–365`

`near_zero_s = Nx.abs(s) < 1.0e-15` is absolute. For large-magnitude observations (e.g., sales in millions), this threshold is appropriate, but for normalized data it may be too small. Same pattern in smoother at lines 94, 234, 352, 393.

**Fix:** Use relative threshold: `Nx.abs(s) < 1.0e-15 * Nx.max(Nx.abs(s), 1.0)`.

#### M3. Float32 Random Draws Upcast to Float64 in Defn Gamma Sampler
**File:** `distributions.ex:207–209`

`Nx.Random.normal` defaults to `{:f, 32}`, then `Nx.as_type(x_raw, {:f, 64})` upcasts. The normal draw has only ~7 decimal digits of entropy before upcast, introducing slight bias in the Marsaglia-Tsang acceptance test tail region.

**Fix:** Pass `type: {:f, 64}` directly to `Nx.Random.normal` and `Nx.Random.uniform`.

#### M4. h_tensor Type Mismatch in Scalar Sampler
**File:** `gibbs_sampler.ex:301`

`h_tensor = Nx.tensor(h_vals, type: {:f, 32})` while observations and variances are `{:f, 64}`. Causes silent mixed-precision arithmetic and potential EXLA recompilations.

**Fix:** Use `{:f, 64}`.

### Algorithmic Efficiency

#### M5. Pure-Elixir Dot Products in Inner Gibbs Loop
**File:** `gibbs_sampler.ex:947–974`

`adjust_observations_for_regression`, `obs_residuals_spike_slab`, and `regression_residual_pairs` use `dot_list` (Elixir `Enum.zip` + `Enum.reduce`) for inner-loop dot products. For p=50, T=500: 25,000 scalar multiply-adds per Gibbs iteration in pure Elixir.

**Fix:** Vectorize as `Nx.subtract(y_tensor, Nx.dot(x_matrix, Nx.tensor(beta)))`.

#### M6. obs_tensor Reconstructed Every Iteration in Spike-Slab Path
**File:** `gibbs_sampler.ex:1345`

`observations_to_filter_tensor(observations)` is called inside `obs_residuals_structured` on every Gibbs iteration, but `observations` is immutable.

**Fix:** Precompute `obs_tensor` once before the sampling loop and pass it through.

#### M7. h_rows Re-stacked Every Iteration in List-Dispatch Path
**File:** `gibbs_sampler.ex:1361–1366`

The list fallback in `obs_residuals_structured` rebuilds `h_rows_tensor` by mapping, flattening, and stacking on every call. The structured path at line 584–585 correctly precomputes this.

**Fix:** Pass `h_struct_tensor` from `ctx` instead of re-stacking.

#### M8. Pinv Fallback in Simulation Smoother is 10x Slower Than Necessary
**File:** `smoother.ex:407–523`

The `_pinv` variant uses `Nx.LinAlg.pinv` (SVD-based) inside a `while` loop for a matrix that is guaranteed PD by construction. 

**Fix:** Use `Nx.LinAlg.eigh` to compute inverse as `V·Λ⁻¹·V'` — both faster and universally supported.

#### M9. project_to_psd_cholesky Wastes an O(n³) Cholesky After Eigen-Projection
**File:** `smoother.ex:851–874`

After projecting via `V·max(Λ,eps)·V'` (guaranteed PSD), the code calls `safe_cholesky` again.

**Fix:** Compute Cholesky directly as `V·sqrt(D)`.

#### M10. ARForecaster Creates Nx.tensor Per Step in Hot Loop
**File:** `ar_forecaster.ex:303–309`

`Nx.tensor(step_key, type: ...)` is called `horizon × num_samples` times inside the reduce loop.

**Fix:** Pre-draw all normal samples at once with `Nx.Random.normal(key, shape: {num_samples, horizon})`.

#### M11. Structured Forward Predictions Ignore State Covariance
**File:** `anomaly_detector.ex:452–501`

`forward_structured_predictions` uses only `H·Q·H'` for predictive variance, ignoring accumulated `H·P·H'`. The scalar path correctly accumulates `step * q + r`. Multi-step forecasts will severely underestimate uncertainty.

**Fix:** Propagate `P` forward: `P_{t+1|t} = F·P_{t|t}·F' + Q`, then use `H·P·H' + R` for observation variance.

#### M12. regularize_cov_for_cholesky Adds Double eps
**File:** `smoother.ex:820–826`

`Nx.add(eps, shift)` where `shift = max(eps - min_diag, 0)` results in adding `2·eps` when `min_diag < eps`. The intent is "ensure all diagonals ≥ eps", which only requires adding `shift`.

**Fix:** `Nx.add(cov_sym, Nx.multiply(i_eye, shift))` — drop the extra `eps`.

### Statistical Correctness

#### M13. Cumulative Variance Inflated by Observation Noise R
**File:** `causal_impact.ex:617`

`cum_var = diag_var + 2.0 * cross_var_sum + n_intervention * r_num`

The `n_intervention * r_num` term treats observation noise as forward uncertainty, but observations are already realized. The baseline uncertainty is only from state uncertainty `h²·P^s`, not observation noise.

**Fix:** Remove the `+ n_intervention * r_num` term.

#### M14. Structured Causal Impact Omits Cross-Covariance
**File:** `causal_impact.ex:851`

`estimate_structured_from_filter` uses `cum_sd = sqrt(diag_var)` with no cross-covariance and no R, while the scalar path includes both. Systematically underestimates uncertainty for correlated interventions.

**Fix:** Add cross-covariance accumulation (port from scalar path) or document the independence assumption and its expected bias.

#### M15. Relative Effect CIs Inconsistent Across Estimation Methods
**Files:** `causal_impact.ex` (lines 156–160, 623–645, 851)

Three paths compute relative effect CIs differently: (1) per-draw ratios with MCMC summary, (2) delta-method Gaussian approximation, (3) no R included. Downstream callers cannot tell which assumption is in play.

**Fix:** Label the estimation method in the output map and unify the mathematical treatment.

#### M16. time_weighted_value_function Violates Shapley Dummy Axiom
**File:** `shapley_allocator.ex:386–399`

`t_max` is computed per-coalition, so removing the most recent spot changes weights for all remaining spots. This violates the null-player property.

**Fix:** Fix `t_max` to the max time over all players in the group (not the coalition).

#### M17. Variance Allocation for Negative Shapley Values is Statistically Unsound
**File:** `spot_attributor.ex:707–718`

Using `|allocated_lift_i| / Σ|allocated_lift_j|` as variance share has no statistical basis and treats detrimental and beneficial spots as equally uncertain.

**Fix:** Document this as a heuristic or implement proper variance decomposition via the delta method on the Shapley formula.

### Code Duplication

#### M18. derive_predict_prng_opts Duplicated Between Forecaster and ARForecaster
**Files:** `forecaster.ex:347–360`, `ar_forecaster.ex:229–242`

Character-for-character identical. **Fix:** Extract to `ModelBuilder`.

#### M19. Trajectory Aggregation Implemented Three Times
**Files:** `forecaster.ex:395–443` (vectorized Nx), `demand_forecaster.ex:385–421` (pure Elixir), `ar_forecaster.ex:324–335` (double-sorted Elixir)

Three diverging implementations of the same mean/sd/CI computation. The Forecaster version is the most efficient.

**Fix:** Extract shared `aggregate_trajectories/3` to a utility module.

#### M20. forward_simulate_demand Duplicates Forecaster.predict_structured
**Files:** `demand_forecaster.ex:326–370`, `forecaster.ex` structured predict path

Near-identical forward simulation with slower per-step key construction.

**Fix:** Reuse `Forecaster.predict_structured` or extract shared simulation logic.

#### M21. summarize_paths Sorts Each Column Twice
**File:** `ar_forecaster.ex:324–335`

`quantile/2` is called twice per time step, each call sorting the same list. Mean and SD also traverse separately.

**Fix:** Sort once per column, then extract all four statistics.

### Design Issues

#### M22. PRNG Key Reuse Risk Between Sampler and Counterfactual Simulator
**File:** `causal_impact.ex:128–129, 246–247`

When `:seed` is provided without `:key`, both the Gibbs sampler and counterfactual simulator independently call `Nx.Random.key(seed_base)`, producing the same root key.

**Fix:** Split a single root key: `{sampler_key, cf_key} = Nx.Random.split(Nx.Random.key(seed))`.

#### M23. Forecaster seed+1 Trick is Fragile
**File:** `forecaster.ex:347–359`

`seed + 1` provides no actual independence guarantee for consecutive Nx.Random keys.

**Fix:** Always derive from a split: `{fit_key, predict_key} = Nx.Random.split(Nx.Random.key(seed))`.

#### M24. Pipeline Enforces Strict Period Contiguity Not Required by CausalImpact
**File:** `pipeline.ex:153–157`

`post_start != pre_end + 1` raises, but CausalImpact permits gaps (e.g., washout periods).

**Fix:** Relax to match CausalImpact's constraints or document the restriction.

#### M25. InterventionAnalysis.is_significant_filter? Hardcodes alpha=0.05
**File:** `intervention_analysis.ex:408–410`

The filter path always uses alpha=0.05 regardless of the configured alpha.

**Fix:** Propagate the runtime alpha value.

---

## Minor Findings

| # | File | Lines | Issue |
|---|------|-------|-------|
| m1 | `utils.ex` | 338–344 | `percentile_interval` formula uses `n-1` multiplier; slightly narrows intervals |
| m2 | `utils.ex` | 385–399 | `erfinv` precision claim of ~1e-12 is overstated for \|x\| > 0.99 |
| m3 | `utils.ex` | 164–165 | `solve_with_jitter` symmetrizes already-symmetric matrices |
| m4 | `distributions.ex` | 243–244 | Defn gamma fallback after 50K iterations returns deterministic `d`, not a sample |
| m5 | `kalman_filter.ex` | 496–501 | `normalize_h_series` round-trips rank-1 tensor through Elixir scalar list |
| m6 | `kalman_filter.ex` | 171–199 | Joseph form covariance update is ~3x more expensive than standard form for linear systems |
| m7 | `smoother.ex` | 787–809 | Redundant 1x1 special-case in `smoother_gain`; `safe_solve` already handles it |
| m8 | `smoother.ex` | 67–251 | `rts_defn` and `rts_defn_with_lag1` both recompute prediction step already done in filter |
| m9 | `smoother.ex` | 596–597 | Eager `rts/3` does not symmetrize covariance update; compiled version does |
| m10 | `gibbs_sampler.ex` | 302 | `obs_present_mask` via `Nx.equal(t, t)` exploits NaN behavior implicitly; prefer `Nx.is_nan` |
| m11 | `gibbs_sampler.ex` | 395,423 | `validate_q_specs!` called twice for validated specs |
| m12 | `gibbs_sampler.ex` | 1135–1145 | `build_full_covariances` index maps rebuilt per retained sample; precompute into ctx |
| m13 | `gibbs_sampler.ex` | 1414–1427 | `rebuild_q` extracts diagonal unnecessarily; build from scratch since all dims are resampled |
| m14 | `components.ex` | 551–562 | `compose_h` with two time-varying lists does stack→concatenate→to_list round-trip |
| m15 | `components.ex` | 286–310 | `local_linear_trend_spec` shares priors for level/slope; no per-component prior API |
| m16 | `model_builder.ex` | 347–360 | `build_future_h` validates time-invariance with T-1 individual `Nx.all_close` calls |
| m17 | `anomaly_detector.ex` | 596–598 | `:critical` severity threshold `2 * z_threshold` is arbitrary/undocumented |
| m18 | `anomaly_detector.ex` | 362–365 | Redundant `* 1.0` float coercions |
| m19 | `demand_forecaster.ex` | 266–272 | Hardcoded priors differ from ModelBuilder defaults; inconsistent and undocumented |
| m20 | `ar_forecaster.ex` | 198–207 | Ridge penalty applied to intercept column; standard ridge excludes it |
| m21 | `ar_forecaster.ex` | 314–316 | `ar_mean` reverses window on every call |
| m22 | `generator.ex` | 264–268 | `build_on_air_set` called twice per coalition evaluation |
| m23 | `generator.ex` | 148–165 | Fourier indexed lists reallocated on every time step |
| m24 | `forecaster.ex` | 245–268 | `decompose/1` allocates full {n,T,d} tensor for mean only; use online accumulation |
| m25 | `forecaster.ex` | 419–432 | `aggregate_trajectories` uses floor/ceil percentiles; off for small n |
| m26 | Multiple | — | Alpha validation guard duplicated across 4 modules |

---

## Recommended Priority Order

### Phase 1 — Mathematical Correctness (High Impact, Moderate Effort)
1. **C4** — Fix relative effect ratio bias (use posterior mean baseline)
2. **C1** — Fix synthetic Shapley ground truth (remove on-air filter)
3. **M13** — Remove spurious R term from cumulative variance
4. **M11** — Propagate state covariance in structured forward predictions
5. **M16** — Fix Shapley time-weight dummy axiom violation
6. **M25** — Propagate alpha in filter significance test

### Phase 2 — Performance (High Impact)
7. **C2/C3** — Optimize spike-and-slab with rank-1 updates and single Cholesky
8. **M5** — Vectorize inner-loop dot products
9. **M6/M7** — Precompute obs_tensor and h_rows outside sampling loop
10. **M10** — Batch-draw all normal samples in ARForecaster
11. **M8** — Replace pinv fallback with eigh-based inverse

### Phase 3 — Code Simplification (Moderate Impact)
12. **C5** — Unify duplicate simulation smoother defns
13. **M18** — Extract shared derive_predict_prng_opts
14. **M19/M20** — Unify trajectory aggregation and forward simulation
15. **M21** — Single-sort summarize_paths
16. **M12** — Fix regularize_cov_for_cholesky double-eps

### Phase 4 — Numerical Robustness (Lower Priority)
17. **M1** — Scale-aware smoother regularization
18. **M2** — Relative near-zero thresholds
19. **M3** — Float64 random draws in defn sampler
20. **M4** — Fix h_tensor type to f64
21. **M22/M23** — PRNG key splitting hygiene

---

## Architectural Observations

1. **Dual-path maintenance burden**: The eager/defn split across kalman_filter, smoother, and gibbs_sampler means every algorithm change must be implemented twice. The defn versions are scalar-only and lag behind the eager versions in features (no predicted arrays, no cross-covariance). Consider a strategy to converge on one path.

2. **Application modules as thin wrappers**: `DemandForecaster`, `AnomalyDetector`, and `ARForecaster` duplicate core logic from `Forecaster` with minor variations. These could be restructured as configuration presets that delegate to `Forecaster` rather than reimplementing simulation/aggregation.

3. **ModelSpec as the unifying abstraction**: The `ModelSpec` + `Components.compose_specs` design is clean and extensible. Pushing more computation into the spec (e.g., `n_regression_dims`, precomputed H tensors, prior hyperparameters per component) would eliminate several post-hoc inference steps in `Forecaster` and `ModelBuilder`.
