# BstsNx Comprehensive Code Review

**Date**: 2026-03-02
**Branch**: `main` (commit `1b32b1c`)
**Scope**: Full codebase — 29 source files (12,315 LOC), 64 test files (22,985 LOC)
**Reviewers**: Architecture, Code Quality, Security/Robustness, Test Engineering (parallel analysis)

---

## Executive Summary

| Dimension | Grade | Notes |
|-----------|-------|-------|
| **Logic / Correctness** | B+ | Strong numerical safety; 2 critical bugs, 1 statistical formula error |
| **Architecture** | A- | Clean DAG layers, no circular deps; one god module (GibbsSampler) |
| **Security** | A | No vulnerabilities; appropriate for a pure numerical library |
| **Robustness** | B+ | Excellent multi-tier fallbacks; some edge case gaps |
| **Maintainability** | B | Good docs; code duplication across 4-5 modules drags score |
| **Test Suite** | B+ | 1,122 tests + 114 property tests; some public functions untested |
| **Performance** | B | O(n) list access in MCMC hot loops; tensor↔list round-trips |
| **DDD / Phoenix / Ash / UI** | N/A | Pure numerical library — these domains don't apply |

**Health Score: 7.5/10**

**Top 3 Risks:**
1. Concurrency hazard: `Process.put` in Smoother (log dedup state leaks across parallel chains)
2. InverseGamma prior mode miscalculated in filter fast-path, systematically widening CIs by 2x
3. `compat_dot` duplicated in 4 modules — divergent bug fixes guaranteed

---

## Table of Contents

1. [Critical Issues](#1-critical-issues)
2. [Major Issues](#2-major-issues)
3. [Medium Issues](#3-medium-issues)
4. [Architecture Assessment](#4-architecture-assessment)
5. [Security & Robustness](#5-security--robustness)
6. [Numerical Safety](#6-numerical-safety)
7. [Performance](#7-performance)
8. [Test Suite Analysis](#8-test-suite-analysis)
9. [Code Duplication & Tech Debt](#9-code-duplication--tech-debt)
10. [Low Priority / Style](#10-low-priority--style)
11. [Positive Highlights](#11-positive-highlights)
12. [Prioritized Action Plan](#12-prioritized-action-plan)
13. [Open Questions](#13-open-questions)
14. [Appendix: Module Inventory](#14-appendix-module-inventory)

---

## 1. Critical Issues

### C1. InverseGamma Prior Mode Formula Is Wrong

**File**: `lib/bsts_nx/causal_impact.ex:717, 727`

The mode of InvGamma(α, β) is `β/(α+1)`, but the code computes `β/α`:

```elixir
# Line 717 — initializes Q diagonal for filter-based estimator
q_spec.prior_scale / q_spec.prior_shape       # WRONG: β/α
# Should be:
q_spec.prior_scale / (q_spec.prior_shape + 1)  # CORRECT: β/(α+1)
```

With default weakly informative priors (shape=1, scale=1), the mode should be 0.5 but the code computes 1.0 — a **2x overestimate**. This biases initial Q diagonal values in `estimate_structured_from_filter`, affecting all `:filter` method results with systematically wider credible intervals than the MCMC path.

**Fix**: Change both lines to use `/ (shape + 1.0)`, or use `q_spec.initial` which is already stored in the spec.

### C2. `Nx.to_number()` Called on Plain Float — Crash Path

**File**: `lib/bsts_nx/gibbs_sampler.ex:1312`

```elixir
Enum.at(ss_vec, qs.dim_index, 0.0) |> Nx.to_number()
```

If `dim_index` exceeds `ss_vec` length, `Enum.at` returns the default `0.0` (a plain float). `Nx.to_number(0.0)` raises because `0.0` is not an Nx tensor.

**Fix**: Guard the conversion:
```elixir
ss_val = Enum.at(ss_vec, qs.dim_index, 0.0)
ss_val = if is_number(ss_val), do: ss_val, else: Nx.to_number(ss_val)
```

---

## 2. Major Issues

### M1. Process Dictionary Mutation Not Safe for Concurrency

**File**: `lib/bsts_nx/smoother.ex:614, 710`

`Process.put(:bsts_smoother_non_pd_logged, true)` is used for log deduplication. While `Task.async_stream` runs each task in its own process (so no data race), the flag is never cleaned up if the function raises mid-execution, leaking state. More importantly, this demonstrates reliance on implicit global state in what should be a pure functional library.

**Fix**: Replace with a counter threaded through the reduce accumulator, or remove the log guard entirely since `safe_cholesky_or_zero` handles non-PD cases silently.

### M2. `compat_dot/2` Duplicated in 4 Modules

**Files**:
- `lib/bsts_nx/utils.ex:262`
- `lib/bsts_nx/smoother.ex:746`
- `lib/bsts_nx/gibbs_sampler.ex:1233`
- `lib/bsts_nx/forecaster.ex:471`

All four implementations are identical (rank dispatch for `Nx.dot`). Bug fixes must be applied to all four copies.

**Fix**: Make `Utils.compat_dot/2` public. Import/call from the other three modules.

### M3. Duplicated Gamma Sampler in CovariateSelection

**File**: `lib/bsts_nx/covariate_selection.ex:575-621`

~50 lines re-implementing the Marsaglia-Tsang gamma sampler that already exists in `Distributions`. The CovariateSelection version uses Erlang `:rand` state threading while Distributions uses Nx keys, but the core algorithm is identical.

**Fix**: Create a shared pure-Elixir gamma sampling function with a `:rand`-state adapter in Distributions.

### M4. Unnecessary Restriction on Pre/Post Period Gap

**File**: `lib/bsts_nx/causal_impact.ex:83-84`

`estimate/4` and `estimate_structured/5` enforce `post_start == pre_end + 1`, rejecting gaps between periods. Washout periods between pre and post are legitimate in causal inference. `InterventionAnalysis.analyze_filter` correctly handles gaps, but the MCMC path rejects them.

**Fix**: Remove or downgrade the `post_start != pre_end + 1` check. The sampler only uses pre-period data for fitting.

### M5. Independence Assumption in Structured Filter Cumulative Variance

**File**: `lib/bsts_nx/causal_impact.ex:818`

`estimate_structured_from_filter/4` drops cross-covariance in cumulative variance computation (documented only in a code comment: "we skip the full backward RTS lag-covariance pass for speed"). The scalar path `estimate_from_filter/3` (lines 567-620) correctly computes it. This produces **underestimated uncertainty** in the structured filter path with no visible flag to the user.

**Fix**: Either implement cross-covariance correction for the structured path, or add `:cross_cov_included => false` to the return map and document prominently.

### M6. `safe_to_number` Silently Coerces NaN/Inf

**File**: `lib/bsts_nx/gibbs_sampler.ex:1361-1373`

`:nan` → `0.0`, `:inf` → `1e10`, `:neg_inf` → `-1e10` — these coercions mask upstream numerical failures. Setting NaN shape to `0.0` for inverse-gamma produces `alpha=0`, causing the gamma sampler to raise. No warnings are logged for the specific atom clauses (only the catch-all at line 1367 logs).

**Fix**: Add `Logger.warning` to each atom clause. Consider propagating the error instead of silently converting.

### M7. File/Module Naming Mismatch

**File**: `lib/bsts_nx/shapley.ex` defines `defmodule BstsNx.ShapleyAllocator`

Elixir convention: file should be `shapley_allocator.ex` or module should be `BstsNx.Shapley`.

**Fix**: Rename file to `shapley_allocator.ex` or rename module to `BstsNx.Shapley`.

---

## 3. Medium Issues

| ID | File:Line | Issue | Fix |
|----|-----------|-------|-----|
| m1 | `gibbs_sampler.ex:1157,1162,1176,1183,1219,1247` | Nested `Enum.at` on lists in hot MCMC loops — 9 O(n) accesses per iteration | Convert to `:array`/tuple for O(1) |
| m2 | `gibbs_sampler.ex:166,511` | `Enum.at(seeds, idx)` is O(n) per chain in `sample_chains` | Use `List.to_tuple` + `elem/2` |
| m3 | `gibbs_sampler.ex:1262` | Unused variable `_ = rows` (debug artifact) | Remove |
| m4 | `gibbs_sampler.ex:302` | `num_diffs = max(t - 1, 0)` counts all state transitions including between consecutive missing obs | Count only valid transitions |
| m5 | `causal_impact.ex:711-712` | Duplicate comment line | Remove duplicate |
| m6 | `causal_impact.ex:331-332` | Nearest-rank percentile approximation; off by several % at small sample sizes (m=10) | Use linear interpolation |
| m7 | `causal_impact.ex:798` | Bare `Nx.squeeze()` removes all size-1 dims — fragile with unexpected shapes | Use `Nx.squeeze(axes: [...])` explicitly |
| m8 | `causal_impact.ex:945` | `import Nx.Defn` mid-file (unconventional) | Extract defn functions to `CausalImpact.Compiled` |
| m9 | `bct/ar_forecaster.ex:179-183` | `Enum.at(observations, idx - lag)` is O(n) per element in `build_lagged_dataset` — O(n²·p) total | Convert observations to tuple |
| m10 | `model_spec.ex` (entire) | No runtime validation of struct fields at construction time | Add `validate!/1` checking tensor shapes/types |
| m11 | `intervention_analysis.ex:127-196` | No explicit bounds checking of post_period indices against `length(observations)` | Add bounds validation for clearer error messages |
| m12 | `applications/anomaly_detector.ex:437` | `length(sds)` called twice | Cache once |
| m13 | `distributions.ex:43-47` | Duplicate `:max_value` documentation paragraph | Remove duplicate |
| m14 | `smoother.ex:344-346` | Scalar simulation conditional variance `p_filt * (1.0 - j * f_in)` can go negative when gain is large | Use matrix-form formula or document bias from clamping |
| m15 | `smoother.ex:136,437` | Smoother gain: solve vs pinv inconsistency between RTS and simulation | Use consistent approach (solve with regularization) |

---

## 4. Architecture Assessment

### Layer Diagram

```
Layer 4 — Applications:  TVAttribution, MarketingLift, DemandForecaster, AnomalyDetector, PolicyEvaluator
                         ↓
Layer 3 — High-Level:    InterventionAnalysis, Forecaster, Pipeline, BCT.ARForecaster
                         ↓
Layer 2 — Model/Attrib:  ModelBuilder, Components, SpotAttributor, ShapleyAllocator, RollingBaseline, CovariateSelection
                         ↓
Layer 1 — Core:          CausalImpact, GibbsSampler, KalmanFilter, Smoother, Distributions, Diagnostics, Validation
                         ↓
Layer 0 — Foundation:    Utils, ModelSpec, StateSpace
```

**Strengths:**
- **No circular dependencies** — clean DAG verified
- Layer boundaries mostly respected
- `ModelSpec` composition pattern is elegant
- Pure functional — no GenServers, no OTP supervision trees

**Minor layer skip**: `DemandForecaster` (L4) directly uses `GibbsSampler` (L1) — justified for custom forward simulation with regressor handling.

### Complexity Hotspots

| File | LOC | Pub Fns | Priv Fns | Action |
|------|-----|---------|----------|--------|
| `gibbs_sampler.ex` | 1,606 | 6 | 66 | **Decompose** into Scalar/Structured/Regression sub-modules |
| `causal_impact.ex` | 1,061 | 5 | 15 | Extract defn functions to submodule |
| `smoother.ex` | 806 | 12 | 5 | OK — well-structured despite size |
| `spot_attributor.ex` | 740 | 12 | 10 | OK |
| `rolling_baseline.ex` | 668 | 6 | 34 | High private count — review for extraction |
| `components.ex` | 723 | 15 | 15 | OK |
| `covariate_selection.ex` | 623 | 4 | 20 | Contains duplicated gamma sampler |

### Other Architecture Issues

- **ModelBuilder accumulating unrelated utilities**: `format_num/1`, `format_pct/1`, `safe_number/1`, `coerce_obs/1`, `ensure_tensor/1` belong in `Utils`
- **`Utils` lacks `@moduledoc`**: Foundational module with 16 public functions hidden from docs
- **Inconsistent test directory structure**: 46 files at root `test/`, 18 under `test/bsts_nx/`
- **BCT.ARForecaster is incomplete scaffold**: Self-describes as "Phase A scaffold" — should be marked experimental

---

## 5. Security & Robustness

### Security (Grade: A)

| Area | Status | Details |
|------|--------|---------|
| Unsafe deserialization | Clean | No `binary_to_term`, `Code.eval_string`, dynamic `File.read` |
| Secrets in source | Clean | Zero hardcoded API keys, tokens, or credentials |
| Dependencies | Current | All deps from hex.pm with checksums; no known CVEs |
| Input validation | Good | Every public function validates inputs with clear error messages |
| Resource exhaustion | Good | Gibbs capped at 1M iterations; Shapley exact limited to 12 players |
| Parallel execution | Good | `Task.async_stream` with timeout + `on_timeout: :kill_task` |
| Atom creation | Clean | No atoms from user input |
| Network I/O | Clean | No HTTP/DB/socket operations |

**Minor concerns:**
- Non-deterministic default seeds from `System.os_time()` (8+ locations) — silently non-reproducible
- Broad `rescue _ ->` clauses in `utils.ex` (5), `smoother.ex` (4), `ar_forecaster.ex` (1) — could mask unexpected errors
- Logger exposes chain failure reasons via `inspect(reason)` (`gibbs_sampler.ex:212,548`)

**License note**: LGPL-2.1-only is more restrictive than typical Elixir library licenses (Apache-2.0/MIT). May limit adoption.

### Robustness

| Scenario | Status | Notes |
|----------|--------|-------|
| Empty inputs | Good | Validated at most public entry points |
| Single observation | Good | Returns prior-dominated results (statistically correct) |
| All missing data | Partial | Posterior collapses to prior (correct) but no validation that ≥1 non-missing exists |
| Tensor shape mismatches | Good | H tensor validated in structured sampler; regressor dims validated in DemandForecaster |
| Memory usage | Acceptable | Full posterior accumulated in memory; `thin` + `max_iters` bound it |
| Crash resilience | Excellent | Multi-tier fallbacks: solve → jitter → SVD pseudoinverse → zero matrix |
| Chain failures | Good | Partial results from surviving chains; raises only if ALL chains fail |
| Timeout handling | Good | Configurable timeout with `kill_task` cleanup |

---

## 6. Numerical Safety

### Division by Zero — EXCELLENT

Comprehensive near-zero denominator guards throughout:
- Kalman gain `s` checked against `1.0e-15` (`kalman_filter.ex:151,358,406`)
- Smoother predicted covariance checked against `1.0e-15`
- CausalImpact baseline mean checked against `1.0e-10` (`causal_impact.ex:160,287,629`)
- CovariateSelection precision/variance floored at `1.0e-12`
- Validation MAPE denominator floored at `1.0`
- PolicyEvaluator slope computations guarded against `ss_xx > 1.0e-10`

### Matrix Singularity — EXCELLENT

Multi-level fallback strategy:
- **`safe_cholesky/1`** (`utils.ex`): progressive jitter (1e-6 → 1e-5 → 1e-4 → 1e-3)
- **`safe_solve/2`** (`utils.ex`): direct → jitter → SVD pseudoinverse → zero matrix
- **`safe_cholesky_or_zero`** (`smoother.ex:758-805`): eigenvalue projection fallback for non-PSD matrices from numerical drift

### Overflow/Underflow — GOOD

- Logistic function clamped at ±35.0 (`gibbs_sampler.ex:1274-1276`)
- Log-domain computation for marginal likelihood (`gibbs_sampler.ex:1050-1067`)
- Gamma values floored at `1e-300` (`distributions.ex:368`)
- Box-Muller `log(0)` prevention (`synthetic/generator.ex:304`)
- Hill saturation numerically stable reformulation (`synthetic/adstock.ex:91-94`)

### NaN/Inf Handling — GOOD (with M6 caveat)

- `has_non_finite?/1` checks in Utils
- Beta coefficient fallback to posterior mean on NaN (`gibbs_sampler.ex:1027`)
- `normalize_number/1` replaces NaN/Inf with 0.0 (`model_builder.ex:476-478`)
- Joseph form covariance update for numerical stability (`kalman_filter.ex:172`)
- Covariance symmetrization before Cholesky (`smoother.ex`, `gibbs_sampler.ex:1272`)

### Sqrt Safety — GOOD

Consistent `max(x, 0.0)` or `max(x, 1e-12)` before all square root operations across smoother, forecaster, applications, and synthetic modules (verified at 12+ locations).

---

## 7. Performance

| Priority | Bottleneck | Location | Impact | Fix |
|----------|-----------|----------|--------|-----|
| HIGH | Tensor↔list round-trips in Gibbs inner loop | `gibbs_sampler.ex:604-614` | Creates/destroys thousands of tensors per MCMC iteration | Keep `sampled_states_tensor` as `{t,n}` tensor |
| HIGH | `rebuild_q` uses `Nx.to_flat_list` + `List.replace_at` | `gibbs_sampler.ex:1376-1385` | List round-trip every iteration | Use `Nx.indexed_put` directly |
| MEDIUM | 9× `Enum.at` on lists in hot loops | `gibbs_sampler.ex:1157-1247` | O(n) per access → O(n²) for long series | Convert to `:array`/tuple |
| MEDIUM | `h_struct_tensor` re-computed every spike-slab iteration | `gibbs_sampler.ex:690-691` | Redundant `Nx.stack` of static data | Pre-compute before loop |
| MEDIUM | `normalize_h_series` materializes full tensor to list | `kalman_filter.ex:488-498` | Creates T separate tensor allocations | Use `Nx.to_batched(1)` or slice on demand |
| MEDIUM | Cholesky retry uses exception-based flow control | `smoother.ex:758-805` | `try/rescue` in backward pass of every iteration | Pre-check for NaN or use regularization |
| MEDIUM | PRNG key splitting via `Nx.to_list()` round-trip | `distributions.ex:265-269` | Called hundreds of times per chain | Use `Nx.slice_along_axis` |
| MEDIUM | O(n²·p) in AR dataset build | `bct/ar_forecaster.ex:179-183` | `Enum.at` is O(n) per element | Convert to tuple |
| LOW | Mixed f32/f64 between filter and smoother | `kalman_filter.ex:321`, `smoother.ex:57-61` | Type conversion doubles memory | Propagate input type |
| LOW | Box-Muller wastes half draws | `synthetic/generator.ex:301-306` | Double uniform draws needed | Cache second draw |
| LOW | `length/1` called repeatedly on same list | Multiple modules | O(n) traversal each time | Cache in variable |

---

## 8. Test Suite Analysis

### Overview

| Metric | Value |
|--------|-------|
| Test files | 64 |
| Test LOC | 22,985 |
| Named test cases | 1,122 |
| Property-based tests | 114 |
| Test:Source LOC ratio | 1.87:1 |
| Files with `async: true` | 45/64 (70%) |
| Files tagged `:external` | 11 (require Python) |

### Untested Public Functions

| Function | File | Notes |
|----------|------|-------|
| `Smoother.simulate_from_filtered_defn/5` | `smoother.ex:278` | Zero direct test invocations |
| `Smoother.simulate_from_filtered_defn_matrix/5` | `smoother.ex:297` | Zero direct test invocations |
| `Distributions.mv_normal_sample_with_chol/4` | `distributions.ex:297` | Only `mv_normal_sample/3` tested |
| `CausalImpact.estimate_structured_from_filter/4` | `causal_impact.ex:680` | Zero test invocations |
| `Components.seasonal/2` error path | `components.ex:159` | `num_seasons < 2` clause untested |

### Thin Coverage

| Module | Issue |
|--------|-------|
| `Utils` | Only `safe_solve` tested directly; missing tests for `percentile_interval`, `z_score`, `erfinv`, `has_non_finite?`, `derive_exsss_seed` |
| `StateSpace` | No dedicated test file (exercised transitively) |
| `ModelSpec` | No dedicated test file (struct-only, exercised through integration) |
| `GibbsSampler.sample_chains/8` | Minimal parallel chain testing; no chain independence verification |
| `RollingBaseline` | 3 of 4 test files tagged `:external`; limited non-Python coverage |

### Missing Edge Case Tests

- Minimum valid pre-period length (2 observations)
- Very long time series (N > 5,000) — numerical stability at scale
- Large state dimensions (5+ composed specs)
- All-NaN input to `filter_defn`
- Negative-valued time series
- Concurrency tests for `sample_chains` (chain independence, thread safety)

### Weak Assertions

**`intervention_analysis_test.exs`** uses `is_map(result.summary)` without field validation (lines 79, 118, 166, 243) and `result.model_spec != nil` without structural checks (lines 80, 165, 189, 212, 241).

**Fix**: Replace with specific field assertions like `assert is_number(result.summary.cumulative_effect.mean)`.

### Files Missing `async: true`

These 9 pure-computation test files have no shared state and should run concurrently:
- `edge_cases_test.exs`, `missing_data_test.exs`, `kalman_gibbs_test.exs`
- `smoother_key_test.exs`, `causal_impact_summary_test.exs`, `distributions_inv_gamma_test.exs`
- `utils_safe_solve_test.exs`, `kalman_filter_length_test.exs`, `gibbs_sampler_missing_observations_test.exs`

### Property Testing Gaps

Current coverage: KalmanFilter, Smoother, StateSpace, Components, CausalImpact, Diagnostics, Distributions, Utils, Shapley, SpotAttributor.

**Missing property tests for**: GibbsSampler (posterior positivity, seed determinism), Forecaster (CI widening with horizon), InterventionAnalysis (result struct well-formedness), CovariateSelection (max_controls constraint).

### Test Organization Issues

- Mixed directory structure: 46 files at root `test/`, 18 in subdirectories
- Multiple files per module (5 smoother, 4 causal impact, 3 gibbs sampler) — consider consolidating into subdirectories
- No shared test helper module — data generation patterns duplicated across 20+ files
- Small MCMC samples (`num_samples: 10, burn_in: 5`) in 15+ tests — fine for structure, but convergence only verified in `:external` parameter recovery test
- Non-external regression tests with golden values would reduce Python dependency

---

## 9. Code Duplication & Tech Debt

| ID | Severity | What | Where | Fix |
|----|----------|------|-------|-----|
| D1 | **High** | `compat_dot/2` — identical in 4 modules | `utils.ex:262`, `smoother.ex:746`, `gibbs_sampler.ex:1233`, `forecaster.ex:471` | Make public in `Utils`, import everywhere |
| D2 | **High** | `gamma_sample`/`inv_gamma_sample` — full reimplementation | `distributions.ex` vs `covariate_selection.ex:575-622` | Factor `:rand`-threaded variant into `Distributions` |
| D3 | Medium | Input validation patterns repeated verbatim | 5+ modules | Extract shared validation helpers |
| D4 | Medium | `h_to_row_tensor/1` identical in 2 modules | `forecaster.ex:466`, `anomaly_detector.ex:424` | Move to `Utils` |
| D5 | Medium | `take_time_slice_at/3` identical in 2 modules | `causal_impact.ex:1057`, `smoother.ex:714` | Move to `Utils` |
| D6 | Medium | Test data generation patterns | 20+ test files | Create shared `TestHelpers` module |
| D7 | Low | `format_num`/`format_pct` delegation wrappers | `intervention_analysis.ex`, `policy_evaluator.ex` | Use `import` |
| D8 | Low | Application `build_analysis_opts` patterns | 5 app modules | Consider shared behavior |

### API Consistency Issues

| Issue | Details |
|-------|---------|
| Error handling inconsistency | Exceptions (input validation) vs `:nan` atoms (Diagnostics) vs silent fallback (Cholesky) — three patterns |
| Default `burn_in` | `sample_general/5` defaults to 0; higher-level modules default to `div(num_samples, 2)` |
| Scattered defaults | `num_samples`, `alpha`, `prior_shape` defined in multiple places with no central source |
| Return types | `InterventionAnalysis.analyze/3` returns bare map, not struct — no compile-time key checking |

---

## 10. Low Priority / Style

- `intervention_analysis.ex:136`: `if not is_number(alpha)` — prefer `unless` or `!` for Elixir idiom
- `diagnostics.ex:324`: `Logger.warning` in `geweke_z` — consider returning error tuple
- `policy_evaluator.ex:211`: unused `_opts` parameter in `pre_trend_check/3`
- `causal_impact.ex:997`: variable `key_out_temp` — misleading name, rename to `key_out`
- `spot_attributor.ex:343`: cross-covariance limitation documented only in code comment — move to `@doc`
- `shapley.ex`: exact→MC transition at 12 players not documented in public API docs
- `components.ex`: `compose_specs/2` could use a docstring example with 3+ specs
- `covariate_selection.ex`: also duplicates `standardize/1`, `variance/1`, `dot/2` helpers from Utils
- Some `@doc false` public functions could be `defp`: `synthetic/generator.ex:132,182,313`, `model_builder.ex:372`
- CLAUDE.md says `nx ~> 0.6` but mix.exs has `~> 0.11.0` — documentation mismatch

---

## 11. Positive Highlights

- **Joseph form covariance update** in Kalman filter (`kalman_filter.ex:172-199`) — numerically superior to naive `P - K*S*K'`
- **Multi-tier Cholesky/solve fallback** (`utils.ex:26-74`, `smoother.ex:758-805`) — progressive jitter + eigenvalue projection
- **NaN-as-missing sentinel** in defn code via `Nx.equal(z, z)` (`kalman_filter.ex:366,426`) — elegant
- **PRNG key threading** — proper splitting and state management throughout; `derive_exsss_seed` correctly bridges Nx↔Erlang
- **1,122 tests + 114 property tests** at 1.87:1 test:source ratio with cross-language validation (Python, R)
- **Clean dependency graph** — strict DAG, no circular dependencies, clear layer boundaries
- **Pure functional design** — no GenServers, no OTP, no global state (except the Process dictionary issue)
- **Comprehensive input validation** — every public function validates with clear error messages
- **Good documentation** — `@moduledoc` on 97% of modules, typespecs on public functions
- **Iteration safety caps** — Gibbs sampler at 1M iterations, Shapley exact at 12 players
- **Robust parallel chains** — configurable timeouts, kill-task on timeout, graceful partial results
- **Marsaglia-Tsang gamma sampler** correctly handles α < 1 adjustment

---

## 12. Prioritized Action Plan

### P0 — Fix Now (Correctness Bugs)

| # | Action | Files |
|---|--------|-------|
| 1 | Fix inverse-gamma mode formula: `β/(α+1)` not `β/α` | `causal_impact.ex:717,727` |
| 2 | Fix `safe_to_number` + `Enum.at` crash path (guard against plain float) | `gibbs_sampler.ex:1312` |
| 3 | Add bounds check for `q_specs` dim_index against state dimension | `gibbs_sampler.ex:1312` |

### P1 — Fix Soon (Major Quality)

| # | Action | Files |
|---|--------|-------|
| 4 | Extract `compat_dot/2` to `Utils` (public) — eliminate 4-way duplication | 4 files |
| 5 | Unify gamma sampler: `CovariateSelection` delegates to `Distributions` | `covariate_selection.ex` |
| 6 | Remove `Process.put/get/delete` in Smoother — thread state explicitly | `smoother.ex:614,710` |
| 7 | Rename `shapley.ex` → `shapley_allocator.ex` (or rename module) | file rename |
| 8 | Document/fix structured filter independence assumption | `causal_impact.ex:818` |
| 9 | Add logging to `safe_to_number` atom clauses (NaN/Inf) | `gibbs_sampler.ex:1361-1373` |
| 10 | Remove pre/post gap restriction or make it a warning | `causal_impact.ex:83-84` |

### P2 — Improve (Maintainability & Performance)

| # | Action | Files |
|---|--------|-------|
| 11 | Keep sampled states as tensor through Gibbs loop (avoid list round-trip) | `gibbs_sampler.ex` |
| 12 | Replace `rebuild_q` list manipulation with `Nx.indexed_put` | `gibbs_sampler.ex:1376-1385` |
| 13 | Replace `Enum.at` with tuple/`:array` in MCMC hot loops | `gibbs_sampler.ex` (9 locations) |
| 14 | Add `ModelSpec.validate!/1` for early error detection | `model_spec.ex` |
| 15 | Add `async: true` to 9 pure-computation test files | test files |
| 16 | Create shared test helper module | `test/support/` |
| 17 | Test untested public functions (4 identified) | new test files |
| 18 | Pre-compute `h_struct_tensor` outside spike-slab loop | `gibbs_sampler.ex:690-691` |
| 19 | Use consistent solve vs pinv approach in smoother | `smoother.ex:136,437` |
| 20 | Add explicit `Nx.squeeze(axes: ...)` | `causal_impact.ex:798` |

### P3 — Nice to Have

| # | Action | Files |
|---|--------|-------|
| 21 | Decompose GibbsSampler into sub-modules | `gibbs_sampler.ex` |
| 22 | Add property tests for GibbsSampler, Forecaster, InterventionAnalysis | test files |
| 23 | Add edge case tests (min pre-period, all-NaN, long series, large state) | test files |
| 24 | Extract defn functions to `CausalImpact.Compiled` submodule | `causal_impact.ex` |
| 25 | Consolidate test directory structure | `test/` → `test/bsts_nx/` |
| 26 | Add direct unit tests for Utils functions | new test file |
| 27 | Add non-external regression tests with golden values | test files |
| 28 | Mark internal-facing ModelBuilder functions `@doc false` or private | `model_builder.ex` |
| 29 | Centralize defaults into `BstsNx.Defaults` module | new module |
| 30 | Unify f32/f64 handling with type propagation | `kalman_filter.ex`, `smoother.ex` |
| 31 | Clean up untracked files (commit or gitignore) | `scripts/`, root test files |
| 32 | Review LGPL-2.1-only license choice | `mix.exs` |
| 33 | Update CLAUDE.md: `nx ~> 0.11` not `~> 0.6`, module name corrections | `CLAUDE.md` |

---

## 13. Open Questions

1. **Is LGPL-2.1-only intentional?** More restrictive than typical Elixir libraries (Apache-2.0/MIT). May limit adoption.
2. **What is the BCT.ARForecaster roadmap?** Currently "Phase A scaffold" — complete or mark experimental?
3. **Should `Utils` be public?** Currently `@moduledoc false` but has 16 widely-used public functions.
4. **Is the eager Kalman filter (`filter/7`) still needed?** Compiled paths cover both scalar and multi-dim.
5. **Is test coverage measured?** No `excoveralls` or equivalent despite strong test investment.
6. **Should the library add a `:strict` mode** that raises on numerical failures instead of silently falling back?
7. **Target audience for domain applications?** Reference implementations or production APIs?
8. **EMLX backend maturity?** `test_helper.exs` defaults to EMLX when available with `max_cases: 1`.

---

## 14. Appendix: Module Inventory

| Module | File | LOC | Pub | Priv |
|--------|------|-----|-----|------|
| GibbsSampler | `gibbs_sampler.ex` | 1,606 | 6 | 66 |
| CausalImpact | `causal_impact.ex` | 1,061 | 5 | 15 |
| Smoother | `smoother.ex` | 806 | 12 | 5 |
| SpotAttributor | `spot_attributor.ex` | 740 | 12 | 10 |
| Components | `components.ex` | 723 | 15 | 15 |
| RollingBaseline | `rolling_baseline.ex` | 668 | 6 | 34 |
| KalmanFilter | `kalman_filter.ex` | 650 | 4 | 10 |
| CovariateSelection | `covariate_selection.ex` | 623 | 4 | 20 |
| ShapleyAllocator | `shapley.ex` | 489 | 15 | 8 |
| Forecaster | `forecaster.ex` | 482 | 5 | 10 |
| ModelBuilder | `model_builder.ex` | 481 | 15 | 27 |
| Validation | `validation.ex` | 475 | 11 | 22 |
| Distributions | `distributions.ex` | 474 | 6 | 18 |
| AnomalyDetector | `applications/anomaly_detector.ex` | 488 | 7 | 14 |
| DemandForecaster | `applications/demand_forecaster.ex` | 417 | 4 | 7 |
| Diagnostics | `diagnostics.ex` | 408 | 9 | 8 |
| MarketingLift | `applications/marketing_lift.ex` | 408 | 5 | 6 |
| Generator | `synthetic/generator.ex` | 408 | 4 | 12 |
| InterventionAnalysis | `intervention_analysis.ex` | 381 | 4 | 7 |
| Utils | `utils.ex` | 378 | 16 | 12 |
| ARForecaster | `bct/ar_forecaster.ex` | 365 | 3 | 19 |
| PolicyEvaluator | `applications/policy_evaluator.ex` | 336 | 3 | 6 |
| Scenarios | `synthetic/scenarios.ex` | 248 | 6 | 1 |
| TVAttribution | `applications/tv_attribution.ex` | 180 | 3 | 1 |
| StateSpace | `state_space.ex` | 166 | 4 | 8 |
| Pipeline | `pipeline.ex` | 166 | 1 | 2 |
| BstsNx | `bsts_nx.ex` | 103 | 0 | 0 |
| Adstock | `synthetic/adstock.ex` | 98 | 4 | 0 |
| ModelSpec | `model_spec.ex` | 64 | 0 | 0 |
| **TOTAL** | **29 files** | **12,315** | **155** | **386** |

---

*Review complete. Awaiting further instruction.*
