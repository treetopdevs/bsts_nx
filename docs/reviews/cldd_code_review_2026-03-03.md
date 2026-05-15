# BstsNx Comprehensive Code Review

**Date:** 2026-03-03
**Scope:** Full codebase — 29 source files, 60+ test files
**Reviewers:** 5 parallel analysis agents (core engine, model layer, high-level APIs, supporting modules, test suite)

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 2 |
| IMPORTANT | 21 |
| MINOR | 27 |

The codebase is well-architected with strong numerical stability practices, thorough documentation, and clean module separation. No showstopping bugs were found. The two CRITICAL issues are edge cases in the causal impact estimation layer. The IMPORTANT issues cluster around: f32 precision constraints in compiled paths, PRNG key handling fragility, and a few statistical methodology concerns. The test suite is extensive but has one tautological assertion that never fails.

---

## CRITICAL Issues

### C1. Inverse-gamma prior mode as initializer in non-MCMC estimator

**File:** `lib/bsts_nx/causal_impact.ex:711-721`

The `estimate_structured_from_filter` (non-MCMC path) initializes Q diagonal and R using `prior_scale / (prior_shape + 1.0)` — the mode of InvGamma(α, β). With defaults `(1.0, 1.0)`, this yields `0.5`, which underestimates the center of a heavily right-skewed InvGamma(1,1). Since this path has no MCMC adaptation, the initial point estimate directly determines the result quality.

**Fix:** Use `prior_scale / max(prior_shape - 1.0, 0.5)` (regularized mean) or document the sensitivity of the filter-only path to prior initialization.

### C2. Rank-1 H tensor mishandled in `estimate_structured_from_filter`

**File:** `lib/bsts_nx/causal_impact.ex:747-750`

When H is a static tensor of shape `{n}` (rank-1), the broadcast produces `{t, n}` (rank-2) instead of the expected `{t, 1, n}` (rank-3) for `filter_defn_multi`. While `ModelSpec` factories always produce `{1, n}`, a user-supplied rank-1 H would silently produce wrong results.

**Fix:** Normalize H to `{1, n}` before broadcasting:
```elixir
normalized = if Nx.rank(static_h) == 1,
  do: Nx.reshape(static_h, {1, Nx.axis_size(static_h, 0)}),
  else: static_h
```

---

## IMPORTANT Issues

### I1. `safe_cholesky` jitter does not scale with matrix magnitude

**File:** `lib/bsts_nx/utils.ex:52-78`

Fixed jitter ladder `[1e-6, 1e-5, 1e-4, 1e-3]` is absolute. For matrices with diagonal ~1e-8 (small variances), even 1e-6 jitter dominates. For diagonals ~1e6, 1e-3 is insufficient.

**Fix:** Scale jitter relative to `max(trace/n, 1e-12)`.

### I2. `scalar_divide_with_jitter_fallback` sign ambiguity

**File:** `lib/bsts_nx/utils.ex:223-227`

Positive jitter on a small negative denominator can push it through zero, flipping the result sign.

**Fix:** Use `denom + sign(denom) * jitter`.

### I3. Distributions key splitting via `Nx.to_list()` roundtrip

**File:** `lib/bsts_nx/distributions.ex:266-273` (also 303-305, 361-363)

`Nx.Random.split(key) |> Nx.to_list() |> Nx.tensor()` is fragile and loses type metadata. Appears in `normal_sample`, `mv_normal_sample_with_chol`, and `sample_with_key_scalar`.

**Fix:** Use `Nx.slice_along_axis` instead of list roundtrip.

### I4. Defn gamma sampler falls back to mode instead of raising

**File:** `lib/bsts_nx/distributions.ex:242`

When the rejection sampler fails after 10,000 iterations in the defn path, `gamma = d = alpha - 1/3` (the mode) is silently used. This biases MCMC chains. The non-defn path raises `RuntimeError` — more honest behavior.

**Fix:** Log a warning or increase the iteration limit. Document the bias risk.

### I5. Smoother defn Cholesky has no NaN fallback

**File:** `lib/bsts_nx/smoother.ex:456`

Inside `simulate_from_filtered_defn_matrix_impl_solve`, Cholesky failure produces NaN that propagates silently through the rest of the trajectory. The eager path has try/rescue + PSD projection; the defn path does not.

**Fix:** Use a larger regularization epsilon or clamp diagonal to be positive before Cholesky.

### I6. `observed_transition_count` vs imputed states mismatch

**File:** `lib/bsts_nx/gibbs_sampler.ex:1535-1540`

`process_sum_of_squares` sums over all `t-1` state transitions (including imputed), but the posterior shape uses `observed_transition_count` (only observed-observed pairs). This makes the posterior shape too small → overdispersed process variance.

**Fix:** Use `t - 1` for the degrees of freedom if states are fully imputed, or document this as a deliberate modeling choice.

### I7. Hardcoded `{:f, 32}` in compiled paths

**Files:** `lib/bsts_nx/kalman_filter.ex:321`, `lib/bsts_nx/gibbs_sampler.ex:1520`

Both `filter_defn_multi` and `observations_to_filter_tensor` force `{:f, 32}`. The scalar `filter_defn` and the smoother (`rts_defn_matrix`) use `{:f, 64}`. This precision mismatch means data enters the high-precision smoother already truncated.

**Fix:** Accept a `:type` option or default to `{:f, 64}`.

### I8. `normalize_h_series` loses dtype through list roundtrip

**File:** `lib/bsts_nx/kalman_filter.ex:488-498`

`Nx.to_flat_list() |> Enum.chunk_every() |> Enum.map(&Nx.tensor/1)` drops the original dtype, silently downgrading f64 inputs to f32.

**Fix:** Pass the original type to `Nx.tensor/2`.

### I9. Duplicate `dim_index` in `q_specs` not validated

**File:** `lib/bsts_nx/model_spec.ex:156-171`

Two q_specs referencing the same dimension would cause double-resampling in the Gibbs sampler, corrupting the posterior.

**Fix:** Add uniqueness check in `validate_q_specs!`.

### I10. `compose_specs/2` silently drops spec2's observation variance priors

**File:** `lib/bsts_nx/components.ex:495-497`

When composing two specs, `obs_var`, `obs_prior_shape`, and `obs_prior_scale` come only from spec1. A user's explicit `obs_var` on spec2 is silently discarded.

**Fix:** Warn when spec2 has different obs parameters, or document the convention clearly.

### I11. `InterventionAnalysis.analyze_filter` defaults `x0 = 0.0`

**File:** `lib/bsts_nx/intervention_analysis.ex:348-349`

For the scalar filter path, `x0 = 0.0` is a poor default. `Forecaster` and `AnomalyDetector` both default to the first observation value for faster convergence.

**Fix:** Default to `List.first(observations) || 0.0`.

### I12. `Forecaster.fit_predict/3` reuses the same seed for fit and predict

**File:** `lib/bsts_nx/forecaster.ex:222-226`

Same `:seed` option is passed to both MCMC sampling and forecast simulation, creating correlated PRNG streams. Same pattern in `BCT.ARForecaster.fit_predict/3:167-171`.

**Fix:** Derive a separate seed for prediction (e.g., `seed + 1`).

### I13. `DemandForecaster.do_forecast` drops `:key` from sampler opts

**File:** `lib/bsts_nx/applications/demand_forecaster.ex:230-236`

`Keyword.take([:seed, :thin])` omits `:key`. If user passes `:key` instead of `:seed`, it is silently ignored.

**Fix:** Include `:key` in the take list.

### I14. `AnomalyDetector.score/2` MCMC path uses naive extrapolation

**File:** `lib/bsts_nx/applications/anomaly_detector.ex:166-173`

Out-of-sample scoring repeats the last training mean and grows uncertainty linearly, ignoring learned trend/seasonality dynamics. Produces systematic false positives for seasonal data.

**Fix:** Use forward simulation from last posterior state, or document the limitation prominently.

### I15. `Pipeline.run/6` does not validate contiguous periods

**File:** `lib/bsts_nx/pipeline.ex:59, 103-104`

No check that `post_start == pre_end + 1`. A gap between periods produces incorrect counterfactuals silently.

**Fix:** Add period contiguity validation.

### I16. `DemandForecaster.safety_stock/2` overestimates via sum of SDs

**File:** `lib/bsts_nx/applications/demand_forecaster.ex:164-168`

Sums per-period safety stocks instead of using `z * sqrt(sum(var_i))`. By triangle inequality, `sum(sd_i) >= sqrt(sum(sd_i^2))`. This is conservative but statistically incorrect for independent demands.

**Fix:** Implement correct aggregation or document as a deliberate conservative choice.

### I17. ShapleyAllocator `default_value_function` breaks symmetry axiom

**File:** `lib/bsts_nx/shapley_allocator.ex:280-286`

String ID tiebreaker in `Enum.sort_by(fn id -> {-abs(lift), id} end)` makes allocation depend on player labels for equal-lift players, violating the Shapley symmetry axiom.

**Fix:** Assign equal rank for tied lifts, or document the limitation.

### I18. Generator: inconsistent aggregation windows between solo and coalition lift

**File:** `lib/bsts_nx/synthetic/generator.ex:196-216 and 278-282`

Solo lift sums within `[window_start, window_end)` but coalition lift sums the full signal including adstock carryover. `v({i}) != solo_lift_i`, making ground truth Shapley allocations inconsistent.

**Fix:** Use the same aggregation window for both computations.

### I19. `SpotAttributor.attribute_posterior` uses O(n^2) `Enum.at` in loop

**File:** `lib/bsts_nx/spot_attributor.ex:267-281`

Repeated `Enum.at(list, idx)` calls are O(n) each, giving O(n^2) total for n spots.

**Fix:** Use `Enum.zip` instead of indexed access.

### I20. Test: tautological assertion in `distributions_mv_normal_test.exs`

**File:** `test/distributions_mv_normal_test.exs:16`

```elixir
assert Nx.all_close(sample_with_chol, sample_direct, atol: 1.0e-8, rtol: 1.0e-8)
```

Missing `|> Nx.to_number() == 1`. An `Nx.Tensor` struct is always truthy — this test **never fails**, even if the samples are completely wrong.

**Fix:** Add `|> Nx.to_number() == 1`.

### I21. `build_future_h` uses `List.last(spec.h)` — fragile for time-varying trend H

**File:** `lib/bsts_nx/model_builder.ex:337-346`

Assumes the non-regression portion of H is constant across time. If trend H is also time-varying, the last value is used for all forecast horizons.

**Fix:** Extract static H from the original component spec before composition, or document the assumption.

---

## MINOR Issues

### M1. `Adstock.geometric_adstock` accepts `lambda >= 1` despite docstring saying `[0, 1)`

**File:** `lib/bsts_nx/synthetic/adstock.ex:44` — Add guard `lambda >= 0 and lambda < 1` or update docs.

### M2. `Validation.assess_placebo` has no `:warn` tier (unlike all other assessments)

**File:** `lib/bsts_nx/validation.ex:326-329` — Consider adding a warning tier for borderline cases.

### M3. `SpotAttributor.validate_inputs!` allows zero-length windows

**File:** `lib/bsts_nx/spot_attributor.ex:577-587` — `window_start == window_end` is silently accepted as a no-op.

### M4. `Generator.normal_sample` exists alongside `:rand.normal_s/1`

**File:** `lib/bsts_nx/synthetic/generator.ex:299-306` — Custom Box-Muller wastes 50% entropy. `:rand.normal_s` is used elsewhere in the codebase.

### M5. `RollingBaseline.forward_simulate_h` ignores `_horizon` parameter

**File:** `lib/bsts_nx/rolling_baseline.ex:518` — Unused parameter with no assertion that `length(h_list) == horizon`.

### M6. `Validation.effect_stability` hardcodes minimum window of 5

**File:** `lib/bsts_nx/validation.ex:255` — Domain-specific (TV attribution). Should be configurable.

### M7. `ShapleyAllocator.time_weighted_value_function` defaults unknown IDs to time 0

**File:** `lib/bsts_nx/shapley_allocator.ex:356-357` — Silent default; consider `Map.fetch!`.

### M8. `PolicyEvaluator.pre_trend_check/3` ignores `_opts` parameter

**File:** `lib/bsts_nx/applications/policy_evaluator.ex:211` — Dead parameter.

### M9. `PolicyEvaluator.pre_trend_check/3` doesn't validate control series lengths

**File:** `lib/bsts_nx/applications/policy_evaluator.ex:261-270` — Short controls silently truncated by `Enum.zip`.

### M10. `DemandForecaster.forecast_with_decomposition/2` sets `:components` to `nil`

**File:** `lib/bsts_nx/applications/demand_forecaster.ex:108-122` — Misleading field.

### M11. `Forecaster.decompose/1` doesn't validate H list length matches training length

**File:** `lib/bsts_nx/forecaster.ex:251-259` — Could silently produce wrong decomposition.

### M12. `Forecaster.h_to_row/1` adds unnecessary `+ 0.0` float coercion

**File:** `lib/bsts_nx/forecaster.ex:462-467` — `Nx.to_flat_list` already returns floats.

### M13. `BCT.ARForecaster` stores full observation history in fit result

**File:** `lib/bsts_nx/bct/ar_forecaster.ex:111` — Only last `order` elements needed for prediction.

### M14. Forward simulation logic duplicated in 3 modules

**Files:** `forecaster.ex:276-339`, `demand_forecaster.ex:319-363`, `causal_impact.ex:925-954` — Consider extracting shared utility.

### M15. `Pipeline.validate_spot_windows!` allows zero-length windows

**File:** `lib/bsts_nx/pipeline.ex:136-150` — Same issue as M3.

### M16. `CausalImpact.to_number/1` crashes on non-numeric atoms

**File:** `lib/bsts_nx/causal_impact.ex:1059` — `FunctionClauseError` rather than descriptive error.

### M17. Unused `_n_steps` parameter in `generate_structured_counterfactual`

**File:** `lib/bsts_nx/causal_impact.ex:931`

### M18. `CovariateSelection.seed_triplet` uses correlated seeds

**File:** `lib/bsts_nx/covariate_selection.ex:437-446` — `{base, base+1, base+2}` vs `derive_exsss_seed` used elsewhere.

### M19. `ModelBuilder.normalize_number` NaN detection via `f == f` is subtle

**File:** `lib/bsts_nx/model_builder.ex:477` — Correct but warrants a comment.

### M20. `ModelBuilder.safe_divide` has redundant clauses for `0` and `0.0`

**File:** `lib/bsts_nx/model_builder.ex:377-379` — Both needed for correctness but `0.0` clause may never match in practice.

### M21. `Validation.known_lift_injection` uses 1-based periods without inline comment

**File:** `lib/bsts_nx/validation.ex:427-433` — Convention differs from 0-based spot windows.

### M22. `Diagnostics.ess_single` formula is correct but not commented

**File:** `lib/bsts_nx/diagnostics.ex:107-136` — Non-obvious Geyer ESS derivation.

### M23. Test: `@compiled_backend?` evaluated at compile time, depends on compilation order

**Files:** `test/causal_impact_structured_test.exs:7-12`, `test/smoother_defn_matrix_test.exs:6-7`

### M24. Test: `finite_number?/1` infinity check is a no-op on guarded branch

**File:** `test/gibbs_sampler_missing_observations_test.exs:68`

### M25. Test: Shapley `three_player_value_fn` crashes if IDs arrive unsorted

**File:** `test/shapley_test.exs:22-33` — Make the function order-agnostic defensively.

### M26. Test: Inconsistent module naming (`BstsNx.SmootherDefnTest` vs `BstsNxEdgeCasesTest`)

**Files:** Various test files.

### M27. Test: No tests for `CausalImpact.estimate` with missing observations in pre-period or empty observations

**Coverage gap** — Realistic scenario not tested.

---

## Positive Observations

1. **Numerical robustness** — Progressive jitter in `safe_cholesky`/`safe_solve`, Joseph-form covariance updates, NaN detection, variance clamping
2. **PRNG discipline** — Consistent key-splitting, functional `:rand` state threading, `derive_exsss_seed` bridge
3. **Modular composition** — `ModelSpec` + `compose_specs` + `Components.*_spec` is clean and extensible
4. **Documentation quality** — Every public function has `@doc`, `@spec`, mathematical notation, and usage examples
5. **Dual implementation strategy** — Eager list-based + compiled `defn` paths serve different use cases well
6. **Validation coverage** — `ModelSpec.validate!/1` checks shapes, positivity, and index bounds thoroughly
7. **Test suite depth** — Property-based tests verify algebraic invariants (Shapley axioms, covariance non-negativity), edge case coverage is systematic
8. **Multi-backend awareness** — Test infrastructure handles EXLA, EMLX, and native backends with appropriate adjustments
9. **Clean API layering** — Domain apps are thin wrappers that add validation without duplicating statistics
10. **Sound mathematical implementations** — Kalman filter, RTS smoother, Carter-Kohn simulation, Marsaglia-Tsang gamma, Hill saturation all correctly implemented

---

## Recommended Priority Order

1. **I20** — Fix tautological test assertion (zero effort, real coverage gap)
2. **C2** — Normalize rank-1 H before broadcast (simple guard, prevents silent wrong results)
3. **I9** — Validate `dim_index` uniqueness (simple, prevents posterior corruption)
4. **I7/I8** — Address f32 precision constraints (most common source of user-visible issues)
5. **C1** — Improve non-MCMC initializer (affects filter-only estimation quality)
6. **I6** — Clarify `observed_transition_count` semantics (statistical methodology)
7. **I1** — Scale-adaptive jitter (robustness across data magnitudes)
8. **I15** — Period contiguity validation (easy, prevents silent wrong results)
9. **I5** — Defn smoother NaN resilience (harder, affects compiled path reliability)
10. **I17/I18** — Shapley consistency fixes (affects ground truth generation)
