# BSTS_NX Comprehensive Code Review (2026-03-02)

## Scope
Reviewed the full repository with focus on:
- Logic and numerical correctness
- DDD boundaries and domain API design
- Phoenix / Ash applicability
- Security and safety
- UI/UX applicability
- Maintainability

## Method
- Static review across `lib/`, `mix.exs`, and representative tests/docs
- Compile signal: `mix compile --warnings-as-errors` (passed)
- Targeted test runs were started but full-suite completion signal was not obtained in this session due long-running execution behavior

## Executive Summary
- High risk findings: 2
- Medium risk findings: 5
- Low risk findings: 3
- Phoenix/Ash/UI codepaths: not present in this codebase (see dedicated section)

---

## Findings (ordered by severity)

### 1) [High] Silent covariance-collapse fallback can hide major inference failures
**Files:**
- `lib/bsts_nx/smoother.ex:758`
- `lib/bsts_nx/smoother.ex:775`
- `lib/bsts_nx/smoother.ex:776`

**Issue:**
When Cholesky fails in `safe_cholesky_or_zero/2`, the function can return a zero matrix fallback (`Nx.broadcast(0.0, Nx.shape(cov_sym))`) with no explicit warning/error. In simulation smoothing, this can collapse stochastic uncertainty to deterministic paths for affected steps.

**Why it matters:**
This can silently bias posterior state draws and downstream intervals while appearing “successful”. For a statistical library, silent numerical degradation is high-impact.

**Recommendation:**
- Fail fast (raise) by default, with an opt-in “best effort” mode.
- At minimum, emit structured warnings/telemetry with step/context and fallback count.
- Surface fallback counts in returned diagnostics where possible.

---

### 2) [High] Regression spec builders do not validate vector option lengths
**Files:**
- `lib/bsts_nx/components.ex:615`
- `lib/bsts_nx/components.ex:618`
- `lib/bsts_nx/components.ex:631`
- `lib/bsts_nx/components.ex:636`
- `lib/bsts_nx/components.ex:645`
- `lib/bsts_nx/components.ex:649`

**Issue:**
`build_dynamic_regression_spec/2` accepts list-valued `:var_beta` and `:initial_cov_beta` but does not validate list length against `p` regressors. Short lists can yield `nil` entries in `q_specs` or malformed initial covariance construction.

**Why it matters:**
This produces late, confusing runtime failures (or invalid numeric state), instead of immediate argument errors.

**Recommendation:**
- Validate exact lengths for list options (`var_beta`, `initial_cov_beta`, `initial_betas`).
- Raise clear `ArgumentError` with expected vs actual lengths.

---

### 3) [Medium] Chain seeding/failure behavior can silently violate reproducibility expectations
**Files:**
- `lib/bsts_nx/gibbs_sampler.ex:161`
- `lib/bsts_nx/gibbs_sampler.ex:180`
- `lib/bsts_nx/gibbs_sampler.ex:211`
- `lib/bsts_nx/gibbs_sampler.ex:507`
- `lib/bsts_nx/gibbs_sampler.ex:524`
- `lib/bsts_nx/gibbs_sampler.ex:547`

**Issue:**
- If `:seeds` length != `num_chains`, code silently falls back to auto-seeding instead of raising.
- Failed chains are logged and dropped; function returns partial chain sets unless all fail.

**Why it matters:**
Silent fallback and partial completion can invalidate diagnostics (R-hat/ESS) and reproducibility assumptions without explicit caller awareness.

**Recommendation:**
- Raise on invalid `:seeds` length.
- Add strict mode (`require_all_chains: true` default true) to fail on any chain loss.
- Return metadata listing failed chains when permissive mode is used.

---

### 4) [Medium] `Forecaster.predict/2` silently ignores list-form `future_regressors`
**Files:**
- `lib/bsts_nx/forecaster.ex:179`
- `lib/bsts_nx/forecaster.ex:191`
- `lib/bsts_nx/forecaster.ex:288`
- `lib/bsts_nx/forecaster.ex:293`

**Issue:**
Validation only handles `%Nx.Tensor{}` or `nil`. Non-tensor values (including list-of-lists) are not rejected and fall through to static-`H` behavior, effectively ignoring provided future regressors.

**Why it matters:**
Forecasts can be materially wrong while caller thinks covariates were applied.

**Recommendation:**
- Accept lists by normalizing via `ModelBuilder.ensure_tensor/1`, or raise explicitly for non-tensor input.
- Add regression-aware tests for list input to prevent silent fallback.

---

### 5) [Medium] `pre_trend_check/3` can crash on short control series
**Files:**
- `lib/bsts_nx/applications/policy_evaluator.ex:262`
- `lib/bsts_nx/applications/policy_evaluator.ex:263`
- `lib/bsts_nx/applications/policy_evaluator.ex:269`

**Issue:**
Control pre-period slices are used without length checks; empty/short control slices can trigger division by zero or invalid diagnostics.

**Why it matters:**
This is user-facing API behavior and currently fails with low-context runtime errors.

**Recommendation:**
- Validate each control series length against observations and requested pre-window.
- Return explicit `ArgumentError` with campaign/intervention context.

---

### 6) [Medium] Filter-based impact APIs silently drop invalid intervention indices
**Files:**
- `lib/bsts_nx/causal_impact.ex:478`
- `lib/bsts_nx/causal_impact.ex:484`
- `lib/bsts_nx/causal_impact.ex:696`
- `lib/bsts_nx/causal_impact.ex:702`

**Issue:**
Out-of-range indices are filtered out, and empty results return zero-effect maps instead of erroring.

**Why it matters:**
Caller mistakes (off-by-one/date-index mismaps) can pass unnoticed and produce misleading “valid” zero-impact outputs.

**Recommendation:**
- Add strict validation mode (default strict) that raises on any invalid index.
- Optionally support permissive mode with explicit warning and returned dropped-index metadata.

---

### 7) [Medium] `safety_stock/2` lacks input guardrails
**Files:**
- `lib/bsts_nx/applications/demand_forecaster.ex:153`
- `lib/bsts_nx/applications/demand_forecaster.ex:158`
- `lib/bsts_nx/applications/demand_forecaster.ex:166`

**Issue:**
No validation on `service_level` or `lead_time`.
- `service_level` outside `(0,1)` can produce nonsensical quantiles.
- negative `lead_time` triggers `Enum.take/2` semantics not intended for planning.

**Why it matters:**
Business decisions (inventory buffers) can be based on invalid calculations.

**Recommendation:**
Validate `service_level in (0,1)` and `lead_time` as positive integer; raise clear errors.

---

### 8) [Low] Structured filter fast path assumes static `H` rank/shape too narrowly
**Files:**
- `lib/bsts_nx/causal_impact.ex:753`
- `lib/bsts_nx/causal_impact.ex:755`

**Issue:**
Static `H` tensor handling uses `Nx.axis_size(static_h, 1)` unconditionally in one branch, which is fragile for rank-1 inputs and can crash with low-context errors.

**Why it matters:**
Custom `ModelSpec` authors can hit hard-to-debug shape failures.

**Recommendation:**
Normalize `H` shape explicitly (`{1,n}` / `{t,1,n}` contracts) and fail early with descriptive shape diagnostics.

---

### 9) [Low] DDD boundary uses loose maps for domain contracts
**Files:**
- `lib/bsts_nx/applications/marketing_lift.ex:121`
- `lib/bsts_nx/applications/policy_evaluator.ex:135`
- `lib/bsts_nx/applications/tv_attribution.ex:91`

**Issue:**
Domain APIs rely on ad-hoc maps with direct key access (dot syntax), without dedicated constructors/validators.

**Why it matters:**
Errors become `KeyError`/shape failures instead of explicit domain validation errors. This increases onboarding and runtime support burden.

**Recommendation:**
Introduce typed structs or validation functions for campaign/intervention/spot inputs and centralize range checks.

---

### 10) [Low] Large core modules increase regression risk and review difficulty
**Files:**
- `lib/bsts_nx/gibbs_sampler.ex:1`
- `lib/bsts_nx/causal_impact.ex:1`
- `lib/bsts_nx/smoother.ex:1`

**Issue:**
Very large, mixed-responsibility modules (state sampling, priors, regression selection, reporting, fast-path logic) reduce locality and raise cognitive overhead.

**Why it matters:**
Future changes are harder to test and reason about; defects are more likely to cross-cut unrelated paths.

**Recommendation:**
Split by responsibility (e.g., `GibbsSampler.Structured`, `GibbsSampler.SpikeSlab`, `CausalImpact.FilterFastPath`, `CausalImpact.MCMCPath`) and enforce narrower public surfaces.

---

## Phoenix / Ash / UI-UX Review Status
- **Phoenix:** No Phoenix application/router/controller/liveview code found in this repository.
- **Ash:** No Ash resources/actions/policies found.
- **UI/UX:** No frontend/UI layer present (library-only project).

Result: those categories are **not directly applicable** in current codebase state.

## Security Posture Notes
- No obvious RCE-style risks in runtime library code (no dynamic eval, no shelling-out in `lib/`).
- Primary security-relevant risk here is **integrity/safety**: silent numerical fallbacks can hide model failure states.

## Commands Run
- `mix compile --warnings-as-errors`
- repository-wide static inspection (`rg`, `nl -ba`, targeted module walkthrough)
