# Concentrate forward-moment propagation

## Scope

Architecture candidate 02. Base: origin/main at ea1a7cb, independent of PR #9.
Exclude the 46 unpublished local-main commits and retain their ADR decisions:
observation mapping remains with the existing caller; posterior interpretation
and counterfactual execution are not redesigned here.

## Design

Deepen Forward with baseline_moments_from_samples/3, taking retained draws,
model, and explicit future observation rows. It owns terminal covariance
selection, compiled state/covariance propagation, and baseline aggregation.
RollingBaseline keeps horizon validation and future regressor preparation.

Share one private compiled recurrence with structured_moments_from_samples/3.
Do not expose a generic policy/options interface. Keep these distinctions:

- Baseline: retained terminal covariance (zero if missing), terminal state dtype,
  per-draw variance clamped at zero, population between-draw variance, and
  observation variance returned separately. Preserve the existing result map.
- Predictive: zero conditional covariance, f64 propagation, unbiased sample
  between-draw variance, clamped observation noise included, variance floor 1e-12.
- Both: full Q including off-diagonal entries; advance before observing each step.
- Operational fixed-variance cumulative covariance and stochastic trajectories
  are outside scope and must remain unchanged.

## Execution and gates

1. Add closed-form public-interface tests for covariance/noise distinctions,
   multiple draws, missing covariance, full Q, and empty horizons. Run them on
   the original implementation to establish behavioral evidence.
2. Move baseline traversal/aggregation to Forward and share compiled propagation.
   Replace the source-location test with tests through the actual interface;
   retain the compiled horizon loop in Forward.
3. Run focused Forward, RollingBaseline, and Operational tests, EXLA smoke tests,
   then bash scripts/ci.sh (warnings-as-errors, non-external suite, format, docs).
4. Independently review Standards and Spec, fix findings, commit, open PR, and
   monitor final-head CI. Report local review and hosted approval separately.

## Validation

- Original implementation: 16 focused tests passed, including the new analytical cases.
- Refactored implementation: 32 focused Forward, RollingBaseline, regression,
  and Operational tests passed.
- Expanded EXLA lane: 40 tests passed. After adding the review-suggested clamp
  case, the Forward EXLA suite passed all 13 tests.
- bash scripts/ci.sh passed: warnings-as-errors compilation, 51 doctests,
  57 properties, 838 tests, zero failures (74 excluded), formatting, and docs.
- Local runtime: Elixir 1.19.5 / OTP 29; hosted CI uses Elixir 1.19 / OTP 28.
- Standards and Spec reviews approved with no blocking findings. The suggested
  negative-variance test was added and passed.
- Operational and stochastic kernel bodies verified unchanged against base.
- Existing predictive empty-row-list failure on remote main remains out of scope.

Hosted CI and review status are recorded on the PR; local review approval is
separate from GitHub approval. Existing numerical test tolerances are unchanged.
