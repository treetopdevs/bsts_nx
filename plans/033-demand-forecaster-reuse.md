# Reuse Forecaster for demand forecasting

## Scope and design

Candidate 01 from the architecture review. Base: `ea1a7cb` (`origin/main`).
The reviewed local main is 46 commits ahead; those commits are excluded from
this PR. Both versions expose the existing Forecaster fit/predict/decompose
interface needed here.

DemandForecaster remains the compatibility adapter for demand model priors,
multiple seasonalities, training/future regressor input, and business calculations.
Forecaster owns sampling, future observation mapping, projection, and summaries.
Return the actual fit internally for decomposition instead of rebuilding a fit map.
Retain the demand random-key split and early validation to preserve their ordering;
no new forecasting interface or changes to shared numerical implementations.

## Compatibility requirements

- Preserve the local-linear-trend defaults and seasonal/regression composition.
- Preserve the summary keys, components: nil, and decomposition from the same fit.
- Preserve seeded/keyed draws, key precedence, burn-in, thinning, and quantile rule.
  Mean/SD may differ only by floating-point reduction rounding.
- Pass training regressors explicitly so constant columns retain their dimensions.
- Keep empty observations, horizon, alpha, sample count, then model validation order.
- Keep unrecognized options ignored; do not activate weighting or output-format options.
- Keep safety stock and promotion impact unchanged.

## Execution

1. Add public-interface compatibility tests and run against the old implementation.
2. Route demand fit/prediction through Forecaster; delete duplicate aggregation
   and future observation mapping; reuse the actual fit for decomposition.
3. Run focused tests and `bash scripts/ci.sh` on the PR worktree.
4. Review the exact change for standards and requirements; address findings.
5. Commit, publish a focused PR against main, and monitor its CI to completion.

## Validation record

- Before refactoring: demand tests passed (20 tests), including the new compatibility cases.
- After refactoring: demand, Forecaster, and custom-regression tests passed (40 tests).
- `bash scripts/ci.sh` passed: warnings-as-errors compilation, 51 doctests,
  57 properties, 836 tests, zero failures (74 excluded), formatting, and docs.
- Local runtime: Elixir 1.19.5 / OTP 29; hosted CI targets Elixir 1.19 / OTP 28.
- Independent Standards and Spec reviews approved with zero findings.
- Production change: 20 lines added, 79 removed; no shared numerical code changed.

Hosted CI results are recorded on the PR. Local review approval is separate from
GitHub review approval; no hosted approval is claimed here.
