# Deepen model-quality validation

## Scope and interface

Architecture candidate 03, based on remote main ea1a7cb; independent of PRs #9
and #10 and the unpublished local-main commits.

Add Validation.evaluate/1. A request supplies actual/baseline rank-1 tensors
and zero-based indices, plus optional coverage, placebo, and effect_stability
configurations. Return residuals, the five detail entries, and their verdicts.
The existing individual checks and assess/1 remain unchanged.

The module owns evidence assembly and assessment order. Callers own model
fitting, refit callbacks, evaluation periods, and presentation. No new adapter
protocol, model-fitting default, threshold policy, or statistical algorithm.

## Required behavior

- Use the caller's indices without conversion; validate bounds and vector shapes.
- Preserve coverage h/confidence options and all existing thresholds.
- Omitted/nil optional configurations yield nil evidence and :skip verdicts.
  Malformed supplied configurations must fail rather than silently skip.
- Run refits synchronously: placebo at most once, stability twice, in that order.
  Delegate existing short-period behavior and propagate callback errors.
- Return the same residuals/details/verdicts the demo and notebooks built manually.
- Preserve demo fitting seeds, callback definitions, graph data, and result shape.
- Notebook summaries describe checks, not a guarantee of release readiness;
  existing pass-or-skip policy remains explicit.

## Execution

1. Add deterministic public-interface tests and demonstrate failure before adding
   evaluate/1. Include callback arguments/counts, options, skips, invalid inputs,
   and error propagation.
2. Implement the workflow; migrate diagnostics demo and both notebook examples.
3. Run library CI parity, site mix precommit, and a deterministic before/after
   demo comparison excluding elapsed timings. Parse changed notebook code cells.
4. Review Standards and Spec independently; fix findings, commit, open PR, and
   monitor final-head CI. Keep local approval distinct from GitHub approval.

## Evidence

- Red: all five initial workflow tests failed because evaluate/1 did not exist.
- Final focused validation suites: 65 tests passed.
- Final bash scripts/ci.sh: warnings-as-errors compilation, 51 doctests,
  57 properties, 839 tests, zero failures (74 excluded), formatting, and docs.
- Final site mix precommit: compilation, formatting, 10 tests passed.
- Final seeded demo snapshot exactly matched original output, excluding timings.
- Both changed notebook evaluation cells parsed and formatted successfully.
- Existing validation calculations and thresholds verified unchanged against base.
- Standards and Spec review findings were corrected: callback message order is
  asserted directly; malformed coverage shapes/scalars/confidence fail even with
  empty evaluation indices. Both reviewers approved the final changes.
- Local runtime: Elixir 1.19.5 / OTP 29. Hosted CI uses Elixir 1.19 / OTP 28.

Final hosted CI and review status are recorded on the PR. Local review approval
is distinct from GitHub approval. Numerical tolerances are unchanged.
