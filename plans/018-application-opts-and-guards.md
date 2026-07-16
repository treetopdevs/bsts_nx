# Plan 018: Fix application-layer option handling and input guards (orphaned passthrough opts, control-length crash, regressor doc contract)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat 9b7cb8d..HEAD -- lib/bsts_nx/model_builder.ex lib/bsts_nx/intervention_analysis.ex lib/bsts_nx/applications/policy_evaluator.ex lib/bsts_nx/applications/tv_attribution.ex lib/bsts_nx/rolling_baseline.ex`
> If any changed since this plan was written, compare the "Current state"
> excerpts against the live code before proceeding; on a mismatch, treat it
> as a STOP condition.

## Status

- **Priority**: P2
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: bug
- **Planned at**: commit `9b7cb8d`, 2026-07-08

## Why this matters

Three small, independently verified defects in the application/model-building
layer, bundled because they are all S-effort input/option-contract fixes:

**(A) Orphaned passthrough options crash instead of passing through.**
`ModelBuilder.build_opts_with_controls/3` documents that it "Returns a keyword
list suitable for passing to `InterventionAnalysis.analyze/3`", and filters
options via `Keyword.take(opts, @analysis_passthrough_opts)`. But that list
contains three keys — `:n_samples`, `:regression_mode`, `:regression_opts` —
that `InterventionAnalysis.analyze/3` does **not** accept: its
`@known_analysis_opts` lacks them and `validate_known_options!/1` raises
`ArgumentError "unknown option ..."` on anything unknown. So a user passing
e.g. `n_samples: 500` to `MarketingLift.measure_lift/3` (a plausible key —
`:n_samples` is a real option of `SpotAttributor`, `Pipeline`, and
`Validation.attribution_options/1`) gets a hard crash, while every *other*
unknown key is silently dropped by the `Keyword.take`. Neither `MarketingLift`
nor `PolicyEvaluator` documents or consumes the three bare keys (they use only
the `control_regression_*` variants, which ARE accepted) — the three entries
are vestigial and should be removed from the passthrough list.

**(B) `PolicyEvaluator.pre_trend_check/3` crashes or silently truncates on
short control series.** The control loop slices each control to the
pre-period with no length validation: an all-too-short control produces
`Enum.sum([]) / 0` → `ArithmeticError`; a partially-short one silently
truncates via `Enum.zip/2` and reports a parallel-trends diagnostic computed
over fewer points than claimed.

**(C) `TVAttribution.rolling_baseline/2` documents an unsatisfiable
`:regressors` contract.** Its doc says the tensor covers "the training
period", but it forwards to `RollingBaseline.fit_and_predict/3`, which raises
unless the tensor has `n_pre + post_period_length` rows. A user following the
doc gets `ArgumentError "full regressors must have N rows ..., got M"`.

## Current state

- `lib/bsts_nx/model_builder.ex` lines 17–45 — the passthrough list (three
  offending keys marked):

  ```elixir
  @analysis_passthrough_opts [
    :alpha,
    :num_samples,
    :burn_in,
    :seed,
    :seasonality,
    :model_spec,
    :method,
    :mode,
    :fallback,
    :allow_mcmc_fallback,
    :return,
    :baseline,
    :key,
    :thin,
    :n_samples,          # <- not accepted by analyze/3
    :initial_state,
    :initial_cov,
    :process_var,
    :obs_var,
    :x0,
    :p0,
    :q,
    :r,
    :regression_mode,    # <- not accepted by analyze/3
    :regression_opts,    # <- not accepted by analyze/3
    :control_regression_mode,
    :control_regression_opts
  ]
  ```

  Used at lines 173 and 226 via `Keyword.take(opts, @analysis_passthrough_opts)`.
  Callers: `MarketingLift.build_analysis_opts/3`
  (`lib/bsts_nx/applications/marketing_lift.ex:233`) and `PolicyEvaluator`
  (`lib/bsts_nx/applications/policy_evaluator.ex:292`).

- `lib/bsts_nx/intervention_analysis.ex` lines 73–101 — `@known_analysis_opts`
  contains `:num_samples`, `:control_regression_mode`,
  `:control_regression_opts` (and the rest of the passthrough list) but NOT
  `:n_samples`, `:regression_mode`, `:regression_opts`. Lines 422–438 —
  `validate_known_options!/1` raises `ArgumentError` for any key not in the
  list.

- `lib/bsts_nx/applications/policy_evaluator.ex` — `pre_trend_check/3`
  (`def pre_trend_check(observations, intervention, _opts \\ [])` at line
  ~209); the unguarded control block at ~lines 253–270:

  ```elixir
  control_check =
    case Map.get(intervention, :control_series) do
      nil ->
        nil

      controls ->
        Enum.map(controls, fn control ->
          ctrl_pre = Enum.slice(control, (pre_start - 1)..(pre_end - 1))
          ctrl_mean = Enum.sum(ctrl_pre) / length(ctrl_pre)

          diff =
            Enum.zip(pre_data, ctrl_pre)
            |> Enum.map(fn {t, c} -> t - c end)

          diff_mean = Enum.sum(diff) / length(diff)
          %{control_mean: ctrl_mean, mean_difference: diff_mean}
        end)
    end
  ```

- `lib/bsts_nx/applications/tv_attribution.ex` line ~127 — the wrong doc:

  ```
  * `:regressors` - a `{T, p}` Nx tensor of regressors for the training period.
    Passed through to `RollingBaseline.fit_and_predict/3`.
  ```

  and the enforcing side, `lib/bsts_nx/rolling_baseline.ex`
  `split_regressor_opts/3` (~lines 623–645):

  ```elixir
  expected_t_total = n_pre + post_period_length

  if t_total != expected_t_total do
    raise ArgumentError,
          "full regressors must have #{expected_t_total} rows (n_pre=#{n_pre} + post_period_length=#{post_period_length}), " <>
            "got #{t_total}"
  end
  ```

- Existing test files (patterns to follow):
  `test/bsts_nx/model_builder_test.exs`,
  `test/bsts_nx/applications/policy_evaluator_test.exs`,
  `test/bsts_nx/applications/marketing_lift_test.exs`.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Targeted tests | `mix test test/bsts_nx/model_builder_test.exs test/bsts_nx/applications/policy_evaluator_test.exs test/bsts_nx/applications/marketing_lift_test.exs` | all pass |
| Docs build (doc change in C) | `mix docs` | exit 0 |
| Full verify | `bash scripts/ci.sh` | exit 0 |

Tooling note: `mix` is a `mise` shim here; prefix `mise exec -- ` if needed.

## Scope

**In scope** (the only files you should modify):
- `lib/bsts_nx/model_builder.ex` (remove 3 list entries)
- `lib/bsts_nx/applications/policy_evaluator.ex` (guard in `pre_trend_check/3`)
- `lib/bsts_nx/applications/tv_attribution.ex` (doc string only)
- `test/bsts_nx/model_builder_test.exs`,
  `test/bsts_nx/applications/policy_evaluator_test.exs`,
  `test/bsts_nx/applications/marketing_lift_test.exs` (new tests)

**Out of scope** (do NOT touch):
- `lib/bsts_nx/intervention_analysis.ex` — do NOT add the three keys to
  `@known_analysis_opts`; `analyze/3` doesn't implement them, and
  accepted-but-ignored options are worse than dropped ones.
- `lib/bsts_nx/rolling_baseline.ex` — its full-tensor contract and error
  message are correct; the doc on the TVAttribution side is what's wrong.
- The `control_regression_mode`/`control_regression_opts` entries — they are
  real, documented, accepted options; leave them.
- `PolicyEvaluator.evaluate/3` — its control validation goes through
  `ModelBuilder` and is fine; only `pre_trend_check/3` lacks the guard.

## Git workflow

- Branch: `advisor/018-application-opts-and-guards` (from `execute-plans`).
- Commit style: `fix: prune orphaned analysis opts, guard pre-trend controls, correct regressor docs`
  (or three separate commits, one per defect — either is acceptable).
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1 (A): Remove the three orphaned keys from `@analysis_passthrough_opts`

Delete `:n_samples`, `:regression_mode`, `:regression_opts` from the list in
`lib/bsts_nx/model_builder.ex`. Keep everything else, including
`:control_regression_mode` / `:control_regression_opts`.

**Verify**: `mix compile --warnings-as-errors` → exit 0, and
`grep -n ":n_samples" lib/bsts_nx/model_builder.ex` → no matches.

### Step 2 (A): Regression tests for the passthrough contract

- In `test/bsts_nx/model_builder_test.exs`, add a test (near the existing
  option pass-through tests around line ~150) asserting the contract
  directly:

  ```elixir
  test "build_opts_with_controls drops options analyze/3 does not accept" do
    opts = [n_samples: 500, regression_mode: :dynamic, regression_opts: [g: 1], num_samples: 10]
    result = BstsNx.ModelBuilder.build_opts_with_controls([1.0, 2.0, 3.0], nil, opts)

    refute Keyword.has_key?(result, :n_samples)
    refute Keyword.has_key?(result, :regression_mode)
    refute Keyword.has_key?(result, :regression_opts)
    assert result[:num_samples] == 10
  end
  ```

- In `test/bsts_nx/applications/marketing_lift_test.exs`, add an end-to-end
  guard modeled on the existing `measure_lift` tests (seeded series, small
  sampler params): call `measure_lift(obs, campaign, n_samples: 5,
  num_samples: 10, burn_in: 5, seed: 42)` and assert it returns a result map
  (no raise). Before Step 1 this raised
  `ArgumentError "unknown option :n_samples"`.

**Verify**: `mix test test/bsts_nx/model_builder_test.exs test/bsts_nx/applications/marketing_lift_test.exs`
→ all pass, new tests included.

### Step 3 (B): Guard control lengths in `pre_trend_check/3`

At the top of the `controls ->` branch (before the `Enum.map`), validate every
control series against the pre-period bound, mirroring the tone of the
library's other `ArgumentError`s:

```elixir
      controls ->
        Enum.each(controls, fn control ->
          if length(control) < pre_end do
            raise ArgumentError,
                  "control series must cover the pre-period " <>
                    "(needs at least #{pre_end} points, got #{length(control)})"
          end
        end)

        Enum.map(controls, fn control ->
          ...
```

This converts both failure modes (empty slice → `ArithmeticError`; short
slice → silent truncation) into one clear error.

**Verify**: `mix compile --warnings-as-errors` → exit 0.

### Step 4 (B): Regression tests for the guard

In `test/bsts_nx/applications/policy_evaluator_test.exs`, following the
existing `pre_trend_check` test setup (see the calls around the file's
pre-trend section — an `intervention` map with `:control_series`):

1. Control shorter than `pre_start` (would previously be `ArithmeticError`):
   assert `assert_raise ArgumentError, ~r/control series must cover the pre-period/, fn -> ... end`.
2. Control longer than `pre_start` but shorter than `pre_end` (previously
   silent truncation): assert the same clear raise.
3. A well-formed control still returns `%{valid: true, ...}` with a
   `control_check` list (guards against over-tightening).

**Verify**: `mix test test/bsts_nx/applications/policy_evaluator_test.exs` →
all pass, 3 new tests.

### Step 5 (C): Fix the `:regressors` doc in `TVAttribution.rolling_baseline/2`

Replace the sentence at `lib/bsts_nx/applications/tv_attribution.ex:127` with
the true contract, e.g.:

```
* `:regressors` - a `{T, p}` Nx tensor covering the training period **plus the
  post-period horizon** (`T = n_pre + horizon` rows; the first `n_pre` rows fit
  the model, the remaining rows drive the counterfactual). Passed through to
  `RollingBaseline.fit_and_predict/3`, which raises if `T` differs.
```

Also check the other `:regressors` doc in the same file (~line 83, for the
other public function) — it describes a `{T, p}` tensor without stating which
window `T` spans; if that function forwards to the same full-tensor contract,
align its wording too; if it genuinely takes training-only regressors, leave
it and note the difference in your report.

**Verify**: `mix docs` → exit 0; rendered text contains the new wording
(`grep -rn "plus the" lib/bsts_nx/applications/tv_attribution.ex` → 1 match).

### Step 6: Full verification

**Verify**: `bash scripts/ci.sh` → exit 0.

## Test plan

Steps 2 and 4 above. Structural patterns: `test/bsts_nx/model_builder_test.exs`
(direct unit assertions on returned keyword lists) and
`test/bsts_nx/applications/*_test.exs` (seeded series, `num_samples: 10,
burn_in: 5, seed: 42` for MCMC speed). Defect C is doc-only; its "test" is
`mix docs` building and the grep.

## Done criteria

- [ ] `grep -En ":n_samples|[^_]:regression_mode|[^_]:regression_opts" lib/bsts_nx/model_builder.ex` → no matches
- [ ] `mix test test/bsts_nx/model_builder_test.exs test/bsts_nx/applications/policy_evaluator_test.exs test/bsts_nx/applications/marketing_lift_test.exs` → 0 failures, ≥5 new tests
- [ ] `grep -c "control series must cover the pre-period" lib/bsts_nx/applications/policy_evaluator.ex` → 1
- [ ] `mix docs` exits 0
- [ ] `bash scripts/ci.sh` exits 0
- [ ] `git status --porcelain` shows only in-scope files modified
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- Any existing test fails after Step 1 — that would mean something in the
  codebase (or a caller you haven't seen) genuinely produces `:n_samples` /
  `:regression_mode` / `:regression_opts` expecting analyze/3 passthrough;
  report which test rather than re-adding the keys.
- `@known_analysis_opts` in `intervention_analysis.ex` now CONTAINS the three
  keys (drift: someone fixed A in the opposite direction) — the passthrough
  removal would then silently change behavior; stop and reconcile.
- `pre_trend_check/3` has been refactored to route through `ModelBuilder`
  validation (B may already be fixed).
- The `:regressors` handling in `tv_attribution.ex` no longer forwards to
  `RollingBaseline.fit_and_predict/3`.

## Maintenance notes

- The invariant to preserve going forward: **every key in
  `@analysis_passthrough_opts` must appear in
  `InterventionAnalysis.@known_analysis_opts`**. The Step 2 unit test encodes
  the three known offenders; if someone adds a new passthrough key, reviewers
  should check the other list. (A property-style test asserting full subset
  inclusion would require exposing both module attributes — noted as optional
  future work, not done here to avoid widening public API.)
- If `:n_samples` support is ever *wanted* in analyze (e.g. as an alias for
  `:num_samples`), implement it explicitly in `intervention_analysis.ex` with
  precedence rules — do not just re-add it to the passthrough.
- Defect B's guard raises; if callers ever want a soft-fail
  (`%{valid: false, reason: ...}`) for diagnostics, that's an API decision for
  the maintainer, not a bug fix.
