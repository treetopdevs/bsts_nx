# Plan 029: Strengthen type-only assertions on Bayesian effect estimates (planted-truth recovery)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat 9b7cb8d..HEAD -- test/bsts_nx/applications/`
> Plans 017/018 add tests to two of these files — expected. On other
> structural drift, compare excerpts before proceeding.

## Status

- **Priority**: P3
- **Effort**: M
- **Risk**: LOW (test-only; may legitimately expose real accuracy issues — that's the point)
- **Depends on**: none (mild interaction: plans 017/018 touch the same test
  files; land after them to avoid merge noise)
- **Category**: tests
- **Planned at**: commit `9b7cb8d`, 2026-07-08

## Why this matters

Across the application modules' MCMC-path tests, the headline effect
estimates are frequently asserted only by type: `is_float(result.total_lift)`
passes when the lift is `0.0`, negative, or a thousand-fold off. The sharpest
case is TVAttribution — its primary entry point `attribute/3` is tested with
deliberately overlapping spots, yet `overlaps` is only checked with
`is_list/1` (which passes on `[]`), and no numeric output is checked for
magnitude or sign. A sign flip or zeroed-out attribution ships green. The
seeded, deterministic test setups already plant known effects — the fix is to
assert the estimates actually *recover* them.

## Current state

- `test/bsts_nx/applications/tv_attribution_test.exs` — the main `attribute/3`
  test builds `pre` around 1000 and `post` around 1200 (a planted ~+200/step
  lift over 10 steps) with overlapping spots
  `%{id: "spot_1", window_start: 0, window_end: 5}` /
  `%{id: "spot_2", window_start: 5, window_end: 10}`, then asserts only:

  ```elixir
  assert length(result.spots) == 2
  assert is_float(result.total_lift)
  assert is_float(result.total_lift_sd)
  assert is_list(result.overlaps)
  assert is_map(result.causal_impact_summary)
  ```

  The regressor variant (~lines 62–64) asserts `is_float(result.total_lift)`
  alone. Contrast the deterministic `attribute_from_baseline/4` test
  (~line 131), which does assert numerically:
  `assert_in_delta attr.lift, expected_lift, 0.01`.

- Same pattern (shape-only on MCMC-path outputs, alongside stronger tests
  elsewhere in the same files):
  `test/bsts_nx/applications/policy_evaluator_test.exs` (~31–34, 131–136,
  242–243), `anomaly_detector_test.exs` (~15, 24–27, 291–292),
  `marketing_lift_test.exs` (~27–30, 88).

- All these tests run with fixed `:rand.seed(:exsss, {...})` + `seed:` opts
  and small sampler params (`num_samples: 10, burn_in: 5`) — outputs are
  **deterministic**, so tightened assertions cannot flake; they can only
  break when behavior changes (which is the goal).

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| The four files | `mix test test/bsts_nx/applications/` | 0 failures |
| Full verify | `bash scripts/ci.sh` | exit 0 |

Tooling note: `mix` is a `mise` shim; prefix `mise exec -- ` if needed.

## Scope

**In scope** (test files only):
- `test/bsts_nx/applications/tv_attribution_test.exs`
- `test/bsts_nx/applications/policy_evaluator_test.exs`
- `test/bsts_nx/applications/anomaly_detector_test.exs`
- `test/bsts_nx/applications/marketing_lift_test.exs`

**Out of scope**:
- Any file under `lib/` — if an assertion exposes a wrong estimate, that's a
  STOP-and-report, not a fix here.
- Deleting or weakening any existing assertion.
- The `:external`-tagged statistical suites (plan 020's territory).

## Git workflow

- Branch: `advisor/029-strengthen-bayesian-assertions` (from `execute-plans`).
- Commit style: `test: planted-truth recovery assertions for Bayesian effect outputs`
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: The assertion recipe (read first, apply everywhere)

For each shape-only assertion site, derive the **planted truth** from the
test's own data construction (e.g. TVAttribution: post ≈ 1200 vs pre ≈ 1000
baseline ⇒ planted per-step lift ≈ 200, total ≈ 200 × 10 = 2000), then add:

1. **Sign**: `assert result.total_lift > 0` (or the planted direction).
2. **Magnitude bracket**: use a sign-safe absolute-error window centered on
   the planted value — for example,
   `tolerance = max(abs(planted) * 2.0, 1.0)` followed by
   `assert_in_delta result.total_lift, planted, tolerance` — wide enough for a
   10-sample MCMC estimate, valid for positive and negative truths, and tight enough to catch
   zeroing/sign/scale bugs. With fixed seeds, this is deterministic: run the
   test, confirm the actual value sits comfortably inside the window
   (record actual values in your report), and if it sits near an edge widen
   to the next factor rather than nudging around the observed number (the
   bracket must derive from the planted truth, NOT from the observed output —
   no change-detector tests).
3. **Finiteness**: `assert result.total_lift == result.total_lift` (NaN guard)
   where relevant.

If a planted truth genuinely cannot be derived from the setup (pure-noise
scenario), assert sign/finiteness only and note it.

### Step 2: TVAttribution

- Main `attribute/3` test: apply the recipe to `total_lift`; additionally
  assert the overlap content:
  `assert [%{} | _] = result.overlaps` and that the overlapping pair
  (spot_1/spot_2 — windows 0–5 and 5–10 share index 5) appears — read
  `ShapleyAllocator.detect_overlaps/1`'s return shape first and assert the
  ids it reports. Also `assert result.total_lift_sd > 0`.
- Regressor-variant test: same recipe on `total_lift`.

**Verify**: `mix test test/bsts_nx/applications/tv_attribution_test.exs` → 0
failures.

### Step 3: PolicyEvaluator, AnomalyDetector, MarketingLift

Apply the recipe at each cited site:

- `policy_evaluator_test.exs`: the effect/slope outputs at ~31–34, 131–136,
  242–243 — the setups plant an intervention effect; assert its sign and
  bracket. For slope diagnostics, assert the slope's sign matches the
  constructed trend.
- `anomaly_detector_test.exs`: at ~15, 24–27, 291–292 — where an anomaly is
  planted, assert the anomalous point's score exceeds the non-anomalous
  points' scores (rank assertion — robust and seed-stable), plus z-threshold
  behavior already implied by the setup.
- `marketing_lift_test.exs`: at ~27–30, 88 — planted campaign lift: sign +
  bracket on `result.effect.cumulative`.

**Verify** after each file: `mix test <file>` → 0 failures.

### Step 4: Full pass

**Verify**: `bash scripts/ci.sh` → exit 0. Report a table: assertion site →
planted truth → observed value → bracket used.

## Test plan

This plan is entirely test strengthening; the table in Step 4 is the
deliverable evidence. No new files — assertions land inside the existing
tests they strengthen.

## Done criteria

- [ ] Every cited shape-only site now also asserts sign + planted-truth
      bracket (or a documented rank/sign-only fallback)
- [ ] `grep -n "is_float(result.total_lift)" test/bsts_nx/applications/tv_attribution_test.exs`
      still matches (kept) but is no longer the *only* assertion on it
- [ ] `mix test test/bsts_nx/applications/` → 0 failures
- [ ] `bash scripts/ci.sh` exits 0
- [ ] Report contains the planted-vs-observed table
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- **An estimate falls outside a generously-derived bracket** — e.g.
  TVAttribution's total_lift comes back ≤0 or an order of magnitude off the
  planted ~2000. That is a real accuracy finding (exactly what the maintainer
  asked to protect); report the numbers, do not widen the bracket to make it
  pass.
- The planted truth is ambiguous for a setup (you can't defend a bracket from
  the data construction) — use sign-only and flag it, or ask.
- Plans 017/018 haven't landed and your edits collide with theirs in
  `marketing_lift_test.exs`/`policy_evaluator_test.exs` — rebase order
  matters; do those files last or after those plans merge.

## Maintenance notes

- The brackets are wide by design (0.3×–3×); they are bug-catchers, not
  calibration checks — calibration lives in the `:slow`/`:external`
  statistical lane (plan 020). Reviewers should reject tightening these into
  flaky precision tests.
- When sampler defaults change (e.g. after plan 021/022 performance work
  allows more samples in tests), brackets can tighten — revisit then.
