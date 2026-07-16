# Plan 017: Fix `MarketingLift.measure_multi/3` crash on overlapping campaigns with staggered baselines

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat 9b7cb8d..HEAD -- lib/bsts_nx/applications/marketing_lift.ex lib/bsts_nx/validation.ex lib/bsts_nx/intervention_analysis.ex test/bsts_nx/applications/marketing_lift_test.exs`
> If any of these changed since this plan was written, compare the "Current
> state" excerpts against the live code before proceeding; on a mismatch,
> treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED (changes the merged analysis window for overlapping-campaign groups; attribution numbers for that path will shift)
- **Depends on**: none
- **Category**: bug
- **Planned at**: commit `9b7cb8d`, 2026-07-08

## Why this matters

`BstsNx.Applications.MarketingLift.measure_multi/3` is a public API for
measuring incremental lift across multiple campaigns, with special handling
for overlapping campaigns. For an overlap group it merges the campaigns into a
single analysis window — but it builds that window incorrectly: the merged
pre-period end is the **maximum** `baseline_end` across the group, while the
merged post-period start is the **minimum** `start_index`. Whenever any
campaign's baseline extends past the earliest campaign's start (the normal
shape for staggered launches), the merged window violates the library's own
period validation (`post_start <= pre_end`) and the call raises
`ArgumentError "post_period must satisfy pre_end < start <= end"`.

The module's **own moduledoc example** triggers the crash: `podcast_a`
(start 10, baseline_end 9) and `podcast_b` (start 12, baseline_end 11) overlap
→ merged pre `{1, 11}`, merged post `{10, 18}` → `10 <= 11` → raise. Existing
tests pass only because they use identical baselines that end before both
campaign starts.

## Current state

- `lib/bsts_nx/applications/marketing_lift.ex` — the buggy merge, lines
  265–281:

  ```elixir
  defp measure_overlapping_group(observations, group, opts) do
    # For overlapping campaigns, find the earliest baseline and latest campaign end,
    # run a single CausalImpact analysis, then use SpotAttributor to allocate
    earliest_baseline = Enum.min_by(group, & &1.baseline_start).baseline_start
    latest_baseline_end = Enum.max_by(group, & &1.baseline_end).baseline_end
    earliest_campaign = Enum.min_by(group, & &1.start_index).start_index
    latest_campaign = Enum.max_by(group, & &1.end_index).end_index

    config = %{
      pre_period: {earliest_baseline, latest_baseline_end},
      post_period: {earliest_campaign, latest_campaign}
    }

    # Use the first campaign's channel for analysis opts
    first = hd(group)
    analysis_opts = build_analysis_opts(observations, first, opts)
    analysis = InterventionAnalysis.analyze(observations, config, analysis_opts)
    ...
  ```

- The moduledoc example that crashes (same file, ~lines 56–61):

  ```elixir
  campaigns = [
    %{name: "podcast_a", start_index: 10, end_index: 15, baseline_start: 1, baseline_end: 9},
    %{name: "podcast_b", start_index: 12, end_index: 18, baseline_start: 1, baseline_end: 11}
  ]

  results = MarketingLift.measure_multi(daily_visits, campaigns)
  ```

  (This example is prose inside the moduledoc, not a doctest, which is why the
  doctest suite doesn't catch it.)

- The validation that (correctly) rejects the merged window —
  `lib/bsts_nx/validation.ex`, in `validate_study_periods!`:

  ```elixir
  if post_start <= pre_end or post_end < post_start do
    raise ArgumentError, "post_period must satisfy pre_end < start <= end"
  end
  ```

  called unconditionally from `lib/bsts_nx/intervention_analysis.ex` (~line
  195) before any mode dispatch, so **both** `:bayesian` and `:operational`
  modes crash.

- The existing multi-campaign test that masks the bug —
  `test/bsts_nx/applications/marketing_lift_test.exs` (~lines 145–164): both
  campaigns use `baseline_start: 1, baseline_end: 20` with starts 21/24, so
  the merged window is valid there.

- Grouping: `measure_multi/3` (lines 164–200) partitions campaigns via
  `detect_campaign_overlaps/1`; singleton groups go through `measure_lift/3`
  (per-campaign windows — unaffected by this bug); groups of ≥2 go through
  `measure_overlapping_group/3`.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Reproduce the bug (before fix) | see Step 1 script | `ArgumentError` raised |
| This module's tests | `mix test test/bsts_nx/applications/marketing_lift_test.exs` | all pass |
| Full library verify | `bash scripts/ci.sh` | exit 0 |

Tooling note: `mix` on this machine is a `mise` shim; prefix with
`mise exec -- ` if a plain invocation misbehaves.

## Scope

**In scope** (the only files you should modify):
- `lib/bsts_nx/applications/marketing_lift.ex` (the `measure_overlapping_group/3`
  merge logic and, if you add a clarifying sentence, the moduledoc)
- `test/bsts_nx/applications/marketing_lift_test.exs` (regression tests)

**Out of scope** (do NOT touch):
- `lib/bsts_nx/validation.ex` and `lib/bsts_nx/intervention_analysis.ex` — the
  validation is correct; do not loosen it to accommodate the bad window.
- `lib/bsts_nx/spot_attributor.ex` — downstream allocation is not the bug.
- `measure_lift/3` (single-campaign path) — its windows come straight from the
  campaign and are already validated correctly.
- `lib/bsts_nx/applications/policy_evaluator.ex` — has its own separate issues
  covered by plan 018.

## Git workflow

- Branch: `advisor/017-marketing-lift-overlap-crash` (from `execute-plans`).
- Commit style: `fix: clamp merged pre-period for overlapping marketing campaigns`
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Reproduce the crash with the moduledoc example

Run:

```bash
mix run -e '
obs = Enum.map(1..20, fn i -> 100.0 + :math.fmod(i * 7.3, 5.0) end)
campaigns = [
  %{name: "podcast_a", channel: :podcast, start_index: 10, end_index: 15, baseline_start: 1, baseline_end: 9},
  %{name: "podcast_b", channel: :podcast, start_index: 12, end_index: 18, baseline_start: 1, baseline_end: 11}
]
BstsNx.Applications.MarketingLift.measure_multi(obs, campaigns, num_samples: 10, burn_in: 5, seed: 1)
'
```

**Verify**: it raises `ArgumentError` with message
`post_period must satisfy pre_end < start <= end`. (If it does NOT raise, the
code has drifted — STOP.)

### Step 2: Clamp the merged pre-period in `measure_overlapping_group/3`

Replace the window construction so the merged pre-period always ends strictly
before the earliest campaign start:

```elixir
    earliest_baseline = Enum.min_by(group, & &1.baseline_start).baseline_start
    latest_baseline_end = Enum.max_by(group, & &1.baseline_end).baseline_end
    earliest_campaign = Enum.min_by(group, & &1.start_index).start_index
    latest_campaign = Enum.max_by(group, & &1.end_index).end_index

    # The merged baseline may extend past the earliest campaign start when
    # baselines are staggered; clamp so the pre-period never overlaps the
    # post-period (validate_study_periods! requires pre_end < post_start).
    merged_pre_end = min(latest_baseline_end, earliest_campaign - 1)

    if merged_pre_end < earliest_baseline do
      raise ArgumentError,
            "overlapping campaign group has no valid shared baseline: " <>
              "earliest baseline_start (#{earliest_baseline}) is not before " <>
              "the earliest campaign start_index (#{earliest_campaign})"
    end

    config = %{
      pre_period: {earliest_baseline, merged_pre_end},
      post_period: {earliest_campaign, latest_campaign}
    }
```

Design notes for correctness:
- Clamping (not shifting the post start) is the right direction: observations
  from `earliest_campaign` onward are potentially treated, so they must not
  contaminate the counterfactual's training window.
- The explicit raise handles the degenerate case where a group's earliest
  baseline starts at/after the earliest campaign (no clean pre-window exists);
  a clear message beats the opaque validator error.

**Verify**: `mix compile --warnings-as-errors` → exit 0.

### Step 3: Confirm the reproducer now succeeds

Re-run the Step 1 script.

**Verify**: no raise; the returned map has `campaigns` with 2 entries (names
`"podcast_a"`, `"podcast_b"`), a numeric `total_lift`, and
`overlap_groups == [["podcast_a", "podcast_b"]]`.

### Step 4: Add regression tests

In `test/bsts_nx/applications/marketing_lift_test.exs`, model the structure on
the existing overlapping-campaign test at ~lines 140–170 (seeded `:rand`
series, `num_samples: 10, burn_in: 5, seed: 42` for speed). Add:

1. **`measure_multi handles overlapping campaigns with staggered baselines`**
   — two campaigns where the later campaign's `baseline_end` is ≥ the earlier
   campaign's `start_index` (use the moduledoc example's index shape scaled
   onto a ~40-point series so post windows fit). Assert: no raise; 2 campaign
   results with the right names; `is_float(result.total_lift)`;
   `overlap_groups` contains both names; **and** each campaign result's
   `effect.cumulative` is finite (`is_float` and not NaN — `r.effect.cumulative == r.effect.cumulative`).
2. **`measure_multi raises a clear error when a group has no valid shared baseline`**
   — two overlapping campaigns whose `baseline_start` is at/after the earliest
   `start_index` (e.g. baseline_start 12 for a campaign starting at 10).
   Assert `assert_raise ArgumentError, ~r/no valid shared baseline/, fn -> ... end`.

**Verify**: `mix test test/bsts_nx/applications/marketing_lift_test.exs` → all
pass, including 2 new tests.

### Step 5: Full verification

**Verify**: `bash scripts/ci.sh` → exit 0 (compile with warnings-as-errors,
full non-external suite, format check, docs build).

## Test plan

Covered by Step 4: the staggered-baseline regression (the bug this plan
fixes), the degenerate no-baseline group (the new explicit raise), plus the
existing suite guarding the identical-baseline group path and singleton path.
Pattern to follow: the existing `measure_multi` test at
`test/bsts_nx/applications/marketing_lift_test.exs:140-170` (small
`num_samples`, fixed `seed`, deterministic `:rand`-seeded series).

## Done criteria

- [ ] Step 1 reproducer script runs without raising
- [ ] `mix test test/bsts_nx/applications/marketing_lift_test.exs` → 0 failures, ≥2 new tests present
- [ ] `grep -n "merged_pre_end" lib/bsts_nx/applications/marketing_lift.ex` → at least one match
- [ ] `bash scripts/ci.sh` exits 0
- [ ] `git status --porcelain` shows only the two in-scope files modified
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- The Step 1 reproducer does NOT raise before the fix (code drifted; the bug
  may already be fixed another way).
- After the fix, existing tests fail because they asserted exact numeric
  effects on the overlapping-group path (the clamp legitimately changes the
  fitted window). Do not weaken assertions silently — report which test and
  the old/new values so a human can confirm the shift is expected.
- The fix appears to require changes to `InterventionAnalysis` or
  `Validation` — that means the approach is wrong; stop.
- You find `measure_overlapping_group/3` no longer merges windows at all
  (refactored since planning).

## Maintenance notes

- Attribution numbers for staggered-baseline overlap groups are produced from
  a shorter pre-period than the (crashing) code nominally intended. Reviewers
  should sanity-check one worked example rather than diff numbers against the
  old path (which never returned).
- If per-campaign baselines diverge wildly, the merged-window approach itself
  is questionable (one shared counterfactual for the whole group); the
  fallback per-campaign path already exists inside
  `measure_overlapping_group/3` for `analysis.impact == nil`. A future
  improvement could expose a `:group_strategy` option — deliberately out of
  scope here.
- The moduledoc example is prose, not a doctest. If someone later converts
  module examples to doctests, this one will start executing — the Step 4
  tests already cover its shape.
