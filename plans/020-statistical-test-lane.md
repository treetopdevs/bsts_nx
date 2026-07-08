# Plan 020: Make the statistical-correctness test suite actually run (retag pure-Elixir `:external` tests, add a scheduled CI lane)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat 9b7cb8d..HEAD -- test/parameter_recovery_test.exs test/diagnostics_test.exs test/components_reproducibility_test.exs test/bsts_nx/rolling_baseline_regression_test.exs test/bsts_nx/rolling_baseline_warm_start_test.exs test/test_helper.exs .github/workflows/ci.yml CLAUDE.md`
> Plans 015/016 are EXPECTED to have changed `ci.yml`; that is not drift. Any
> change to the test files: compare the "Current state" excerpts against the
> live code before proceeding; on a mismatch, treat it as a STOP condition.

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: LOW (test/CI-config only; no production code)
- **Depends on**: plans/016-ci-hardening-and-toolchain.md (same `ci.yml`; copy its hardened job patterns). Can run before 016 if needed — then use the current unhardened job style and note it.
- **Category**: tests
- **Planned at**: commit `9b7cb8d`, 2026-07-08

## Why this matters

The library's deepest correctness guarantees — "the Gibbs sampler recovers the
true variances", "diagnostics math is right", "components are reproducible",
"rolling-baseline regression behavior holds" — live in test files tagged
`@moduletag :external`, even though they are **pure Elixir with no external
tool dependency**. The merge-gating CI job runs `mix test --exclude external`;
the R-parity job is manual-dispatch-only; the EXLA job runs two named smoke
files. Net effect: the statistical layer runs in **zero** automated lanes, and
`parameter_recovery_test.exs` (additionally skip-tagged on EMLX) may never run
anywhere at all. A regression that makes the sampler produce wrong posteriors
ships with green CI.

The `:external` tag should mean "needs an external runtime (R/Python)". Pure
but slow statistical tests belong under `:slow` (already excluded from the
default gate by `test_helper.exs`, so local ergonomics don't change) plus a
**scheduled nightly CI lane** that runs them; pure and *fast* ones belong in
the default gate with no tag at all.

## Current state

- `.github/workflows/ci.yml` — default `test` job runs
  `mix test --exclude external` (line ~62); `test_exla` runs two named files;
  `r_parity` is `workflow_dispatch`-gated. No `schedule:` trigger exists.
- `test/test_helper.exs` — both `ExUnit.start` branches exclude `:slow`
  (lines 36 and 38); `:external` is excluded only via the CI flag /
  `mix test --exclude external`.
- The mis-tagged pure-Elixir files (all read and verified to contain no
  `System.cmd`/`Port.open`/shell-out):
  - `test/parameter_recovery_test.exs` — `@moduletag :external`,
    `@moduletag timeout: 300_000`, plus a conditional `@tag skip: "...too slow
    on EMLX..."` when the default backend is EMLX. Content: local-linear-trend
    sampler covers true variances with 95% HPD intervals.
  - `test/diagnostics_test.exs` — `@moduletag :external` (line 5). Content:
    R-hat / ESS on small fake chains (looks fast).
  - `test/components_reproducibility_test.exs` — `@moduletag :external`
    (line 7). Content: seeded reproducibility of component samplers.
  - `test/bsts_nx/rolling_baseline_regression_test.exs` — `@moduletag
    :external` (line 8).
  - `test/bsts_nx/rolling_baseline_warm_start_test.exs` — `@moduletag
    :external` (line 6).
- Legitimately-`:external` files that must NOT be retagged (they shell out or
  need external runtimes): `test/r_parity_test.exs`, `test/r_sidecar_test.exs`,
  `test/python_comparison_test.exs`. Before finalizing, enumerate every file
  carrying `:external` (`grep -rln "moduletag :external" test/`) and classify
  each: shell-out/external-runtime → keep `:external`; pure Elixir → retag per
  Step 2. Files not named in this plan get classified by the same rule.
- `test/bsts_nx/rolling_baseline_test.exs` — has per-test `@tag :slow` on 8
  tests; already correctly tagged; the nightly lane will pick these up
  automatically.
- `CLAUDE.md` documents the test lanes ("Run full test suite (excludes
  `@tag :slow` by default)", `mix test --only external`, etc.).
- Known and NOT in scope here: the `CausalImpact.estimate/4` default-gate
  smoke test exists on branch `advisor/008-*` awaiting cherry-pick
  (plan 008).

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Time one file | `time mix test test/diagnostics_test.exs --include external` | tests run (not "0 tests"), wall time printed |
| Run the future slow lane locally | `mix test --only slow` | the retagged + existing slow tests run, 0 failures |
| Default gate unchanged | `mix test --exclude external` | 0 failures, same-or-more tests than before |
| YAML validity | `ruby -ryaml -e 'YAML.load_file(".github/workflows/ci.yml"); puts "ok"'` | prints `ok` |
| Full verify | `bash scripts/ci.sh` | exit 0 |

Tooling note: `mix` is a `mise` shim here; prefix `mise exec -- ` if needed.

## Scope

**In scope** (the only files you should modify):
- `test/parameter_recovery_test.exs`, `test/diagnostics_test.exs`,
  `test/components_reproducibility_test.exs`,
  `test/bsts_nx/rolling_baseline_regression_test.exs`,
  `test/bsts_nx/rolling_baseline_warm_start_test.exs` (tag changes only — do
  not alter test bodies or assertions)
- Any additional file the Step 1 classification proves is pure-Elixir but
  `:external`-tagged (tag change only; list them in your report)
- `.github/workflows/ci.yml` (one new scheduled job + trigger)
- `CLAUDE.md` (lane documentation update)

**Out of scope** (do NOT touch):
- `test/r_parity_test.exs`, `test/r_sidecar_test.exs`,
  `test/python_comparison_test.exs` — genuinely external.
- `test/test_helper.exs` — the `:slow` default exclusion is correct and
  relied on locally; leave it.
- Test assertion bodies — retagging only. Strengthening type-only assertions
  is a separately recorded finding (TEST-04/07).
- `scripts/ci.sh` — the nightly lane is CI-only by design (it's slow); local
  users opt in via `mix test --only slow`.

## Git workflow

- Branch: `advisor/020-statistical-test-lane` (from `execute-plans`, after
  016 if possible).
- Commit style: `test: move pure-Elixir statistical suites to :slow, add nightly CI lane`
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Classify and time every `:external` file

1. `grep -rln "moduletag :external" test/` → enumerate.
2. For each file: `grep -n "System.cmd\|Port.open\|rscript\|python" <file>`
   (case-insensitive) and skim — classify as *external-runtime* (keep) or
   *pure* (retag).
3. For each *pure* file, measure:
   `time mix test <file> --include external --include slow` and record wall
   time. (For `parameter_recovery_test.exs` expect minutes — its own timeout
   allows 300 s.)

**Verify**: you have a written classification table (goes in your final
report) covering every `:external` file, with timings for the pure ones.

### Step 2: Retag the pure files

Decision rule per pure file, applied to its measured time:
- **< 10 s** → delete the `@moduletag :external` line entirely (it joins the
  default merge gate).
- **≥ 10 s** → replace `@moduletag :external` with `@moduletag :slow`.

Expected outcome given planning-time reads (confirm with your own timings):
`diagnostics_test.exs` and probably `components_reproducibility_test.exs` go
untagged into the default gate; `parameter_recovery_test.exs` and the two
rolling-baseline files become `:slow`. Keep `@moduletag timeout: 300_000` and
the EMLX skip-tag in `parameter_recovery_test.exs` exactly as they are.

**Verify**:
- `mix test --exclude external` → 0 failures, and the count of tests **rose**
  by the newly-untagged fast tests (compare against a pre-change run).
- `mix test --only slow` → the retagged files + the 8 pre-existing
  `rolling_baseline_test.exs` slow tests all run, 0 failures. Record total
  wall time — the nightly job's timeout needs it.

### Step 3: Add the scheduled nightly lane to `ci.yml`

Add a `schedule` trigger and a job (copy the hardened patterns from plan 016
if it has landed — pinned SHAs, `version-file: .tool-versions`, deps+`_build`
cache; otherwise mirror the current `test` job style):

```yaml
on:
  push:
    ...existing...
  pull_request:
  workflow_dispatch:
    ...existing...
  schedule:
    - cron: "17 6 * * *"   # nightly, 06:17 UTC

jobs:
  ...existing jobs...

  slow_statistical:
    name: Nightly statistical suite
    runs-on: ubuntu-latest
    if: ${{ github.event_name == 'schedule' || github.event_name == 'workflow_dispatch' }}
    timeout-minutes: 45   # adjust: ~3x the wall time measured in Step 2
    env:
      MIX_ENV: test
    steps:
      # checkout / setup-beam / cache / hex+rebar / deps.get — same as the test job
      - name: Run slow statistical tests
        run: mix test --only slow
```

Notes:
- `if:` keeps the job out of every push/PR run (it would otherwise also fire
  there since job-level triggers don't exist); `workflow_dispatch` inclusion
  lets a human run it on demand.
- `mix test --only slow` implies `--include slow`, overriding the
  `test_helper.exs` default exclusion — verified locally in Step 2.
- Scheduled workflows only run from the **default branch** (`main`); until
  this lands on main, the lane is dormant. Say so in your report.

**Verify**: `ruby -ryaml -e 'YAML.load_file(".github/workflows/ci.yml"); puts "ok"'`
→ `ok`; `grep -c "cron:" .github/workflows/ci.yml` → 1.

### Step 4: Update `CLAUDE.md` lane documentation

In the "Build & Test Commands" section, adjust the lane description to the new
reality, e.g. add/update lines:

```bash
mix test --only slow                # Pure-Elixir statistical suite (nightly CI lane)
mix test --only external            # Tests needing external runtimes (R/Python)
```

and make sure the comment on `mix test` still says it excludes `:slow` by
default (unchanged behavior).

**Verify**: `grep -n "only slow" CLAUDE.md` → ≥1 match.

### Step 5: Full verification

**Verify**: `bash scripts/ci.sh` → exit 0 (the default gate must not have
gotten slower than the team tolerates — if the newly-untagged fast tests add
more than ~30 s to the local run, reconsider their classification and say so).

## Test plan

No new tests — this plan moves existing tests into lanes that execute. The
verification IS the moved tests running: Step 2's two `mix test` invocations,
plus the timing table in the report. A reviewer should `workflow_dispatch` the
`slow_statistical` job after merge to main and confirm green.

## Done criteria

- [ ] `grep -rln "moduletag :external" test/` returns ONLY files that
      genuinely shell out to R/Python (verified in the Step 1 table)
- [ ] `mix test --only slow` → 0 failures, includes parameter recovery + the
      rolling-baseline slow/regression/warm-start tests
- [ ] `mix test --exclude external` → 0 failures, test count increased vs
      pre-change
- [ ] `ci.yml` has a `schedule` trigger and a `slow_statistical` job gated to
      schedule/dispatch; YAML parses
- [ ] `CLAUDE.md` documents the `--only slow` lane
- [ ] `bash scripts/ci.sh` exits 0
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- Any test **fails** when you first run it in Step 1 — these suites haven't
  run in CI for a while; a pre-existing failure is a real finding, not
  something to fix or skip silently. Report file, test name, and output.
- A "pure" classification is ambiguous (e.g. a file conditionally shells out
  based on env vars) — report it rather than guessing.
- `parameter_recovery_test.exs` exceeds its own 300 s timeout on this machine
  — report the timing; the nightly lane may need EXLA (a follow-up, not an
  improvisation).
- `mix test --only slow` runs 0 tests (tag-interaction surprise with
  `test_helper.exs`) — the lane design assumption is wrong; stop.

## Maintenance notes

- Tag semantics after this plan: `:external` = needs external runtime;
  `:slow` = pure but heavy, nightly lane; untagged = merge gate. New
  statistical tests should follow this rule — reviewers should reject new
  `:external` tags on pure-Elixir tests.
- The nightly lane only protects `main`; failures arrive by scheduled-run
  email/UI, not PR checks. If the team wants pre-merge protection for the
  statistical layer, promote a small planted-truth subset into the default
  gate later (see recorded findings TEST-04/TEST-07 about strengthening
  assertions).
- If plan 016 lands after this one, extend its hardening (SHA pins,
  version-file, `_build` cache) to the `slow_statistical` job too.
