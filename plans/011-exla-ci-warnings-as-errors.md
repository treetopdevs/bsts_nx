# Plan 011: Align the EXLA CI job's compile flags with the main test job

> **Executor instructions**: Follow this plan step by step. This plan changes CI
> configuration whose effect is only fully observable when GitHub Actions runs the
> EXLA lane. Do the local investigation in Step 1 FIRST and let its result decide
> Step 2 — do not blindly flip the flag. If anything in "STOP conditions" occurs,
> stop and report. When done, update the status row in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat e4654c5..HEAD -- .github/workflows/ci.yml`
> If `ci.yml` changed since this plan was written, compare the "Current state"
> excerpts below against the live file before proceeding; on a mismatch, STOP.

## Status

- **Priority**: P3
- **Effort**: S
- **Risk**: MED — adding `--warnings-as-errors` to the EXLA lane could turn
  pre-existing EXLA-only compile warnings into CI failures.
- **Depends on**: none
- **Category**: dx
- **Planned at**: commit `e4654c5`, 2026-06-30

## Why this matters

The main `test` job compiles with warnings treated as errors, but the EXLA job does
not:

- `test` job: `mix compile --warnings-as-errors`
- `test_exla` job: `mix compile` (no flag)

So a warning that only appears when compiling against the EXLA backend (or any
warning newly introduced in code first exercised on the EXLA lane) slips through CI.
The fix is either to add the flag (if the EXLA compile is already warning-clean) or
to document why it is intentionally omitted — but the current state does neither,
which is just an unexplained inconsistency. This plan resolves it **based on
evidence**, not assumption.

## Current state

`.github/workflows/ci.yml` — the main `test` job's compile step:

```yaml
      - name: Compile
        run: mix compile --warnings-as-errors

      - name: Run tests
        run: mix test --exclude external
```

The `test_exla` job's compile step (the inconsistency):

```yaml
      - name: Compile
        run: mix compile

      - name: Run EXLA smoke tests
        run: mix test test/structured_performance_smoke_test.exs test/utils_safe_solve_test.exs
```

The `test_exla` job sets `env: BSTS_NX_TEST_BACKEND: exla` and `MIX_ENV: test`.

## Commands you will need

| Purpose | Command | Expected |
|---------|---------|----------|
| EXLA-backed clean compile (investigation) | `BSTS_NX_TEST_BACKEND=exla MIX_ENV=test mix deps.get && BSTS_NX_TEST_BACKEND=exla MIX_ENV=test mix compile --warnings-as-errors` | exit 0, **no warnings** — or a list of warnings |
| Confirm EXLA is usable locally at all | `MIX_ENV=test mix deps.get` then check `deps/exla` exists and compiles | EXLA fetched/compiled, or a clear failure |
| YAML sanity (no parser available — visual) | `grep -n "warnings-as-errors" .github/workflows/ci.yml` | 2 matches after the edit (was 1) |

(If local `mix` errors with a `mise exec` usage message, prefix with `mise exec -- `.
EXLA downloads/builds a large native XLA artifact; this can take minutes or fail on
constrained machines — that failure mode is handled by Step 1's escape hatch.)

## Scope

**In scope** (the only file you may modify):
- `.github/workflows/ci.yml` — the single compile step inside the `test_exla` job.

**Out of scope** (do NOT touch):
- The `test`, `quality`, and `r_parity` jobs.
- Any `lib/` code. If the EXLA compile surfaces real warnings, **fixing them is a
  separate plan** — do not edit source under this plan (STOP and report instead).
- `scripts/ci.sh` — it intentionally does not run the EXLA lane.

## Git workflow

- Branch: `advisor/011-exla-ci-warnings-as-errors`
- Commit message: `ci: treat warnings as errors in the EXLA compile job`
  (or `ci: document why the EXLA job omits warnings-as-errors`, per Step 2's branch).
- Do NOT push or open a PR unless instructed.

## Steps

### Step 1: Investigate — does the EXLA compile produce warnings?

Run the EXLA-backed warnings-as-errors compile locally:

`BSTS_NX_TEST_BACKEND=exla MIX_ENV=test mix deps.get`
`BSTS_NX_TEST_BACKEND=exla MIX_ENV=test mix compile --warnings-as-errors`

Three possible outcomes:

- **(A) Exits 0, no warnings** → the flag is safe. Go to Step 2A.
- **(B) Fails with warnings-as-errors** → capture the warning text. Go to Step 2B.
- **(C) EXLA cannot be fetched/compiled on this machine** (XLA download failure,
  unsupported platform) → you cannot verify locally. STOP and report outcome (C)
  with the error; recommend the operator apply Step 2A and let CI verify, or that a
  maintainer with EXLA runs the investigation. Do not guess.

**Verify**: you have a definitive (A), (B), or (C) result recorded.

### Step 2A: EXLA compile is clean — add the flag

Edit the `test_exla` job's compile step so it reads:

```yaml
      - name: Compile
        run: mix compile --warnings-as-errors
```

**Verify**: `grep -n "warnings-as-errors" .github/workflows/ci.yml` → exactly 2
matches (the original `test` job + this one). Confirm the change is inside the
`test_exla` job (the one with `BSTS_NX_TEST_BACKEND: exla`), not duplicated into
another job.

Skip Step 2B.

### Step 2B: EXLA compile warns — document instead of breaking CI

Do NOT add the flag (it would make CI red). Instead add an explanatory comment so
the omission is intentional and reviewable:

```yaml
      # NOTE: --warnings-as-errors is intentionally omitted here: the EXLA backend
      # currently emits compile warnings (see plan 011 investigation). Re-enable
      # once those are resolved in a dedicated fix.
      - name: Compile
        run: mix compile
```

Then STOP and report the captured warning list so a follow-up plan can address the
warnings themselves.

**Verify**: the comment is present immediately above the EXLA `Compile` step.

### Step 3: Confirm no other job was affected

`git diff .github/workflows/ci.yml`

**Verify**: the diff touches only the `test_exla` job's compile step (and its comment
in the 2B case). No change to `test`, `quality`, or `r_parity`.

## Test plan

- This is CI configuration; there is no unit test. Verification is:
  - The local EXLA compile result from Step 1 (the evidence the change is based on).
  - `grep` confirming the YAML edit landed in the correct job (Step 2A) or the
    documenting comment is present (Step 2B).
  - The authoritative check is the GitHub Actions `test_exla` job on the PR — the
    operator confirms it stays green after this change.

## Done criteria

Exactly ONE of these branches must hold (record which in the README status note):

- **2A path**: EXLA compile is clean locally; `.github/workflows/ci.yml`'s
  `test_exla` job runs `mix compile --warnings-as-errors`;
  `grep -c "warnings-as-errors" .github/workflows/ci.yml` == 2.
- **2B path**: EXLA compile warns; the EXLA `Compile` step is left as `mix compile`
  with an explanatory comment, and the warning list is reported for follow-up.

In all cases:

- [ ] Only `.github/workflows/ci.yml` is modified (`git status`).
- [ ] No `lib/` source changed.
- [ ] `plans/README.md` status row for 011 updated (DONE for 2A; BLOCKED-with-reason
      pointing at the follow-up for 2B).

## STOP conditions

Stop and report (do not improvise) if:

- `ci.yml` doesn't match the "Current state" excerpts (drift).
- Outcome (C): EXLA cannot be built locally, so the change cannot be verified — report
  and let the operator/CI decide.
- The EXLA compile surfaces warnings (outcome B): apply Step 2B and report the
  warnings; do NOT fix source code under this plan.

## Maintenance notes

- If the EXLA warnings are fixed later, flip the documented `mix compile` back to
  `mix compile --warnings-as-errors` and remove the NOTE comment.
- Reviewer should confirm the EXLA `test_exla` job is actually green on the PR run,
  not just that the YAML looks right.
