# Plan 005: Untrack the stale `optimize-plan.md` planning artifact and remove the crash dump

> **Executor instructions**: Follow this plan step by step. Run every verification
> command and confirm the expected result before moving to the next step. If
> anything in the "STOP conditions" section occurs, stop and report — do not
> improvise. When done, update the status row for this plan in `plans/README.md`.
>
> **Drift check (run first)**: `git ls-files | grep -E 'optimize-plan|erl_crash'`
> If `optimize-plan.md` is no longer tracked, this plan is already partly done —
> re-read "Current state" and adjust. If unexpected files appear, STOP.

## Status

- **Priority**: P3
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: dx
- **Planned at**: commit `e4654c5`, 2026-06-20

## Why this matters

`optimize-plan.md` is a ~30 KB development-planning artifact tracked in version
control at the repo root. It is **not** in the Hex package `files:` list (so it
does not ship to users), but it bloats the repo, reads as a source of truth that
has drifted from the actual code, and creates "is this still accurate?" friction
for contributors and agents. Separately, a 1.5 MB `erl_crash.dump` sits in the
working tree (it is already gitignored and untracked, so it just needs deleting
locally). Removing both reduces clutter and removes a misleading status document
from the tree without losing the content on disk.

## Current state

- `optimize-plan.md` — tracked by git (`git ls-files` lists it), repo root, ~30 KB.
  `.gitignore` does **not** mention it.
- `erl_crash.dump` — present in the working tree (~1.5 MB), **not** tracked
  (`.gitignore` line 17 already ignores `erl_crash.dump`).
- Current `.gitignore` tail (relevant region):

```
.expert
/.claude
docs/~$
/docs/~$VPV.docx
/logs
/.specstory
bench/results/current.json
```

## Commands you will need

| Purpose | Command | Expected |
|---------|---------|----------|
| Check tracking | `git ls-files \| grep -E 'optimize-plan\|erl_crash'` | only `optimize-plan.md` before, nothing after |
| Check references | `git grep -n "optimize-plan.md"` | (used in Step 1) |
| Compile (sanity) | `mix compile --warnings-as-errors` | exit 0 |
| Full gate | `bash scripts/ci.sh` | exit 0 |

(If local `mix` errors with a `mise` usage message, prefix with `mise exec -- `.)

## Scope

**In scope**:
- `.gitignore` (add one line)
- Untrack `optimize-plan.md` from git (keep the file on disk)
- Delete the untracked `erl_crash.dump` from the working tree

**Out of scope** (do NOT touch):
- The contents of `optimize-plan.md` — do not edit it; it stays on disk for
  reference, just untracked.
- `docs/` internal artifacts (`docs/reviews/`, `docs/plans/`, etc.) — Plan 003
  handles whether those ship; this plan is only the root `optimize-plan.md` and
  the crash dump.
- `mix.exs` — `optimize-plan.md` is not referenced there.

## Git workflow

- Branch: `advisor/005-untrack-stale-planning-artifacts`
- Commit message: conventional commits, e.g. `chore: untrack optimize-plan.md and ignore it`.
- Do NOT push or open a PR unless instructed.

## Steps

### Step 1: Confirm nothing tracked depends on `optimize-plan.md`

Run `git grep -n "optimize-plan.md"`. Expected: matches only inside `plans/`
(these advisor plans) and possibly the file referencing itself. If any **library
code, `mix.exs`, CI config, or doc that ships** references `optimize-plan.md` as a
required path, STOP and report — untracking it could break a link or build.

**Verify**: `git grep -n "optimize-plan.md"` shows no references from `lib/`,
`mix.exs`, `.github/`, or shipped docs.

### Step 2: Add `optimize-plan.md` to `.gitignore`

Append a line ignoring the root planning artifact. Add, near the end of
`.gitignore`:

```
# Local development planning artifact (kept on disk, not tracked).
/optimize-plan.md
```

**Verify**: `grep -n "optimize-plan.md" .gitignore` → 1 match.

### Step 3: Untrack the file (keep it on disk)

Run `git rm --cached optimize-plan.md`. This removes it from the git index but
leaves the file in the working directory.

**Verify**:
- `git ls-files | grep optimize-plan` → no matches.
- `test -f optimize-plan.md && echo present` → prints `present` (still on disk).
- `git status` shows `optimize-plan.md` as deleted-from-index and now ignored
  (it should NOT reappear as untracked because of the `.gitignore` entry).

### Step 4: Delete the crash dump from the working tree

`erl_crash.dump` is untracked and gitignored. Remove it:
`rm -f erl_crash.dump`.

**Verify**: `test -e erl_crash.dump && echo present || echo gone` → prints `gone`.

### Step 5: Sanity-check the build

**Verify**: `mix compile --warnings-as-errors` → exit 0 (nothing in the build
depends on these files; this just confirms no surprise).

## Test plan

No unit tests apply. Verification is the git/file-state checks above plus a clean
compile. Optionally run `bash scripts/ci.sh` to confirm the full lane is green.

## Done criteria

ALL must hold:

- [ ] `git ls-files | grep -E 'optimize-plan|erl_crash'` → no matches.
- [ ] `optimize-plan.md` still exists on disk (`test -f optimize-plan.md`).
- [ ] `.gitignore` contains `/optimize-plan.md`.
- [ ] `erl_crash.dump` no longer exists in the working tree.
- [ ] `mix compile --warnings-as-errors` exits 0.
- [ ] Only `.gitignore` changed in tracked files (plus the index removal of
      `optimize-plan.md`); `git status` shows nothing else unexpected.
- [ ] `plans/README.md` status row for 005 updated to DONE.

## STOP conditions

Stop and report (do not improvise) if:

- `git grep` shows `optimize-plan.md` is referenced by shipped code/config/docs.
- The maintainer's intent is unclear — if there is any signal that
  `optimize-plan.md` is meant to remain tracked (e.g. it is linked from `README`
  or `CONTRIBUTING`), set this plan BLOCKED with that note and report instead of
  untracking.
- `git rm --cached` reports the file is not tracked (then it was already done;
  just ensure the `.gitignore` line exists and report).

## Maintenance notes

- If the team later wants a tracked, accurate roadmap, prefer GitHub issues or a
  short `docs/roadmap.md` that is actually maintained, rather than reviving the
  large stale plan file.
- This deliberately keeps `optimize-plan.md` on disk so no information is lost; it
  simply stops travelling with the repo history going forward.
