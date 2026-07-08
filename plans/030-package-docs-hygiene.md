# Plan 030: Package & docs hygiene — stop shipping internal docs to Hex, prune stale artifacts, fix the leftover ignore rule

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat 9b7cb8d..HEAD -- mix.exs docs .gitignore`
> On mismatch with the excerpts below, STOP.

## Status

- **Priority**: P2 (pre-requisite hygiene for any Hex publish — see the
  D-LAUNCH direction item)
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: tech-debt + docs
- **Planned at**: commit `9b7cb8d`, 2026-07-08

## Why this matters

Two internal maintainer-process documents ship inside the Hex package and
render as public "Guides" on hexdocs: `docs/hex-publishing-checklist.md`
("This checklist is for releasing `bsts_nx` to Hex…") and
`docs/release-readiness-plan.md` (titled "Release Readiness Plan (No Public
Publish Yet)" — literally instructing "Do **not** run `mix hex.publish`
yet"). A downstream user browsing the docs would read the maintainer's
unpublished-release planning as product documentation. Separately, `docs/`
mixes real guides with internal artifacts: a byte-identical duplicate review
file, a methodology doc written for a *different project* ("Scripps Verify"),
a stale implementation plan, and exported review scratch; and `.gitignore`
still carries a `scrippsverify-*.tar` rule from that other project. None of
this blocks anything today, but every item becomes public embarrassment the
moment the package publishes — and D-LAUNCH is on the direction list.

## Current state

- `mix.exs` package `:files` (lines ~61–78) includes among real guides:

  ```elixir
  "docs/hex-publishing-checklist.md",
  "docs/release-readiness-plan.md",
  ```

  and ExDoc config lists both under `extras` (~lines 97–111) AND
  `groups_for_extras: [Guides: [...]]` (~lines 112–127).

- Verified facts about `docs/` (as of `9b7cb8d`):
  - `docs/bsts_nx_bayesian_review.md` is **byte-identical** to
    `docs/reviews/bsts_nx_bayesian_review.md` (`diff -q` confirms).
  - `docs/validation-methodology.md` line 3: "…the Bayesian Structural Time
    Series (BSTS) lift estimation pipeline in **Scripps Verify**…" — wrong
    project; not referenced by `mix.exs`.
  - `docs/implementation-plan.md` — stale ("Current State (as of
    2026-02-10)"); not referenced by `mix.exs`.
  - `docs/plans/2026-07-02-showcase-site-design.md` and `docs/reviews/*` (six
    review artifacts) — internal working docs; not referenced by `mix.exs`.
  - The other extras in `mix.exs` docs() all exist and are genuine guides.
- `.gitignore` line 26: `scrippsverify-*.tar` (this repo's Hex tarball would
  be `bsts_nx-*.tar`).

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Docs build | `mix docs` | exit 0 |
| Package tarball audit | `mix hex.build` then `tar -tf bsts_nx-*.tar` (and `rm` the tarball after) | file list without the two internal docs |
| Full verify | `bash scripts/ci.sh` | exit 0 |

Tooling note: `mix` is a `mise` shim; prefix `mise exec -- ` if needed.

## Scope

**In scope**:
- `mix.exs` (`:files`, `extras`, `groups_for_extras` entries only)
- Moves/deletions within `docs/` as specified in Step 2
- `.gitignore` (line 26)

**Out of scope**:
- Editing the *content* of any guide (README truthfulness is plan 031).
- `CHANGELOG.md`, `README.md`.
- Deleting `docs/hex-publishing-checklist.md` / `release-readiness-plan.md`
  from the repo — they stay as maintainer docs; only their *packaging*
  changes.
- Anything under `site/` (its docs are fine).

## Git workflow

- Branch: `advisor/030-package-docs-hygiene` (from `execute-plans`).
- Commit style: `chore(docs): keep internal process docs out of the Hex package; prune stale artifacts`
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Unpublish the two internal docs

In `mix.exs`, remove `"docs/hex-publishing-checklist.md"` and
`"docs/release-readiness-plan.md"` from all three lists: package `:files`,
docs `extras`, and `groups_for_extras[Guides]`.

**Verify**: `mix docs` → exit 0; `grep -c "hex-publishing-checklist\|release-readiness-plan" mix.exs` → 0.

### Step 2: Reorganize the internal artifacts

Create `docs/internal/` with a 3-line README ("Maintainer working documents —
not packaged, not published"). Then:

1. `git mv docs/hex-publishing-checklist.md docs/release-readiness-plan.md docs/implementation-plan.md docs/internal/`
2. `git mv docs/plans docs/internal/plans` and `git mv docs/reviews docs/internal/reviews`
3. Delete the duplicate: `git rm docs/bsts_nx_bayesian_review.md` (the copy
   under the reviews dir survives, now at `docs/internal/reviews/`).
4. `git mv docs/validation-methodology.md docs/internal/` — it's
   wrong-project content; do NOT rewrite it for bsts_nx (that's a content
   decision for the maintainer; note it in your report as a candidate for
   deletion or a rewrite).

Then re-grep the repo for references to the moved paths
(`grep -rn "docs/reviews\|docs/plans\|hex-publishing-checklist\|release-readiness-plan\|validation-methodology\|implementation-plan" --include='*.md' --include='*.exs' --include='*.ex' . | grep -v _build | grep -v deps | grep -v docs/internal`)
and fix any hits (e.g. README's doc index mentions
`docs/release-readiness-plan.md` — update that line to the new path or drop
the bullet; plans/ files may cite old paths — leave `plans/*.md` citations
as-is, they are historical records).

**Verify**: `mix docs` → exit 0 (no missing extras); the grep above returns
only `plans/` historical citations.

### Step 3: Fix the ignore rule

In `.gitignore`, replace line 26 `scrippsverify-*.tar` with
`bsts_nx-*.tar`.

**Verify**: `grep -n "scrippsverify" .gitignore` → no matches.

### Step 4: Prove the package is clean

`mix hex.build` → inspect with `tar -tf bsts_nx-0.1.0.tar` (contents listing
includes an inner `contents.tar.gz`; `tar -xOf bsts_nx-0.1.0.tar CHECKSUM VERSION metadata.config | head` and
`tar -xOf bsts_nx-0.1.0.tar contents.tar.gz | tar -tzf - | sort` for the real
file list). Delete the tarball afterwards (`rm bsts_nx-*.tar`) — it's now
correctly gitignored either way.

**Verify**: the inner file list contains the real guides but NOT
`hex-publishing-checklist.md`, `release-readiness-plan.md`, or anything under
`docs/internal/`.

### Step 5: Full pass

**Verify**: `bash scripts/ci.sh` → exit 0 (its `mix docs` step revalidates
the extras).

## Test plan

No ExUnit tests. Machine checks: the `mix hex.build` tarball listing (Step
4), `mix docs` builds, and the greps in the done criteria.

## Done criteria

- [ ] `mix.exs` no longer references the two internal docs anywhere
- [ ] `docs/internal/` exists; plans/reviews/stale/wrong-project artifacts moved under it; duplicate deleted
- [ ] Hex tarball contents verified free of internal docs (listing in report)
- [ ] `.gitignore` has `bsts_nx-*.tar`, not `scrippsverify-*.tar`
- [ ] `mix docs` and `bash scripts/ci.sh` exit 0
- [ ] Only in-scope files changed (`git status --porcelain`)
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- Any moved doc turns out to be load-bearing (referenced by code, tests, or
  CI — the Step 2 grep finds it) in a way that isn't a one-line path fix.
- `mix hex.build` fails for an unrelated packaging reason (missing files in
  `:files`) — report; don't reshape the package config beyond this plan's
  removals.
- You are tempted to delete rather than move an artifact whose value you
  can't judge — move it; deletion is the maintainer's call (the report lists
  candidates).

## Maintenance notes

- Rule going forward: `docs/` top level = published guides only; everything
  else goes to `docs/internal/`. Reviewers should check new `mix.exs` extras
  entries against that rule.
- `docs/internal/validation-methodology.md` is flagged for maintainer
  decision: rewrite for bsts_nx (it describes genuinely useful validation
  tests) or delete as foreign content.
- This plan is a soft prerequisite of D-LAUNCH (Hex publish); the tarball
  audit in Step 4 is the repeatable pre-publish check.
