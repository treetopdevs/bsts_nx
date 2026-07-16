# Plan 003: Stop publishing internal code-review docs in ExDoc/Hex

> **Executor instructions**: Follow this plan step by step. Run every verification
> command and confirm the expected result before moving to the next step. If
> anything in the "STOP conditions" section occurs, stop and report — do not
> improvise. When done, update the status row for this plan in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat e4654c5..HEAD -- mix.exs`
> If `mix.exs` changed since this plan was written, compare the "Current state"
> excerpts below against the live code before proceeding; on a mismatch, treat it
> as a STOP condition.

## Status

- **Priority**: P2
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: docs
- **Planned at**: commit `e4654c5`, 2026-06-20

## Why this matters

`mix.exs` lists five internal code-review markdown files in the ExDoc `extras` and
in a `groups_for_extras` "Reviews" group, so `mix docs` renders them into the
published HTML and they appear on HexDocs. In addition, the Hex package `files`
list ships the entire `docs/` directory, so the same internal review/planning docs
travel inside the package tarball. These are working artifacts (dated review
notes, reviewer back-and-forth), not user documentation. Publishing them dilutes
the docs, looks unprofessional on a public package, and exposes internal review
discussion. Removing them is a pure subtraction with no functional impact.

## Current state

`lib/`-independent change — `mix.exs` only.

`mix.exs:86-106` (the `extras:` list — note the five `docs/reviews/*` entries at
the end):

```elixir
      extras: [
        "README.md",
        "CHANGELOG.md",
        "docs/overview.md",
        "docs/hex-publishing-checklist.md",
        "docs/release-readiness-plan.md",
        "docs/getting-started.md",
        "docs/core-modeling.md",
        "docs/causal-inference-and-attribution.md",
        "docs/forecasting-and-applications.md",
        "docs/synthetic-data-and-validation.md",
        "docs/module-reference.md",
        "docs/components.md",
        "docs/causal-impact.md",
        "docs/reviews/bsts_nx_bayesian_review.md",
        "docs/reviews/CODE_REVIEW.md",
        "docs/reviews/cdx_code_review_2026-03-02.md",
        "docs/reviews/cld_code_review.md",
        "docs/reviews/cldd_code_review_2026-03-03.md"
      ],
```

`mix.exs:107-129` (the `groups_for_extras:` map — note the whole `Reviews:` group):

```elixir
      groups_for_extras: [
        Guides: [
          "CHANGELOG.md",
          "docs/overview.md",
          "docs/hex-publishing-checklist.md",
          "docs/release-readiness-plan.md",
          "docs/getting-started.md",
          "docs/core-modeling.md",
          "docs/causal-inference-and-attribution.md",
          "docs/forecasting-and-applications.md",
          "docs/synthetic-data-and-validation.md",
          "docs/module-reference.md",
          "docs/components.md",
          "docs/causal-impact.md"
        ],
        Reviews: [
          "docs/reviews/bsts_nx_bayesian_review.md",
          "docs/reviews/CODE_REVIEW.md",
          "docs/reviews/cdx_code_review_2026-03-02.md",
          "docs/reviews/cld_code_review.md",
          "docs/reviews/cldd_code_review_2026-03-03.md"
        ]
      ],
```

`mix.exs:61-68` (the Hex package `files:` list — `"docs"` ships the whole dir,
including `docs/reviews/` and other internal docs):

```elixir
      files: [
        "lib",
        "docs",
        "CHANGELOG.md",
        "mix.exs",
        "README.md",
        "LICENSE"
      ],
```

For reference, `docs/` currently contains both published guides (overview,
getting-started, core-modeling, causal-impact, components, module-reference,
forecasting-and-applications, causal-inference-and-attribution,
synthetic-data-and-validation, hex-publishing-checklist, release-readiness-plan)
**and** internal artifacts (`docs/reviews/`, `docs/plans/`, `implementation-plan.md`,
`validation-methodology.md`, `bsts_nx_bayesian_review.md`, and a long
`causal-inference-for-tv-to-web-attribution-*_text_markdown.md`).

## Commands you will need

| Purpose | Command | Expected |
|---------|---------|----------|
| Build docs | `mix docs` | exit 0; writes to `doc/` (gitignored) |
| Check no review pages rendered | `ls doc/ \| grep -i -E 'review\|code_review'` | no matches |
| Check mix.exs has no review refs | `grep -n "docs/reviews" mix.exs` | no matches |
| Format | `mix format && mix format --check-formatted` | exit 0 |
| Full gate | `bash scripts/ci.sh` | exit 0 |

(If local `mix` errors with a `mise` usage message, prefix with `mise exec -- `.)

## Scope

**In scope** (only file you may modify):
- `mix.exs`

**Out of scope** (do NOT touch):
- The files under `docs/reviews/` themselves — leave them on disk and tracked;
  this plan only stops *publishing* them. Deleting/relocating them is Plan 005's
  and the maintainer's concern, not this plan's.
- The `extras`/`groups_for_extras` entries for the actual guides — keep them.

## Git workflow

- Branch: `advisor/003-remove-internal-review-docs`
- Commit message: conventional commits, e.g. `docs: stop publishing internal review notes`.
- Do NOT push or open a PR unless instructed.

## Steps

### Step 1: Remove the review entries from `extras`

Delete the five `docs/reviews/*` lines from the `extras:` list so it ends at
`"docs/causal-impact.md"`. Take care with the trailing comma — `"docs/causal-impact.md"`
must no longer have a comma after it (it becomes the last element).

**Verify**: `grep -n "docs/reviews" mix.exs` still shows the `groups_for_extras`
copies (handled next) but the `extras:` list no longer contains them.

### Step 2: Remove the entire `Reviews:` group from `groups_for_extras`

Delete the `Reviews: [ ... ]` entry (and its five lines) from the
`groups_for_extras:` keyword list, leaving only the `Guides:` group. Ensure the
`Guides:` list keeps valid keyword-list syntax (no dangling comma issues).

**Verify**: `grep -n "docs/reviews" mix.exs` → **no matches**.

### Step 3: Stop shipping internal docs in the package tarball

Replace the bare `"docs"` entry in the `files:` list (line ~63) with an explicit
list of only the published guide docs, so internal review/planning files are not
included in the Hex tarball. The new `files:` list:

```elixir
      files: [
        "lib",
        "docs/overview.md",
        "docs/getting-started.md",
        "docs/core-modeling.md",
        "docs/causal-inference-and-attribution.md",
        "docs/forecasting-and-applications.md",
        "docs/synthetic-data-and-validation.md",
        "docs/module-reference.md",
        "docs/components.md",
        "docs/causal-impact.md",
        "docs/hex-publishing-checklist.md",
        "docs/release-readiness-plan.md",
        "CHANGELOG.md",
        "mix.exs",
        "README.md",
        "LICENSE"
      ],
```

(This is exactly the guide set that remains in `extras` after Steps 1-2, plus the
existing top-level files. If you are not confident this is the intended published
set — e.g. the maintainer may consider `hex-publishing-checklist` internal — see
the STOP conditions; completing Steps 1-2 alone is still a valid partial outcome.)

**Verify**: `grep -n '"docs"' mix.exs` → no match for the bare `"docs"` entry.

### Step 4: Build docs and confirm reviews are gone

**Verify**:
- `mix docs` → exit 0.
- `ls doc/ | grep -i -E 'review|code_review'` → no matches (no review HTML pages
  were generated).
- `mix format --check-formatted` → exit 0.

## Test plan

No unit tests apply (build-config change). Verification is the `mix docs` build
plus the grep checks above. Run `bash scripts/ci.sh` to confirm the full quality
lane (which includes `mix docs`) stays green.

## Done criteria

ALL must hold:

- [ ] `grep -n "docs/reviews" mix.exs` → no matches.
- [ ] `grep -n '"docs",' mix.exs` → no match for the bare `"docs"` package-files entry.
- [ ] `mix docs` exits 0 and `ls doc/ | grep -i -E 'review|code_review'` → no matches.
- [ ] `mix format --check-formatted` exits 0.
- [ ] Only `mix.exs` modified (`git status`).
- [ ] `plans/README.md` status row for 003 updated to DONE.

## STOP conditions

Stop and report (do not improvise) if:

- `mix.exs` "Current state" excerpts don't match the live file (drift since `e4654c5`).
- `mix docs` fails after the edits (likely a stray comma / broken keyword list —
  re-check the edited lists; if it still fails, report).
- You are unsure which docs belong in the published set for Step 3. In that case,
  complete Steps 1-2 (the high-confidence delisting), set this plan's status to
  BLOCKED with the note "Step 3 package-files scope needs maintainer decision",
  and report.

## Maintenance notes

- When new guide docs are added, add them to **both** `extras` and the `files:`
  list (now explicit). The previous `"docs"` glob hid this requirement.
- Plan 005 covers untracking/relocating the on-disk planning artifacts; this plan
  intentionally leaves `docs/reviews/` files in place.
- Reviewer should confirm the rendered HexDocs sidebar no longer shows a "Reviews"
  section.
