# Plan 007: Stop `.formatter.exs` from importing optional/dev-only deps

> **Executor instructions**: Follow this plan step by step. Run every verification
> command and confirm the expected result before moving to the next step. If
> anything in the "STOP conditions" section occurs, stop and report — do not
> improvise. When done, update the status row for this plan in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat e4654c5..HEAD -- .formatter.exs`
> Also open `.formatter.exs` and confirm it still matches the "Current state"
> excerpt below (the working tree may already carry uncommitted changes from
> plans 001–006; rely on the excerpt, not just the SHA). On a mismatch, treat it
> as a STOP condition.

## Status

- **Priority**: P2
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: dx
- **Planned at**: commit `e4654c5`, 2026-06-30

## Why this matters

`.formatter.exs` declares `import_deps: [:nx, :exla, :ex_doc]`. Two of those three
are not always available when `mix format` runs:

- `:exla` is an **optional** dependency (`mix.exs:42` — `{:exla, "~> 0.12.0", optional: true}`).
- `:ex_doc` is **dev-only** (`mix.exs:44` — `{:ex_doc, "~> 0.34", only: :dev, runtime: false}`),
  so it is absent under `MIX_ENV=test`.

`import_deps` requires each listed dependency to be fetched and loadable so the
formatter can read its exported `:locals_without_parens`. When a contributor runs
`MIX_ENV=test mix format` (or runs `mix format` before optional deps have been
fetched/compiled — EXLA pulls a large native XLA build that can fail), the
formatter aborts with an error like
`Unknown dependency :ex_doc given to :import_deps in the formatter configuration`.

Only `:nx` actually exports formatter rules this project uses (the `defn`/
`deftransform` macros throughout `lib/`). EXLA and ex_doc export nothing the code
relies on, so importing them buys nothing and only adds a failure mode. Dropping
them makes `mix format` work in every environment without changing how a single
line is formatted.

## Current state

`.formatter.exs` (entire file):

```elixir
[
  import_deps: [
    :nx,
    :exla,
    :ex_doc
  ],
  inputs: ["*.{ex,exs}", "{config,lib,test}/**/*.{ex,exs}"]
]
```

Relevant `mix.exs` dependency declarations (DO NOT MODIFY — reference only):

```elixir
{:nx, "~> 0.12.0"},
{:emlx, "~> 0.3.0", optional: true},
{:exla, "~> 0.12.0", optional: true},
{:xla, "~> 0.10.0", optional: true},
{:ex_doc, "~> 0.34", only: :dev, runtime: false},
{:stream_data, "~> 1.0", only: [:test, :dev]}
```

The key fact this plan relies on: `:nx` is the only one of the three imported deps
that exports a non-empty `:locals_without_parens` consumed by this project. Step 2
verifies this empirically — if removing `:exla`/`:ex_doc` reformats any file, that
assumption is wrong and you STOP.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Format (dev) | `mix format` | exit 0, **no files changed** |
| Format check | `mix format --check-formatted` | exit 0 |
| Format under test env | `MIX_ENV=test mix format --check-formatted` | exit 0 (this is the bug being fixed) |
| Compile | `mix compile --warnings-as-errors` | exit 0 |

(If local `mix` errors with a `mise exec` usage message or `unexpected argument '-n'`,
prefix the command with `mise exec -- `, e.g. `mise exec -- mix format`. `.tool-versions`
pins Erlang 28.3.1 / Elixir 1.19.5-otp-28.)

## Scope

**In scope** (the only file you may modify):
- `.formatter.exs`

**Out of scope** (do NOT touch):
- `mix.exs` / `mix.lock` — the dependency declarations are correct; this plan does
  not change them.
- Any `lib/` or `test/` source file — no code is reformatted by this change. If any
  file *would* be reformatted, STOP (see STOP conditions).

## Git workflow

- Branch: `advisor/007-formatter-drop-optional-deps`
- Commit message style: conventional commits (recent history uses `fix:`, `chore:`,
  `refactor:`). Example: `chore: stop importing optional deps in formatter config`.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Establish the pre-change baseline

Confirm the tree is already formatted under dev (so any post-change diff is caused
only by your edit):

`mix format --check-formatted`

**Verify**: exit 0. If it fails here, STOP — the tree is not formatted and this
plan's "no files changed" check in Step 3 would be meaningless.

Optionally reproduce the bug to confirm the premise:
`MIX_ENV=test mix format --check-formatted` → expected to **error** with a message
naming `:ex_doc` (or `:exla`) as an unknown/unavailable dependency for
`:import_deps`. If it does NOT error, note that and continue (the fix is still
correct hygiene; the failure may require ex_doc to be unfetched in test).

### Step 2: Edit `.formatter.exs`

Replace the entire file contents with:

```elixir
[
  import_deps: [
    :nx
  ],
  inputs: ["*.{ex,exs}", "{config,lib,test}/**/*.{ex,exs}"]
]
```

(Only `:exla` and `:ex_doc` are removed from `import_deps`; `:nx` and `inputs` are
unchanged.)

**Verify**: `mix compile --warnings-as-errors` → exit 0.

### Step 3: Confirm formatting is byte-for-byte unchanged

Run the formatter and confirm it rewrites nothing:

`mix format`

Then:

`git status --porcelain`

**Verify**: the only modified path is `.formatter.exs`. If `git status` shows ANY
`lib/` or `test/` file as modified after `mix format`, the removed imports *were*
contributing formatter rules — revert your edit (`git checkout .formatter.exs`) and
STOP and report.

### Step 4: Confirm the bug is fixed in the test environment

`MIX_ENV=test mix format --check-formatted`

**Verify**: exit 0 (previously this errored on the missing `:ex_doc`/`:exla` dep).

### Step 5: Full dev gate

`mix format --check-formatted`

**Verify**: exit 0.

## Test plan

This change has no runtime behavior and needs no unit test. Its correctness is
fully captured by the verification commands:

- `mix format` rewrites no source files (Step 3) — proves formatting is unchanged.
- `MIX_ENV=test mix format --check-formatted` now exits 0 (Step 4) — proves the
  bug is fixed.
- There is no test-suite impact; you do not need to run `mix test` for this plan,
  though `bash scripts/ci.sh` should still pass end-to-end.

## Done criteria

ALL must hold:

- [ ] `.formatter.exs` `import_deps` contains exactly `[:nx]`.
- [ ] `grep -n ":exla\|:ex_doc" .formatter.exs` → no matches.
- [ ] `mix format` modifies no file other than `.formatter.exs` (`git status --porcelain`).
- [ ] `mix format --check-formatted` exits 0.
- [ ] `MIX_ENV=test mix format --check-formatted` exits 0.
- [ ] `mix compile --warnings-as-errors` exits 0.
- [ ] `plans/README.md` status row for 007 updated to DONE.

## STOP conditions

Stop and report (do not improvise) if:

- `.formatter.exs` does not match the "Current state" excerpt (drift).
- Running `mix format` after the edit reformats any `lib/` or `test/` file
  (the removed imports were load-bearing — revert and report).
- The pre-change `mix format --check-formatted` (Step 1) fails on the untouched tree.

## Maintenance notes

- If a future dependency ships macros that need special formatting (a new `defn`-like
  DSL), add only that dependency to `import_deps` — and only if it is available in
  every `MIX_ENV` where `mix format` runs, or the same bug returns.
- Reviewer should confirm the diff touches only `.formatter.exs` and that CI's
  `quality` job (which runs `mix format --check-formatted` under `MIX_ENV=dev`)
  still passes.
