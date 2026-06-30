# Plan 012: Document the undocumented `BSTS_NX_RSCRIPT` and benchmark env vars

> **Executor instructions**: Follow this plan step by step. Run every verification
> command and confirm the expected result before moving to the next step. If
> anything in the "STOP conditions" section occurs, stop and report. When done,
> update the status row in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat e4654c5..HEAD -- README.md lib/bsts_nx/r_sidecar.ex bench/structured_backend_benchmark.exs`
> If any changed since this plan was written, compare the "Current state" excerpts
> below against the live files before proceeding; on a mismatch, STOP.

## Status

- **Priority**: P3
- **Effort**: S
- **Risk**: LOW (documentation only)
- **Depends on**: none
- **Category**: docs
- **Planned at**: commit `e4654c5`, 2026-06-30

## Why this matters

Two sets of runtime knobs are read from the environment but not documented where a
user would look:

1. **`BSTS_NX_RSCRIPT`** — `lib/bsts_nx/r_sidecar.ex:152` reads this to override the
   `Rscript` binary path, but the `RSidecar` docs only mention `BSTS_NX_R_TIMEOUT_MS`
   and `BSTS_NX_R_LIBS_USER/SITE`. A user whose `Rscript` is not on the sidecar's
   trusted PATH (`/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin`) has no documented
   way to point at it and must read the source.
2. **Three benchmark vars** — `bench/structured_backend_benchmark.exs` reads
   `BSTS_NX_BENCH_CONTROLS`, `BSTS_NX_BENCH_SEED`, and `BSTS_NX_BENCH_WARM_RUNS`, but
   the README's "useful environment overrides" sentence lists only six of the nine.

Both are pure documentation gaps with a concrete cost (undiscoverable configuration).
This plan documents them; it changes no code behavior.

## Current state

README env-var sentence — `README.md:160-162`:

```markdown
Useful environment overrides include `BSTS_NX_BENCH_BACKENDS`, `BSTS_NX_BENCH_T`,
`BSTS_NX_BENCH_SAMPLES`, `BSTS_NX_BENCH_BURN_IN`, `BSTS_NX_BENCH_DTYPE`, and
`BSTS_NX_BENCH_OUTPUT`.
```

The benchmark script actually reads these nine (confirmed via grep of
`bench/structured_backend_benchmark.exs`): `BSTS_NX_BENCH_BACKENDS`, `_T`, `_SAMPLES`,
`_BURN_IN`, `_CONTROLS`, `_WARM_RUNS`, `_DTYPE`, `_SEED`, `_OUTPUT`. The three
missing from the README are `_CONTROLS`, `_SEED`, `_WARM_RUNS`.

`RSidecar` doc, `lib/bsts_nx/r_sidecar.ex:63-66` (inside the `run_report/4` `@doc`):

```elixir
  The sidecar runs with an isolated R startup environment. Ambient `R_LIBS*`,
  `R_HOME`, `HOME`, and `PATH` are not inherited. If your R packages live
  outside system libraries, configure trusted paths with `BSTS_NX_R_LIBS_USER`
  or `BSTS_NX_R_LIBS_SITE`.
```

`BSTS_NX_RSCRIPT` is read at `lib/bsts_nx/r_sidecar.ex:152`
(`case System.get_env("BSTS_NX_RSCRIPT") do`) — it overrides the resolved `Rscript`
path. It is currently undocumented.

## Commands you will need

| Purpose | Command | Expected |
|---------|---------|----------|
| Build docs | `mix docs` | exit 0 (no warnings about the edited moduledoc) |
| Compile (moduledoc is compiled) | `mix compile --warnings-as-errors` | exit 0 |
| Format | `mix format --check-formatted` | exit 0 |
| Full gate | `bash scripts/ci.sh` | exit 0 |

(If local `mix` errors with a `mise exec` usage message, prefix with `mise exec -- `.)

## Scope

**In scope** (the only files you may modify):
- `README.md` — the env-overrides sentence at lines 160-162.
- `lib/bsts_nx/r_sidecar.ex` — the `run_report/4` `@doc` only (add one sentence).

**Out of scope** (do NOT touch):
- Any runtime code in `r_sidecar.ex` (the `System.get_env("BSTS_NX_RSCRIPT")` logic at
  line 152 already works — this plan only documents it).
- `bench/structured_backend_benchmark.exs` — it is the source of truth; do not change it.
- The reproduction of any secret or path beyond the literal env-var names.

## Git workflow

- Branch: `advisor/012-document-undocumented-env-vars`
- Commit message: `docs: document BSTS_NX_RSCRIPT and remaining benchmark env vars`.
- Do NOT push or open a PR unless instructed.

## Steps

### Step 1: Add the three missing benchmark vars to the README

Replace `README.md:160-162` with:

```markdown
Useful environment overrides include `BSTS_NX_BENCH_BACKENDS`, `BSTS_NX_BENCH_T`,
`BSTS_NX_BENCH_SAMPLES`, `BSTS_NX_BENCH_BURN_IN`, `BSTS_NX_BENCH_CONTROLS`,
`BSTS_NX_BENCH_WARM_RUNS`, `BSTS_NX_BENCH_DTYPE`, `BSTS_NX_BENCH_SEED`, and
`BSTS_NX_BENCH_OUTPUT`.
```

**Verify**: `grep -c "BSTS_NX_BENCH_" README.md` increases by 3 vs the pre-edit count;
`grep -n "BSTS_NX_BENCH_CONTROLS\|BSTS_NX_BENCH_SEED\|BSTS_NX_BENCH_WARM_RUNS" README.md`
→ 3 matches.

### Step 2: Document `BSTS_NX_RSCRIPT` in the RSidecar doc

In the `run_report/4` `@doc` of `lib/bsts_nx/r_sidecar.ex`, append one sentence to the
isolated-environment paragraph (lines 63-66) so it ends:

```elixir
  The sidecar runs with an isolated R startup environment. Ambient `R_LIBS*`,
  `R_HOME`, `HOME`, and `PATH` are not inherited. If your R packages live
  outside system libraries, configure trusted paths with `BSTS_NX_R_LIBS_USER`
  or `BSTS_NX_R_LIBS_SITE`. If `Rscript` is not on the sidecar's trusted `PATH`,
  set `BSTS_NX_RSCRIPT` to its full path.
```

**Verify**: `mix compile --warnings-as-errors` → exit 0 (the `@doc` compiles);
`grep -n "BSTS_NX_RSCRIPT" lib/bsts_nx/r_sidecar.ex` → now includes the doc line in
addition to the existing `System.get_env` line at 152.

### Step 3: Build the docs

`mix docs`

**Verify**: exit 0 with no new warnings referencing the edited `RSidecar` doc.

### Step 4: Format + gate

**Verify**: `mix format --check-formatted` → exit 0; `bash scripts/ci.sh` → exit 0.

## Test plan

- No unit test (documentation only). Verification is the `grep` checks (Steps 1-2),
  a clean `mix docs` build (Step 3), and `mix compile --warnings-as-errors`
  (moduledocs/`@doc` are compiled, so a malformed edit fails here).
- Optional sanity: open `doc/BstsNx.RSidecar.html` after `mix docs` and confirm the
  `BSTS_NX_RSCRIPT` sentence renders in the `run_report/4` entry.

## Done criteria

ALL must hold:

- [ ] README lists all nine `BSTS_NX_BENCH_*` vars (includes `_CONTROLS`, `_SEED`, `_WARM_RUNS`).
- [ ] `lib/bsts_nx/r_sidecar.ex` `run_report/4` `@doc` mentions `BSTS_NX_RSCRIPT`.
- [ ] `mix compile --warnings-as-errors` exits 0.
- [ ] `mix docs` exits 0.
- [ ] `mix format --check-formatted` exits 0.
- [ ] Only `README.md` and `lib/bsts_nx/r_sidecar.ex` are modified (`git status`).
- [ ] `plans/README.md` status row for 012 updated to DONE.

## STOP conditions

Stop and report (do not improvise) if:

- The "Current state" excerpts don't match the live files (drift) — in particular, if
  `bench/structured_backend_benchmark.exs` no longer reads one of the three vars
  you're about to add, do not document a non-existent knob; re-derive the actual list
  via `grep -onE "BSTS_NX_[A-Z_]+" bench/structured_backend_benchmark.exs`.
- `mix docs` fails (an unrelated doc error) — report it; do not "fix" unrelated docs.

## Maintenance notes

- If a new `BSTS_NX_BENCH_*` knob is added to the benchmark later, add it to the same
  README sentence; the grep in Done criteria is the canonical cross-check.
- Reviewer should confirm the documented var names exactly match what the code reads
  (`System.get_env(...)` call sites), with no typos.
