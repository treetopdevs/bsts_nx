# Plan 022: Profile-gated optimization of the spike-and-slab Gibbs hot loop (bit-identical outputs required)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat 9b7cb8d..HEAD -- lib/bsts_nx/gibbs_sampler/spike_slab.ex bench/optimize_plan.exs test/covariate_selection_test.exs`
> If any changed since this plan was written, compare the "Current state"
> excerpts against the live code before proceeding; on a mismatch, treat it
> as a STOP condition.

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: MED (hot-loop refactor in a statistical sampler — controlled by a
  bit-identical-output gate)
- **Depends on**: none
- **Category**: perf
- **Planned at**: commit `9b7cb8d`, 2026-07-08

## Why this matters

`BstsNx.GibbsSampler.SpikeSlab` implements spike-and-slab variable selection —
the per-iteration inner loop resamples every regression inclusion indicator
(`gamma`) with a g-prior marginal-likelihood computation. Two prior audits
flagged pure-Elixir inefficiencies here (O(p²) `List.replace_at` gamma
updates, per-flip submatrix assembly and solve) but rejected the work as
"needs benchmarking to justify" — the module was already deliberately
optimized once (commit `da81702`, pre-computed XtX/Xty) and the aggressive
fix (incremental rank-1 updates) is high-risk for accuracy.

The maintainer has now asked for further performance work **with accuracy
explicitly protected**. This plan does that in the only defensible order:
measure → profile → optimize *only* transformations that cannot change any
floating-point result (data-structure and allocation changes, not math
changes) → prove bit-identical outputs on seeded runs → measure again. If
profiling shows the targeted patterns don't actually dominate, the plan stops
and reports instead of churning a correct sampler.

## Current state

- `lib/bsts_nx/gibbs_sampler/spike_slab.ex` (806 lines, `@moduledoc false`).
  The per-iteration gamma sweep (~lines 314–336):

  ```elixir
  stats = spike_slab_sufficient_stats(y_obs, x_obs_rows, xtx_stats)

  ...
    Enum.reduce(0..(p - 1), {gamma, model0}, fn j, {gamma_curr, model_curr} ->
      ...
        List.replace_at(gamma_curr, j, gamma_j),
      ...
    end)
  ```

  Pre-computed sufficient statistics already exist (~lines 341–362:
  `spike_slab_xtx_stats/1` builds XtX once via `compat_dot`;
  `spike_slab_sufficient_stats/3` adds Xty). Per-flip cost lives around lines
  382–393: an active-column submatrix is assembled with nested
  `Enum.map(active, &xtx_value(stats, row_idx, &1))` and inverted via
  `BstsNx.Utils.safe_solve(xtx, Nx.eye(k))` — i.e. **per column flip**, an
  O(k²) rebuild plus an O(k³) solve, giving O(p·k³) per Gibbs iteration.
  There is also existing rank-update machinery gated by
  `@rank_update_pivot_floor 1.0e-12` (line 11) — read the whole module before
  touching anything; some optimization may already exist behind flags/paths
  you must not duplicate or break.

- Benchmark harness (already exists): `mix bench.optimize` runs
  `bench/optimize_plan.exs`, which builds `spike_slab_dataset(72, 24)` among
  its scenarios and writes results under `bench/results/` (gitignored).
  `mix bench.structured_backends` also exists.

- Tests exercising this path: `test/covariate_selection_test.exs` (spike-slab
  selection), `test/gibbs_structured_test.exs` (structured sampler including
  regression modes). PRNG is seed-threaded (`:key`/`:seed` options), so a
  fixed seed ⇒ a deterministic draw sequence ⇒ **bit-identical outputs are a
  meaningful gate** for refactors that don't reorder float operations.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Benchmark | `mix bench.optimize` | timing table printed; results file under `bench/results/` |
| Profile | `mix profile.tprof -e '<script from Step 2>'` (OTP 28 has tprof; fall back to `mix profile.eprof`) | per-function time table |
| Focused tests | `mix test test/covariate_selection_test.exs test/gibbs_structured_test.exs` | 0 failures |
| Full verify | `bash scripts/ci.sh` | exit 0 |

Tooling note: `mix` is a `mise` shim; prefix `mise exec -- ` if needed.

## Scope

**In scope** (the only files you should modify):
- `lib/bsts_nx/gibbs_sampler/spike_slab.ex`
- `test/bsts_nx/spike_slab_parity_test.exs` (create — the golden-fixture gate)

**Out of scope** (do NOT touch):
- Any change that alters the **order or association of floating-point
  operations** — no reordered sums, no algebraic rewrites, no incremental
  rank-1/Cholesky-update rewrite of the per-flip solve (that is the known
  HIGH-risk item; it stays rejected until someone signs off on a statistical
  re-validation plan).
- `lib/bsts_nx/utils.ex` (`safe_solve`), `lib/bsts_nx/distributions.ex`,
  `lib/bsts_nx/gibbs_sampler/{structured,residuals}.ex`.
- `bench/optimize_plan.exs` — measure with the harness as-is so numbers are
  comparable.
- Tolerances or assertions in existing tests.

## Git workflow

- Branch: `advisor/022-spike-slab-profile-optimize` (from `execute-plans`).
- Commit style: `perf: reduce allocation overhead in spike-slab gamma sweep (bit-identical)`
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Freeze a golden fixture from the CURRENT code

Create `test/bsts_nx/spike_slab_parity_test.exs` that runs a short seeded
spike-and-slab workload through the public API and asserts against values you
capture NOW (before any change). Use `test/covariate_selection_test.exs` as
the structural pattern for building a spike-slab call (small T, p≈6–8, fixed
`seed:`, few samples so the test stays <5 s). Capture, at full float
precision (`inspect(x, limit: :infinity)` / exact list equality — not
`assert_in_delta`):

- the final inclusion-probability vector (or gamma draw sequence), and
- the first and last posterior coefficient draws.

The test must assert **exact equality** with the captured values, with a
comment explaining: "bit-parity gate for allocation-level refactors; if an
intentional math change ever lands here, this fixture must be regenerated in
the same commit with a statistical re-validation."

**Verify**: `mix test test/bsts_nx/spike_slab_parity_test.exs` → passes on
unmodified HEAD.

### Step 2: Baseline benchmark + profile

1. `mix bench.optimize` → record the spike-slab scenario's timing.
2. Profile a representative run, e.g.
   `mix profile.tprof -e 'BstsNx.CovariateSelection.select(<same fixture args as Step 1>)'`
   (build the expression from the Step 1 test; `mix profile.eprof` if tprof is
   unavailable). Record the top-15 functions by own-time and specifically the
   shares of: `List.replace_at`-driven list rebuilding, `xtx_value`/submatrix
   assembly, `Utils.safe_solve`, `Nx.eye`, and `Distributions` sampling.

**Verify**: you have a written baseline table (bench ms + profile shares) —
it goes in the final report. **Decision gate**: if the pure-Elixir
data-structure patterns (list ops + submatrix assembly + repeated `Nx.eye`
allocation, excluding the solve itself) sum to **<15%** of run time, STOP
(condition 1) — the loop is algebra-dominated and only the out-of-scope
rewrite would help.

### Step 3: Apply allocation-level optimizations (only what profiling justified)

Candidate transformations, all value-preserving by construction — implement
the ones Step 2 showed matter, skip the rest:

1. **Gamma as a tuple** — replace the `gamma` list +
   `List.replace_at(gamma_curr, j, gamma_j)` with a tuple +
   `put_elem/3` (O(p) copy but no cons-cell churn) or `:array`; convert at
   the sweep boundary so the public shape is unchanged.
2. **Hoist invariant allocations** — `Nx.eye(k)` and any per-flip constant
   tensors rebuilt inside the sweep get built once per distinct `k` (a small
   map keyed by `k`), since `k` only steps by ±1 across flips.
3. **Submatrix assembly** — if `xtx_value/3` does per-element map/tuple
   lookups, pre-extract the needed rows/columns into a flat structure once
   per flip instead of nested `Enum.map`; the assembled matrix must contain
   the exact same floats in the exact same positions.

Each transformation lands as its own commit, and after each:

**Verify**: `mix test test/bsts_nx/spike_slab_parity_test.exs
test/covariate_selection_test.exs test/gibbs_structured_test.exs` → 0
failures. The parity test failing = you changed a value = revert that commit
(see STOP conditions).

### Step 4: Post-benchmark and the keep/revert decision

`mix bench.optimize` again, same machine, same scenario.

**Verify / decision gate**: spike-slab scenario improves **≥15%** vs Step 2's
baseline → keep. Under 15% → revert the optimization commits (keep the parity
test — it's valuable regardless) and record "not worth the churn" in the
report. Honesty over sunk cost.

### Step 5: Full verification

**Verify**: `bash scripts/ci.sh` → exit 0.

## Test plan

- New: `test/bsts_nx/spike_slab_parity_test.exs` (Step 1) — the bit-parity
  gate, exact-equality on seeded outputs, patterned on
  `test/covariate_selection_test.exs`.
- Existing: `covariate_selection_test.exs`, `gibbs_structured_test.exs`, and
  the full non-external suite via `scripts/ci.sh`.
- Report artifacts: baseline + post benchmark table, profile top-15, and the
  keep/revert decision with numbers.

## Done criteria

- [ ] `test/bsts_nx/spike_slab_parity_test.exs` exists and passes (whether or
      not optimizations were kept)
- [ ] Report contains: profile table, before/after `mix bench.optimize`
      numbers, and an explicit ≥15%-kept or reverted outcome
- [ ] `mix test test/covariate_selection_test.exs test/gibbs_structured_test.exs` → 0 failures
- [ ] `bash scripts/ci.sh` exits 0
- [ ] `git status --porcelain` shows only in-scope files (plus reverts leaving
      no residue if the revert path was taken)
- [ ] `plans/README.md` status row updated (DONE with kept-or-reverted note)

## STOP conditions

Stop and report back (do not improvise) if:

- **The parity test fails after any transformation.** Do not "fix" the
  fixture; revert the offending commit. If you believe the transformation
  *should* have been value-preserving, that belief is the bug — report the
  diff of outputs.
- Step 2's decision gate fires (<15% attributable to the in-scope patterns).
- The module's existing rank-update machinery (`@rank_update_pivot_floor`)
  turns out to already implement one of the candidate optimizations behind a
  branch — report how it's gated instead of adding a parallel path.
- `mix profile.tprof`/`eprof` cannot attribute time sensibly (all time in
  anonymous funs) — say so; don't optimize blind.

## Maintenance notes

- The parity fixture is deliberately brittle: any future *intentional* change
  to spike-slab math must regenerate it in the same commit and say why. That
  is the accuracy tripwire the maintainer asked for.
- The rank-1 incremental-update rewrite remains the big prize (O(p·k³) →
  O(p·k²)) and remains out of scope pending a statistical re-validation plan
  (R-parity lane + posterior-recovery checks). If the Step 2 profile shows
  `safe_solve` dominating, that's the data point that justifies scoping it
  properly — put it in the report.
- If plan 021 (EXLA on the site) lands, spike-slab's eager-Elixir loop is NOT
  accelerated by it (this loop is host-side Elixir, not defn) — the two plans
  are complementary, not redundant.
