# Plan 004: Deduplicate the scalar RTS smoother backward pass

> **Executor instructions**: Follow this plan step by step. Run every verification
> command and confirm the expected result before moving to the next step. This
> plan touches numerically-sensitive `Nx.Defn` code in the highest-churn file in
> the repo. The existing smoother test suite is your **oracle**: if it disagrees
> after your change, the refactor is NOT behavior-preserving — STOP and report,
> do not tune tolerances or improvise. When done, update the status row for this
> plan in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat e4654c5..HEAD -- lib/bsts_nx/smoother.ex`
> If `smoother.ex` changed since this plan was written, compare the "Current state"
> excerpts below against the live code before proceeding; on a mismatch, STOP.

## Status

- **Priority**: P3
- **Effort**: M
- **Risk**: MED
- **Depends on**: plans/002-missing-observation-correctness-tests.md
- **Category**: tech-debt
- **Planned at**: commit `e4654c5`, 2026-06-20

## Why this matters

`smoother.ex` is the highest-churn file in the repo (≈893 lines, 21 commits). It
contains two **scalar** compiled RTS backward passes whose while-loop bodies are
identical except that one also accumulates the lag-one cross-covariance:

- `rts_defn_impl/4` (lines 82-125) — smoothed means/covariances only.
- `rts_defn_with_lag1_impl/4` (lines 221-270) — the same backward pass plus a
  `lag1` accumulator.

(The two *matrix* variants, `rts_defn_matrix_impl` and `rts_defn_matrix_impl_pinv`,
are already deduplicated via a `for {name, gain_fun} <- ...` macro at line 127, so
they are **out of scope** — the only real duplication is the two scalar passes.)

Every numerical fix to the scalar backward pass must currently be made in two
places, which is exactly how subtle drift between paths begins. Collapsing
`rts_defn_impl/4` into a thin delegate of `rts_defn_with_lag1_impl/4` (discarding
the lag-1 output) removes ~44 lines of duplicated recursion with no behavior
change — *provided the smoother test suite stays green*.

## Current state

`lib/bsts_nx/smoother.ex` public wrappers:

- `rts_defn/4` (lines 49-57) calls `rts_defn_impl(xs, ps, f_t, q_t)`.
- `rts_defn_with_lag1/4` (lines 200-219) calls `rts_defn_with_lag1_impl(xs, ps, f_t, q_t)`
  and slices off the lag-1 padding before returning `{sxs, sps, lag1}`.

`rts_defn_impl/4` — the function to REPLACE (lines 82-125):

```elixir
  Nx.Defn.defn rts_defn_impl(xs, ps, f, q) do
    t = Nx.axis_size(xs, 0)
    x_type = Nx.type(xs)
    p_type = Nx.type(ps)
    sxs = Nx.broadcast(Nx.tensor(0.0, type: x_type), {t})
    sps = Nx.broadcast(Nx.tensor(0.0, type: p_type), {t})
    last_idx = t - 1
    sxs = Nx.put_slice(sxs, [last_idx], Nx.reshape(take_scalar_at(xs, last_idx), {1}))
    sps = Nx.put_slice(sps, [last_idx], Nx.reshape(take_scalar_at(ps, last_idx), {1}))
    num_steps = t - 1

    {_, sxs_out, sps_out, _, _, _, _} =
      while {k = Nx.tensor(0), sxs_acc = sxs, sps_acc = sps, xs_in = xs, ps_in = ps, f_in = f,
             q_in = q},
            k < num_steps do
        i = last_idx - 1 - k
        x_filt = take_scalar_at(xs_in, i)
        p_filt = take_scalar_at(ps_in, i)
        x_pred_next = f_in * x_filt
        p_pred_next = f_in * p_filt * f_in + q_in
        near_zero_p = Nx.abs(p_pred_next) < @near_zero_covariance
        safe_pred = Nx.select(near_zero_p, 1.0, p_pred_next)
        c = Nx.select(near_zero_p, 0.0, p_filt * f_in / safe_pred)
        x_smooth_next = take_scalar_at(sxs_acc, i + 1)
        p_smooth_next = take_scalar_at(sps_acc, i + 1)
        x_smooth = x_filt + c * (x_smooth_next - x_pred_next)
        p_smooth = p_filt + c * (p_smooth_next - p_pred_next) * c
        sxs_new = Nx.put_slice(sxs_acc, [i], Nx.reshape(x_smooth, {1}))
        sps_new = Nx.put_slice(sps_acc, [i], Nx.reshape(p_smooth, {1}))
        {k + 1, sxs_new, sps_new, xs_in, ps_in, f_in, q_in}
      end

    {sxs_out, sps_out}
  end
```

`rts_defn_with_lag1_impl/4` — the function to KEEP UNCHANGED and delegate to
(lines 221-270). Its backward-pass body is identical to the above except it casts
`f`/`q` to `Nx.type(xs)`, uses a single `out_type = Nx.type(xs)` for both
accumulators, and additionally computes `lag1_val = p_smooth_next * c` and stores
it. It returns `{sxs_out, sps_out, lag1_out}`.

**Behavioral note (read before editing):** `rts_defn_impl` uses *separate* types
for the two accumulators (`x_type` for `sxs`, `p_type` for `sps`) and does **not**
cast `f`/`q`. `rts_defn_with_lag1_impl` uses `Nx.type(xs)` for both and casts
`f`/`q` to it. When `Nx.type(xs) == Nx.type(ps)` (the normal case from
`filter_defn` output, and always the case for `rts_defn_matrix` which pre-casts to
f64), the two are equivalent. They could differ only if a caller passes filtered
means and covariances with *different* dtypes. The smoother test suite is the
oracle for whether this matters in practice.

## Commands you will need

| Purpose | Command | Expected |
|---------|---------|----------|
| Compile | `mix compile --warnings-as-errors` | exit 0 |
| Smoother suite (the oracle) | `mix test test/smoother_test.exs test/smoother_defn_test.exs test/smoother_defn_matrix_test.exs test/smoother_robustness_test.exs test/smoother_statistical_test.exs test/smoother_key_test.exs` | all pass |
| Missing-data correctness (Plan 002) | `mix test test/missing_data_correctness_test.exs` | all pass |
| New parity test | `mix test test/smoother_rts_dedup_test.exs` | all pass |
| Doctests | `mix test test/doctest_test.exs` | all pass |
| Format | `mix format && mix format --check-formatted` | exit 0 |
| Full gate | `bash scripts/ci.sh` | exit 0 |

(If local `mix` errors with a `mise` usage message, prefix with `mise exec -- `.)

## Suggested executor toolkit

- This is `Nx.Defn` code. If a `vendored`/`nx` best-practices skill is available in
  your environment, consult it before editing the `while` loop. Otherwise, the
  delegation below requires no new `defn` knowledge — you are deleting a body, not
  writing one.

## Scope

**In scope**:
- `lib/bsts_nx/smoother.ex` — replace the body of `rts_defn_impl/4` (lines 82-125)
  with a delegation to `rts_defn_with_lag1_impl/4`. Nothing else in the file.
- `test/smoother_rts_dedup_test.exs` (create) — parity guard.

**Out of scope** (do NOT touch):
- `rts_defn_with_lag1_impl/4` and `rts_defn_with_lag1/4` — leave entirely unchanged.
- The matrix variants and their generating `for` macro (lines 127-174) — already
  deduplicated.
- The public `rts_defn/4` wrapper (lines 49-57) — it keeps calling `rts_defn_impl`.
- Any tolerance/jitter constants (`@near_zero_covariance`, `@min_covariance`,
  `@cholesky_jitter`).

## Git workflow

- Branch: `advisor/004-deduplicate-scalar-rts-smoother-defn`
- Commit message: conventional commits, e.g. `refactor: delegate scalar RTS smoother to the lag1 implementation`.
- Do NOT push or open a PR unless instructed.

## Steps

### Step 0 (prerequisite): confirm Plan 002 has landed

This plan depends on Plan 002's `test/missing_data_correctness_test.exs`. Confirm
it exists and passes: `mix test test/missing_data_correctness_test.exs`. If it does
not exist, STOP — execute Plan 002 first.

### Step 1: Establish the green baseline (before changing anything)

Run the oracle suite and record that it passes **before** your edit:

`mix test test/smoother_test.exs test/smoother_defn_test.exs test/smoother_defn_matrix_test.exs test/smoother_robustness_test.exs test/smoother_statistical_test.exs test/smoother_key_test.exs test/doctest_test.exs`

**Verify**: all pass. If anything fails on the untouched code, STOP — the baseline
is not green and this plan cannot proceed safely.

### Step 2: Replace the `rts_defn_impl/4` body with a delegation

Replace the entire `rts_defn_impl/4` definition (lines 82-125) with:

```elixir
  Nx.Defn.defn rts_defn_impl(xs, ps, f, q) do
    # Single source of truth for the scalar backward pass lives in
    # rts_defn_with_lag1_impl/4; this variant simply discards the lag-1 output.
    {sxs, sps, _lag1} = rts_defn_with_lag1_impl(xs, ps, f, q)
    {sxs, sps}
  end
```

**Verify**: `mix compile --warnings-as-errors` → exit 0.

### Step 3: Run the oracle suite again — it MUST still be green

Run the exact same command as Step 1.

**Verify**: all pass, identical to the baseline. If **any** test now fails
(numerical mismatch, dtype assertion, doctest shape), STOP and report. A failure
means the separate-dtype / `f`/`q`-casting difference noted in "Current state"
actually matters for tested inputs, and the naive delegation is not safe — that
needs a more careful refactor (see Maintenance notes), not this plan.

### Step 4: Add a parity test

Create `test/smoother_rts_dedup_test.exs` asserting `rts_defn/4` and
`rts_defn_with_lag1/4` agree on smoothed means/covariances:

```elixir
defmodule BstsNxSmootherRtsDedupTest do
  use ExUnit.Case, async: true

  alias BstsNx.KalmanFilter
  alias BstsNx.Smoother

  test "rts_defn matches the means/covariances from rts_defn_with_lag1" do
    obs = Nx.tensor([1.0, 2.0, 1.5, 3.0, 2.5, 4.0], type: {:f, 64})
    {xs, ps} = KalmanFilter.filter_defn(obs, 1.0, 1.0, 0.2, 0.6, 0.0, 1.0)

    {sxs_a, sps_a} = Smoother.rts_defn(xs, ps, 1.0, 0.2)
    {sxs_b, sps_b, _lag1} = Smoother.rts_defn_with_lag1(xs, ps, 1.0, 0.2)

    assert Nx.to_flat_list(sxs_a) == Nx.to_flat_list(sxs_b)
    assert Nx.to_flat_list(sps_a) == Nx.to_flat_list(sps_b)
  end

  test "rts_defn preserves f64 output dtype" do
    obs = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: {:f, 64})
    {xs, ps} = KalmanFilter.filter_defn(obs, 1.0, 1.0, 0.1, 0.5, 0.0, 1.0)
    {sxs, sps} = Smoother.rts_defn(xs, ps, 1.0, 0.1)

    assert Nx.type(sxs) == {:f, 64}
    assert Nx.type(sps) == {:f, 64}
  end
end
```

If the first assertion fails, STOP — the delegation changed results (it should be
bit-identical since both wrappers now route through the same `defn`). If the
second fails, the dtype handling differs from the original; STOP and report.

**Verify**: `mix test test/smoother_rts_dedup_test.exs` → 2 tests, 0 failures.

### Step 5: Full gate

**Verify**:
- `mix test test/smoother_test.exs test/smoother_defn_test.exs test/smoother_defn_matrix_test.exs test/smoother_robustness_test.exs test/smoother_statistical_test.exs test/smoother_key_test.exs test/missing_data_correctness_test.exs test/smoother_rts_dedup_test.exs test/doctest_test.exs` → all pass.
- `mix format --check-formatted` → exit 0.

## Test plan

- New file `test/smoother_rts_dedup_test.exs`: parity between `rts_defn/4` and
  `rts_defn_with_lag1/4`, plus an f64 dtype-preservation assertion.
- The existing smoother suite (six files) + Plan 002's missing-data correctness
  test are the regression oracle for numerical equivalence.
- Pattern reference: `test/smoother_defn_test.exs` for `filter_defn` → `rts_defn`
  call shape and tolerance conventions.
- Verification: full gate in Step 5.

## Done criteria

ALL must hold:

- [ ] `rts_defn_impl/4` is a 2-statement delegate to `rts_defn_with_lag1_impl/4`
      (the duplicated `while` loop is gone). `grep -c "while" lib/bsts_nx/smoother.ex`
      decreases by exactly 1 vs the pre-edit count.
- [ ] `mix compile --warnings-as-errors` exits 0.
- [ ] The full Step 5 gate passes (every listed suite green).
- [ ] `test/smoother_rts_dedup_test.exs` exists and passes.
- [ ] `mix format --check-formatted` exits 0.
- [ ] Only `lib/bsts_nx/smoother.ex` and the new test file modified (`git status`).
- [ ] `plans/README.md` status row for 004 updated to DONE.

## STOP conditions

Stop and report (do not improvise) if:

- The "Current state" excerpts don't match the live code (drift since `e4654c5`).
- The oracle suite is not green **before** your edit (Step 1).
- Any smoother test, doctest, or Plan 002 test fails **after** the delegation
  (Step 3) — this signals the dtype / `f`/`q`-cast difference is observable.
  Do NOT adjust tolerances or test expectations to make it pass.
- The parity or dtype test in Step 4 fails.
- You find a third scalar backward-pass duplicate not mentioned here.

## Maintenance notes

- If Step 3 reveals the delegation is not numerically identical (because some
  caller passes `xs` and `ps` with different dtypes, or relies on `f`/`q` *not*
  being cast to `xs`'s type), the safe alternative is the reverse: extract a
  single private `defn` that takes the inputs once, computes the backward pass,
  and returns `{sxs, sps, lag1}` with the *original* `rts_defn_impl` type
  handling (separate `x_type`/`p_type`, no `f`/`q` cast), then have
  `rts_defn_with_lag1_impl` add only the lag-1 slice. That is a larger change —
  report back rather than attempting it under this plan.
- After this lands, the scalar backward pass exists in exactly one place; future
  numerical fixes go to `rts_defn_with_lag1_impl/4` only.
- Reviewer should confirm `rts_defn/4`'s public contract (`{sxs, sps}` shapes and
  dtype) is unchanged and that the matrix smoother paths were not touched.
