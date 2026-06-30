# Plan 006: Reject ambiguous-length rank-2 observation matrices explicitly

> **Executor instructions**: Follow this plan step by step. Run every verification
> command and confirm the expected result before moving to the next step. If
> anything in the "STOP conditions" section occurs, stop and report — do not
> improvise. When done, update the status row for this plan in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat e4654c5..HEAD -- lib/bsts_nx/kalman_filter.ex`
> If it changed since this plan was written, compare the "Current state" excerpt
> below against the live code before proceeding; on a mismatch, STOP.

## Status

- **Priority**: P3
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: bug (hardening / error-clarity)
- **Planned at**: commit `e4654c5`, 2026-06-20

## Why this matters

`normalize_h_series/2` (the eager Kalman filter's per-step observation-matrix
normalizer) already raises clear `ArgumentError`s for almost every malformed `H`:
rank-1 length mismatch (line 491), multivariate static row-count mismatch (line
508), and the rank-3 ambiguous case (line 520). One narrow gap remains: for
**scalar observations** (`obs_dim == 1`), a rank-2 `H` of shape `{len, n}` where
`len` is neither `1` (a static row) nor `t_len` (time-varying) falls into the
"static matrix" branch unchecked. It is then replicated across all timesteps and
fails later with a confusing shape error (or, if dimensions coincidentally align,
could silently use the wrong matrix). This plan adds the missing guard so the
filter rejects such input up front with a clear message.

This is **error-clarity hardening**, not a silent-correctness fix — most malformed
`H` already raises. Scope it accordingly.

## Current state

`lib/bsts_nx/kalman_filter.ex` — public `filter/7` → `filter_with_pred/7` calls
`normalize_h_series(h, obs_list)` at line 117. The relevant `cond` branches inside
`normalize_h_series` are at lines 500-525. The branch to change is the
`rank == 2 ->` static branch at lines 504-514:

```elixir
            rank == 2 and obs_dim == 1 and len == t_len ->
              # Scalar-observation time-varying H encoded as {T, n}
              h |> Nx.to_batched(1) |> Enum.map(&Nx.squeeze(&1, axes: [0]))

            rank == 2 ->
              # Rank-2 tensors are static observation matrices by default.
              # This avoids misclassifying static multivariate H matrices when
              # rows(H) == T by coincidence.
              if obs_dim != 1 and len != obs_dim do
                raise ArgumentError,
                      "static observation matrix row count #{len} must match observation dimension #{obs_dim}"
              end

              h_t = to_tensor(h)
              Enum.map(0..(t_len - 1), fn _ -> h_t end)
```

Note the clause directly above (`rank == 2 and obs_dim == 1 and len == t_len`)
already handles the scalar time-varying case, so by the time control reaches
`rank == 2 ->` with `obs_dim == 1`, we know `len != t_len`. A legitimate static
scalar-observation `H` is a single row `{1, n}` (`len == 1`). Any other `len` is
ambiguous.

## Commands you will need

| Purpose | Command | Expected |
|---------|---------|----------|
| Compile | `mix compile --warnings-as-errors` | exit 0 |
| New + existing H tests | `mix test test/kalman_filter_h_normalization_test.exs test/kalman_filter_h_length_test.exs` | all pass |
| Broad Kalman suites | `mix test test/kalman_filter_r_shape_test.exs test/kalman_filter_length_test.exs test/kalman_filter_multivariate_observation_test.exs test/missing_data_test.exs` | all pass |
| Format | `mix format && mix format --check-formatted` | exit 0 |
| Full gate | `bash scripts/ci.sh` | exit 0 |

(If local `mix` errors with a `mise` usage message, prefix with `mise exec -- `.)

## Scope

**In scope**:
- `lib/bsts_nx/kalman_filter.ex` — the single `rank == 2 ->` branch (lines 504-514).
- `test/kalman_filter_h_length_test.exs` (create).

**Out of scope** (do NOT touch):
- The other branches of `normalize_h_series` (rank-1, the time-varying `{T,n}`
  clause, rank-3+, and the `true ->` fallback). They already validate correctly.
- `normalize_h_multi/3` (the compiled-multi path, lines 565-598) — it already
  raises for unsupported shapes via its `true ->` clause.
- The downstream filter math.

## Git workflow

- Branch: `advisor/006-validate-time-varying-h-length`
- Commit message: conventional commits, e.g. `fix: reject ambiguous rank-2 observation matrix lengths`.
- Do NOT push or open a PR unless instructed.

## Steps

### Step 1: Replace the single-`if` guard with a `cond` that also rejects scalar-obs ambiguity

Replace the `if obs_dim != 1 and len != obs_dim do ... end` guard (lines 508-511)
with a `cond` that preserves the existing multivariate check and adds the
scalar-observation check. The branch becomes:

```elixir
            rank == 2 ->
              # Rank-2 tensors are static observation matrices by default.
              # This avoids misclassifying static multivariate H matrices when
              # rows(H) == T by coincidence.
              cond do
                obs_dim == 1 and len != 1 ->
                  raise ArgumentError,
                        "rank-2 observation matrix with #{len} rows is ambiguous for " <>
                          "#{t_len} scalar observations; pass a {1, n} row for a static H " <>
                          "or a {#{t_len}, n} matrix for time-varying H"

                obs_dim != 1 and len != obs_dim ->
                  raise ArgumentError,
                        "static observation matrix row count #{len} must match observation dimension #{obs_dim}"

                true ->
                  :ok
              end

              h_t = to_tensor(h)
              Enum.map(0..(t_len - 1), fn _ -> h_t end)
```

This preserves all currently-valid inputs:
- scalar-obs static `{1, n}` (`obs_dim == 1`, `len == 1`) → `true` branch, accepted;
- multivariate static `{m, n}` with `m == obs_dim` → `true` branch, accepted;
- scalar-obs time-varying `{t_len, n}` → handled by the earlier clause, never reaches here.

**Verify**: `mix compile --warnings-as-errors` → exit 0.

### Step 2: Add the regression test

Create `test/kalman_filter_h_length_test.exs`:

```elixir
defmodule BstsNxKalmanFilterHLengthTest do
  use ExUnit.Case, async: true

  alias BstsNx.KalmanFilter

  test "rejects a rank-2 H whose row count is neither 1 nor the series length (scalar obs)" do
    obs = [1.0, 2.0, 3.0]
    # {4, 1}: 4 rows, but the series has 3 scalar observations and a static H
    # would be {1, 1}. 4 is ambiguous.
    h = Nx.tensor([[1.0], [1.0], [1.0], [1.0]])

    assert_raise ArgumentError, ~r/ambiguous/, fn ->
      KalmanFilter.filter(obs, 1.0, h, 0.1, 0.5, 0.0, 1.0)
    end
  end

  test "still accepts a {1, n} static row for scalar observations" do
    obs = [1.0, 2.0, 3.0]
    h = Nx.tensor([[1.0, 0.0]])
    x0 = Nx.tensor([0.0, 0.0])
    p0 = Nx.eye(2)

    result = KalmanFilter.filter(obs, Nx.eye(2), h, 0.1, 0.5, x0, p0)
    assert length(result) == 3
  end

  test "still accepts a {T, n} time-varying H for scalar observations" do
    obs = [1.0, 2.0, 3.0]
    h = Nx.tensor([[1.0], [1.0], [1.0]])
    result = KalmanFilter.filter(obs, 1.0, h, 0.1, 0.5, 0.0, 1.0)
    assert length(result) == 3
  end
end
```

**Confirm the bug first (recommended)**: stash the `lib/` edit
(`git stash push -- lib/bsts_nx/kalman_filter.ex`), run
`mix test test/kalman_filter_h_length_test.exs`, and confirm the first test does
NOT pass (it either gets no exception, or a non-`ArgumentError`, or an
`ArgumentError` whose message lacks "ambiguous"). Then `git stash pop` and confirm
all three pass. If the first test already passes against the unfixed code with an
"ambiguous" message, STOP and report — the gap may not exist as described.

**Verify**: `mix test test/kalman_filter_h_length_test.exs` → 3 tests, 0 failures.

### Step 3: Confirm no existing valid usage broke

**Verify**:
- `mix test test/kalman_filter_h_normalization_test.exs test/kalman_filter_r_shape_test.exs test/kalman_filter_length_test.exs test/kalman_filter_multivariate_observation_test.exs test/missing_data_test.exs` → all pass.

If any existing test now fails because it relied on a rank-2 scalar-obs `H` with
`len ∉ {1, t_len}`, STOP and report — that would mean the "ambiguous" shape was an
intended (if odd) usage, and the guard needs reconsideration.

## Test plan

- New file `test/kalman_filter_h_length_test.exs`: one negative test (ambiguous
  `{4,1}` H raises with "ambiguous") and two positive tests (static `{1,n}` and
  time-varying `{T,n}` still accepted).
- Pattern reference: `test/kalman_filter_h_normalization_test.exs` for H-shape test
  construction.
- Verification: `mix test test/kalman_filter_h_length_test.exs` → all pass, plus
  the broad Kalman suites in Step 3.

## Done criteria

ALL must hold:

- [ ] `mix compile --warnings-as-errors` exits 0.
- [ ] `test/kalman_filter_h_length_test.exs` exists and passes (3 tests).
- [ ] The Step 3 broad Kalman suites all pass (no valid usage regressed).
- [ ] `mix format --check-formatted` exits 0.
- [ ] Only `lib/bsts_nx/kalman_filter.ex` and the new test file modified (`git status`).
- [ ] `plans/README.md` status row for 006 updated to DONE.

## STOP conditions

Stop and report (do not improvise) if:

- The "Current state" excerpt doesn't match the live code (drift since `e4654c5`).
- An existing Kalman test breaks because it depended on the previously-accepted
  ambiguous shape.
- The negative test cannot be made to go red on the unfixed code (the gap may
  already be closed differently).

## Maintenance notes

- This only hardens the **eager** `normalize_h_series` path. The compiled-multi
  path (`normalize_h_multi`) already rejects unsupported shapes via its `true ->`
  clause (lines 594-596); no change needed there.
- If a future feature legitimately needs a multi-row static scalar-observation `H`,
  this guard is where to revisit.
- Reviewer should confirm the three accepted-shape cases (scalar static `{1,n}`,
  multivariate static `{m,n}`, scalar time-varying `{T,n}`) are all still covered.
