# Plan 010: Add a scalar-vs-multi agreement test for the compiled Kalman filter

> **Executor instructions**: Follow this plan step by step. Run every verification
> command and confirm the expected result before moving to the next step. If
> anything in the "STOP conditions" section occurs, stop and report — do not
> improvise. When done, update the status row for this plan in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat e4654c5..HEAD -- lib/bsts_nx/kalman_filter.ex test/kalman_filter_defn_multi_test.exs`
> If either changed since this plan was written, compare the "Current state"
> excerpts below against the live code before proceeding; on a mismatch, STOP.

## Status

- **Priority**: P3
- **Effort**: S
- **Risk**: LOW (adds a test; touches no library code)
- **Depends on**: none
- **Category**: tests
- **Planned at**: commit `e4654c5`, 2026-06-30

## Why this matters

This repo's top correctness risk is divergence between its **dual
implementations** of the same algorithm. The compiled Kalman filter has two:

- `KalmanFilter.filter_defn/7` — scalar-state path, returns `{xs, ps}` with shapes
  `{t}` and `{t}`.
- `KalmanFilter.filter_defn_multi/7` — multi-dimensional-state path, returns
  `{xs, ps}` with shapes `{t, n}` and `{t, n, n}`.

`test/kalman_filter_defn_multi_test.exs` checks `filter_defn_multi` against the
**eager** `filter_with_pred`, and the scalar `filter_defn` is exercised by the
smoother defn suite — but **nothing checks that the two compiled paths agree at
`n == 1`.** For a scalar local-level model a caller can legitimately use either
(`filter_defn` directly, or `filter_defn_multi` with `1×1` matrices, e.g. a
regression model with a single predictor reduces to `n == 1`). If the two diverge
in near-zero-innovation handling or missing-data (NaN) skipping, scalar results
quietly differ depending on which entry point a higher-level module happened to
call. This plan pins them together with a focused agreement test, including the
missing-observation path.

## Current state

`KalmanFilter.filter_defn/7` (scalar) — `lib/bsts_nx/kalman_filter.ex`. It builds a
per-step `h_vec` and calls the compiled `filter_defn_vec`, which initializes
`xs = Nx.broadcast(zero_x, {t})` and `ps = Nx.broadcast(zero_p, {t})` — i.e. scalar
state, output shapes `{t}` / `{t}`. Missing observations are encoded as NaN. The
arity-7 signature is `filter_defn(observations, f, h, q, r, x0, p0)` (the argument
error at `kalman_filter.ex:286` names `filter_defn/7`).

`KalmanFilter.filter_defn_multi/7` — `lib/bsts_nx/kalman_filter.ex:301-336`:

```elixir
  @doc """
  Compiled Kalman filter for multi-dimensional state with scalar observations.

  Inputs:
    * `observations` shape `{t}` (missing values encoded as NaN)
    * `f` shape `{n, n}`
    * `h` static row shape `{n}` / `{1, n}` or time-varying shape `{t, n}`
    * `q` shape `{n, n}` (or scalar when `n == 1`)
    * `r` scalar
    * `x0` shape `{n}` (or scalar when `n == 1`)
    * `p0` shape `{n, n}` (or scalar when `n == 1`)

  Returns `{xs, ps}` with shapes `{t, n}` and `{t, n, n}`.
  """
  def filter_defn_multi(observations, f, h, q, r, x0, p0) do
    type = {:f, 64}
    obs_t = Nx.as_type(observations, type)
    ...
  end
```

The existing agreement test to model on — `test/kalman_filter_defn_multi_test.exs`:

```elixir
  test "filter_defn_multi matches eager filter_with_pred for scalar observations" do
    nan = Nx.Constants.nan() |> Nx.to_number()
    observations = [1.0, 1.4, nan, 2.1, 2.6]
    obs_t = Nx.tensor(observations, type: {:f, 32})

    f = Nx.tensor([[1.0, 1.0], [0.0, 1.0]])
    h = Nx.tensor([[1.0, 0.0]])
    ...
    {xs_defn, ps_defn} = KalmanFilter.filter_defn_multi(obs_t, f, h, q, r, x0, p0)

    assert Nx.all_close(xs_defn, xs_eager, atol: 1.0e-4, rtol: 1.0e-4) |> Nx.to_number() == 1
  end
```

This plan adds a new test asserting `filter_defn` (scalar) and
`filter_defn_multi` (with `1×1` inputs) agree once the multi output is squeezed
from `{t, 1}`/`{t, 1, 1}` down to `{t}`/`{t}`.

## Commands you will need

| Purpose | Command | Expected |
|---------|---------|----------|
| Run the new test | `mix test test/kalman_filter_defn_scalar_multi_agreement_test.exs` | all pass |
| Existing multi test still green | `mix test test/kalman_filter_defn_multi_test.exs` | all pass |
| Compile | `mix compile --warnings-as-errors` | exit 0 |
| Format | `mix format && mix format --check-formatted` | exit 0 |
| Full gate | `bash scripts/ci.sh` | exit 0 |

(If local `mix` errors with a `mise exec` usage message, prefix with `mise exec -- `.)

## Scope

**In scope** (the only file you may create/modify):
- `test/kalman_filter_defn_scalar_multi_agreement_test.exs` (create)

**Out of scope** (do NOT touch):
- `lib/bsts_nx/kalman_filter.ex` — this plan tests existing behavior; it must not
  change it. If the test reveals a real divergence, that is a separate fix (STOP).
- `test/kalman_filter_defn_multi_test.exs` — leave it unchanged.

## Git workflow

- Branch: `advisor/010-filter-defn-scalar-multi-agreement-test`
- Commit message: `test: assert scalar and multi compiled Kalman filters agree at n=1`.
- Do NOT push or open a PR unless instructed.

## Steps

### Step 1: Create the agreement test

Create `test/kalman_filter_defn_scalar_multi_agreement_test.exs`:

```elixir
defmodule BstsNx.KalmanFilterDefnScalarMultiAgreementTest do
  use ExUnit.Case, async: true

  alias BstsNx.KalmanFilter

  # Squeeze multi outputs ({t,1} and {t,1,1}) down to scalar shape {t}.
  defp to_scalar({xs, ps}) do
    {Nx.squeeze(xs, axes: [1]), Nx.squeeze(ps, axes: [1, 2])}
  end

  test "filter_defn and filter_defn_multi(n=1) agree on a clean series" do
    obs = Nx.tensor([1.0, 2.0, 1.5, 3.0, 2.5, 4.0], type: {:f, 64})

    {xs_s, ps_s} = KalmanFilter.filter_defn(obs, 1.0, 1.0, 0.1, 0.5, 0.0, 1.0)

    {xs_m, ps_m} =
      KalmanFilter.filter_defn_multi(
        obs,
        Nx.tensor([[1.0]]),
        Nx.tensor([[1.0]]),
        Nx.tensor([[0.1]]),
        0.5,
        Nx.tensor([0.0]),
        Nx.tensor([[1.0]])
      )
      |> to_scalar()

    assert Nx.all_close(xs_s, xs_m, atol: 1.0e-6, rtol: 1.0e-6) |> Nx.to_number() == 1
    assert Nx.all_close(ps_s, ps_m, atol: 1.0e-6, rtol: 1.0e-6) |> Nx.to_number() == 1
  end

  test "filter_defn and filter_defn_multi(n=1) agree across a missing observation" do
    nan = Nx.Constants.nan() |> Nx.to_number()
    obs = Nx.tensor([1.0, 2.0, nan, 3.0, 2.5], type: {:f, 64})

    {xs_s, ps_s} = KalmanFilter.filter_defn(obs, 1.0, 1.0, 0.2, 0.4, 0.0, 1.0)

    {xs_m, ps_m} =
      KalmanFilter.filter_defn_multi(
        obs,
        Nx.tensor([[1.0]]),
        Nx.tensor([[1.0]]),
        Nx.tensor([[0.2]]),
        0.4,
        Nx.tensor([0.0]),
        Nx.tensor([[1.0]])
      )
      |> to_scalar()

    assert Nx.all_close(xs_s, xs_m, atol: 1.0e-6, rtol: 1.0e-6) |> Nx.to_number() == 1
    assert Nx.all_close(ps_s, ps_m, atol: 1.0e-6, rtol: 1.0e-6) |> Nx.to_number() == 1
  end
end
```

**Verify**: `mix test test/kalman_filter_defn_scalar_multi_agreement_test.exs` →
2 tests, 0 failures.

### Step 2: Confirm shapes before asserting closeness (debug aid only)

If either test fails on a **shape** error (not a value mismatch), the squeeze axes
are wrong for the actual output ranks. Confirm `filter_defn` returns `{t}`/`{t}` and
`filter_defn_multi` returns `{t,1}`/`{t,1,1}` by temporarily inspecting
`Nx.shape/1`, then fix the `to_scalar/1` axes. Do not change library code.

**Verify**: tests pass after any squeeze-axis correction.

### Step 3: Full gate

**Verify**:
- `mix test test/kalman_filter_defn_scalar_multi_agreement_test.exs test/kalman_filter_defn_multi_test.exs` → all pass.
- `mix format --check-formatted` → exit 0.

## Test plan

- New file `test/kalman_filter_defn_scalar_multi_agreement_test.exs`, two tests:
  clean-series agreement and missing-observation (NaN) agreement between the scalar
  and `n=1` multi compiled paths.
- Pattern reference: `test/kalman_filter_defn_multi_test.exs` for the
  `filter_defn_multi` call shape, NaN encoding, and `Nx.all_close` idiom.
- Verification: Steps 1 and 3.

## Done criteria

ALL must hold:

- [ ] `test/kalman_filter_defn_scalar_multi_agreement_test.exs` exists and passes (2 tests).
- [ ] `mix test test/kalman_filter_defn_multi_test.exs` still passes.
- [ ] `mix compile --warnings-as-errors` exits 0.
- [ ] `mix format --check-formatted` exits 0.
- [ ] Only the new test file is added; no `lib/` file modified (`git status`).
- [ ] `plans/README.md` status row for 010 updated to DONE.

## STOP conditions

Stop and report (do not improvise) if:

- The "Current state" excerpts don't match the live `kalman_filter.ex` (drift).
- After fixing squeeze axes, the **values** still disagree beyond `1e-6`: that is a
  genuine scalar-vs-multi divergence (a real bug). Report it with the two output
  tensors — do NOT loosen the tolerance to force a pass.
- The missing-observation test disagrees while the clean test passes: the two paths
  handle the NaN skip differently — report it specifically (this is the highest-value
  thing this test can catch).
- `filter_defn` or `filter_defn_multi` does not exist with the arity-7 signature in
  the "Current state" excerpt.

## Maintenance notes

- If `filter_defn_multi`'s output rank ever changes, `to_scalar/1`'s squeeze axes
  must be updated here.
- This is a behavior-pinning test: any future numerical change to one compiled path
  must be mirrored in the other, or this test goes red — which is the intent.
- A parallel agreement test already exists for the smoother
  (`test/smoother_rts_dedup_test.exs`, added by plan 004); this completes the same
  guarantee for the filter.
