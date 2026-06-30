# Plan 013: Clamp the percentile slice indices in `SpotAttributor.attribute_posterior`

> **Executor instructions**: Follow this plan step by step. Run every verification
> command and confirm the expected result before moving on. This is a defensive
> consistency change with NO behavior change for valid inputs — the existing
> posterior test suite is your regression oracle. If anything in "STOP conditions"
> occurs, stop and report. When done, update the status row in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat e4654c5..HEAD -- lib/bsts_nx/spot_attributor.ex lib/bsts_nx/forecaster.ex`
> If either changed, compare the "Current state" excerpts below against the live
> code before proceeding; on a mismatch, STOP.

## Status

- **Priority**: P3
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: bug (defensive)
- **Planned at**: commit `e4654c5`, 2026-06-30

## Why this matters

`SpotAttributor.attribute_posterior/5` computes credible-interval bounds by slicing
a sorted draw tensor at computed percentile indices — but, unlike the equivalent
code in `Forecaster`, it does **not** clamp those indices into range:

- `spot_attributor.ex` slices at `[lower_idx, 0]` and `[upper_idx, 0]` directly.
- `forecaster.ex:389,395` slices at `[max(lower_idx, 0), 0]` and
  `[min(upper_idx, last), 0]`.

For every **valid** `alpha ∈ (0, 1)` the indices are provably in range
(`upper_idx = ceil((1 − alpha/2)·(n_draws−1)) ≤ n_draws−1`, `lower_idx ≥ 0`), so this
is **not** a live bug — it is a defensive-consistency gap. The two near-identical
interval routines should clamp identically so a future refactor, an off-by-one, or a
caller passing an out-of-range `alpha` fails safe (clamped) in both places rather
than producing an `Nx.slice` error in one and not the other. This plan brings
`SpotAttributor` in line with `Forecaster`. Because there is no valid input that
reaches the unclamped branch, there is no red-before-green regression test to write;
correctness is verified by the existing posterior suite staying green plus a grep
that the clamps are present.

## Current state

`lib/bsts_nx/spot_attributor.ex` — `n_draws` is defined at line 165
(`n_draws = length(counterfactual_draws)`). The interval computation
(around lines 242-256):

```elixir
    sorted = Nx.sort(lifts_t, axis: 0)
    lower_idx = trunc(Float.floor(alpha / 2.0 * max(n_draws - 1, 0)))
    upper_idx = trunc(Float.ceil((1.0 - alpha / 2.0) * max(n_draws - 1, 0)))

    lowers =
      sorted
      |> Nx.slice([lower_idx, 0], [1, n_spots])
      |> Nx.squeeze(axes: [0])
      |> Nx.to_flat_list()

    uppers =
      sorted
      |> Nx.slice([upper_idx, 0], [1, n_spots])
      |> Nx.squeeze(axes: [0])
      |> Nx.to_flat_list()
```

The exemplar to match — `lib/bsts_nx/forecaster.ex:382-397` (DO NOT MODIFY — reference
only). Note `last = n - 1` there is the equivalent of `n_draws - 1` here:

```elixir
      sorted = Nx.sort(traj_t, axis: 0)
      last = n - 1
      lower_idx = trunc(Float.floor(alpha / 2.0 * last))
      upper_idx = trunc(Float.ceil((1.0 - alpha / 2.0) * last))

      lower =
        sorted
        |> Nx.slice([max(lower_idx, 0), 0], [1, horizon])
        |> Nx.squeeze(axes: [0])
        |> Nx.to_flat_list()

      upper =
        sorted
        |> Nx.slice([min(upper_idx, last), 0], [1, horizon])
        |> Nx.squeeze(axes: [0])
        |> Nx.to_flat_list()
```

The sort is over axis 0 (the `n_draws` draws), so the upper index must be clamped to
`n_draws - 1`, not `n_spots`.

## Commands you will need

| Purpose | Command | Expected |
|---------|---------|----------|
| Regression oracle | `mix test test/spot_attributor_posterior_test.exs test/spot_attributor_test.exs` | all pass |
| Compile | `mix compile --warnings-as-errors` | exit 0 |
| Format | `mix format && mix format --check-formatted` | exit 0 |
| Full gate | `bash scripts/ci.sh` | exit 0 |

(If local `mix` errors with a `mise exec` usage message, prefix with `mise exec -- `.)

## Scope

**In scope** (the only files you may modify):
- `lib/bsts_nx/spot_attributor.ex` — the two `Nx.slice` index expressions only.
- `test/spot_attributor_posterior_test.exs` — optionally add the boundary test in
  Step 3.

**Out of scope** (do NOT touch):
- `lib/bsts_nx/forecaster.ex` — it is already correct; it is the reference.
- The `lower_idx` / `upper_idx` *formulas* (lines 243-244) — leave the
  `max(n_draws - 1, 0)` factor as-is; only the **slice** indices change.
- Any other percentile/interval code elsewhere in the repo.

## Git workflow

- Branch: `advisor/013-spot-attributor-clamp-percentile-index`
- Commit message: `fix: clamp posterior percentile slice indices in SpotAttributor`.
- Do NOT push or open a PR unless instructed.

## Steps

### Step 1: Establish the green baseline

`mix test test/spot_attributor_posterior_test.exs test/spot_attributor_test.exs`

**Verify**: all pass. If not, STOP — the baseline is not green.

### Step 2: Clamp the two slice indices

In `lib/bsts_nx/spot_attributor.ex`, change the `lowers` slice from
`Nx.slice([lower_idx, 0], [1, n_spots])` to
`Nx.slice([max(lower_idx, 0), 0], [1, n_spots])`, and the `uppers` slice from
`Nx.slice([upper_idx, 0], [1, n_spots])` to
`Nx.slice([min(upper_idx, n_draws - 1), 0], [1, n_spots])`.

Resulting block:

```elixir
    lowers =
      sorted
      |> Nx.slice([max(lower_idx, 0), 0], [1, n_spots])
      |> Nx.squeeze(axes: [0])
      |> Nx.to_flat_list()

    uppers =
      sorted
      |> Nx.slice([min(upper_idx, n_draws - 1), 0], [1, n_spots])
      |> Nx.squeeze(axes: [0])
      |> Nx.to_flat_list()
```

**Verify**: `mix compile --warnings-as-errors` → exit 0.

### Step 3 (optional but recommended): Lock ordered intervals at small `n_draws`

Add a test to `test/spot_attributor_posterior_test.exs` (inside a new `describe` or
at the end) that runs `attribute_posterior` with a small number of draws and asserts
each attribution's interval is ordered. Model the setup on the existing helpers
(`make_spot/3`, `generate_draws/5`) at the top of that file:

```elixir
  describe "credible-interval boundary safety" do
    test "lift_lower <= lift_upper holds with few draws" do
      obs = [110.0, 112.0, 108.0, 115.0, 109.0]
      spots = [make_spot("s1", 0, 3), make_spot("s2", 3, 5)]
      draws = generate_draws(100.0, 2.0, 5, 3, 7)

      result = SpotAttributor.attribute_posterior(obs, spots, draws, 0.0)

      Enum.each(result.attributions, fn attr ->
        assert attr.lift_lower <= attr.lift_upper
      end)
    end
  end
```

(This test passes both before and after the fix — the indices are in range for valid
`alpha`. It documents the boundary contract and guards future refactors; it is not a
red-before-green test, because no valid input reaches the unclamped branch.)

**Verify**: `mix test test/spot_attributor_posterior_test.exs` → all pass.

### Step 4: Confirm no behavior change

Re-run the regression oracle from Step 1:

`mix test test/spot_attributor_posterior_test.exs test/spot_attributor_test.exs`

**Verify**: all pass, identical to the Step 1 baseline. Any newly-failing test means
the clamp changed a previously-correct result (it should not, since valid indices are
unaffected by `max(_, 0)` / `min(_, n_draws - 1)`) — STOP and report.

## Test plan

- Regression oracle: the existing `spot_attributor_posterior_test.exs` and
  `spot_attributor_test.exs` must stay green — they exercise the interval path and
  prove the clamp introduces no behavior change.
- Optional new boundary test (Step 3) documenting `lift_lower <= lift_upper` with few
  draws.
- No red-before-green test exists by construction (the unclamped branch is
  unreachable for valid `alpha`); the change is defensive parity with `Forecaster`.

## Done criteria

ALL must hold:

- [ ] `grep -n "max(lower_idx, 0)" lib/bsts_nx/spot_attributor.ex` → 1 match.
- [ ] `grep -n "min(upper_idx, n_draws - 1)" lib/bsts_nx/spot_attributor.ex` → 1 match.
- [ ] `mix compile --warnings-as-errors` exits 0.
- [ ] `mix test test/spot_attributor_posterior_test.exs test/spot_attributor_test.exs` → all pass (unchanged from baseline).
- [ ] `mix format --check-formatted` exits 0.
- [ ] Only `spot_attributor.ex` (and optionally `spot_attributor_posterior_test.exs`) modified (`git status`).
- [ ] `plans/README.md` status row for 013 updated to DONE.

## STOP conditions

Stop and report (do not improvise) if:

- The "Current state" excerpt doesn't match `spot_attributor.ex` (drift).
- Any existing spot-attributor test fails after the clamp (it must not — that would
  mean the indices were NOT always in range, which is a different and bigger finding).
- You find the sort axis is not axis 0 (then `n_draws - 1` is the wrong clamp bound).

## Maintenance notes

- If `SpotAttributor` and `Forecaster` interval logic are ever consolidated into a
  shared `Utils` helper (a separate, larger refactor), this clamp must be preserved
  in the shared version.
- Reviewer should confirm the upper clamp uses `n_draws - 1` (the sorted axis size),
  not `n_spots`.
