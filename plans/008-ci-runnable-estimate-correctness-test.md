# Plan 008: Add a CI-runnable correctness test for `CausalImpact.estimate/4`

> **Executor instructions**: Follow this plan step by step. Run every verification
> command and confirm the expected result before moving to the next step. If
> anything in the "STOP conditions" section occurs, stop and report — do not
> improvise. When done, update the status row for this plan in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat e4654c5..HEAD -- lib/bsts_nx/causal_impact.ex test/causal_impact_test.exs`
> If either changed since this plan was written, compare the "Current state"
> excerpts below against the live code before proceeding; on a mismatch, STOP.

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: LOW (adds a test; touches no library code)
- **Depends on**: none
- **Category**: tests
- **Planned at**: commit `e4654c5`, 2026-06-30

## Why this matters

`CausalImpact.estimate/4` is the library's flagship public API (the README's
headline example). Yet **the default CI gate never runs it.** Its only direct
tests live in `test/causal_impact_test.exs`, which is `@moduletag :external`
(line 5) — and `scripts/ci.sh` / the GitHub `test` job both run
`mix test --exclude external`. The MCMC parameter-recovery test
(`test/parameter_recovery_test.exs`) is *also* `@moduletag :external` and carries a
`@tag skip:`. So a change that silently breaks `estimate/4`'s end-to-end behavior
(e.g. a regression in Gibbs sampling, forward simulation, or summary assembly)
would pass `bash scripts/ci.sh` cleanly.

This plan adds **one fast, fully deterministic, non-external** test that runs in the
default suite and asserts the estimator's load-bearing invariants on a synthetic
series with a large, unambiguous injected effect: the result shape is correct, the
summary numbers are finite, the credible interval is ordered, and a large positive
intervention is detected as positive. It is a smoke + light-recovery guard, not a
tight numerical assertion (which would flake), so it is safe for CI.

## Current state

`CausalImpact.estimate/4` contract — from `lib/bsts_nx/causal_impact.ex:68-123`:

- Signature: `estimate(observations, pre_period, post_period, opts \\ [])`.
- `pre_period` / `post_period` are **1-based inclusive** `{start, end}` tuples.
- Options include `:num_samples` (default 200), `:burn_in` (default num_samples/2),
  `:thin`, and `:seed` (integer; derives the Nx.Random key via
  `derive_estimate_keys/1`). A fixed `:seed` makes the run deterministic.
- The returned struct/map exposes (see the `:external` tests in
  `test/causal_impact_test.exs:20-22`): `result.point_effects` (list),
  `result.cumulative_effects` (list, length == `num_samples`),
  `result.actual` (list, length == post-period length).

`CausalImpact.summary/1` contract — from `lib/bsts_nx/causal_impact.ex:304-385`:

- `summary(result)` returns
  `%{point_effects: [...], cumulative_effect: %{...}, relative_effect: %{...}}`.
  `cumulative_effect` and `relative_effect` are each `%{mean, sd, lower, upper}`;
  `point_effects` is a LIST of such maps (one per post-period step). **There is NO
  `average_effect` field** — assert on `cumulative_effect`, `relative_effect`, or
  the per-step `point_effects`.
- With `num_samples: 1` the spread fields (`sd`, `lower`, `upper`) are the atom
  `:nan` (`test/causal_impact_test.exs:60-62`); `mean` is the single value. With
  several samples they are floats.

The existing external "detects positive effect" test to model the data shape on
(`test/causal_impact_test.exs:65-87`):

```elixir
  test "detects positive effect in synthetic data" do
    :rand.seed(:exsss, {1, 2, 3})
    pre_data = Enum.map(1..100, fn _ -> 50.0 + :rand.normal() * 5 end)
    post_data = Enum.map(1..50, fn _ -> 80.0 + :rand.normal() * 5 end)
    observations = pre_data ++ post_data

    result =
      CausalImpact.estimate(observations, {1, 100}, {101, 150},
        num_samples: 50,
        burn_in: 25,
        seed: 789
      )

    summary = CausalImpact.summary(result)
    assert summary.cumulative_effect.mean > 0
    assert summary.cumulative_effect.upper > 0
  end
```

This plan's new test is the **non-external, smaller, contract-locking** complement
of that one. It stays in its own file so the `:external` moduletag on
`causal_impact_test.exs` is untouched.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Run the new test | `mix test test/causal_impact_estimate_smoke_test.exs` | all pass |
| Confirm it runs in the default lane | `mix test --exclude external test/causal_impact_estimate_smoke_test.exs` | all pass (not skipped) |
| Compile | `mix compile --warnings-as-errors` | exit 0 |
| Format | `mix format && mix format --check-formatted` | exit 0 |
| Full gate | `bash scripts/ci.sh` | exit 0 |

(If local `mix` errors with a `mise exec` usage message, prefix with `mise exec -- `.)

## Scope

**In scope** (the only file you may create/modify):
- `test/causal_impact_estimate_smoke_test.exs` (create)

**Out of scope** (do NOT touch):
- `test/causal_impact_test.exs` — leave its `@moduletag :external` and tests as-is.
- `test/parameter_recovery_test.exs` — do not unskip or retag it; its slow recovery
  assertions are intentionally external.
- Any `lib/` file — this plan adds coverage only; it must not change behavior.

## Git workflow

- Branch: `advisor/008-ci-runnable-estimate-correctness-test`
- Commit message style: conventional commits, e.g.
  `test: cover CausalImpact.estimate/4 in the default CI lane`.
- Do NOT push or open a PR unless instructed.

## Steps

### Step 1: Create the test file

Create `test/causal_impact_estimate_smoke_test.exs` with the content below. It is
**not** tagged `:external`, so the default suite runs it. The data uses a fixed
`:rand` seed and a fixed estimator `:seed`, so the run is deterministic; the
asserted invariants are robust (a +30 effect on a ~50 baseline is detected by even
a simple local-level counterfactual).

```elixir
defmodule BstsNxCausalImpactEstimateSmokeTest do
  use ExUnit.Case, async: false

  alias BstsNx.CausalImpact

  # A value is finite if it equals itself (rules out the codebase's :nan atom and
  # float NaN) and is not an infinity atom.
  defp finite?(v) when is_number(v), do: v == v and v not in [:infinity, :neg_infinity]
  defp finite?(_), do: false

  test "estimate/4 recovers a large positive effect with finite, ordered summary" do
    :rand.seed(:exsss, {11, 22, 33})
    pre = Enum.map(1..60, fn _ -> 50.0 + :rand.normal() * 4.0 end)
    # +30 intervention — unambiguous for a local-level baseline.
    post = Enum.map(1..30, fn _ -> 80.0 + :rand.normal() * 4.0 end)
    obs = pre ++ post

    result =
      CausalImpact.estimate(obs, {1, 60}, {61, 90},
        num_samples: 60,
        burn_in: 30,
        seed: 4242
      )

    # Result shape (locks the public contract).
    assert length(result.actual) == 30
    assert length(result.cumulative_effects) == 60

    summary = CausalImpact.summary(result)

    # Finite summary (regression guard: a NaN init/posterior would poison these).
    assert finite?(summary.cumulative_effect.mean)
    assert finite?(summary.cumulative_effect.lower)
    assert finite?(summary.cumulative_effect.upper)
    assert finite?(summary.relative_effect.mean)

    # Credible interval is ordered.
    assert summary.cumulative_effect.lower <= summary.cumulative_effect.mean
    assert summary.cumulative_effect.mean <= summary.cumulative_effect.upper

    # A large positive intervention is detected as positive — both the cumulative
    # effect and the average pointwise effect are above zero.
    assert summary.cumulative_effect.mean > 0

    avg_point_mean =
      summary.point_effects
      |> Enum.map(& &1.mean)
      |> then(fn means -> Enum.sum(means) / length(means) end)

    assert avg_point_mean > 0
  end

  test "estimate/4 is deterministic for a fixed seed" do
    obs = List.duplicate(10.0, 40) ++ List.duplicate(18.0, 20)

    run = fn ->
      CausalImpact.estimate(obs, {1, 40}, {41, 60},
        num_samples: 40,
        burn_in: 20,
        seed: 7
      )
      |> CausalImpact.summary()
      |> Map.get(:cumulative_effect)
      |> Map.get(:mean)
    end

    assert run.() == run.()
  end
end
```

**Verify**: `mix test test/causal_impact_estimate_smoke_test.exs` → 2 tests, 0 failures.

### Step 2: Confirm field names against the live code

The test reads `result.actual`, `result.cumulative_effects`,
`summary.cumulative_effect.{mean,lower,upper}`, `summary.relative_effect.mean`, and
the per-step `summary.point_effects` means.
Confirm these exist: open `lib/bsts_nx/causal_impact.ex` and search for
`cumulative_effect`, `relative_effect`, `point_effects`, and the result builder
(`build_impact_result`).
If any asserted key does not exist, adjust the test to read the equivalent existing
field (do NOT change library code). If `summary/1` exposes no per-field map, STOP
and report.

**Verify**: `mix test test/causal_impact_estimate_smoke_test.exs` → all pass.

### Step 3: Confirm the test runs in the default (non-external) lane

`mix test --exclude external test/causal_impact_estimate_smoke_test.exs`

**Verify**: 2 tests run and pass (NOT "0 tests" / all skipped). If they are
excluded, the file accidentally inherited an `:external` tag — remove it.

### Step 4: Determinism sanity check

If the second test ("deterministic for a fixed seed") ever fails, the estimator is
not reproducible under a fixed `:seed`. That is a real finding but **out of scope**
for this plan — STOP and report rather than weakening the assertion to make it pass.

### Step 5: Full gate

**Verify**:
- `bash scripts/ci.sh` → exit 0 (the new test now runs inside the non-external lane).
- `mix format --check-formatted` → exit 0.

## Test plan

- New file `test/causal_impact_estimate_smoke_test.exs`, two tests:
  1. Large-positive-effect recovery with finite, ordered summary (the headline
     contract + the NaN-poisoning regression guard).
  2. Determinism under a fixed seed.
- Pattern reference: `test/causal_impact_test.exs:65-87` for data/call shape;
  `test/causal_impact_missing_init_test.exs` (added by plan 001) for the
  `finite?/1` helper idiom.
- Verification: Steps 1, 3, 5.

## Done criteria

ALL must hold:

- [ ] `test/causal_impact_estimate_smoke_test.exs` exists and passes.
- [ ] `grep -n "external" test/causal_impact_estimate_smoke_test.exs` → no matches.
- [ ] `mix test --exclude external test/causal_impact_estimate_smoke_test.exs` runs 2 tests, 0 failures (not skipped).
- [ ] `mix compile --warnings-as-errors` exits 0.
- [ ] `mix format --check-formatted` exits 0.
- [ ] `bash scripts/ci.sh` exits 0.
- [ ] Only `test/causal_impact_estimate_smoke_test.exs` is added; no `lib/` file modified (`git status`).
- [ ] `plans/README.md` status row for 008 updated to DONE.

## STOP conditions

Stop and report (do not improvise) if:

- The "Current state" `estimate/4` / `summary/1` contract excerpts don't match the
  live code (drift).
- Any asserted summary field does not exist and there is no clear equivalent.
- The recovery test fails to detect the +30 effect (`cumulative_effect.mean > 0`):
  do NOT loosen it — a failure here is a genuine estimator regression to report.
- The determinism test fails (report it; do not delete the test).
- The test proves flaky across repeated runs (`mix test --seed 0` then
  `mix test --seed 1` on this file): report it; the synthetic data/seed may need
  hardening, but do not silence the assertions.

## Maintenance notes

- Keep this test fast (small `num_samples`); it runs on every CI invocation. If the
  eager Gibbs path slows down, prefer reducing sample count over tagging it
  `:slow`/`:external` — the point is that it runs in the default gate.
- If `estimate/4`'s result keys are ever renamed, this test breaks loudly — that is
  intended (it documents the public contract).
- A natural follow-up (separate plan) is a CI-runnable component-variance recovery
  check for `estimate_structured/5`, complementing the external
  `parameter_recovery_test.exs`. Deferred here to keep this plan single-purpose.
