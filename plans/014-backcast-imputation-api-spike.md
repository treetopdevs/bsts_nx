# Plan 014 (SPIKE): Design a public backcast / missing-data imputation API

> **Executor instructions**: This is a **design + feasibility SPIKE**, not a
> ship-a-feature plan. Your deliverables are (1) a written design recommendation and
> (2) a minimal *prototype* that proves the existing machinery can back it, plus a
> feasibility test. Do NOT build a full production feature, expose public docs, or
> wire this into existing modules. Follow the steps, and if anything in "STOP
> conditions" occurs, stop and report. When done, update the status row in
> `plans/README.md` and leave your findings in
> `plans/014-backcast-imputation-findings.md`.
>
> **Drift check (run first)**:
> `git diff --stat e4654c5..HEAD -- lib/bsts_nx/causal_impact.ex lib/bsts_nx/smoother.ex lib/bsts_nx/kalman_filter.ex`
> If any changed since this plan was written, compare the "Current state" excerpts
> below against the live code before proceeding; on a mismatch, STOP.

## Status

- **Priority**: P3
- **Effort**: M (spike — investigation + prototype, not a full feature)
- **Risk**: LOW (prototype is `@moduledoc false`, not merged into the public surface)
- **Depends on**: none
- **Category**: direction
- **Planned at**: commit `e4654c5`, 2026-06-30

## Why this matters

The library can already impute missing observations — it just doesn't expose it.
`CausalImpact.estimate_from_filter/3` masks the post-period observations to NaN, runs
the compiled Kalman filter and RTS smoother, and reads back state-space estimates at
the masked positions (that *is* imputation). But there is **no public API** for a user
who simply has a time series with gaps (sensor dropouts, delayed reporting) and wants
the model's best estimate — with uncertainty — at the missing positions. Forecasting
is forward-only; backcast/imputation (smoothing backward in time to fill interior
gaps) is a common data-cleaning need that the existing smoother machinery already
supports.

This spike answers: **can a clean, public `impute`/`backfill` API be built as a thin
wrapper over the existing filter+smoother, and what should its surface be?** The
output is a recommendation the maintainer can act on — not a shipped feature.

## Current state (the machinery to reuse)

`CausalImpact.estimate_from_filter/3` already composes the full pipeline —
`lib/bsts_nx/causal_impact.ex:455-471`:

```elixir
      masked_obs = mask_observations_at_indices(obs_tensor, t, valid_indices)

      # Filter
      {xs, ps} = BstsNx.KalmanFilter.filter_defn(masked_obs, f, h, q, r, x0, p0)

      # Smooth
      {sxs, sps} = BstsNx.Smoother.rts_defn(xs, ps, f, q)

      # Extract intervention period values
      idx_tensor = Nx.tensor(valid_indices, type: {:s, 64})
      actual_vals = Nx.take(obs_tensor, idx_tensor) |> Nx.to_flat_list()
      raw_state_vals = Nx.take(sxs, idx_tensor) |> Nx.to_flat_list()
      state_var_vals = Nx.take(sps, idx_tensor) |> Nx.to_flat_list()
      # Project smoothed states through h to get baseline in observation space
      h_vals = h_intervention_values(h, valid_indices)
      baseline_vals = Enum.zip(h_vals, raw_state_vals) |> Enum.map(fn {hi, xi} -> hi * xi end)
```

Supporting pieces (public, already tested):

- `BstsNx.KalmanFilter.filter_defn/7` — scalar compiled filter; missing obs encoded as
  NaN are skipped (the filter does prediction-only at those steps).
- `BstsNx.Smoother.rts_defn/4` — scalar compiled RTS smoother
  (`lib/bsts_nx/smoother.ex:49-57`); fills masked positions with smoothed estimates.
- `BstsNx.Smoother.rts_defn_matrix/4` — multi-dimensional analogue
  (`lib/bsts_nx/smoother.ex:65-`).
- Missing-data convention: `nil` in eager code, `NaN` in defn/tensor code
  (see `mask_observations_at_indices` at `causal_impact.ex:1024`, which builds a NaN
  mask).

The gap is purely **interface**: there is no public entry point that takes a series
with gaps + a model and returns imputed values + uncertainty at the gaps.

## Commands you will need

| Purpose | Command | Expected |
|---------|---------|----------|
| Compile (incl. prototype) | `mix compile --warnings-as-errors` | exit 0 |
| Feasibility test | `mix test test/imputer_prototype_test.exs` | all pass |
| Format | `mix format --check-formatted` | exit 0 |
| Full gate | `bash scripts/ci.sh` | exit 0 |

(If local `mix` errors with a `mise exec` usage message, prefix with `mise exec -- `.)

## Scope

**In scope** (you may create these):
- `plans/014-backcast-imputation-findings.md` — the design recommendation (primary deliverable).
- `lib/bsts_nx/imputer.ex` — a **prototype** module marked `@moduledoc false` (so ExDoc
  does not publish it and it is clearly not yet public API).
- `test/imputer_prototype_test.exs` — feasibility tests for the prototype.

**Out of scope** (do NOT touch):
- `lib/bsts_nx/causal_impact.ex`, `smoother.ex`, `kalman_filter.ex` — read them, reuse
  their public functions, but do not modify them. In particular, do NOT refactor
  `estimate_from_filter/3` to share code with the prototype yet (that is a follow-up
  the maintainer decides on).
- `mix.exs` docs config — do not add the prototype to `groups_for_modules`.
- Any structured/ModelSpec or multivariate generalization — the prototype is
  **scalar-only**; structured is an open question, not a deliverable.

## Git workflow

- Branch: `advisor/014-backcast-imputation-api-spike`
- Commit message: `spike: prototype scalar backcast/imputation over filter+smoother`.
- Do NOT push or open a PR unless instructed.

## Steps

### Step 1: Confirm the machinery does prediction-only at NaN (the premise)

Write a throwaway check (you can do it in `iex -S mix` or as the first feasibility
test): build a scalar series `[1.0, 2.0, NaN, 4.0, 5.0]`, run
`KalmanFilter.filter_defn/7` then `Smoother.rts_defn/4` with a local-level model
(`f=1.0, h=1.0`, modest `q`, `r`), and confirm the smoothed state at index 2 is a
finite value between its neighbors (≈ 3.0 for this ramp). If the smoothed value at the
NaN position is NaN or wildly off, STOP — the premise (smoother fills gaps) is wrong
and the rest of the spike is invalid.

**Verify**: smoothed value at the gap is finite and plausibly interpolated.

### Step 2: Prototype `BstsNx.Imputer` (scalar, `@moduledoc false`)

Create `lib/bsts_nx/imputer.ex`:

```elixir
defmodule BstsNx.Imputer do
  @moduledoc false
  # PROTOTYPE (plan 014 spike). Scalar local-level imputation over the existing
  # compiled filter + RTS smoother. Not public API — pending design review.

  alias BstsNx.{KalmanFilter, Smoother}

  @doc false
  # observations: list of numbers with `nil` for gaps.
  # Returns a list of maps, one per input index:
  #   %{index:, observed: number | nil, imputed: float, std: float, missing?: boolean}
  def impute(observations, opts \\ []) when is_list(observations) do
    f = Keyword.get(opts, :f, 1.0)
    h = Keyword.get(opts, :h, 1.0)
    q = Keyword.get(opts, :q, 1.0)
    r = Keyword.get(opts, :r, 1.0)
    x0 = Keyword.get(opts, :x0, first_finite(observations))
    p0 = Keyword.get(opts, :p0, 1.0)

    nan = Nx.Constants.nan() |> Nx.to_number()
    masked = Enum.map(observations, fn v -> if is_nil(v), do: nan, else: v * 1.0 end)
    obs_t = Nx.tensor(masked, type: {:f, 64})

    {xs, ps} = KalmanFilter.filter_defn(obs_t, f, h, q, r, x0, p0)
    {sxs, sps} = Smoother.rts_defn(xs, ps, f, q)

    imputed = sxs |> Nx.multiply(h) |> Nx.to_flat_list()
    # observation-space std at each step: sqrt(h^2 * P^s + r)
    stds =
      sps
      |> Nx.multiply(h * h)
      |> Nx.add(r)
      |> Nx.max(0.0)
      |> Nx.sqrt()
      |> Nx.to_flat_list()

    observations
    |> Enum.with_index()
    |> Enum.map(fn {v, i} ->
      %{
        index: i,
        observed: v,
        imputed: Enum.at(imputed, i),
        std: Enum.at(stds, i),
        missing?: is_nil(v)
      }
    end)
  end

  defp first_finite(obs), do: Enum.find(obs, 0.0, &is_number/1) * 1.0
end
```

(This composition mirrors `estimate_from_filter/3`. Adjust the `std` formula or
`x0` default if Step 1 showed a better choice — record any deviation in the findings.)

**Verify**: `mix compile --warnings-as-errors` → exit 0.

### Step 3: Feasibility tests

Create `test/imputer_prototype_test.exs`:

```elixir
defmodule BstsNx.ImputerPrototypeTest do
  use ExUnit.Case, async: true

  alias BstsNx.Imputer

  test "imputes an interior gap on a linear ramp near the true value" do
    # True series is the ramp 1..7; index 3 (value 4.0) is missing.
    obs = [1.0, 2.0, 3.0, nil, 5.0, 6.0, 7.0]
    rows = Imputer.impute(obs, f: 1.0, h: 1.0, q: 1.0, r: 0.25)

    gap = Enum.find(rows, & &1.missing?)
    assert gap.index == 3
    assert gap.missing?
    # Smoothed estimate should land near the true 4.0 and carry finite uncertainty.
    assert_in_delta gap.imputed, 4.0, 1.0
    assert gap.std > 0.0 and gap.std == gap.std
  end

  test "leaves observed points finite and returns one row per input" do
    obs = [10.0, nil, 12.0, 13.0]
    rows = Imputer.impute(obs)
    assert length(rows) == length(obs)
    assert Enum.all?(rows, fn r -> is_float(r.imputed) and r.imputed == r.imputed end)
  end
end
```

**Verify**: `mix test test/imputer_prototype_test.exs` → 2 tests, 0 failures. If the
ramp imputation is not within the delta, widen the delta *once* and record the actual
value in the findings — do not chase tight numerics in a spike. If it is wildly off
(e.g. 0.0 or NaN), STOP and report.

### Step 4: Write the design findings

Create `plans/014-backcast-imputation-findings.md` answering, with evidence from your
prototype:

1. **Feasibility**: does the prototype impute interior gaps correctly? Include the
   ramp test's actual imputed value and std.
2. **API home**: recommend ONE — a new public `BstsNx.Imputer`, a `Forecaster.backfill/2`,
   or a `Smoother` convenience. Give the proposed public signature and return shape.
3. **Missing-data convention**: should the public API accept `nil` (eager idiom),
   `NaN`, or both? (The prototype uses `nil`.)
4. **Uncertainty semantics**: state-space std vs observation-space std (`sqrt(h²P+r)`);
   which should the public API return, and labelled how?
5. **Generalization**: what would structured/`ModelSpec` and multivariate support
   require? (Note `rts_defn_matrix/4` exists for the multi case.) Keep this as open
   questions, not work.
6. **Code-sharing**: should `estimate_from_filter/3` and the new API extract a shared
   private "mask → filter → smooth → project" helper? Sketch the seam; do not build it.

**Verify**: the findings file exists and addresses all six points.

### Step 5: Gate

**Verify**: `mix format --check-formatted` → exit 0; `bash scripts/ci.sh` → exit 0
(the prototype + its test are part of the suite but `@moduledoc false` keeps the
module out of published docs).

## Deliverables / Done criteria

ALL must hold:

- [ ] `plans/014-backcast-imputation-findings.md` exists and answers the six questions in Step 4.
- [ ] `lib/bsts_nx/imputer.ex` exists, is `@moduledoc false`, and compiles with `--warnings-as-errors`.
- [ ] `test/imputer_prototype_test.exs` passes (2 tests).
- [ ] `mix compile --warnings-as-errors` exits 0; `mix format --check-formatted` exits 0; `bash scripts/ci.sh` exits 0.
- [ ] No existing `lib/` module was modified (`git status` shows only the new files).
- [ ] `mix.exs` `groups_for_modules` was NOT changed (prototype stays unpublished).
- [ ] `plans/README.md` status row for 014 updated to DONE (with a one-line pointer to the findings).

## STOP conditions

Stop and report (do not improvise) if:

- The "Current state" excerpts don't match the live code (drift).
- Step 1 shows the smoother does NOT fill NaN positions with finite, interpolated
  values — the whole premise is wrong; report what you observed.
- The ramp imputation returns NaN or is grossly wrong (not just imprecise).
- Implementing the prototype would require modifying `causal_impact.ex`/`smoother.ex`/
  `kalman_filter.ex` (it should not — only their public functions are called).

## Maintenance notes

- This is exploratory. The `BstsNx.Imputer` prototype is intentionally `@moduledoc false`
  and must not be advertised as public API until the maintainer approves the design.
- The obvious follow-up (a separate plan, if the maintainer greenlights) is to:
  extract the shared "mask → filter → smooth → project" helper so `estimate_from_filter/3`
  and `Imputer` share one code path; decide the structured/multivariate story; and
  promote the module into `mix.exs` docs with a real `@moduledoc` + guide section.
- Reviewer should evaluate the *design recommendation* first; the prototype is evidence,
  not the product.
