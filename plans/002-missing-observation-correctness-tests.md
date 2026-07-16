# Plan 002: Add correctness tests for missing-observation handling (compiled filter + structured Gibbs)

> **Executor instructions**: Follow this plan step by step. Run every verification
> command and confirm the expected result before moving to the next step. If
> anything in the "STOP conditions" section occurs, stop and report — do not
> improvise. When done, update the status row for this plan in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat e4654c5..HEAD -- lib/bsts_nx/kalman_filter.ex lib/bsts_nx/smoother.ex lib/bsts_nx/gibbs_sampler test/missing_data_test.exs test/gibbs_sampler_missing_observations_test.exs`
> If any of those changed since this plan was written, compare the "Current state"
> excerpts below against the live code before proceeding; on a mismatch, treat it
> as a STOP condition.
>
> This plan **adds tests only**. It must not modify any file under `lib/`.

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: LOW
- **Depends on**: none
- **Category**: tests
- **Planned at**: commit `e4654c5`, 2026-06-20

## Why this matters

Missing-observation support (`nil` / `NaN`) is a documented feature (README
"Missing observations" section). Current coverage is uneven:

- The **eager** Kalman filter has a precise analytic test
  (`test/missing_data_test.exs`): for `[1.0, nil, 3.0]` the missing step's
  estimate equals the prediction-only step.
- The **compiled** (`Nx.Defn`) filter's missing-data path
  (`filter_defn`, NaN sentinel) has **no dedicated correctness test** — only
  end-to-end usage.
- The **Gibbs** missing-data tests (`test/gibbs_sampler_missing_observations_test.exs`,
  and `test/gibbs_structured_test.exs` "handles nil observations") assert the
  sampler runs and produces finite, positive variances, but never assert the
  *handling is correct* (e.g. that the compiled and eager filters agree, or that
  results are reproducible under a seed when data is missing).

Plan 001 (the leading-`NaN` init bug) shows these paths are fragile and
under-guarded. This plan adds three deterministic, machine-checkable tests that
pin the behavior down, and it is a prerequisite for Plan 004 (which refactors the
compiled scalar smoother those tests exercise).

## Current state

Relevant files (read for context; DO NOT modify):

- `lib/bsts_nx/kalman_filter.ex` — `filter/7` (eager, list of `{x, p}` tuples,
  detects missing via `missing_observation?`), `filter_defn/7` (compiled scalar,
  returns `{xs, ps}` as rank-1 `{t}` tensors, detects missing via `Nx.equal(z, z)`
  i.e. a `NaN` observation; see lines 371-375).
- `lib/bsts_nx/smoother.ex` — `rts_defn/4` consumes `filter_defn` output
  `{xs, ps}` and returns `{smoothed_xs, smoothed_ps}` (doctest at lines 43-47).
- `lib/bsts_nx/gibbs_sampler.ex` — `sample_structured/4`; samples are maps with
  `:q_matrix` and `:obs_var` tensor fields.

Existing analytic eager test, `test/missing_data_test.exs` (pattern to mirror):

```elixir
  describe "KalmanFilter.filter/7 with missing data" do
    test "skips update for nil observations" do
      observations = [1.0, nil, 3.0]
      f = 1.0; h = 1.0; q = 0.1; r = 0.5; x0 = 0.0; p0 = 1.0
      estimates = KalmanFilter.filter(observations, f, h, q, r, x0, p0)
      # ... computes expected {x_t, p_t} analytically and asserts assert_in_delta ...
    end
  end
```

Existing structured missing-data finite-check, `test/gibbs_sampler_missing_observations_test.exs:52-66` (pattern to mirror):

```elixir
  test "sample_structured skips NaN observations in variance update" do
    observations = [1.0, Nx.Constants.nan(), 2.0]
    spec = Components.local_level_spec(process_var: 0.5, obs_var: 1.0)
    samples = GibbsSampler.sample_structured(observations, spec, 4, burn_in: 0, seed: 123)
    assert length(samples) == 4
    # asserts q diagonal + obs_var are finite numbers
  end
```

## Commands you will need

| Purpose | Command | Expected |
|---------|---------|----------|
| Compile | `mix compile --warnings-as-errors` | exit 0 |
| New tests | `mix test test/missing_data_correctness_test.exs` | all pass |
| Format | `mix format && mix format --check-formatted` | exit 0 |
| Full gate | `bash scripts/ci.sh` | exit 0 |

(If local `mix` errors with a `mise` usage message, prefix with `mise exec -- `.)

## Scope

**In scope** (only file you may create):
- `test/missing_data_correctness_test.exs` (create)

**Out of scope** (do NOT touch):
- Everything under `lib/`. This plan is tests-only. If a test reveals a real bug,
  STOP and report it — do not fix library code under this plan.
- `test/missing_data_test.exs` and `test/gibbs_sampler_missing_observations_test.exs`
  — leave the existing tests as they are; add a new file.

## Git workflow

- Branch: `advisor/002-missing-observation-correctness-tests`
- Commit message: conventional commits, e.g. `test: pin missing-observation handling in compiled filter and structured Gibbs`.
- Do NOT push or open a PR unless instructed.

## Steps

### Step 1: Create the test file with the three tests

Create `test/missing_data_correctness_test.exs`:

```elixir
defmodule BstsNxMissingDataCorrectnessTest do
  use ExUnit.Case, async: true

  alias BstsNx.Components
  alias BstsNx.GibbsSampler
  alias BstsNx.KalmanFilter
  alias BstsNx.Smoother

  # Builds a rank-1 {n} tensor equal to `values` but with NaN at `nan_index`.
  defp with_nan(values, nan_index) do
    n = length(values)
    base = Nx.tensor(values, type: {:f, 64})
    nan = Nx.broadcast(Nx.Constants.nan() |> Nx.as_type({:f, 64}), {n})
    mask = Nx.tensor(for i <- 0..(n - 1), do: if(i == nan_index, do: 1, else: 0), type: {:u, 8})
    Nx.select(mask, nan, base)
  end

  defp finite?(v) when is_number(v), do: v == v and v not in [:infinity, :neg_infinity]
  defp finite?(_), do: false

  describe "compiled filter agrees with eager filter on missing data" do
    # The eager path encodes missing as nil; the compiled path encodes it as NaN.
    # Both must produce the same filtered means/covariances.
    test "filter_defn (NaN) matches filter (nil) when an interior obs is missing" do
      f = 1.0; h = 1.0; q = 0.1; r = 0.5; x0 = 0.0; p0 = 1.0

      eager = KalmanFilter.filter([1.0, nil, 3.0], f, h, q, r, x0, p0)
      eager_x = Enum.map(eager, fn {x, _p} -> Nx.to_number(x) end)
      eager_p = Enum.map(eager, fn {_x, p} -> Nx.to_number(p) end)

      obs_defn = with_nan([1.0, 0.0, 3.0], 1)
      {xs, ps} = KalmanFilter.filter_defn(obs_defn, f, h, q, r, x0, p0)
      defn_x = Nx.to_flat_list(xs)
      defn_p = Nx.to_flat_list(ps)

      Enum.zip(eager_x, defn_x)
      |> Enum.each(fn {a, b} -> assert_in_delta(a, b, 1.0e-6) end)

      Enum.zip(eager_p, defn_p)
      |> Enum.each(fn {a, b} -> assert_in_delta(a, b, 1.0e-6) end)
    end
  end

  describe "compiled smoother stays finite through a missing observation" do
    test "rts_defn produces finite smoothed states when filter saw a NaN obs" do
      obs_defn = with_nan([1.0, 0.0, 3.0, 4.0, 5.0], 2)
      {xs, ps} = KalmanFilter.filter_defn(obs_defn, 1.0, 1.0, 0.1, 0.5, 0.0, 1.0)
      {sxs, sps} = Smoother.rts_defn(xs, ps, 1.0, 0.1)

      Enum.each(Nx.to_flat_list(sxs), fn v -> assert finite?(v) end)
      Enum.each(Nx.to_flat_list(sps), fn v -> assert finite?(v) end)
    end
  end

  describe "structured Gibbs is reproducible with missing data" do
    test "same seed yields identical samples when an interior obs is missing" do
      observations = [1.0, 2.0, Nx.Constants.nan(), 4.0, 5.0, 6.0]
      spec = Components.local_level_spec(process_var: 0.5, obs_var: 1.0)

      run = fn ->
        GibbsSampler.sample_structured(observations, spec, 5, burn_in: 2, seed: 99)
      end

      a = run.()
      b = run.()

      assert length(a) == length(b)

      Enum.zip(a, b)
      |> Enum.each(fn {sa, sb} ->
        assert Nx.to_flat_list(sa.q_matrix) == Nx.to_flat_list(sb.q_matrix)
        assert Nx.to_number(sa.obs_var) == Nx.to_number(sb.obs_var)
      end)
    end
  end
end
```

**Verify**: `mix test test/missing_data_correctness_test.exs` → 3 tests, 0 failures.

### Step 2: Confirm the helpers match the real APIs

Before assuming the test compiles, confirm against the live code (do not change
library code — adapt the *test* if an API differs):

- `KalmanFilter.filter/7` returns a list of `{x, p}` tensor tuples (see
  `test/missing_data_test.exs`). If it returns something else, adjust the
  `eager_x`/`eager_p` extraction.
- `KalmanFilter.filter_defn/7` returns `{xs, ps}` rank-1 tensors (doctest at
  `lib/bsts_nx/smoother.ex:43-47`). If the arity/shape differs, adjust.
- `Smoother.rts_defn/4` signature is `rts_defn(xs, ps, f, q)`.
- `GibbsSampler.sample_structured/4` samples carry `:q_matrix` and `:obs_var`
  (see `test/gibbs_sampler_missing_observations_test.exs:61-64`).

If any of these APIs differ from the excerpts, STOP and report rather than
guessing — the failure indicates drift.

### Step 3: Run the touched neighborhood and the full gate

**Verify**:
- `mix test test/missing_data_test.exs test/gibbs_sampler_missing_observations_test.exs test/missing_data_correctness_test.exs` → all pass.
- `mix format --check-formatted` → exit 0.

## Test plan

- New file `test/missing_data_correctness_test.exs`, three deterministic tests:
  1. **Cross-implementation equivalence**: `filter_defn` (NaN) == `filter` (nil)
     on a series with one interior missing point — the previously-untested
     compiled NaN path.
  2. **Compiled smoother finiteness** through a missing observation.
  3. **Structured Gibbs reproducibility** under a fixed seed with interior missing
     data (guards seed-threading through the missing-data branch).
- Patterns mirrored: `test/missing_data_test.exs` (filter assertions) and
  `test/gibbs_sampler_missing_observations_test.exs` (finite-number checks).
- Verification: `mix test test/missing_data_correctness_test.exs` → all pass.

## Done criteria

ALL must hold:

- [ ] `test/missing_data_correctness_test.exs` exists with the three tests and passes.
- [ ] `mix compile --warnings-as-errors` exits 0.
- [ ] `mix test test/missing_data_test.exs test/gibbs_sampler_missing_observations_test.exs test/missing_data_correctness_test.exs` → all pass.
- [ ] `mix format --check-formatted` exits 0.
- [ ] No files under `lib/` modified (`git status`).
- [ ] `plans/README.md` status row for 002 updated to DONE.

## STOP conditions

Stop and report (do not improvise) if:

- Any of the three tests fails in a way that indicates a **real library bug**
  (e.g. compiled and eager filters genuinely disagree, or samples are not
  reproducible under a fixed seed). That is a finding, not something to "fix" by
  loosening the test — report it.
- An API in Step 2 differs materially from the excerpts (drift since `e4654c5`).
- `with_nan/2` does not produce a tensor containing a `NaN` (verify with
  `Nx.to_flat_list(with_nan([1.0, 0.0, 3.0], 1))` showing a non-finite middle
  element) — if NaN construction fails on this Nx version, report it.

## Maintenance notes

- Test 1 is the most valuable: it locks the compiled and eager missing-data paths
  together. If someone changes the NaN sentinel handling in `filter_defn`
  (`lib/bsts_nx/kalman_filter.ex:371-375`), this test should catch a divergence.
- Plan 004 refactors `Smoother.rts_defn_impl`; test 2 here is part of its safety
  net alongside `test/smoother_defn_test.exs`.
- Tolerances are `1.0e-6` on f64; if the default backend is changed to one that
  computes in f32, revisit the tolerance.
