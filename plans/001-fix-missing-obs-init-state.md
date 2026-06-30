# Plan 001: Use `ModelBuilder.first_obs/1` for initial state at the two wrapper sites the 5035d69 sweep missed

> **Executor instructions**: Follow this plan step by step. Run every verification
> command and confirm the expected result before moving to the next step. If
> anything in the "STOP conditions" section occurs, stop and report — do not
> improvise. When done, update the status row for this plan in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat e4654c5..HEAD -- lib/bsts_nx/causal_impact.ex lib/bsts_nx/intervention_analysis.ex lib/bsts_nx/model_builder.ex`
> If any of those files changed since this plan was written, compare the "Current
> state" excerpts below against the live code before proceeding; on a mismatch,
> treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: bug
- **Planned at**: commit `e4654c5`, 2026-06-20

## Why this matters

Commit `5035d69` ("fix: preserve observation inputs across wrappers") introduced
`BstsNx.ModelBuilder.first_obs/1`, which returns the first **finite** observation
(skipping leading `nil` / `:nan` / `NaN`), and swept the old
`List.first(obs) || 0.0` idiom out of `forecaster.ex`, `anomaly_detector.ex`,
`demand_forecaster.ex`, and `model_builder.ex`. Two sites were missed:

- `lib/bsts_nx/causal_impact.ex:83` — initial state for the pre-period Gibbs chain.
- `lib/bsts_nx/intervention_analysis.ex:393` — initial state for the operational filter spec.

The old idiom is subtly wrong when the first observation is missing. In Elixir
`NaN` is **truthy**, so `List.first([NaN, ...]) || 0.0` returns `NaN`, not `0.0`.
At `causal_impact.ex:83` this seeds the Gibbs sampler with a `NaN` initial state,
which propagates through the Kalman filter and corrupts the entire posterior
(every effect/credible-interval becomes `NaN`). Replacing both with
`ModelBuilder.first_obs/1` makes these two public entry points consistent with the
rest of the codebase and removes the corruption path.

## Current state

Files and roles:

- `lib/bsts_nx/causal_impact.ex` — `BstsNx.CausalImpact.estimate/4` (MCMC causal
  impact). The bug is at line 83. This module does **not** currently alias
  `ModelBuilder`; its alias block is at lines 26-29.
- `lib/bsts_nx/intervention_analysis.ex` — `analyze_filter!/5` (operational filter
  path). The bug is at line 393. This module **already** aliases `ModelBuilder`
  (line 55).
- `lib/bsts_nx/model_builder.ex` — defines the fix function (do not modify).

`lib/bsts_nx/causal_impact.ex:83` (inside `estimate/4`):

```elixir
    init_state = Keyword.get(opts, :initial_state, List.first(period.pre_data) || 0.0)
```

`lib/bsts_nx/causal_impact.ex:26-29` (current alias/import block):

```elixir
  alias BstsNx.Forward
  alias BstsNx.GibbsSampler
  alias BstsNx.Validation
  import BstsNx.Utils, only: [split_key_at: 2]
```

`lib/bsts_nx/intervention_analysis.ex:390-397` (inside `analyze_filter!/5`):

```elixir
    spec =
      model_spec ||
        Components.local_level_spec(
          initial_state: Keyword.get(opts, :x0, List.first(observations) || 0.0),
          initial_cov: Keyword.get(opts, :p0, 1.0),
          process_var: Keyword.get(opts, :q, 1.0),
          obs_var: Keyword.get(opts, :r, 1.0)
        )
```

The fix function, `lib/bsts_nx/model_builder.ex:247-261` (DO NOT MODIFY — reference only):

```elixir
  @doc """
  Returns the first observed value as a float, defaulting to 0.0.
  """
  @spec first_obs([number() | nil | atom() | Nx.t()]) :: float()
  def first_obs([]), do: 0.0

  def first_obs(observations) do
    Enum.find_value(observations, fn obs ->
      if BstsNx.Utils.missing_observation?(obs) do
        nil
      else
        coerce_observation(obs)
      end
    end) || 0.0
  end
```

Convention this follows: the four modules fixed in `5035d69` all call
`ModelBuilder.first_obs(obs_list)` where they previously had `List.first(...) || 0.0`.
For example `lib/bsts_nx/forecaster.ex:122`: `first = ModelBuilder.first_obs(obs_list)`.
Match that exactly. The existing unit test for `first_obs/1` is at
`test/bsts_nx/model_builder_test.exs:26-29`:

```elixir
    test "first_obs/1 skips missing observations before falling back" do
      assert ModelBuilder.first_obs([nil, :nan, Nx.Constants.nan(), 12.5]) == 12.5
      assert ModelBuilder.first_obs([nil, :nan, Nx.Constants.nan()]) == 0.0
    end
```

## Commands you will need

| Purpose   | Command | Expected on success |
|-----------|---------|---------------------|
| Compile   | `mix compile --warnings-as-errors` | exit 0, no warnings |
| New test  | `mix test test/causal_impact_missing_init_test.exs` | all pass |
| Touched suites | `mix test test/causal_impact_test.exs test/bsts_nx/intervention_analysis_test.exs` | all pass |
| Format    | `mix format` then `mix format --check-formatted` | exit 0 |
| Full gate | `bash scripts/ci.sh` | exit 0 |

(If local `mix` errors with a `mise` usage message, prefix with `mise exec -- `.)

## Scope

**In scope** (the only files you may modify):
- `lib/bsts_nx/causal_impact.ex` (line 83 + add one alias)
- `lib/bsts_nx/intervention_analysis.ex` (line 393)
- `test/causal_impact_missing_init_test.exs` (create)

**Out of scope** (do NOT touch):
- `lib/bsts_nx/model_builder.ex` — `first_obs/1` is already correct and tested.
- The two-clause private `to_number/1` in `causal_impact.ex` (lines 1037-1038).
  It has no `nil`/`:nan`-atom clause, so a leading `nil`/`:nan` *atom* will raise
  earlier in `period_context`. That is a **separate** boundary; this plan only
  fixes the `NaN`-encoded leading-observation path. Do not add a `nil` clause here.
- Any change to the public result shape of `estimate/4` or the analyze APIs.

## Git workflow

- Branch: `advisor/001-fix-missing-obs-init-state`
- Commit message style: conventional commits (recent history uses `fix:`,
  `test:`, `chore:`). Example from `git log`: `fix: preserve observation inputs across wrappers`.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Add the `ModelBuilder` alias to `causal_impact.ex`

Insert `alias BstsNx.ModelBuilder` into the alias block so it stays alphabetical
with the existing aliases. After the edit, lines 26-29 read:

```elixir
  alias BstsNx.Forward
  alias BstsNx.GibbsSampler
  alias BstsNx.ModelBuilder
  alias BstsNx.Validation
  import BstsNx.Utils, only: [split_key_at: 2]
```

**Verify**: `mix compile --warnings-as-errors` → exit 0, no "unused alias" warning.

### Step 2: Fix the initial state in `causal_impact.ex:83`

Replace:

```elixir
    init_state = Keyword.get(opts, :initial_state, List.first(period.pre_data) || 0.0)
```

with:

```elixir
    init_state = Keyword.get(opts, :initial_state, ModelBuilder.first_obs(period.pre_data))
```

**Verify**: `grep -n "List.first(period.pre_data)" lib/bsts_nx/causal_impact.ex` → no matches.

### Step 3: Fix the initial state in `intervention_analysis.ex:393`

`ModelBuilder` is already aliased in this module (line 55). Replace:

```elixir
          initial_state: Keyword.get(opts, :x0, List.first(observations) || 0.0),
```

with:

```elixir
          initial_state: Keyword.get(opts, :x0, ModelBuilder.first_obs(observations)),
```

**Verify**: `grep -rn "List.first(observations) || 0.0" lib/bsts_nx/intervention_analysis.ex` → no matches.

### Step 4: Add the regression test

Create `test/causal_impact_missing_init_test.exs` with the content below. The test
builds a pre-period whose **first** value is missing (`Nx.Constants.nan()`), runs
`estimate/4`, and asserts the summary numbers are finite. Before this plan's fix
this test FAILS (init state becomes `NaN`, poisoning the posterior → `NaN`
summary). After the fix it PASSES.

```elixir
defmodule BstsNxCausalImpactMissingInitTest do
  use ExUnit.Case, async: true

  alias BstsNx.CausalImpact

  defp finite?(v) when is_number(v), do: v == v and v not in [:infinity, :neg_infinity]
  defp finite?(_), do: false

  test "estimate/4 tolerates a missing (NaN) first pre-period observation" do
    nan = Nx.Constants.nan()
    # First pre-period point is missing; the rest carry the real ~50 level.
    pre = [nan | Enum.map(1..29, fn _ -> 50.0 end)]
    post = Enum.map(1..15, fn _ -> 60.0 end)
    obs = pre ++ post

    result =
      CausalImpact.estimate(obs, {1, 30}, {31, 45},
        num_samples: 40,
        burn_in: 20,
        seed: 7
      )

    summary = CausalImpact.summary(result)

    assert finite?(summary.cumulative_effect.mean)
    assert finite?(summary.average_effect.mean)
  end
end
```

**Confirm the bug first (recommended)**: stash your `lib/` edits
(`git stash push -- lib/bsts_nx/causal_impact.ex lib/bsts_nx/intervention_analysis.ex`),
run `mix test test/causal_impact_missing_init_test.exs` and confirm it FAILS, then
`git stash pop` and confirm it PASSES. If it passes even without the fix, STOP and
report (the assumption that a NaN init state poisons the posterior is wrong, and
this test does not protect the fix).

**Verify**: `mix test test/causal_impact_missing_init_test.exs` → 1 test, 0 failures.

### Step 5: Confirm the summary field names

The test reads `summary.cumulative_effect.mean` and `summary.average_effect.mean`.
Open `lib/bsts_nx/causal_impact.ex` and confirm `CausalImpact.summary/1` returns a
map with those keys (search for `cumulative_effect` and `average_effect`). If the
keys differ, adjust the test to read two finite numeric summary fields that exist
(do not change library code). If `summary/1` exposes no per-field map you can
assert on, STOP and report.

**Verify**: `mix test test/causal_impact_missing_init_test.exs` → all pass.

## Test plan

- New file `test/causal_impact_missing_init_test.exs`, one test: leading-`NaN`
  pre-period → finite causal-impact summary (the exact regression this plan fixes).
- Pattern reference: `test/causal_impact_test.exs` for `estimate/4` call shape and
  option keywords; `test/gibbs_sampler_missing_observations_test.exs` for the
  finite-number helper idiom.
- Optionally extend with an `InterventionAnalysis` analogue if a filter-path entry
  point with a leading-`NaN` series is easy to construct; not required for done.
- Verification: `mix test test/causal_impact_missing_init_test.exs` → all pass.

## Done criteria

ALL must hold:

- [ ] `mix compile --warnings-as-errors` exits 0 with no warnings.
- [ ] `grep -rn "List.first(period.pre_data)" lib/bsts_nx/causal_impact.ex` → no matches.
- [ ] `grep -rn "List.first(observations) || 0.0" lib/bsts_nx/intervention_analysis.ex` → no matches.
- [ ] `test/causal_impact_missing_init_test.exs` exists and passes.
- [ ] `mix test test/causal_impact_test.exs test/bsts_nx/intervention_analysis_test.exs` → all pass.
- [ ] `mix format --check-formatted` exits 0.
- [ ] No files outside the in-scope list are modified (`git status`).
- [ ] `plans/README.md` status row for 001 updated to DONE.

## STOP conditions

Stop and report (do not improvise) if:

- The "Current state" excerpts don't match the live code (drift since `e4654c5`).
- The regression test passes against the *unfixed* code (Step 4 red check fails to
  go red) — the fix would then be unprotected and the premise needs review.
- `CausalImpact.summary/1` exposes no finite numeric field you can assert on.
- Compiling reveals `ModelBuilder` would create a dependency cycle with
  `causal_impact.ex` (it should not — `ModelBuilder` is a lower-level module and
  `intervention_analysis.ex` already depends on both).

## Maintenance notes

- A leading `nil` or `:nan` **atom** in the observations passed to
  `CausalImpact.estimate/4` still raises in `period_context` (the private
  `to_number/1` has only `%Nx.Tensor{}` and `is_number` clauses). If the team
  wants `estimate/4` to accept atom-encoded missing values like the Kalman filter
  does, that is a separate, larger change — out of scope here.
- Reviewer should confirm the new test actually exercises the *first* pre-period
  element being missing (not an interior one) — that is the only position
  `List.first` inspected.
