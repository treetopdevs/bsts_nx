# Plan 009: Consolidate the duplicated positive-integer validation into `Utils`

> **Executor instructions**: Follow this plan step by step. Run every verification
> command and confirm the expected result before moving to the next step. If
> anything in the "STOP conditions" section occurs, stop and report — do not
> improvise. When done, update the status row in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat e4654c5..HEAD -- lib/bsts_nx/utils.ex lib/bsts_nx/gibbs_sampler.ex lib/bsts_nx/validation.ex lib/bsts_nx/covariate_selection.ex`
> If any changed since this plan was written, compare the "Current state" excerpts
> below against the live code before proceeding; on a mismatch, STOP.

## Status

- **Priority**: P3
- **Effort**: S
- **Risk**: LOW (message-preserving; see "Why this is low-risk")
- **Depends on**: none
- **Category**: tech-debt
- **Planned at**: commit `e4654c5`, 2026-06-30

## Why this matters

The exact same "positive integer" argument check is reimplemented in at least three
modules, each raising the **identical** message
`"#{name} must be a positive integer, got: #{inspect(value)}"`:

- `lib/bsts_nx/gibbs_sampler.ex:580-584` — `validate_positive!/2` (private helper).
- `lib/bsts_nx/validation.ex:642-653` — three inline `if not is_integer(...) ...`
  blocks (`n_pre`, `n_post`, `num_samples`).
- `lib/bsts_nx/covariate_selection.ex:389,397` — two inline blocks (`num_samples`, `thin`).

Because the message string is already byte-identical across these sites, they can be
routed through a single shared helper **without changing any behavior or any error
message** — a clean, low-risk consolidation. This plan adds one helper to
`BstsNx.Utils` (the dependency-free base module everything already uses) and migrates
these sites to it.

### Why this is low-risk

- The new helper produces the **exact** existing message, so tests that assert on
  the message text keep passing (Step 1 captures the baseline; Step 5 re-runs it).
- `Utils` is the bottom of the dependency stack (it imports nothing from the library),
  so every migrated module can call it without creating a cycle. **Note:** the helper
  must NOT go in `BstsNx.Validation` — `Validation.known_lift_injection/3`
  (`validation.ex:631`) runs a sampler, so `Validation` sits *above* `GibbsSampler`;
  putting the helper there and calling it from `GibbsSampler` would create a cycle.

### Explicitly deferred (do NOT attempt here)

- **Non-empty checks** (`gibbs_sampler.ex:568-578`, `intervention_analysis.ex:181`):
  their messages **differ** ("observations must contain at least one value" vs
  "observations must be non-empty"), so consolidating them risks breaking
  message-asserting tests. Out of scope — see Maintenance notes.
- **Period-tuple validators** (`operational.ex:511`, `model_builder.ex:497`,
  `r_sidecar.ex:445`): they return different shapes and `operational`'s does not
  bound-check against the observation length, so unifying them could change behavior
  in the numerical index path. Out of scope — higher-risk, needs its own plan.

## Current state

The shared message shape, from `lib/bsts_nx/gibbs_sampler.ex:580-584`:

```elixir
  defp validate_positive!(_name, value) when is_integer(value) and value > 0, do: :ok

  defp validate_positive!(name, value) do
    raise ArgumentError, "#{name} must be a positive integer, got: #{inspect(value)}"
  end
```

`lib/bsts_nx/validation.ex:642-653` (inside `known_lift_injection/3`):

```elixir
    if not is_integer(n_pre) or n_pre <= 0 do
      raise ArgumentError, "n_pre must be a positive integer, got: #{inspect(n_pre)}"
    end

    if not is_integer(n_post) or n_post <= 0 do
      raise ArgumentError, "n_post must be a positive integer, got: #{inspect(n_post)}"
    end

    if not is_integer(num_samples) or num_samples <= 0 do
      raise ArgumentError,
            "num_samples must be a positive integer, got: #{inspect(num_samples)}"
    end
```

`lib/bsts_nx/covariate_selection.ex` — the two single-line raises (confirm by opening
the surrounding `if` blocks at the cited lines):

```elixir
      raise ArgumentError, "num_samples must be a positive integer, got: #{inspect(num_samples)}"   # line 389
      raise ArgumentError, "thin must be a positive integer, got: #{inspect(thin)}"                 # line 397
```

`BstsNx.Utils` is a public module of shared primitives (it already exposes
`percentile_interval/3`, `z_score/1`, etc.). The new helper goes here. `validation.ex`,
`covariate_selection.ex`, and `gibbs_sampler.ex` all already use `BstsNx.Utils`.

## Commands you will need

| Purpose | Command | Expected |
|---------|---------|----------|
| New Utils test | `mix test test/utils_validate_positive_integer_test.exs` | all pass |
| Baseline/regression suites | `mix test test/validation_test.exs test/covariate_selection_test.exs test/gibbs_general_test.exs test/gibbs_structured_test.exs` | all pass |
| Compile | `mix compile --warnings-as-errors` | exit 0 |
| Format | `mix format && mix format --check-formatted` | exit 0 |
| Full gate | `bash scripts/ci.sh` | exit 0 |

(If local `mix` errors with a `mise exec` usage message, prefix with `mise exec -- `.)

## Scope

**In scope** (the only files you may modify):
- `lib/bsts_nx/utils.ex` — add the public helper.
- `lib/bsts_nx/gibbs_sampler.ex` — route `validate_positive!/2` through the helper.
- `lib/bsts_nx/validation.ex` — replace the three `known_lift_injection` inline checks.
- `lib/bsts_nx/covariate_selection.ex` — replace the two inline checks.
- `test/utils_validate_positive_integer_test.exs` (create).

**Out of scope** (do NOT touch):
- The non-empty observation validators (different messages — deferred).
- The `burn_in` (non-negative) and `noise_sd` (non-negative number) checks in
  `validation.ex:655-661` — different semantics; leave them exactly as-is.
- All period-tuple validators (`operational.ex`, `model_builder.ex`, `r_sidecar.ex`).
- Any change to a raised message's text.

## Git workflow

- Branch: `advisor/009-consolidate-positive-integer-validation`
- Commit message: `refactor: centralize positive-integer validation in Utils`.
- Do NOT push or open a PR unless instructed.

## Steps

### Step 1: Capture the green baseline

`mix test test/validation_test.exs test/covariate_selection_test.exs test/gibbs_general_test.exs test/gibbs_structured_test.exs`

**Verify**: all pass. If not, STOP — baseline is not green.

### Step 2: Add the helper to `BstsNx.Utils`

Add this public function to `lib/bsts_nx/utils.ex` (place it near the other public
validation/number helpers; match the file's `@doc`/`@spec` style):

```elixir
  @doc """
  Raises `ArgumentError` unless `value` is a positive integer.

  `name` is interpolated into the message so callers get a field-specific error.
  Returns `:ok` when valid.

  ## Examples

      iex> BstsNx.Utils.validate_positive_integer!("num_samples", 10)
      :ok
  """
  @spec validate_positive_integer!(String.t(), term()) :: :ok
  def validate_positive_integer!(_name, value) when is_integer(value) and value > 0, do: :ok

  def validate_positive_integer!(name, value) do
    raise ArgumentError, "#{name} must be a positive integer, got: #{inspect(value)}"
  end
```

**Verify**: `mix compile --warnings-as-errors` → exit 0.

### Step 3: Migrate the call sites

The message is identical everywhere, so these are mechanical swaps.

**`gibbs_sampler.ex:580-584`** — replace the two `validate_positive!/2` clauses with a
single delegating clause (keep the private name so its call sites are unchanged):

```elixir
  defp validate_positive!(name, value), do: BstsNx.Utils.validate_positive_integer!(name, value)
```

**`validation.ex:642-653`** — replace the three inline `if` blocks with:

```elixir
    BstsNx.Utils.validate_positive_integer!("n_pre", n_pre)
    BstsNx.Utils.validate_positive_integer!("n_post", n_post)
    BstsNx.Utils.validate_positive_integer!("num_samples", num_samples)
```

(Leave the `burn_in` and `noise_sd` checks below them untouched.)

**`covariate_selection.ex:389,397`** — replace each inline `if ... raise` block with
`BstsNx.Utils.validate_positive_integer!("num_samples", num_samples)` and
`BstsNx.Utils.validate_positive_integer!("thin", thin)` respectively. Open the file
and confirm each replacement preserves the surrounding control flow (the original is
an `if not is_integer(...) or ... <= 0 do raise ... end` guard; the helper replaces
the entire guard).

**Verify**: `mix compile --warnings-as-errors` → exit 0, no unused-variable or
unreachable-clause warnings.

### Step 4: Add the Utils unit test

Create `test/utils_validate_positive_integer_test.exs`:

```elixir
defmodule BstsNx.UtilsValidatePositiveIntegerTest do
  use ExUnit.Case, async: true

  alias BstsNx.Utils

  test "accepts positive integers" do
    assert Utils.validate_positive_integer!("n", 1) == :ok
    assert Utils.validate_positive_integer!("n", 1000) == :ok
  end

  test "rejects zero, negatives, and non-integers with a field-specific message" do
    assert_raise ArgumentError, "n_pre must be a positive integer, got: 0", fn ->
      Utils.validate_positive_integer!("n_pre", 0)
    end

    assert_raise ArgumentError, ~r/^thin must be a positive integer, got: -3$/, fn ->
      Utils.validate_positive_integer!("thin", -3)
    end

    assert_raise ArgumentError, ~r/must be a positive integer/, fn ->
      Utils.validate_positive_integer!("num_samples", 2.5)
    end
  end
end
```

**Verify**: `mix test test/utils_validate_positive_integer_test.exs` → 2 tests, 0 failures.

### Step 5: Confirm no message changed (regression)

Re-run the Step 1 suites:

`mix test test/validation_test.exs test/covariate_selection_test.exs test/gibbs_general_test.exs test/gibbs_structured_test.exs`

**Verify**: all pass, identical to the Step 1 baseline. If any test that asserts on a
raised message now fails, the message text drifted — STOP and report (the helper's
message must match the originals exactly).

### Step 6: Full gate

**Verify**: `bash scripts/ci.sh` → exit 0; `mix format --check-formatted` → exit 0.

## Test plan

- New file `test/utils_validate_positive_integer_test.exs` covering the helper
  (accept, reject zero/negative/non-integer, field-specific message).
- Regression oracle: the four existing suites in Step 1 must stay green — they prove
  the migrated call sites behave identically (same messages).
- Pattern reference: the existing public-helper tests for `Utils` (e.g.
  `test/utils_safe_solve_test.exs`) for module/test structure.

## Done criteria

ALL must hold:

- [ ] `BstsNx.Utils.validate_positive_integer!/2` exists and is covered by the new test.
- [ ] `grep -rn "must be a positive integer" lib/bsts_nx/gibbs_sampler.ex lib/bsts_nx/validation.ex lib/bsts_nx/covariate_selection.ex` → no matches (the raise strings moved into `utils.ex`).
- [ ] `grep -n "must be a positive integer" lib/bsts_nx/utils.ex` → exactly 1 match.
- [ ] `mix compile --warnings-as-errors` exits 0, no warnings.
- [ ] All Step 1 suites + the new Utils test pass.
- [ ] `mix format --check-formatted` exits 0.
- [ ] Only the five in-scope files are modified (`git status`).
- [ ] `plans/README.md` status row for 009 updated to DONE.

## STOP conditions

Stop and report (do not improvise) if:

- Any "Current state" excerpt doesn't match the live code (drift).
- A regression suite fails after migration because a message changed.
- Migrating a site requires touching a non-empty or period validator (out of scope).
- Compiling reveals a dependency cycle from any migrated module to `Utils`
  (it should not — `Utils` imports nothing from the library).

## Maintenance notes

- **Follow-up (deferred non-empty dedup)**: the non-empty observation checks could be
  unified the same way, but only after standardizing their differing messages and
  updating the tests that assert on them. A focused follow-up plan should do that.
- **Follow-up (deferred period-validator unification)**: `operational.ex:511`,
  `model_builder.ex:497`, and `r_sidecar.ex:445` each hand-roll period validation with
  different return contracts (and `operational`'s omits the obs-length bound check
  that `Validation.validate_study_periods!` enforces). Unifying them is a behavior-
  sensitive change in the numerical index path — give it its own plan with parity
  tests, do not fold it into a "simple dedup".
- Reviewer should confirm no raised message text changed (diff the strings).
