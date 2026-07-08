# Plan 028: Test the `BstsNx.Execution` fallback machinery (and pin its fragile error-string match)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat 9b7cb8d..HEAD -- lib/bsts_nx/execution.ex lib/bsts_nx/intervention_analysis.ex`
> On mismatch with the excerpts below, STOP.

## Status

- **Priority**: P2
- **Effort**: S
- **Risk**: LOW (tests + at most one string-to-attribute refactor)
- **Depends on**: none
- **Category**: tests
- **Planned at**: commit `9b7cb8d`, 2026-07-08

## Why this matters

`BstsNx.Execution` decides when the fast operational (compiled-filter) path
falls back to MCMC: it recognizes the "backend can't do LU decomposition"
error **by substring match on the exception message** and routes accordingly.
The module has **zero test references**. If a future Nx release rewords that
message, `backend_lu_missing?/1` silently returns `false` and the behavior
degrades in two ways with no test failing: the explicit `fallback: :mcmc`
path stops falling back (users get a raw internal error), and the clear
"structured filter unsupported" error never raises. This matters most on
EMLX, where the missing-LU case is real today (per CLAUDE.md's backend
notes).

## Current state

- `lib/bsts_nx/execution.ex` (~lines 95–115):

  ```elixir
  @spec explicit_mcmc_fallback?(keyword()) :: boolean()
  def explicit_mcmc_fallback?(opts) do
    Keyword.get(opts, :fallback) == :mcmc or Keyword.get(opts, :allow_mcmc_fallback, false)
  end

  @doc """
  Recognizes the current Nx missing-LU backend error.
  """
  @spec backend_lu_missing?(Exception.t()) :: boolean()
  def backend_lu_missing?(%RuntimeError{message: message}) do
    String.contains?(message, "Nx.Backend.lu/3 not implemented")
  end

  def backend_lu_missing?(_), do: false
  ```

  The module also has `resolve_mode!/2` (legacy mode mapping + an
  `ArgumentError` branch, ~lines 47–56) and `metadata/5` — read the whole
  module (192 lines? check) before writing tests; cover every public
  function.

- The consumer, `lib/bsts_nx/intervention_analysis.ex` (~lines 366–385):

  ```elixir
  try do
    analyze_filter!(observations, pre_period, post_period, alpha, opts)
  rescue
    e ->
      stacktrace = __STACKTRACE__

      cond do
        Execution.backend_lu_missing?(e) and Execution.explicit_mcmc_fallback?(opts) ->
          observations
          |> analyze_mcmc(pre_period, post_period, alpha, Keyword.put(opts, :mode, :bayesian))
          |> mark_fallback(Exception.message(e))

        Execution.backend_lu_missing?(e) ->
          Execution.raise_structured_filter_unsupported!(e)

        true ->
          reraise e, stacktrace
      end
  end
  ```

- No test file references `Execution`, `backend_lu_missing`, or
  `raise_structured_filter_unsupported` (verified by grep at planning time).

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| New tests | `mix test test/bsts_nx/execution_test.exs` | 0 failures |
| Full verify | `bash scripts/ci.sh` | exit 0 |

Tooling note: `mix` is a `mise` shim; prefix `mise exec -- ` if needed.

## Scope

**In scope**:
- `test/bsts_nx/execution_test.exs` (create)
- `lib/bsts_nx/execution.ex` — ONLY the optional Step 3 refactor (extract the
  matched string into a public `@doc`-annotated module attribute accessor so
  tests and code share one source); no behavior change.

**Out of scope**:
- `lib/bsts_nx/intervention_analysis.ex` — do not refactor the rescue block
  to make it more injectable; the Step 2 tests cover the decision logic
  without it.
- Any change to fallback semantics or error messages beyond the single-source
  extraction.

## Git workflow

- Branch: `advisor/028-execution-fallback-tests` (from `execute-plans`).
- Commit style: `test: cover Execution fallback decisions; single-source the LU error match`
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Unit-test every public `Execution` function

Create `test/bsts_nx/execution_test.exs` (plain ExUnit, `async: true`;
pattern: any small unit test file, e.g. `test/bsts_nx/model_builder_test.exs`):

- `backend_lu_missing?/1`: true for
  `%RuntimeError{message: "... Nx.Backend.lu/3 not implemented ..."}`;
  false for a reworded RuntimeError; false for non-RuntimeError exceptions
  (`%ArgumentError{}`).
- `explicit_mcmc_fallback?/1`: `[fallback: :mcmc]` → true;
  `[allow_mcmc_fallback: true]` → true; `[]` → false;
  `[fallback: :other]` → false.
- `resolve_mode!/2`: each accepted mode + the legacy mapping (read the
  function; assert each branch) and `assert_raise ArgumentError` on an
  invalid mode.
- `raise_structured_filter_unsupported!/1`: `assert_raise RuntimeError,
  ~r/structured filter/..., fn -> ... end` (match on the real message —
  read it) and confirm the message includes guidance about `fallback: :mcmc`
  if it does.
- `metadata/5` (and any other public fun): happy-path shape assertions.

**Verify**: `mix test test/bsts_nx/execution_test.exs` → 0 failures, ~10+
assertions across the surface.

### Step 2: Exercise the rescue path end-to-end (decision-table test)

Full injection through `analyze_filter!` would require a crippled backend;
instead, pin the *decision logic* with a table test mirroring the `cond` in
`intervention_analysis.ex` (documented as such):

```elixir
lu_err = %RuntimeError{message: "Nx.Backend.lu/3 not implemented for ..."}
other_err = %RuntimeError{message: "something else"}

# {error, opts, expected_route}
cases = [
  {lu_err, [fallback: :mcmc], :mcmc_fallback},
  {lu_err, [], :clear_unsupported_raise},
  {other_err, [fallback: :mcmc], :reraise}
]
```

asserting `backend_lu_missing?/1` + `explicit_mcmc_fallback?/1` compose to
the expected route. Add a comment: "mirrors the cond in
`InterventionAnalysis.analyze_filter rescue` — update both together."

**Verify**: test passes; the comment references the exact function.

### Step 3: Single-source the fragile string

In `execution.ex`, extract the match fragment:

```elixir
@lu_missing_fragment "Nx.Backend.lu/3 not implemented"

@doc "The Nx error-message fragment identifying a backend without LU support."
def lu_missing_fragment, do: @lu_missing_fragment
```

use it in `backend_lu_missing?/1`, and add a test asserting
`Execution.lu_missing_fragment()` equals the literal — with a comment that
this is the canary to check against Nx release notes on every `nx` upgrade
(the string lives in Nx, not here; the test can't detect Nx rewording it,
but it makes the coupling visible and greppable).

**Verify**: `mix test test/bsts_nx/execution_test.exs` → 0 failures;
`bash scripts/ci.sh` → exit 0.

## Test plan

Steps 1–3 above; ~12–15 new assertions in one new file. No existing tests
change.

## Done criteria

- [ ] `test/bsts_nx/execution_test.exs` exists; every public `Execution`
      function has ≥1 test
- [ ] The decision-table test covers the three rescue routes
- [ ] `grep -c "Nx.Backend.lu/3 not implemented" lib/bsts_nx/execution.ex` → 1
      (single-sourced)
- [ ] `bash scripts/ci.sh` exits 0
- [ ] Only in-scope files changed
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- `Execution`'s public surface differs materially from the excerpt
  (refactored since planning).
- You're tempted to refactor `intervention_analysis.ex` for injectability —
  that's a design change needing maintainer sign-off; the decision-table
  test is the agreed substitute.
- Any new test fails against current code — that's a real bug; report it.

## Maintenance notes

- On every `nx` version bump, grep the Nx changelog/source for the LU error
  wording and compare against `Execution.lu_missing_fragment()` — this is now
  a one-line check. Better long-term fix (out of scope): Nx raising a typed
  error the library can match structurally; worth an upstream issue if the
  string ever breaks.
- The mirrored decision-table comment means changes to the rescue `cond`
  must update the test in the same commit — reviewers should enforce that.
