# Plan 025: Site code-quality sweep (shared formatters/clamps/timing, dead component trim, noise-quiz honest fallback)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat 9b7cb8d..HEAD -- site/lib`
> Plans 019/024 will have touched `site/lib` — expected. Re-grep each
> duplication claim below against live code before acting on it; counts may
> have shifted slightly.

## Status

- **Priority**: P3
- **Effort**: M
- **Risk**: LOW (display helpers, dead code, and one latent fallback path)
- **Depends on**: plans/024-async-scaffold-and-error-states.md (avoids editing
  the same LiveViews concurrently)
- **Category**: tech-debt
- **Planned at**: commit `9b7cb8d`, 2026-07-08

## Why this matters

Four small, verified duplication/debt clusters in the site, none urgent but
all cheap and all compounding with every new page:

1. **Formatting helpers**: `defp fmt` (12 files), `signed` (8), `pct` (6),
   `fmt2` (5), `fmt1` (4) — 40+ copies of one-line
   `:erlang.float_to_binary/2` wrappers.
2. **Param clamping**: a byte-identical 3-clause `clamp(value, range)` in
   `demos/demand.ex:153-164`, `marketing.ex:190-201`, `policy.ex:135-146`,
   plus variant families in `counterfactual.ex:125-149` and
   `anomaly.ex:160-174`.
3. **Timing conventions**: the "computed live in N ms" badge — the site's
   signature honesty device — is fed by **five different** elapsed-ms
   formulas: ceil-div `div(max(us,0)+999, 1000)` (`hindsight.ex:62,90`,
   `noise_quiz.ex`, `kalman_tuner.ex`), floor `System.convert_time_unit`
   (`demand.ex`, `gibbs.ex`), `Float.round(us/1000, 1)` (`compose.ex:88`),
   `Float.round(_, 2)` (`anomaly.ex`), and monotonic-ms subtraction
   (`counterfactual.ex:92`). Same badge, inconsistent precision.
4. **Dead generator scaffolding**: ~330 of 506 lines in
   `site/lib/bsts_site_web/components/core_components.ex` have zero call
   sites (`button` 102, `input` clauses 193–305, `error` 308, `header` 324,
   `table` 366, `list` 418, `translate_error` 485, `translate_errors` 503 —
   verified: only their own docstrings reference them). Only `flash` (56),
   `icon` (452), `show` (460), `hide` (471) are used (from `layouts.ex`).

Plus one latent content bug: `NoiseQuiz.batch/1`
(`site/lib/bsts_site/demos/noise_quiz.ex:38-43`) uses
`Enum.reduce_while(0..(@max_attempts - 1), nil, ...)` where the `{:cont,
candidate}` accumulator means that if all attempts fail vetting it silently
returns the last NON-clean candidate — and `NoiseLive` renders "the only
interval that excludes zero" without checking `clean?`, which would then be
false on-screen. Rare (clean sets are nearly always found), but it's an
honesty bug on the trust-themed page.

## Current state

Key excerpts (re-verify each before editing):

- `site/lib/bsts_site/demos/noise_quiz.ex:38-43`:

  ```elixir
  def batch(counter) when is_integer(counter) and counter >= 0 do
    Enum.reduce_while(0..(@max_attempts - 1), nil, fn attempt, _last ->
      candidate = build(counter, attempt)
      if candidate.clean?, do: {:halt, candidate}, else: {:cont, candidate}
    end)
  end
  ```

- Timing examples: `compose.ex:88` →
  `elapsed_ms: max(Float.round(elapsed_us / 1000, 1), 0.1)`;
  `hindsight.ex:62` → `elapsed_ms: div(max(elapsed_us, 0) + 999, 1000)`;
  `counterfactual.ex:92` → `System.monotonic_time(:millisecond) - t0`.
- `reveal_truth` handler: `handle_event("reveal_truth", _, socket) ->
  {:noreply, assign(socket, revealed: true)}` — byte-identical in 14
  LiveViews.
- Component usage verification: `grep -rn "<\.button\|<\.input\|<\.table\|<\.header" site/lib`
  matches only inside `core_components.ex` docstrings.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Site tests | `cd site && mix test` | 0 failures |
| Compile (dead-code check) | `cd site && mix compile --warnings-as-errors` | exit 0 |
| Format | `cd site && mix format` then `--check-formatted` | exit 0 |

Tooling note: `mix` is a `mise` shim; prefix `mise exec -- ` if needed.

## Scope

**In scope**:
- Create: `site/lib/bsts_site/format.ex`, `site/lib/bsts_site/demos/params.ex`,
  `site/lib/bsts_site/demos/timing.ex`
- Edit: the demo modules and LiveViews that currently hold the duplicated
  helpers; `core_components.ex` (deletions only); `noise_quiz.ex` +
  `engine/noise_live.ex`
- Tests: `site/test/bsts_site/format_test.exs`,
  extend `site/test/bsts_site/demos/shapes_test.exs` (from plan 023) for the
  noise-quiz fallback

**Out of scope**:
- `AsyncDemo` and the async control flow (plan 024's territory).
- `story.ex`/`charts.ex` internals beyond call-site updates.
- The `reveal_truth` dedup **if** plan 024's helper didn't already create a
  natural home: dedup it only via a trivial shared clause include if one
  exists; otherwise leave the 14 copies and note it (a `use`-macro just for
  this is over-engineering — see Step 5).

## Git workflow

- Branch: `advisor/025-site-code-quality-sweep` (from `execute-plans`, after
  024).
- Commit style: one commit per cluster —
  `refactor(site): shared number formatting`, `...: shared param clamps`,
  `...: one elapsed_ms convention`, `chore(site): trim dead core_components`,
  `fix(site): noise quiz reports non-clean batches honestly`.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: `BstsSite.Format`

Create the module with `fmt/1` (default decimals), `fmt/2`, `signed/1`,
`pct/1` — copy the semantics from the most common existing implementations
(open 2–3 of the `defp fmt` copies and match their output exactly, including
`* 1.0` coercion). Replace every private copy with
`import BstsSite.Format` (or fully-qualified calls) and delete the locals.
Where copies genuinely differ (e.g. a page-specific decimal count), keep the
call-site behavior identical via the `fmt/2` arity — output must not change.

**Verify**: `grep -rn "defp fmt\b\|defp signed\|defp pct\|defp fmt2\|defp fmt1" site/lib | wc -l`
→ 0; `cd site && mix test` → 0 failures.

### Step 2: `BstsSite.Demos.Params`

Create `clamp_int/2` and `clamp_float/2` from the byte-identical 3-clause
implementations in `demand.ex`/`marketing.ex`/`policy.ex`; delegate those
three plus the variant families in `counterfactual.ex`/`anomaly.ex` where the
variant is expressible as the shared function + a range (if a variant does
more than clamp — e.g. scaling — keep its wrapper but route the clamp core
through `Params`). Public per-module `clamp/2` functions that LiveViews call
(e.g. `Demand.clamp/2`) keep their names as thin delegations — don't churn
call sites.

**Verify**: `cd site && mix test` (plan 023's event tests exercise the
clamps) → 0 failures.

### Step 3: One timing convention

Create `BstsSite.Demos.Timing.elapsed_ms/1` taking microseconds and using the
ceil-integer convention (`max(div(max(us, 0) + 999, 1000), 1)`) — it's the most
common copy and "rounds up to the millisecond you actually waited", which
suits the honesty badge. Route all five variants through it (for
`counterfactual.ex`, capture with `:timer.tc` or monotonic µs instead of ms
subtraction). The badge display changes for `compose` (was 1-decimal float,
min 0.1) and `anomaly` (2-decimal) — they become integers ≥1; that's the
point (one convention), and it's a display-metadata change only. Note it in
the commit message.

**Verify**: `grep -rn "elapsed_ms" site/lib/bsts_site/demos | grep -v timing.ex`
shows only calls to `Timing.elapsed_ms` (no inline formulas);
`cd site && mix test` → 0 failures (plan 023's shape tests assert
`execution.elapsed_ms` presence — if any asserted a float type for
compose/anomaly, update those assertions in the same commit and say so).

### Step 4: Trim `core_components.ex`

Delete `button/1`, all `input/1` clauses, `error/1`, `header/1`, `table/1`,
`list/1`, `translate_error/1`, `translate_errors/2` and their `attr`/`slot`
declarations and docstrings. Keep `flash/1`, `icon/1`, `show/2`, `hide/2`
(and any private helpers only they use). Also delete now-unused aliases and
the Gettext-related plumbing if only `translate_error` used it — follow
compiler warnings.

**Verify**: `cd site && mix compile --warnings-as-errors` → exit 0 (this
catches any reference you missed); `wc -l site/lib/bsts_site_web/components/core_components.ex`
→ roughly 150–200 lines (from 506); `cd site && mix test` → 0 failures.

### Step 5: `reveal_truth` — dedup only if free

If plan 024 introduced a shared `on_mount`/hook module where a
`handle_event("reveal_truth", ...)` clause can live without inventing new
macro machinery, move it there and delete the 14 copies. Otherwise leave
them and record "kept: dedup requires a macro layer that costs more than 14
one-liners" in your report. Either outcome is acceptable.

**Verify** (if deduped): `grep -rn "reveal_truth" site/lib/bsts_site_web/live | wc -l`
counts only template usages, not handler clauses; site tests green.

### Step 6: Honest noise-quiz fallback

1. `noise_quiz.ex`: make exhaustion explicit — after the `reduce_while`,
   the candidate is returned regardless; keep that, but ensure the returned
   map's `clean?` field survives (it already exists on the candidate) and
   document the exhaustion case in the function doc.
2. `engine/noise_live.ex`: check `batch.clean?` wherever the verdict copy
   renders ("the only interval that excludes zero" — `verdict_line/2` area,
   ~lines 286–294): when `clean?` is false, render neutral copy ("this deal
   was ambiguous — regenerate for a cleaner one") instead of the absolute
   claim.
3. Test (extend plan 023's shapes test): build a candidate with
   `clean?: false` and assert the LiveView/verdict helper renders the neutral
   copy (unit-test the helper directly if it's pure).

**Verify**: `cd site && mix test` → 0 failures, ≥1 new test.

### Step 7: Format + full pass

**Verify**: `cd site && mix format --check-formatted` → exit 0;
`git status --porcelain` → only in-scope files.

## Test plan

- New: `format_test.exs` (table-driven: `fmt/signed/pct` on positives,
  negatives, zero, floats needing coercion), the noise-quiz neutral-verdict
  test (Step 6).
- Regression net: plan 023's route smoke + event + shape tests after every
  step; `mix compile --warnings-as-errors` as the dead-code detector.

## Done criteria

- [ ] Zero `defp fmt/signed/pct/fmt1/fmt2` duplicates remain in `site/lib`
- [ ] Zero inline `elapsed_ms` formulas outside `Timing`
- [ ] `core_components.ex` ≤ ~200 lines; compile clean with warnings-as-errors
- [ ] Noise quiz: non-clean batch renders neutral copy (test proves it)
- [ ] `cd site && mix test` → 0 failures; format check clean
- [ ] `plans/README.md` status row updated (note the Step 5 outcome)

## STOP conditions

Stop and report back (do not improvise) if:

- Two `fmt` copies turn out to have genuinely different semantics that a
  shared function can't express without changing some page's output — report
  the pair instead of picking a winner silently.
- Deleting a "dead" component breaks compilation (a call site the grep
  missed) — re-verify usage before deleting further.
- Plan 023's tests don't exist (run this plan after 023/024).
- The noise-quiz change requires touching the scoring/vetting math in
  `build/2` — out of scope; only the exhaustion *reporting* is in scope.

## Maintenance notes

- New demos should reach for `Format`, `Params`, and `Timing` — reviewers
  should flag fresh private `fmt`/`clamp`/timing helpers.
- The exec-badge convention is now "integer ms, rounded up"; the design
  contract's honesty framing depends on it staying uniform.
- If forms/tables ever come to the site, regenerate the deleted components
  from the Phoenix generator rather than resurrecting from git history.
