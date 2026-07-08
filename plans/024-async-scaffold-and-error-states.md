# Plan 024: Extract the shared async-MCMC LiveView scaffold; stop rendering crashes as "busy"

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat 9b7cb8d..HEAD -- site/lib/bsts_site_web/live site/lib/bsts_site_web/components`
> Plans 019/023 may have touched mounts/tests — expected. If the
> `handle_async` clauses below no longer match any of the eight files, STOP.

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: MED (rewrites the interactive path of all 8 MCMC pages — gated by plan 023's smoke tests)
- **Depends on**: plans/023-site-liveview-demo-tests.md (characterization net first)
- **Category**: bug + tech-debt
- **Planned at**: commit `9b7cb8d`, 2026-07-08

## Why this matters

Eight LiveViews implement the same ~25-line scaffold by copy-paste: a
`running`-guard on the trigger event, `start_async` wrapping
`Limiter.run`, and three `handle_async` clauses. Two audited defects live in
those copies:

1. **Crashes are rendered as "busy" and the reason is thrown away.** Every
   copy maps `handle_async(_, {:exit, _reason}, socket)` to the same assign
   as the rate-limiter's polite `{:ok, :busy}` path. A genuine exception
   inside a fit therefore shows visitors "Another visitor is sampling, try
   again in a moment" forever, is indistinguishable from load, and is never
   logged — zero operator signal on a broken demo.
2. **Behavioral drift across copies.** Only the parameter-varying pages
   (`counterfactual`, `marketing`, `policy`) re-check that the async result
   still matches current params; the fixed-scenario pages don't need it —
   but the next parameter-varying page written by copy-paste from the wrong
   donor will silently lack the guard. Centralizing makes the staleness check
   a named option instead of tribal knowledge.

One shared helper fixes the error handling once, adds logging once, and turns
~200 duplicated lines into one audited implementation.

## Current state

The eight files (each with the trigger guard + `start_async` + 3
`handle_async` clauses): `site/lib/bsts_site_web/live/speed_live.ex`,
`demos/demand_live.ex`, `demos/marketing_live.ex`, `demos/policy_live.ex`,
`engine/gibbs_live.ex`, `engine/counterfactual_live.ex`,
`trust/calibration_live.ex`, `trust/diagnostics_live.ex`.

Representative copy — `demand_live.ex` (~lines 30–81), including the
conflated exit clause:

```elixir
def handle_event("fit", _params, %{assigns: %{running: true}} = socket) do
  {:noreply, socket}
end

def handle_event("fit", _params, socket) do
  scenario = socket.assigns.scenario
  socket = assign(socket, running: true, busy: false)

  {:noreply,
   start_async(socket, :fit, fn ->
     case BstsSite.Demos.Limiter.run(fn -> Demand.fit_and_forecast(scenario) end) do
       {:ok, result} -> result
       :busy -> :busy
     end
   end)}
end
...
def handle_async(:fit, {:ok, :busy}, socket) do
  {:noreply, assign(socket, running: false, busy: true)}
end
...
def handle_async(:fit, {:exit, _reason}, socket) do
  {:noreply, assign(socket, running: false, busy: true)}   # <- crash rendered as "busy"
end
```

The staleness-guard exemplar — `engine/counterfactual_live.ex` (~lines 83–92):

```elixir
def handle_async(:fit, {:ok, fit}, socket) do
  if fit.effect == socket.assigns.effect do
    stats = Demo.stats(fit.result, socket.assigns.alpha)
    {:noreply, assign(socket, running: false, busy: false, fit: fit, stats: stats)}
  else
    # The effect slider moved while the sampler ran; this result answers
    # a question the page is no longer asking. Drop it.
    {:noreply, assign(socket, running: false)}
  end
end
```

Note the success-clause bodies are page-specific (each derives different
assigns from the result — e.g. `demand_live` recomputes `decision`/`coverage`
from *current* assigns). The scaffold can own the control flow, but the
result-application must stay a per-page callback.

`speed_live.ex` uses assign names `racing`/`race` instead of
`running`/`fit` — normalize via the helper's options, not by renaming its
template assigns wholesale.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Site tests (characterization) | `cd site && mix test` | 0 failures |
| Compile | `cd site && mix compile --warnings-as-errors` | exit 0 |
| Format | `cd site && mix format` then `--check-formatted` | exit 0 |

Tooling note: `mix` is a `mise` shim; prefix `mise exec -- ` if needed.

## Scope

**In scope** (the only files you should modify/create):
- `site/lib/bsts_site_web/live/async_demo.ex` (create — the shared helper)
- The eight LiveViews listed above
- `site/lib/bsts_site_web/components/story.ex` ONLY if the error card needs a
  tiny component (prefer reusing existing verdict/busy markup with different
  copy)
- `site/test/bsts_site_web/live/async_demo_test.exs` (create)

**Out of scope** (do NOT touch):
- `site/lib/bsts_site/demos/*.ex` including `limiter.ex` — the demo layer and
  the semaphore are correct.
- Non-MCMC LiveViews (hub, tv, anomaly, kalman, smoother, noise, start,
  indexes).
- Catch-all `handle_event` hardening for *instant-lane* pages (the
  missing-param-key crash, TEST-08) — only add the catch-all to the eight
  files you're already editing; a site-wide sweep belongs to plan 025.

## Git workflow

- Branch: `advisor/024-async-scaffold-and-error-states` (from
  `execute-plans`, after plan 023 lands).
- Commit style: `refactor(site): shared async demo scaffold; log + surface fit crashes`
  — one commit for the helper, then one per migrated page (bisectable).
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 0: Run the characterization net

`cd site && mix test` (plan 023's suite must exist and pass).

**Verify**: 0 failures. If plan 023 isn't merged, STOP.

### Step 1: Build the helper

Create `site/lib/bsts_site_web/live/async_demo.ex`:

```elixir
defmodule BstsSiteWeb.AsyncDemo do
  @moduledoc """
  Shared control flow for Limiter-gated MCMC demos.

  `run_guarded/4` starts the async fit unless one is already running.
  `handle_result/4` folds the three async outcomes:

    * `{:ok, :busy}`   — semaphore full → `busy: true` (polite card)
    * `{:ok, result}`  — success → caller's `apply_result` callback, after an
      optional `stale?` check (result answers a question the page stopped asking)
    * `{:exit, reason}` — the fit CRASHED → logged with the page + reason,
      rendered as `error: true` (distinct from busy)
  """
  import Phoenix.Component, only: [assign: 2]
  import Phoenix.LiveView, only: [start_async: 3]
  require Logger

  def run_guarded(socket, key, running_flag \\ :running, work) do
    if socket.assigns[running_flag] do
      socket
    else
      socket
      |> assign([{running_flag, true}, {:busy, false}, {:error, false}])
      |> start_async(key, fn ->
        case BstsSite.Demos.Limiter.run(work) do
          {:ok, result} -> result
          :busy -> :busy
        end
      end)
    end
  end

  def handle_result(socket, outcome, opts) do
    running_flag = Keyword.get(opts, :running_flag, :running)
    stale? = Keyword.get(opts, :stale?, fn _result, _socket -> false end)
    apply_result = Keyword.fetch!(opts, :apply_result)
    page = Keyword.fetch!(opts, :page)

    case outcome do
      {:ok, :busy} ->
        assign(socket, [{running_flag, false}, {:busy, true}])

      {:ok, result} ->
        if stale?.(result, socket) do
          assign(socket, [{running_flag, false}])
        else
          socket |> assign([{running_flag, false}, {:busy, false}]) |> apply_result.(result)
        end

      {:exit, reason} ->
        Logger.error("demo fit crashed page=#{page} reason=#{inspect(reason)}")
        assign(socket, [{running_flag, false}, {:busy, false}, {:error, true}])
    end
  end
end
```

(Adjust to what compiles cleanly on LiveView 1.2 — e.g. `assign/2` with a map.
The load-bearing requirements: busy ≠ error, `Logger.error` with the reason,
optional staleness callback, configurable running-flag name for `speed_live`.)

**Verify**: `cd site && mix compile --warnings-as-errors` → exit 0.

### Step 2: Migrate the eight pages, one commit each

For each page: replace the trigger-event pair with one `handle_event` calling
`AsyncDemo.run_guarded/4`; replace the three `handle_async` clauses with one
that delegates to `AsyncDemo.handle_result/3`, passing:

- `page:` the module name,
- `apply_result:` a fun containing that page's existing success-assign logic
  **unchanged** (e.g. demand's `decision`/`coverage` recompute from current
  assigns),
- `stale?:` for `counterfactual`/`marketing`/`policy`, the existing param
  comparison moved verbatim into the callback; omit for fixed-scenario pages,
- `running_flag: :racing` for `speed_live`.

Each page also: initialize `error: false` in mount, add an error card in the
template where the busy card renders (distinct copy, e.g. "This computation
failed on our end — it's been logged. Reload to try again."), and add a
final catch-all `def handle_event(_event, _params, socket), do: {:noreply, socket}`
as the LAST handle_event clause.

**Verify after EACH page**: `cd site && mix test` → 0 failures (the plan-023
smoke + event tests are the net); `mix compile --warnings-as-errors` → 0.

### Step 3: Test the new behavior

Create `site/test/bsts_site_web/live/async_demo_test.exs`:

1. Crash → error, not busy: mount one migrated page (pick `speed`), fire the
   trigger with the fit rigged to crash — simplest is a direct unit test of
   `AsyncDemo.handle_result(socket_stub, {:exit, {%RuntimeError{}, []}}, opts)`
   asserting `error: true`, `busy: false`, plus
   `ExUnit.CaptureLog.capture_log` containing `"demo fit crashed"`.
2. Busy → busy: `handle_result(..., {:ok, :busy}, ...)` asserts `busy: true`,
   `error: false`.
3. Staleness: `stale?` returning true → result not applied, running cleared.
4. (LiveView-level, optional if a socket stub is awkward: build assigns as a
   plain `%Phoenix.LiveView.Socket{}` — `handle_result` only touches assigns.)

**Verify**: `cd site && mix test` → 0 failures, ≥3 new tests.

### Step 4: Confirm the dedup landed

**Verify**:
- `grep -rln "handle_async" site/lib/bsts_site_web/live | wc -l` → the eight
  files still declare `handle_async` (one thin clause each) but
  `grep -rn "{:exit, _reason}" site/lib/bsts_site_web/live | wc -l` → 0
  (no page-local exit handling remains).
- `grep -rn "Limiter.run" site/lib/bsts_site_web/live | wc -l` → 0 (only the
  helper calls it: `grep -c "Limiter.run" site/lib/bsts_site_web/live/async_demo.ex` → 1).
- `cd site && mix format --check-formatted` → exit 0.

## Test plan

Plan 023's route smoke + event tests are the regression net (run per page in
Step 2). New tests in Step 3 pin the three outcomes (busy / error+log /
stale-drop). Manual: load one MCMC page, click its run button, see the normal
result; no way to manually force a crash without code edits — the unit test
covers it.

## Done criteria

- [ ] `site/lib/bsts_site_web/live/async_demo.ex` exists; all 8 pages delegate to it
- [ ] `grep -rn "busy: true" site/lib/bsts_site_web/live/*.ex site/lib/bsts_site_web/live/**/*.ex` shows no `{:exit` clause mapping to busy anywhere
- [ ] `Logger.error` fires on `{:exit, _}` (asserted via CaptureLog test)
- [ ] Every one of the 8 pages has a catch-all `handle_event` fallthrough
- [ ] `cd site && mix test` → 0 failures (incl. ≥3 new)
- [ ] `cd site && mix compile --warnings-as-errors` and `mix format --check-formatted` → exit 0
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- Plan 023's tests don't exist or don't pass at Step 0.
- A page's success-clause logic can't be expressed as an `apply_result`
  callback without changing behavior (report which page and why).
- The helper's `import Phoenix.LiveView` surface differs on LiveView 1.2
  (e.g. `start_async/3` arity) in a way that needs API redesign — report,
  don't invent a macro layer.
- Any plan-023 smoke/event test fails after a migration and the cause isn't
  an obvious mechanical slip you can fix within that page's commit.

## Maintenance notes

- New MCMC demo pages should use `AsyncDemo` from day one; reviewers should
  reject fresh copies of the old scaffold. Parameter-varying pages MUST pass
  `stale?:` — the helper makes that a visible, named decision.
- The error card is deliberately generic; if operators want more, the logged
  `reason` is the hook for error tracking (e.g. a future Sentry/AppSignal).
- Plan 026 (per-client Limiter fairness) will change the `Limiter.run` call
  signature — after this plan there is exactly ONE call site to update
  (`async_demo.ex`), which is the point.
