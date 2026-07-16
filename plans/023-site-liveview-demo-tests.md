# Plan 023: Add site LiveView smoke tests and demo shape-contract tests

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat 9b7cb8d..HEAD -- site/lib site/test`
> Plans 019/021 may legitimately have touched `site/lib` mounts — that is
> expected drift; test what is live. Structural changes beyond that (routes
> added/removed, demos renamed): reconcile with `site/lib/bsts_site_web/router.ex`
> before proceeding.

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: LOW (tests only)
- **Depends on**: plans/015-site-ci-verification-lane.md (so these tests run in CI). Execute BEFORE plan 024 (they are its characterization safety net).
- **Category**: tests
- **Planned at**: commit `9b7cb8d`, 2026-07-08

## Why this matters

The site has ~20 LiveViews and 17 demo modules; only the `Limiter` and the
error controllers have tests (7 tests total). The demo modules hard-depend on
exact library return-map shapes — e.g. `site/lib/bsts_site/demos/marketing.ex`
reads `result.summary.baseline`, `result.counterfactual.variance`,
`result.summary.cumulative_effect`, `result.summary.relative_effect`,
`result.execution` — so any library change to those maps breaks the deployed
site at runtime with nothing catching it. These tests serve three purposes at
once: (1) mount smoke coverage for every route, (2) a shape contract between
the site and the library (they fail in CI when the library drifts), and
(3) the characterization net under plan 024's LiveView refactor.

## Current state

- Existing site tests: `site/test/bsts_site/demos/limiter_test.exs`,
  `site/test/bsts_site_web/controllers/error_html_test.exs`,
  `error_json_test.exs`. Support: `site/test/support/conn_case.ex`
  (`use BstsSiteWeb.ConnCase`). `{:lazy_html, ">= 0.1.0", only: :test}` is in
  `site/mix.exs` — `Phoenix.LiveViewTest` is available (LiveView ~> 1.2).
- Routes: enumerate live in `site/lib/bsts_site_web/router.ex` (the `scope "/"`
  block, `live "..."` lines — `/`, `/start`, `/speed`, demos index + 5 demo
  pages, engine index + ~6 chapters, trust index + 3 pages, hub; read the
  file, don't trust this list).
- A demo shape dependency example, `site/lib/bsts_site/demos/marketing.ex`
  (~lines 95–112):

  ```elixir
  {band_lower, band_upper} =
    [result.summary.baseline, result.counterfactual.variance]
    |> Enum.zip_with(fn [mean, var] ->
      sd = :math.sqrt(max(var, 0.0))
      {mean - band_z * sd, mean + band_z * sd}
    end)
    |> Enum.unzip()

  %{
    counterfactual_mean: result.summary.baseline,
    band_lower: band_lower,
    band_upper: band_upper,
    cumulative: result.summary.cumulative_effect,
    relative: result.summary.relative_effect,
    execution: result.execution,
    significant?: interval_excludes_zero?(result.summary.cumulative_effect)
  }
  ```

- Latency reality (drives test selection): instant-lane demo functions
  (`Hero.run`, `Marketing.fast_lane`, `KalmanTuner`, `Anomaly.run`,
  `NoiseQuiz`, `Hindsight`, `Compose`, `Tv.run`, `Scenarios.*`) are
  milliseconds — testable directly. MCMC functions
  (`Counterfactual.fit`, `Demand.fit_and_forecast`, `Speed.race`,
  `Gibbs`, `Calibration`, `Diagnostics`, `Policy`, `Marketing.run_mcmc`) take
  ~0.5–2 s each — tag their direct tests `@tag :slow` so the default site
  suite stays fast, and do NOT trigger them via LiveView events in tests.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Compile | `cd site && mix compile --warnings-as-errors` | exit 0 |
| Format | `cd site && mix format` | exit 0 |
| Site tests | `cd site && mix test` | 0 failures (post-event-test budget: <30 s) |
| Incl. slow demo tests | `cd site && mix test --include slow` | 0 failures |

Tooling note: `mix` is a `mise` shim; prefix `mise exec -- ` if needed.

## Scope

**In scope** (create only; the only existing file you may edit is
`site/test/test_helper.exs` if slow-tag exclusion needs configuring):
- `site/test/bsts_site_web/live/routes_smoke_test.exs`
- `site/test/bsts_site/demos/shapes_test.exs`
- (optional split) one extra file per demo if `shapes_test.exs` grows unwieldy

**Out of scope** (do NOT touch):
- Any file under `site/lib/` — if a test exposes a bug, report it; don't fix
  here (plan 024 owns LiveView changes).
- `site/test/support/conn_case.ex` unless a one-line import of
  `Phoenix.LiveViewTest` is genuinely needed there (prefer importing in the
  test files).
- The library and its tests.

## Git workflow

- Branch: `advisor/023-site-liveview-demo-tests` (from `execute-plans`).
- Commit style: `test(site): route smoke tests + demo shape contracts`
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Configure slow-tag exclusion

In `site/test/test_helper.exs`, ensure `:slow` tests are excluded by default.
If it's a bare `ExUnit.start()`, change it to `ExUnit.start(exclude: [:slow])`.

**Verify**: `cat site/test/test_helper.exs` shows the exclusion config.

### Step 2: Route smoke tests

Create `site/test/bsts_site_web/live/routes_smoke_test.exs`:

```elixir
defmodule BstsSiteWeb.RoutesSmokeTest do
  use BstsSiteWeb.ConnCase, async: true
  import Phoenix.LiveViewTest

  @routes [
    "/",
    # ...every live route from router.ex — enumerate them all
  ]

  for route <- @routes do
    test "mounts #{route}" do
      conn = Phoenix.ConnTest.build_conn()
      {:ok, _view, html} = live(conn, unquote(route))
      assert html =~ "<svg" or html =~ "BstsNx" # cheap sanity; refine per page if trivial
    end
  end
end
```

Fill `@routes` from `router.ex` (every `live` route with no params). This
exercises the full `mount` — including each demo's default computation — so
it doubles as the "library shapes still feed the pages" canary. Keep
`async: true` unless a shared resource (the Limiter, `persistent_term` cache
from plan 019) makes it flaky — if it does, drop async and note why.

**Verify**: `cd site && mix test test/bsts_site_web/live/routes_smoke_test.exs`
→ one passing test per route, 0 failures. Record the wall time; if any single
mount takes >2 s, note which (candidate for a mount that's doing MCMC-tier
work — report it, that would contradict the audit).

### Step 3: One instant-lane event test per interactive page

Extend the smoke file (or a sibling) with a `render_change`/`render_click`
test for one representative **instant-lane** event per page that has one —
e.g. hub `adjust` with `%{"lift" => "10", "noise" => "5"}`, kalman `tune`,
anomaly slider, tv `adjust`, marketing `adjust`, counterfactual `adjust`,
noise `choose`/`regenerate`, `reveal_truth` on one page. Get exact event
names + param keys from each LiveView's `handle_event` heads — copy them, do
not guess. Assert the render result is a binary containing an expected marker
(e.g. the page still contains `phx-` bindings / an `<svg`).

Do NOT fire MCMC-button events (`fit`, `race`, `run_mcmc`, `run_checks`,
`run_sweep`) — they cost seconds and go through the Limiter.

**Verify**: `cd site && mix test` → 0 failures, total suite still <30 s.

### Step 4: Demo shape-contract tests

Create `site/test/bsts_site/demos/shapes_test.exs` with a test per demo
module asserting the exact keys and length invariants the templates/chart
prep consume. For instant-lane demos, call directly; for MCMC demos, tag
`@tag :slow` and use the module's smallest workload. Pattern:

```elixir
test "Marketing.fast_lane/1 returns the shape the page consumes" do
  result = BstsSite.Demos.Marketing.fast_lane(<default args — read the module>)

  assert is_list(result.counterfactual_mean)
  assert length(result.band_lower) == length(result.counterfactual_mean)
  assert length(result.band_upper) == length(result.counterfactual_mean)
  assert is_number(result.cumulative) and result.cumulative == result.cumulative
  assert %{elapsed_ms: _, method_used: _} = result.execution
  assert is_boolean(result.significant?)
end
```

Cover for each demo: (a) every key the LiveView/HEEx reads exists, (b) series
that are plotted together have equal lengths, (c) numeric outputs are finite
(`x == x` catches NaN), (d) `execution.elapsed_ms` present where the exec
badge is rendered. Include `Scenarios.hero/2` determinism (two calls, equal
results) if plan 019 hasn't already added it.

**Verify**: `cd site && mix test` (fast lane) and
`cd site && mix test --include slow` (MCMC shapes) → 0 failures each.

### Step 5: Format + full pass

**Verify**: `cd site && mix format --check-formatted` → exit 0;
`cd site && mix compile --warnings-as-errors` → exit 0.

## Test plan

This plan IS the test plan. Deliverables: one mount smoke per route, one
instant-lane event test per interactive page, one shape-contract test per
demo module (16 modules; limiter already covered), slow-tagged where the
demo is MCMC-tier. Structural pattern: `limiter_test.exs` for plain ExUnit,
`Phoenix.LiveViewTest.live/2` for the smoke tests.

## Done criteria

- [ ] Every live route in `router.ex` has a mount smoke test (count them: routes in file == tests in `routes_smoke_test.exs`)
- [ ] Every demo module in `site/lib/bsts_site/demos/` except `limiter.ex` has ≥1 shape test (16 modules)
- [ ] `cd site && mix test` → 0 failures, wall time <30 s
- [ ] `cd site && mix test --include slow` → 0 failures
- [ ] `cd site && mix format --check-formatted` and `mix compile --warnings-as-errors` → exit 0
- [ ] Only files under `site/test/` changed (`git status --porcelain`)
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- Any mount or shape test FAILS against current code — that's a live bug or
  a shape mismatch the audit predicted; report it verbatim (do not adjust the
  site code, and do not write the test to match broken behavior without
  flagging it).
- A "fast" demo takes >2 s in tests (mis-tiered demo — report).
- LiveView tests can't run because of a missing test dep (would contradict
  the `lazy_html` line in `site/mix.exs` — check before improvising deps).
- Route enumeration finds routes requiring session/params this plan doesn't
  cover (none expected — everything is anonymous `live` routes).

## Maintenance notes

- These shape tests are the early-warning system for library↔site drift:
  when a library return map changes, fix the demo AND the test in the same
  commit. Reviewers should treat a shape-test edit without a demo edit (or
  vice versa) as a smell.
- Plan 024 (async scaffold refactor) relies on the Step 1/2 tests passing
  before and after its refactor — run them at its start.
- When plan 021 (EXLA) merges, `mix test --include slow` timing on the MCMC
  shape tests is a cheap regression check on the backend speedup.
