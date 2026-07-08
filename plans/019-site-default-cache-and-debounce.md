# Plan 019: Stop the showcase site recomputing deterministic demos (default-result cache, slider debounce, hero hot-path fix)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat 9b7cb8d..HEAD -- site/lib`
> If files under `site/lib` changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: LOW-MED (touches the mount path of the busiest pages; mitigated by the site test suite + per-page manual checks)
- **Depends on**: none (pairs well with plan 015, which makes site tests run in CI)
- **Category**: perf (also mitigates the availability-DoS exposure SEC-01)
- **Planned at**: commit `9b7cb8d`, 2026-07-08

## Why this matters

The showcase site (`site/`, Phoenix LiveView, deployed on a Fly.io
shared-cpu-2x / 2 GB machine per `fly.toml`) computes every demo in-process.
Three compounding wastes, all verified in the code:

1. **Deterministic results recomputed per visitor.** Every demo's default
   figure is built from fixed seeds (`Scenarios` uses seeded `:rand.seed_s`;
   e.g. `Scenarios.hero(lift, noise_sd, seed: 42)`), so the default result is
   byte-identical for every visitor — yet there is **no caching mechanism
   anywhere in the app** (`grep persistent_term|:ets|Cachex site/lib` → 0
   matches). The landing page (`HubLive`) rebuilds 12 `Operational.prepare`
   models (~7 ms each per `Hero`'s own docstring) plus a full run on every
   mount.
2. **Everything computes twice per page load.** LiveView calls `mount/3` once
   for the static HTTP render and again on WebSocket connect;
   `grep connected? site/lib` → **0 matches**, so all eight compute-in-mount
   pages pay double.
3. **Sliders fire a full model fit per drag tick.** `grep phx-debounce
   site/lib` → **0 matches**. The shared `param_slider` component renders a
   bare `<input type="range">`; each intermediate tick triggers a synchronous
   Kalman/fit event handler (hub `adjust` → `Hero.run`, tv `adjust` →
   `Tv.run` incl. Shapley, anomaly slider → full detector re-fit-and-score,
   etc.). None of these "instant lane" events go through the `Limiter` (which
   only guards the MCMC buttons), so one dragging visitor floods the 2-vCPU
   box — this is also the main ingredient of the availability-DoS finding
   SEC-01.

There is also a small hot-path bug: `Scenarios.hero/3` indexes a plain list
with `Enum.at(noise, t)` inside a map over all 144 indices — O(n²) on the
landing page's recompute path.

The fix strategy deliberately avoids the risky variant (nil-skeleton static
renders): cache the deterministic *default* artifacts once, have `mount` read
the cache on both the static and connected render (cheap either way), debounce
the sliders, and fix the O(n²) loop. Non-default recomputes only ever happen
via user events, which only occur on connected sockets.

## Current state

- `site/lib/bsts_site_web/live/hub_live.ex` lines 11–19 — the landing-page
  mount:

  ```elixir
  def mount(_params, _session, socket) do
    prepared = Hero.prepare()
    demo = Hero.run(prepared, 12, 4)

    {:ok,
     socket
     |> assign(page_title: "Was it you, or was it noise?")
     |> assign(prepared: prepared, demo: demo, revealed: false)}
  end
  ```

- `site/lib/bsts_site/demos/hero.ex` — `prepare/0` builds all 12 models
  eagerly (`Map.new(@noise_range, fn noise_sd -> ... Operational.prepare(...)`)
  and `run/3` picks one via `Map.fetch!(prepared_by_noise, noise_sd)`.

- `site/lib/bsts_site_web/components/story.ex` (~lines 315–330) — the shared
  slider, no debounce:

  ```heex
  <input
    type="range"
    name={@name}
    min={@min}
    max={@max}
    step={@step}
    value={@value}
    class="range range-xs range-primary mt-1 w-full"
  />
  ```

- `site/lib/bsts_site/demos/scenarios.ex` lines 45–65 — the O(n²) loop:

  ```elixir
  def hero(lift, noise_sd, opts \\ []) do
    seed = Keyword.get(opts, :seed, 42)
    total = 144
    pre_end = 96
    noise = noise(total, noise_sd, seed)

    {observations, truth_effect} =
      0..(total - 1)
      |> Enum.map(fn t ->
        baseline = 84.0
        effect = ...
        {baseline + effect + Enum.at(noise, t), effect}
      end)
      |> Enum.unzip()
  ```

- The eight compute-in-mount LiveViews (all unguarded, verified):
  `hub_live.ex`, `speed_live.ex`, `demos/tv_live.ex`, `demos/anomaly_live.ex`,
  `demos/marketing_live.ex`, `engine/kalman_live.ex`,
  `engine/smoother_live.ex`, `engine/noise_live.ex` (whose
  `NoiseQuiz.batch(0)` runs up to 10 candidate generations × 4
  `estimate_from_filter` passes).

- MCMC demo pages (`gibbs`, `counterfactual`, `calibration`, `diagnostics`,
  `policy`, `demand`) do **only cheap scenario setup in mount** — heavy
  sampling is already correctly behind buttons + `Limiter` + `start_async`.
  They are NOT in scope for the cache.

- The site's design contract (`site/DESIGN_CONTRACT.md`) defines latency
  tiers; the instant first paint is a stated product requirement — which the
  cache preserves better than a skeleton.

- Site tests: `cd site && mix test` → 7 tests, 0 failures at planning time.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Site tests | `cd site && mix test` | 0 failures |
| Site compile | `cd site && mix compile --warnings-as-errors` | exit 0 |
| Site format | `cd site && mix format` then `mix format --check-formatted` | exit 0 |
| Run the site locally | `cd site && mix phx.server` then open http://localhost:4000 | pages render |

Tooling note: `mix` is a `mise` shim here; prefix `mise exec -- ` if needed.
Local quirk: if `mix assets.build`/`phx.server` dies with SIGKILL on the
tailwind/esbuild binaries (macOS Gatekeeper), re-sign them:
`codesign --force --sign - site/_build/tailwind-macos-arm64*` (same for
esbuild), then retry.

## Scope

**In scope** (the only files you should modify):
- `site/lib/bsts_site/demos/default_cache.ex` (create)
- `site/lib/bsts_site_web/live/hub_live.ex`, `speed_live.ex`,
  `demos/tv_live.ex`, `demos/anomaly_live.ex`, `demos/marketing_live.ex`,
  `engine/kalman_live.ex`, `engine/smoother_live.ex`, `engine/noise_live.ex`
  (mount changes only)
- `site/lib/bsts_site_web/components/story.ex` (one attribute on the slider)
- `site/lib/bsts_site/demos/scenarios.ex` (the `hero/3` loop)
- `site/test/bsts_site/demos/default_cache_test.exs` (create)

**Out of scope** (do NOT touch):
- `site/lib/bsts_site/demos/limiter.ex` — verified correct; per-IP fairness is
  a separate recorded finding (SEC-01 residual), not this plan.
- The MCMC LiveViews (`gibbs`, `counterfactual`, `calibration`,
  `diagnostics`, `policy`, `demand`) — their mounts are already cheap.
- `handle_async`/`handle_event` bodies — the exit-vs-busy conflation and the
  async-scaffold dedup are a separate recorded finding (SITE-BUG-02/DEBT-01).
- Chart components (`charts.ex`) — SVG payload downsampling (PERF-04) was
  deliberately deferred until after debounce lands; do not attempt it here.
- The library (`lib/` at repo root).

## Git workflow

- Branch: `advisor/019-site-default-cache-and-debounce` (from `execute-plans`).
- Commit style: `perf(site): cache deterministic demo defaults, debounce sliders`
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Add `phx-debounce` to the shared slider

In `site/lib/bsts_site_web/components/story.ex`, add `phx-debounce="150"` to
the `<input type="range" ...>` in `param_slider/1`. One attribute; because
every demo slider goes through this component, this throttles every
instant-lane recompute in the app at once.

**Verify**: `grep -rn "phx-debounce" site/lib | wc -l` → `1`;
`cd site && mix test` → 0 failures.

### Step 2: Create the default-result cache

Create `site/lib/bsts_site/demos/default_cache.ex`:

```elixir
defmodule BstsSite.Demos.DefaultCache do
  @moduledoc """
  Lazy, process-independent memoization for deterministic demo defaults.

  Every demo's default figure is computed from fixed seeds, so the result is
  identical for every visitor. First reader computes and stores; later
  readers (including the disconnected static render and the WebSocket mount
  of the same visit) get a `:persistent_term` read.

  Only cache values derived from compile-time constants — never anything
  derived from user params. A duplicate concurrent first-compute is benign
  (both writers store the same deterministic value).
  """

  @spec get(term(), (-> result)) :: result when result: var
  def get(key, fun) when is_function(fun, 0) do
    pt_key = {__MODULE__, key}

    case :persistent_term.get(pt_key, :__miss__) do
      :__miss__ ->
        value = fun.()
        :persistent_term.put(pt_key, value)
        value

      value ->
        value
    end
  end
end
```

Design constraints (why this shape):
- Lazy (no app-start precompute) so boot stays fast and only visited pages pay.
- `:persistent_term` is write-once/read-many-optimized; the site runs the
  BinaryBackend, so cached Nx tensors are plain Erlang terms (no NIF resource
  refs) — safe to store. Values are never erased, so total entries are bounded
  by the fixed set of demo keys.

**Verify**: `cd site && mix compile --warnings-as-errors` → exit 0.

### Step 3: Wire the eight mounts through the cache

For each in-scope LiveView, wrap the mount's deterministic default compute in
`DefaultCache.get/2` with a stable key. The transformation for `hub_live.ex`
(apply the same pattern elsewhere):

```elixir
alias BstsSite.Demos.DefaultCache

def mount(_params, _session, socket) do
  %{prepared: prepared, demo: demo} =
    DefaultCache.get(:hub, fn ->
      prepared = Hero.prepare()
      %{prepared: prepared, demo: Hero.run(prepared, 12, 4)}
    end)

  {:ok,
   socket
   |> assign(page_title: "Was it you, or was it noise?")
   |> assign(prepared: prepared, demo: demo, revealed: false)}
end
```

Key/value per page (cache the *default* artifacts only — event handlers keep
computing fresh results from user params exactly as today):

| LiveView | Cache key | Cache value |
|----------|-----------|-------------|
| `hub_live.ex` | `:hub` | `%{prepared:, demo:}` (prepared map + default run) |
| `speed_live.ex` | `:speed` | whatever `Speed.prepare()` (and any default run in mount) produces |
| `demos/tv_live.ex` | `:tv` | the mount's scenario + default attribution result |
| `demos/anomaly_live.ex` | `:anomaly` | base series + default fit/score result |
| `demos/marketing_live.ex` | `:marketing` | the mount's scenario/fast-lane default |
| `engine/kalman_live.ex` | `:kalman` | prepared artifacts + default tune result |
| `engine/smoother_live.ex` | `:smoother` | `Hindsight.prepare()` output + default draw |
| `engine/noise_live.ex` | `{:noise_batch, 0}` | `NoiseQuiz.batch(0)` |

Notes:
- `NoiseLive`: only the **initial** batch (counter 0) is cached; the
  "regenerate" event's `batch(counter)` for `counter > 0` stays uncached (it
  varies per click). Key by counter so a future decision to cache more is a
  one-liner.
- Read each mount carefully: cache exactly the part that depends only on
  constants. Anything touching `params`/`session` stays outside the cache.
- Do not add `connected?/2` branching — with the cache, both the static and
  the connected mount do a cheap read, and the static render keeps showing the
  real figure (the design contract's instant first paint).

**Verify** after each file: `cd site && mix compile --warnings-as-errors` →
exit 0. After all eight: `grep -rn "DefaultCache.get" site/lib | wc -l` → `8`.

### Step 4: Fix the O(n²) loop in `Scenarios.hero/3`

Replace the `Enum.at(noise, t)` indexing with a single zip pass, preserving
output exactly, e.g.:

```elixir
{observations, truth_effect} =
  noise
  |> Enum.with_index()
  |> Enum.map(fn {noise_t, t} ->
    baseline = 84.0

    effect =
      if t >= pre_end do
        lift * (1.0 - :math.exp(-(t - pre_end + 1) / 6.0))
      else
        0.0
      end

    {baseline + effect + noise_t, effect}
  end)
  |> Enum.unzip()
```

**Verify**: `grep -n "Enum.at(noise" site/lib/bsts_site/demos/scenarios.ex` →
no matches; `cd site && mix compile --warnings-as-errors` → exit 0.

### Step 5: Add tests for the cache

Create `site/test/bsts_site/demos/default_cache_test.exs`, modeled
structurally on `site/test/bsts_site/demos/limiter_test.exs`:

1. `get/2` computes on first call and returns the stored value on the second
   (use a counter in a `:counters` ref or an Agent to assert the fun ran once).
2. Distinct keys don't collide.
3. A determinism guard for the biggest cached page:
   `Scenarios.hero(12, 4)` called twice returns identical maps (protects the
   cache's core assumption after Step 4's refactor).

Use unique keys per test (e.g. `{:test, make_ref()}`) so `persistent_term`
state can't leak between tests.

**Verify**: `cd site && mix test` → 0 failures, ≥3 new tests.

### Step 6: Manual smoke of the four highest-traffic pages

Run `cd site && mix phx.server`, then load `/` (hub), the speed page, the
anomaly demo, and the noise quiz in a browser. Drag one slider on the hub.

**Verify**: figures render identically to before (compare against a
pre-change run if in doubt); slider still updates (with ≤~150 ms lag);
server log shows no errors.

### Step 7: Format and finish

**Verify**: `cd site && mix format --check-formatted` → exit 0;
`git status --porcelain` → only in-scope files.

## Test plan

Step 5's cache tests plus the existing 7 site tests. The determinism test
(hero output stable across calls) is the load-bearing one — it guards the
assumption the whole cache rests on. Manual smoke in Step 6 covers rendering,
which has no automated coverage yet (site LiveView tests are a separately
recorded finding, TEST-02).

## Done criteria

- [ ] `grep -rn "phx-debounce" site/lib | wc -l` → `1` (in `story.ex`)
- [ ] `grep -rn "DefaultCache.get" site/lib | wc -l` → `8`
- [ ] `grep -n "Enum.at(noise" site/lib/bsts_site/demos/scenarios.ex` → no matches
- [ ] `cd site && mix test` → 0 failures, with ≥3 new cache tests
- [ ] `cd site && mix compile --warnings-as-errors` → exit 0
- [ ] `cd site && mix format --check-formatted` → exit 0
- [ ] `git status --porcelain` shows only in-scope files modified
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- Any mount computes from `params`/`session` in a way that can't be cleanly
  split from the constant part — report the page instead of caching
  user-derived data.
- A demo module turns out to be nondeterministic across calls (the Step 5
  determinism test fails before your changes) — the cache assumption is wrong
  for that page; exclude it and report.
- Cached tensors misbehave (errors mentioning backend references or
  `:persistent_term` — would indicate a non-BinaryBackend configuration) —
  stop; do not switch backends.
- Step 6 shows any figure rendering differently from before the change.

## Maintenance notes

- **Never cache user-parameterized results** in `DefaultCache` — only
  compile-time-constant defaults. `persistent_term` writes trigger a global GC
  scan; a per-params cache there would be a footgun. If per-params caching is
  ever wanted, use ETS instead.
- If a demo's constants (seeds, ranges, series lengths) change, the cache
  self-heals on next boot (keys are recomputed lazily; terms don't survive the
  VM). No invalidation logic needed.
- The debounce value (150 ms) is a UX/thoroughput tradeoff; if sliders feel
  laggy, 100 ms is fine — the point is collapsing dozens-per-second to
  a-few-per-second.
- Deferred follow-ups recorded in `plans/README.md`: SVG payload downsampling
  (PERF-04, measure after debounce), per-IP fairness on the Limiter (SEC-01
  residual), async scaffold dedup + crash-vs-busy split (DEBT-01/SITE-BUG-02).
