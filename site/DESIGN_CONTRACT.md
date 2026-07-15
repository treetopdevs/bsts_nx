# BstsNx Showcase Site: Build Contract

Read this in full before writing any code. It is the binding contract for every
page on this site. The reference implementation is the hub:
`lib/bsts_site_web/live/hub_live.ex` + `lib/bsts_site/demos/hero.ex` +
`lib/bsts_site/demos/scenarios.ex`. Mirror its structure exactly.

## The story (why every page exists)

One question: **"The numbers changed. Was it you, or was it noise?"**
One method: **decompose → project → compare.**
One recurring device: **the planted truth.** Every demo runs on synthetic data
we generated, so the exact answer is known. The demo estimates first, then the
visitor clicks "Reveal the planted truth" and the model gets graded in public.
One proof device: **the execution badge.** The library self-reports
`execution.method_used` and `execution.elapsed_ms`; every figure carries it.
Nothing on this site is a screenshot.

## Page anatomy (required, in order)

1. `<Layouts.app flash={@flash} track={:questions|:engine|:trust|:speed|:start}>`
2. `<.section_heading level={1} eyebrow="..." title="...">` for the page's
   primary heading; nested sections keep the default `level={2}`. Eyebrow is
   the question or chapter label; body is 2–4 sentences of scene-setting prose.
3. Controls (sliders/selects in a `phx-change` form) where interactive.
4. `<.figure no="1" caption="..." execution={...}>` wrapping a chart component.
   Number figures sequentially per page ("1", "2", ...). Caption = one factual
   sentence about what is drawn. Add `<:legend>` with `<.swatch>` entries.
5. `<.verdict_card>`: the plain-language answer with stats.
6. `<.reveal_truth>` where a truth was planted.
7. `<.under_the_hood code={...}>`: the REAL library calls the page runs,
   kept as a `code_snippet/0` function in the page's demo module, adjacent to
   the code it mirrors so it cannot drift.
8. Cross-links: 1-3 `<.cross_link>` cards to related pages
   (demo -> the engine chapter that powers it; engine -> the trust page that
   stress-tests it; everything may link to `/speed` when latency is the point).
9. `<.mantra active={:decompose|:project|:compare}>` near the heading on engine
   pages (pick the stage the chapter teaches); optional elsewhere.

## Module layout

- Statistical logic: `lib/bsts_site/demos/<name>.ex`: a module like
  `BstsSite.Demos.Hero`: pure functions, deterministic seeds, returns plain
  maps with everything the LiveView needs (series as lists of floats, truth,
  execution metadata, verdict inputs). NO Phoenix imports in these modules.
- Page: `lib/bsts_site_web/live/...`: thin LiveView: mount prepares, events
  re-run, render composes contract components. Keep statistical code out.
- Shared deterministic noise: `BstsSite.Demos.Scenarios.noise/3` (do not edit
  `scenarios.ex`; put new generators in your own demo module).

## Design system laws

The physical scene is a technical report printed on cool graph paper, read by
an engineer deciding whether the live computations are honest. The site is
light-only by design. Do not add dark-mode treatments unless the product
context changes.

- Use OKLCH tokens from `assets/css/app.css`. Owned code should not introduce
  hard-coded hex or `rgba(0, 0, 0, ...)` colors. If JavaScript needs color, use
  OKLCH values aligned with the CSS tokens, or CSS custom properties when the
  library supports them.
- Keep the semantic color contract strict: ink is observed data, model blue is
  estimates and counterfactuals, lift magenta is effects only, and truth green
  is planted ground truth only. Topbar/loading chrome follows the same model
  blue plus tinted ink system.
- Typography is part of the proof voice: STIX Two Text for display/prose, IBM
  Plex Sans for UI, IBM Plex Mono for data and code. Use the existing font
  classes and keep prose near 65-75ch.
- No side-stripe borders. Quote blocks, callouts, and verdicts use full borders
  with paper/background tints. Never use `border-l-2`, `border-l-4`, or wider
  as a visual accent.
- No gradient text, decorative backdrop blur, glass cards, or hero-metric
  templates. Use solid color, weight, layout, and live computation for emphasis.
- Cards are for figures, cross-links, and framed tools. Do not nest card
  surfaces or turn whole page sections into floating cards.
- Every page has one primary heading: `<.section_heading level={1} ...>` or a
  literal `<h1>`. Nested sections keep the default `level={2}`.
- Every SVG chart call must provide authored `title` and `desc` values. The
  component fallback exists only as a guardrail for missed call sites.
- Touch targets must stay comfortable on coarse pointers. Use the existing
  `.nav-pill`, `.touch-target`, `.btn-sm`, `.select-sm`, and range control
  support rather than shrinking controls.
- Motion must respect `prefers-reduced-motion`. Animate stroke, opacity, color,
  or transform, not layout properties. Avoid `transition-all`.

## Components API (import is automatic via `use BstsSiteWeb, :live_view`)

`BstsSiteWeb.Charts`:
- `<.line_figure id= title= desc= series=[%{points: [float], x_offset: 0, color: :ink|:model|:lift|:truth, dash: bool, width: float, opacity: float, draw: bool}] bands=[%{lower:, upper:, x_offset:, fill: "var(--color-model-band)"|"var(--color-lift-band)"}] vlines=[%{x:, label:}] hlines=[%{y:, label:, color:}] markers=[%{x:, y:, color:, shape: :dot|:x}] y_label= height=300 y_domain={min,max}|:auto />`
- `<.bar_figure id= title= desc= items=[%{label:, value:, lower:, upper:, color:}] unit="" height=220 />`
- `<.gantt id= title= desc= rows=[%{id:, start:, stop:, color:}] domain={min,max} height=140 />` (half-open windows)
- `<.matrix_grid id= title= desc= matrix=[[num]] blocks=[%{from:, size:, label:}] />`
- `<.histogram id= title= desc= values=[float] bins=24 vlines=[%{x:, label:, color:}] height=170 />`

Chart `title` and `desc` feed the SVG accessibility tree. Keep them short,
factual, and aligned with the figure caption. The component fallback is only a
guardrail for missed call sites, not the desired authored state.

Semantic colors are LAW: ink = observed data; `:model` blue = model estimates /
counterfactuals; `:lift` magenta = effects and ONLY effects; `:truth` green =
planted ground truth and ONLY that.

`BstsSiteWeb.Story`: `figure`, `swatch`, `exec_badge`, `verdict_card` (tone:
:lift|:truth|:neutral|:warning, `<:stat label= value= hint=>`), `reveal_truth`
(assigns `revealed` boolean + `event` name, default "reveal_truth"),
`under_the_hood`, `mantra`, `cross_link`, `section_heading`, `param_slider`.

## Latency tiers (measured on BinaryBackend: obey strictly)

- **Instant (<50 ms)**: Kalman/RTS/FFBS on ≤200 pts, `Operational.prepare+run`,
  `CausalImpact.estimate_from_filter`, AnomalyDetector fit/score, Shapley given
  existing counterfactual, spec composition. Run directly in `handle_event`.
- **Short (0.5–3 s)**: scalar MCMC (`GibbsSampler.sample`,
  `CausalImpact.estimate`, scalar `Forecaster.fit_predict`) at ≤100 samples,
  <=150 points. Use `start_async` + spinner, wrapped in
  `BstsSite.Demos.Limiter.run/1`; on `:busy` show a polite "another visitor is
  sampling, try again in a moment" note (verdict_card tone: :warning).
- **FORBIDDEN live**: anything with a seasonal/structured ModelSpec through
  MCMC (minutes per run), `seasonality:` options on MCMC paths, num_samples > 100,
  series > 400 points. If a page needs structured-MCMC output, fake nothing:
  use the scalar path or redesign the demo.
- All numeric inputs come from bounded sliders/selects. Clamp server-side
  anyway (see `Hero.clamp/2`). Never allow `num_samples < 2` (NaN summaries).

Async pattern:
```elixir
# Guard first: double-clicks must not start two samplers.
def handle_event("run", _p, %{assigns: %{running: true}} = socket), do: {:noreply, socket}

def handle_event("run", _p, socket) do
  socket = assign(socket, running: true, busy: false)
  {:noreply,
   start_async(socket, :fit, fn ->
     case BstsSite.Demos.Limiter.run(fn -> Demo.fit(...) end) do
       {:ok, result} -> result
       :busy -> :busy
     end
   end)}
end

def handle_async(:fit, {:ok, :busy}, socket), do: {:noreply, assign(socket, running: false, busy: true)}
def handle_async(:fit, {:ok, result}, socket), do: {:noreply, assign(socket, running: false, result: result)}
def handle_async(:fit, {:exit, reason}, socket) do
  Logger.error("async fit failed: #{inspect(reason)}")
  {:noreply, assign(socket, running: false, busy: false, error: "The fit failed. Please try again.")}
end
```

## Library conventions (get these right or the page crashes/lies)

- `pre_period`/`post_period` are **1-based inclusive** tuples; post must
  immediately follow pre. Spot windows are **0-based half-open** maps
  `%{id:, window_start:, window_end:}` **indexed within the post-period**.
- `CausalImpact.estimate_from_filter/3` takes a list of 0-based intervention
  indices (e.g. `Enum.to_list(96..143)`), NOT period tuples. Its baseline is an
  RTS interpolation that HAS SEEN post data. Label it "fast interpolation",
  never "counterfactual forecast". The honest fast counterfactual is
  `BstsNx.Operational` with its forecast-first baseline.
- `Operational.run(prepared, obs, spots, return: :lists)` result:
  `summary.baseline` / `summary.actual` are POST-PERIOD-length lists;
  `summary.point_effects.{mean,lower,upper}`, `summary.cumulative_effect.{mean,sd,lower,upper}`,
  `summary.relative_effect.{mean,sd,lower,upper}` (fractions, multiply by 100 for %),
  `counterfactual.{mean,variance}`, `attributions.{attributions,total_lift,total_lift_sd}`,
  `execution.{method_used,elapsed_ms,backend,...}`.
- Verify every other signature against `../lib/bsts_nx/*.ex` BEFORE using it.
  Do not trust your memory of the docs. The module for Shapley is
  `BstsNx.ShapleyAllocator` / `BstsNx.SpotAttributor` (there is no `BstsNx.Shapley`).
- Seed everything (`seed:` opts, `Scenarios.noise/3`). Same inputs → same page.

## Copy voice

Plain-spoken, curious, honest. Like the livebooks: "The black line is what
actually happened." Never "blazingly", "powerful", "seamless", "magic".
Uncertainty is a feature. Write verdicts that respect it ("Honestly? Can't
tell; the interval straddles zero."). Numbers in copy must come from the live
computation, never hardcoded. Sentence case everywhere. Errors instruct, don't
apologize. Do not use em dashes in UI copy or in this contract.

Accuracy guardrails (site-wide law):
1. The library is early-stage (v0.1.0), validated against the R bsts /
   CausalImpact reference stack. Never claim parity with or superiority to them.
2. License is LGPL-2.1-only. Not on Hex yet. `/start` handles install copy;
   other pages never show `{:bsts_nx, "~> 0.1"}` as a working install.
3. No fabricated benchmark numbers: performance claims either come from the
   page's own live `execution` metadata or say "measured on this server".
4. Spike-and-slab regression EXISTS (`Components.regression_spec(mode: :spike_and_slab)`,
   `CovariateSelection.select_spike_slab/3`).

## Working rules

- Touch ONLY your assigned files. Never edit: contract components
  (`charts.ex`, `story.ex`), layouts, router, `scenarios.ex`, `hero.ex`,
  `hub_live.ex`, `mix.exs`, CSS, or anything in `../lib` (the library is
  READ-ONLY). If a component is missing an ability, work within what exists:
  compose SVG-bearing components you already have; do not invent new shared ones.
- Replace your assigned stub LiveViews entirely (keep module names/routes).
- Verify: from the repo root, run `cd site && mix compile --warnings-as-errors`
  to confirm compilation, then `mix format <your files>` only on your files.
  Do NOT start the phx server (ports collide with other agents); to sanity-check
  runtime behavior of your demo module, use `mix run -e '...'` one-liners.
- Compile lock is shared across agents. If mix appears to hang briefly, it's
  waiting on the lock; let it finish.
