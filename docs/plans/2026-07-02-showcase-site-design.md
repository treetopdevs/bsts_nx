# BstsNx Showcase Site — Design

**Date:** 2026-07-02 · **Status:** approved for build · **Target:** fly.io · **Stack:** Phoenix 1.8 LiveView (in `site/`), path dep on the library, no database (nothing to persist; SQLite reserved if that changes)

## The story

The site is not a feature list. It is one question, asked well, and answered honestly:

> **"The numbers changed. Was it you — or was it noise?"**

Everything hangs off the library's own mantra (from the guide livebook epilogue):
**decompose → project → compare**, and its closing line: *"Every section used the same
underlying engine. The only thing that changed was the question."*

### The recurring device: the planted truth

Every demo on the site runs on synthetic data **where we know the ground truth** —
because we generated it (`BstsNx.Synthetic.*`). Each demo ends with a **"Reveal the
truth"** moment: overlay what the model estimated against what we actually planted.
The site never asks for trust; it demonstrates recovery. This is the library's own
validation loop (`Validation.known_lift_injection`) turned into public theater.

### The proof device: live execution badges

Every demo shows the library's self-reported `execution` metadata as a badge:
*"method: `:scalar_forecast_filter` · 6 ms · computed live on this server."*
Nothing is a video or a canned screenshot. The "how well it works" claim is made by
the server the visitor is talking to, at the moment they ask.

### Non-linear flow: a hub and three doors

Visitors arrive with different intents. The hub routes by intent; every path
cross-links into the others, so any entry point eventually walks the whole graph.

```
                        ┌──────────────────────────────┐
                        │  HUB: "Was it you, or noise?" │
                        │  (live hero demo, 3 doors)    │
                        └──────┬───────┬───────┬───────┘
              "I have a question"  "How does it work?"  "Should I trust it?"
                        │              │                │
                 QUESTIONS track   ENGINE track      TRUST track
                 (5 domain demos)  (6 chapters)      (3 pages)
                        │              │                │
                        └──── every demo links to the engine chapter
                              that powers it, and every engine chapter
                              links to the trust page that stress-tests it
```

## Page map

### Hub — `/`
Hero: an instant interactive causal-impact demo (Operational lane, ~6 ms warm).
Sliders for effect size and noise; the counterfactual band and lift shading re-render
on every drag. When the planted effect is small relative to noise, the interval
honestly straddles zero — the hero teaches the core lesson before a word is read.
Below: the three doors, the mantra, and a real terminal-style quick-start snippet.

### Questions track — `/demos/*` (one question per page)
| Route | Question | Module | Latency tier |
|---|---|---|---|
| `/demos/marketing` | "Did the campaign actually work?" | `Applications.MarketingLift` | button (scalar MCMC, capped) + operational instant mode |
| `/demos/tv` | "Which TV spots earned their airtime?" | `Applications.TVAttribution` + `SpotAttributor` | instant (Operational + Shapley); draggable spot windows re-split credit live |
| `/demos/demand` | "How much should we stock?" | `Applications.DemandForecaster` | posterior fitted at build time; service-level slider replays via `Forward` instantly |
| `/demos/anomaly` | "Is this spike an anomaly?" | `Applications.AnomalyDetector` | instant (3 ms fit); visitor clicks on the chart to inject anomalies |
| `/demos/policy` | "Did the policy change anything?" | `Applications.PolicyEvaluator` | button (scalar MCMC, capped) |

Each demo page: scenario intro (2 sentences) → interactive chart → verdict card
(plain-language answer with CI) → **Reveal the truth** → execution badge →
"Under the hood" collapsible with the real, copy-pasteable Elixir that just ran →
cross-links ("this ran the same engine as → Kalman chapter").

### Engine track — `/engine/*` (the tutorial, adapted from the livebooks)
| Route | Chapter | Interactive |
|---|---|---|
| `/engine/noise` | Why raw data lies | "Spot the real effect" quiz: N random-walk charts, one has a planted lift; visitor guesses, model answers |
| `/engine/kalman` | Tracking a signal through noise | Q/R sliders: watch the filter shift from trusting-the-model to trusting-the-data |
| `/engine/smoother` | The power of hindsight | filtered↔smoothed toggle + "draw another possible history" (FFBS) button |
| `/engine/gibbs` | Honest about the unknown | **MCMC theater**: 3 chains stream live over LiveView, trace plots grow, R-hat badge flips red→green as they converge |
| `/engine/compose` | Models as Lego bricks | Component chips (trend / seasonal / regression) live-render the block-diagonal F matrix + state dimension |
| `/engine/counterfactual` | The counterfactual | Spaghetti plot of posterior counterfactual draws (precomputed structured fit; interval-width slider replays instantly) |

### Trust track — `/trust/*`
| Route | Page | Content |
|---|---|---|
| `/trust/calibration` | The planted truth, systematically | Run recovery live across effect sizes; coverage table; the site grading its own homework |
| `/trust/diagnostics` | Break your own results | The five checks (prediction error, coverage, Durbin-Watson, placebo, stability) as a green/amber/red dashboard with explanations |
| `/trust/honesty` | What this is and isn't | v0.1 early-stage status; R bsts/CausalImpact lineage as reference implementation; what's not built yet; LGPL-2.1; indexing conventions; "measured on this server" methodology |

### Utility
- `/speed` — the two-lanes page: milliseconds lane (forecast-first filter) vs MCMC lane
  (full posterior), when to use which, plus **live micro-benchmarks measured on the
  serving machine**. Also the honest footnote: `estimate_from_filter` interpolates
  (sees post-period data); `Operational` forecast-first never peeks.
- `/start` — install (git dep until Hex publish; caveat rendered from a single config
  flag so flipping to Hex later is one change), quick start, decision table
  ("which question → which function", from the guide epilogue), links to the two
  livebooks as the deep-dive companions.

## Latency tiers (measured, BinaryBackend)

| Tier | Budget | Pattern | Examples |
|---|---|---|---|
| Instant | <50 ms | run in `handle_event`, every interaction | Operational (6 ms), filter CI (12 ms), Kalman (10 ms), anomaly (3 ms), Shapley re-split |
| Short | 0.5–2.5 s | async `Task` + spinner, button-triggered | scalar `CausalImpact.estimate` / `GibbsSampler.sample` at ≤100 samples |
| Streamed | seconds–minutes | LiveView streams iterations as they compute | MCMC theater (small chains); structured fits |
| Baked | build time | mix task at Docker build → `priv/precomputed/*.etf` → `persistent_term`; visitors explore via `Forward` replay | demand posterior, counterfactual spaghetti, diagnostics dashboard |

**Abuse guards:** hard server-side caps (series ≤ 400 pts, samples ≤ 100 scalar / ≤ 30
streamed), a global counting semaphore around Short/Streamed tiers (429-style "busy,
try again" message beyond N concurrent), all inputs from bounded sliders/selects —
no free-form numeric input reaches the samplers.

## Architecture

```
site/
  lib/bsts_site/
    demos/           # one module per demo: pure functions, fixed seeds,
                     # returns %{series, result, execution, truth, code_snippet}
    demos/scenarios.ex   # shared synthetic scenario builders (seeded)
    demos/limiter.ex     # semaphore for heavy tiers
    precompute.ex        # mix bsts_site.precompute → priv/precomputed/*.etf
  lib/bsts_site_web/
    components/charts.ex # server-rendered SVG: line+band, spaghetti, bars,
                         # gantt, trace plot, F-matrix grid, fan chart
    components/story.ex  # verdict card, execution badge, reveal-truth,
                         # under-the-hood, mantra bar, cross-link cards
    live/                # one LiveView per page
```

- **Charts are server-rendered SVG** (HEEx function components). No JS chart lib, no
  CDN; LiveView diffs keep drag interactions cheap; fully on-brand for "pure Elixir."
- **No Ecto/SQLite**: nothing is persisted. Revisit only if we add e.g. shared
  scenario permalinks.
- **No EXLA in the site**: default backend stays BinaryBackend (measured fine for
  every live tier; EXLA-as-default-backend slows the eager paths).
- Code snippets shown on pages are the *actual* code the demo module runs, extracted
  from one source of truth so they can't drift.

## Deployment (fly.io)

- Dockerfile at repo root (build context must include both `site/` and the library);
  standard hexpm/elixir → debian-slim release build.
- Precomputed artifacts: once `mix bsts_site.precompute` is added to the Docker
  build, its ETF outputs will ship in the image.
  Seeds fixed → reproducible builds.
- fly.toml: single small machine to start (shared-cpu-2x / 2 GB suggested),
  `min_machines_running = 1` (persistent_term warm), health check on `/`.
- Re-measure the published latency numbers on the fly machine post-deploy; the
  `/speed` page measures live anyway.

## Copy & accuracy guardrails (from research)

1. Install: git dependency until Hex publish; never render the bare `{:bsts_nx, "~> 0.1"}` as working today.
2. License: LGPL-2.1-only, stated plainly. Not "MIT", not "permissive".
3. Repo link: use the confirmed public URL `https://github.com/treetopdevs/bsts_nx` consistently in package metadata and site links.
4. Maturity: "early-stage, validated against the R reference stack" — never "faster than R", never "feature parity with Google CausalImpact".
5. Module names: `BstsNx.ShapleyAllocator` (there is no `BstsNx.Shapley`).
6. Spike-and-slab exists (`regression_spec(mode: :spike_and_slab)`); the stale claim in `docs/bsts_nx_bayesian_review.md` must not leak into copy.
7. Conventions: pre/post periods 1-based inclusive; spot windows 0-based half-open. Show them exactly as the top-level `BstsNx` moduledoc states them.
8. Performance numbers on pages come from live measurement or say "measured on this server"; no hardcoded dev-machine numbers.
9. `estimate_from_filter` = fast interpolation; `Operational` = honest forecast-first counterfactual. Never conflate.
10. Single-sample MCMC summaries return NaN — sliders never allow `num_samples < 2`.

## Visual direction (to be developed in the polish phase)

Flow first — but the standing intent: editorial/technical-report aesthetic (the site
is an argument, not a SaaS landing page), restrained palette with one accent reserved
exclusively for "lift/effect" across every chart, monospace for numbers and
execution badges, generous prose measure for the tutorial track. Distinctive, not
templated. Final direction set with the frontend-design skill before foundation UI.
