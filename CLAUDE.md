# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BstsNx is a pure-Elixir library for Bayesian Structural Time Series (BSTS) models built on Nx. It provides Kalman filtering/smoothing, Gibbs sampling for state-space models, causal impact estimation, forecasting, and domain-specific applications (marketing lift, TV attribution, anomaly detection, etc.).

## Build & Test Commands

```bash
mix deps.get          # Fetch dependencies
bash scripts/ci.sh    # Local CI-parity verifier: compile, non-external tests, format check, docs
mix compile           # Compile the library
mix test              # Run full test suite (excludes @tag :slow by default)
mix test path/to/test.exs           # Run a single test file
mix test path/to/test.exs:42        # Run a specific test at line
mix test --only external            # Run tests tagged @moduletag :external
mix test --include slow             # Include slow tests
mix format            # Format all code
mix format --check-formatted        # Check formatting without modifying
BSTS_NX_TEST_BACKEND=exla mix test test/structured_performance_smoke_test.exs test/utils_safe_solve_test.exs
BSTS_NX_TEST_BACKEND=emlx mix test test/structured_performance_smoke_test.exs test/utils_safe_solve_test.exs
BSTS_NX_ENABLE_R_PARITY=1 mix test test/r_parity_test.exs
mix bench.structured_backends       # Compare structured sampler backend behavior
```

`bash scripts/ci.sh` mirrors the default GitHub Actions `test` and `quality`
jobs. EXLA, EMLX, slow, external, and R parity checks are intentionally separate
optional lanes.

## Architecture

### Core Statistical Engine (bottom-up dependency order)

1. **`BstsNx.Utils`** — Tensor conversion (`to_tensor/1`), safe Cholesky with jitter, percentile intervals, z-scores
2. **`BstsNx.Distributions`** — Inverse-gamma sampler (Marsaglia-Tsang gamma + inversion), normal/multivariate-normal samplers using Nx.Random keys
3. **`BstsNx.KalmanFilter`** — Two implementations:
   - `filter/7` / `filter_with_pred/7` — Eager Elixir, supports multi-dimensional state, time-varying H, missing observations (`nil`)
   - `filter_defn/7` — `Nx.Defn` compiled, scalar-only, missing data via NaN sentinel. JIT-compilable with EXLA.
4. **`BstsNx.Smoother`** — RTS smoother and Carter-Kohn simulation smoother. Also has compiled `rts_defn/4` and `rts_defn_with_lag1/4` for scalar systems.
5. **`BstsNx.GibbsSampler`** — Two sampler families:
   - **Scalar**: `sample/7`, `sample_general/5`, `sample_chains/8` — single Q and R
   - **Structured**: `sample_structured/4`, `sample_structured_chains/5` — multi-dimensional state with per-diagonal-Q-entry resampling via `ModelSpec`
6. **`BstsNx.CausalImpact`** — Three estimators: `estimate/4` (scalar MCMC), `estimate_structured/5` (structured MCMC), `estimate_from_filter/3` (non-MCMC, compiled filter)

### Model Building Layer

- **`BstsNx.ModelSpec`** — Struct defining a complete state-space model (F, H, Q specs, x0, p0, priors). Central to the structured sampler.
- **`BstsNx.Components`** — Factory functions for model components (`local_level`, `local_linear_trend`, `seasonal`, `regression`) and their `*_spec` variants for `ModelSpec`. `compose_specs/2` combines specs with block-diagonal matrices.
- **`BstsNx.StateSpace`** — `block_diag/1` and `compose/2` for building composite state-space matrices.
- **`BstsNx.ModelBuilder`** — Shared model-building utilities: resolves spec from options (seasonality, regressors, explicit spec), coercion, formatting.

### High-Level APIs

- **`BstsNx.InterventionAnalysis`** — Domain-agnostic causal inference with friendly config map interface. Wraps CausalImpact.
- **`BstsNx.Forecaster`** — `fit/2` + `predict/2` or `fit_predict/3` with credible intervals and component decomposition.
- **`BstsNx.Pipeline`** — End-to-end CausalImpact → SpotAttributor pipeline for TV attribution.

### Domain Applications (`BstsNx.Applications.*`)

- `MarketingLift` — Digital marketing incrementality with control series
- `TVAttribution` — TV spot-to-web attribution with Shapley allocation
- `DemandForecaster` — Demand forecasting with causal drivers
- `AnomalyDetector` — Anomaly detection with calibrated uncertainty
- `PolicyEvaluator` — Interrupted time series for policy evaluation

### Supporting Modules

- **`BstsNx.SpotAttributor`** — Per-spot lift attribution with Shapley values for overlapping windows
- **`BstsNx.Shapley`** — Monte Carlo Shapley value computation with time-decay
- **`BstsNx.Diagnostics`** — MCMC convergence: R-hat, effective sample size
- **`BstsNx.CovariateSelection`** — Spike-and-slab variable selection
- **`BstsNx.RollingBaseline`** — Rolling-window baseline estimation
- **`BstsNx.Validation`** — Input validation for time series and periods
- **`BstsNx.Synthetic.*`** — Data generation for testing (Generator, Scenarios, Adstock)

## Key Patterns

- **PRNG threading**: All stochastic functions accept `:key` (Nx.Random key) or `:seed` options. Keys are split via `Nx.Random.split/2` and threaded through. The `Distributions` module bridges Nx keys to Erlang `:rand` state via `derive_exsss_seed/1`.
- **Missing data**: `nil` in eager code, `NaN` in defn code. The Kalman filter skips the update step; the Gibbs sampler excludes missing observations from variance updates.
- **Dual implementations**: Core algorithms have both eager (list-based, multi-dim) and compiled (`Nx.Defn`, scalar-only) versions. The `_defn` suffix indicates JIT-compilable functions.
- **`ModelSpec` composition**: Complex models are built by composing simple specs via `Components.compose_specs/2`, which block-diagonalizes F/Q, concatenates H, and shifts q_specs indices.
- **Parallel chains**: Both `sample_chains` and `sample_structured_chains` use `Task.async_stream` for parallel MCMC chains with per-chain PRNG seeds.

## Dependencies

- Elixir `~> 1.19`
- `nx ~> 0.12.0` — Core numerical computing
- `emlx ~> 0.3.0` (optional) — MLX-backed CPU/GPU execution
- `exla ~> 0.12.0` (optional) — JIT compilation and hardware acceleration
- `stream_data ~> 1.0` (test/dev) — Property-based testing
- `ex_doc ~> 0.34` (dev) — Documentation generation

Structured EMLX GPU runs are currently limited by missing linalg primitive
coverage. Treat EMLX CPU, EXLA, and BinaryBackend as the practical fallback
paths until those GPU primitives are available.

## AI Tooling Reference (updated 2026-07-01)

Current Claude model lineup (all active):

| Model | ID | Context / Max out | Price (in/out per MTok) | Best for |
|---|---|---|---|---|
| Claude Fable 5 | `claude-fable-5` | 1M / 128K | $10 / $50 | Hardest long-horizon & agentic work (most capable) |
| Claude Opus 4.8 | `claude-opus-4-8` | 1M / 128K | $5 / $25 | Default high-capability, autonomous work |
| Claude Sonnet 5 | `claude-sonnet-5` | 1M / 128K | $3 / $15 ($2/$10 intro thru 2026-08-31) | Most coding/agentic work — near-Opus at lower cost |
| Claude Haiku 4.5 | `claude-haiku-4-5` | 200K / 64K | $1 / $5 | Fast, cheap, simple tasks |

Thinking/effort: Fable 5 / Sonnet 5 / Opus 4.8 use adaptive thinking (`thinking: {type: "adaptive"}`); set depth with `output_config.effort` (`low`…`high`/`xhigh`/`max` — `xhigh` is the sweet spot for coding/agentic). `budget_tokens`, `temperature`, `top_p`, and `top_k` are rejected (400) on these models. Sonnet 5 uses a new tokenizer (~30% more tokens than Sonnet 4.6) — re-baseline token budgets.

CLI coding tools available for delegation / cross-checking:
- **Claude Code** — primary agentic CLI (this). Fast Mode via `/fast` (Opus 4.8/4.7).
- **Codex** (`/codex:*`, `codex` runtime) — GPT-based; rescue passes, second opinions, deep root-cause investigation.
- **Antigravity / `agy`** (`/agy:*`, `agy` CLI) — Google Gemini-backed, recently re-added; delegate coding/research (`/agy:delegate`, `/agy:research`), review a diff (`/agy:review`), or generate images (`/agy:image`, Imagen).
