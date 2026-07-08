# Plan 021: Run the showcase site's numerics on EXLA (JIT backend for prod, with full accuracy gates)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat 9b7cb8d..HEAD -- site/mix.exs site/config/runtime.exs site/lib/bsts_site/application.ex Dockerfile test/test_helper.exs`
> If any changed since this plan was written, compare the "Current state"
> excerpts against the live code before proceeding; on a mismatch, treat it
> as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED (backend swap on the deployed numerics path — mitigated by the full library suite as an accuracy gate)
- **Depends on**: none hard; **soft: plan 019** (its `DefaultCache` is the natural JIT-warmup hook — Step 5 has a fallback if 019 hasn't landed)
- **Category**: perf
- **Planned at**: commit `9b7cb8d`, 2026-07-08

## Why this matters

The showcase site runs all demos in-process on Nx's default **BinaryBackend**
(pure-Elixir tensor ops — the slowest option). Measured consequences, per the
site's own design docs: the MCMC demo tier takes hundreds of ms to seconds per
fit behind a 3-slot limiter, and structured/seasonal MCMC is **banned from the
site entirely** (~0.5–2.6 s *per iteration* on BinaryBackend). Meanwhile the
library ships optional EXLA support (`{:exla, "~> 0.12", optional: true}`),
maintains an EXLA CI lane, and its hot paths are `Nx.Defn` functions
(`filter_defn`, `rts_defn`, forecast moments) that EXLA JIT-compiles to native
code — typically orders of magnitude faster than the defn evaluator on
BinaryBackend.

Switching the **site** (not the library — it stays backend-agnostic) to EXLA
on its Linux/Fly deployment is the single biggest performance lever available:
it speeds every Limiter-gated MCMC demo, widens the DoS headroom (same
finding SEC-01), and potentially unlocks the currently-forbidden structured
demo tier. The accuracy contract: **the full library test suite must pass
under EXLA before the site adopts it** — that is this plan's hard gate.

## Current state

- `site/mix.exs` deps — no `:exla` entry; the site gets the library via
  `{:bsts_nx, path: ".."}` and inherits plain Nx on BinaryBackend.
- The canonical backend-switching mechanics, `test/test_helper.exs:13-18`
  (root project):

  ```elixir
  unless Code.ensure_loaded?(EXLA.Backend) do
    raise "BSTS_NX_TEST_BACKEND=exla requested, but EXLA.Backend is unavailable"
  end

  Application.put_env(:nx, :default_backend, EXLA.Backend)
  Nx.global_default_backend(EXLA.Backend)
  ```

  (Note it sets the *backend* only; for JIT of `defn` code you also set the
  defn compiler — Step 2.)
- Root `mix.exs`: `{:exla, "~> 0.12.0", optional: true}`, `{:xla, "~> 0.10.0",
  optional: true}`. The `xla` package downloads a precompiled XLA binary at
  compile time (needs network + `ca-certificates`, both present in the Docker
  builder stage).
- `Dockerfile`: builder installs `build-essential git ca-certificates`; runner
  installs `libstdc++6 openssl libncurses6 locales ca-certificates`. EXLA's
  NIF additionally wants OpenMP (`libgomp1`) at runtime on Debian.
- `fly.toml`: `shared-cpu-2x`, `memory = "2gb"`.
- `site/config/runtime.exs`: standard Phoenix release config
  (SECRET_KEY_BASE/PHX_HOST/PORT); no Nx config anywhere in `site/config/`.
- `site/lib/bsts_site/application.ex`: standard supervision tree (Telemetry,
  PubSub, Limiter, Endpoint...).
- CLAUDE.md context: "Structured EMLX GPU runs are currently limited by
  missing linalg primitive coverage. Treat EMLX CPU, EXLA, and BinaryBackend
  as the practical fallback paths" — EXLA CPU is the sanctioned fast path.
- Benchmark harness: `mix bench.structured_backends`
  (`bench/structured_backend_benchmark.exs`) compares backends on the
  structured sampler.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| **Accuracy gate** (library under EXLA) | `BSTS_NX_TEST_BACKEND=exla mix test --exclude external` (repo root; first run compiles EXLA — slow) | 0 failures |
| EXLA smoke lane (as CI runs it) | `BSTS_NX_TEST_BACKEND=exla mix test test/structured_performance_smoke_test.exs test/utils_safe_solve_test.exs` | 0 failures |
| Site deps | `cd site && mix deps.get && mix compile` | exit 0 |
| Site tests (stay on BinaryBackend) | `cd site && mix test` | 0 failures |
| Backend benchmark | `mix bench.structured_backends` | timing table incl. EXLA column |
| Site console for measurements | `cd site && BSTS_SITE_NX_BACKEND=exla iex -S mix` | boots; `Nx.default_backend()` → `{EXLA.Backend, ...}` |

Tooling note: `mix` is a `mise` shim; prefix `mise exec -- ` if needed.

## Scope

**In scope** (the only files you should modify):
- `site/mix.exs` (+ resulting `site/mix.lock`)
- `site/config/runtime.exs`
- `site/lib/bsts_site/application.ex` (warmup only)
- `Dockerfile` (runner apt line)
- `site/DEPLOY.md` (document the knob)

**Out of scope** (do NOT touch):
- Root `mix.exs` / the library — it stays backend-agnostic; do NOT make exla
  non-optional or set a default backend in the library.
- `site/config/test.exs` — site tests stay on BinaryBackend (deterministic,
  no EXLA compile cost in CI). Note this in DEPLOY.md.
- Demo iteration counts / Limiter `@max_concurrent` — retuning budgets after
  the speedup is a follow-up decision (see Maintenance notes), not this plan.
- `fly.toml` — no resizing.
- Any relaxation of test tolerances anywhere — if a test fails under EXLA,
  that is a STOP, not a tolerance bump.

## Git workflow

- Branch: `advisor/021-site-exla-backend` (from `execute-plans`).
- Commit style: `perf(site): run demos on EXLA in prod with accuracy gates`
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Run the accuracy gate BEFORE any site change

From the repo root:
`BSTS_NX_TEST_BACKEND=exla mix test --exclude external`

This compiles EXLA locally (minutes on first run; the `xla` package downloads
a precompiled binary for your platform) and runs the **entire** non-external
library suite on the EXLA backend — far broader than CI's two smoke files.

**Verify**: 0 failures. Record the test count and wall time in your report.
Any failure is a STOP (see STOP conditions) — it means EXLA and BinaryBackend
disagree somewhere, which is exactly what this gate exists to catch.

### Step 2: Add EXLA to the site and gate it by env

1. `site/mix.exs` deps: add `{:exla, "~> 0.12"}` (not optional — the site
   explicitly opts in). Run `cd site && mix deps.get`.
2. `site/config/runtime.exs` — add near the top (before the endpoint block),
   so it applies to releases and `mix phx.server` alike:

   ```elixir
   # Numerics backend. EXLA (JIT) in prod; BinaryBackend elsewhere unless
   # explicitly overridden. Library code stays backend-agnostic — this is
   # the site's deployment choice.
   nx_backend =
     System.get_env("BSTS_SITE_NX_BACKEND") ||
       if(config_env() == :prod, do: "exla", else: "binary")

   case nx_backend do
     "exla" ->
       config :nx, default_backend: EXLA.Backend
       config :nx, default_defn_options: [compiler: EXLA]

     "binary" ->
       :ok

     other ->
       raise "unsupported BSTS_SITE_NX_BACKEND=#{inspect(other)} (expected exla or binary)"
   end
   ```

   Both keys matter: `default_backend` moves eager tensor ops to EXLA
   buffers; `default_defn_options: [compiler: EXLA]` makes every `defn`
   (the Kalman/smoother hot paths) JIT-compile.

**Verify**: `cd site && BSTS_SITE_NX_BACKEND=exla iex -S mix` boots and
`Nx.default_backend()` returns `{EXLA.Backend, _}`;
`Nx.Defn.default_options()[:compiler]` returns `EXLA`. Then plain
`iex -S mix` (no env) still reports `Nx.BinaryBackend`.

### Step 3: Measure the win

In the EXLA console from Step 2, time representative demos cold and warm
(first call pays JIT compilation), e.g.:

```elixir
:timer.tc(fn -> BstsSite.Demos.Speed.race() end)          # run twice: cold, warm
:timer.tc(fn -> BstsSite.Demos.Counterfactual.fit(8.0) end)  # check the module for the real arity/args first
```

(Open each demo module to get the exact function/args — they are plain public
functions.) Repeat the same calls in a plain BinaryBackend console.

**Verify**: record a small table (demo × backend × cold/warm ms) in your
report. Expect the warm MCMC-lane numbers to improve ≥5×. If warm improvement
is <2×, STOP (the compiler is probably not engaged — do not ship the change
on vibes).

### Step 4: Keep site tests green and deterministic

`cd site && mix test` (no env var — BinaryBackend).

**Verify**: 0 failures, and confirm `site/config/test.exs` needed no changes
(the runtime.exs gate defaults non-prod to binary).

### Step 5: Warm the JIT at boot

First-visitor latency must not pay JIT compilation. In
`site/lib/bsts_site/application.ex`, after the supervision tree starts (end
of `start/2`, before returning), fire a non-blocking warmup:

```elixir
if Application.get_env(:nx, :default_defn_options)[:compiler] do
  Task.start(fn -> BstsSite.Demos.Warmup.run() end)
end
```

Create the tiny warmup (in the same file or `site/lib/bsts_site/demos/warmup.ex`):
if plan 019 has landed, call the `DefaultCache`-wrapped mount computations for
the hub and speed pages (they populate the cache AND compile the kernels); if
019 has NOT landed, call the underlying demo functions directly and discard
the results. Rescue-and-log any error — warmup must never crash the app.

**Verify**: `BSTS_SITE_NX_BACKEND=exla mix phx.server` boots; the log shows no
warmup crash; loading `/` immediately after boot renders fast (subjective —
note it).

### Step 6: Docker runtime dependency

In the `Dockerfile` runner stage, extend the apt install line with `libgomp1`
(EXLA's NIF links OpenMP; the builder already has what it needs and the XLA
download happens during `mix deps.compile` with network available).

**Verify**: if Docker is available locally,
`docker build -t bsts-site-exla .` from the repo root completes and
`docker run --rm -e SECRET_KEY_BASE=$(openssl rand -hex 32) -e PHX_HOST=localhost -e PORT=8080 -p 8080:8080 bsts-site-exla`
serves `/`. If Docker is NOT available, state so in your report — the deploy
smoke in `site/DEPLOY.md` becomes the verification point.

### Step 7: Document the knob

Add a short section to `site/DEPLOY.md`: prod runs EXLA by default;
`BSTS_SITE_NX_BACKEND=binary` (Fly secret/env) is the instant rollback to the
old behavior; site tests intentionally stay on BinaryBackend; first boot pays
a one-time JIT warmup in the background.

**Verify**: `grep -n "BSTS_SITE_NX_BACKEND" site/DEPLOY.md` → ≥1 match.

## Test plan

No new ExUnit tests in the site (behavior is identical by contract; the gate
is the library suite). The accuracy evidence trail your report must contain:

- Step 1: full library suite green under EXLA (count + time).
- Step 3: before/after timing table, cold and warm.
- Step 4: site suite green on BinaryBackend.
- `mix bench.structured_backends` output (optional but nice — quantifies the
  structured-tier headroom for the Maintenance follow-up).

## Done criteria

- [ ] `BSTS_NX_TEST_BACKEND=exla mix test --exclude external` → 0 failures (root)
- [ ] `grep -n "default_defn_options" site/config/runtime.exs` → 1 match
- [ ] `grep -n ":exla" site/mix.exs` → 1 match; `site/mix.lock` contains `exla` + `xla`
- [ ] `cd site && mix test` → 0 failures (BinaryBackend)
- [ ] Warm speedup ≥2× on at least one MCMC-lane demo, recorded in the report
- [ ] `Dockerfile` runner installs `libgomp1`
- [ ] `site/DEPLOY.md` documents `BSTS_SITE_NX_BACKEND`
- [ ] `git status --porcelain` shows only in-scope files
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- **Any library test fails under EXLA** (Step 1). Report the failing tests
  verbatim — this is a real accuracy finding about backend divergence and must
  be triaged by a human. Never fix by widening tolerances or skipping tests.
- The `xla` precompiled binary is unavailable for your platform or the
  download fails.
- Warm speedup <2× (Step 3) — the config isn't doing what we think.
- Boot-time RSS with EXLA warm exceeds ~1.2 GB locally (the Fly VM has 2 GB
  total; leave headroom for 200 connections) — report the number.
- `runtime.exs` config of `:nx` appears to take no effect in a release (e.g.
  Nx already started before config applied) — report rather than moving the
  config somewhere exotic.

## Maintenance notes

- **Follow-up decision for the maintainer** (do not do it in this plan): with
  EXLA the demo budgets were tuned for BinaryBackend — MCMC sample counts,
  the Limiter's `@max_concurrent 3`, and the DESIGN_CONTRACT's "structured
  MCMC is forbidden live" rule can all be revisited. `mix
  bench.structured_backends` gives the data. A structured-MCMC live demo is a
  genuinely new capability if the numbers allow it.
- Rollback is one env var (`BSTS_SITE_NX_BACKEND=binary`) — no redeploy of
  code needed if EXLA misbehaves in prod.
- Image size grows (~150–300 MB for XLA). If Fly image limits ever bite,
  that's the cost center.
- Plan 032 (dependency refresh) aligns nx locks; EXLA pins should ride the
  same refresh cadence.
