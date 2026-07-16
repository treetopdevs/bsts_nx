# Deploying the showcase site to fly.io

The Docker build context is the **repo root** (the site depends on the library
via `{:bsts_nx, path: ".."}`), so `Dockerfile`, `fly.toml`, and `.dockerignore`
live at the repo root. All commands below run from the repo root.

## First deploy

```bash
# 1. Create the app (skip the launch wizard's generated files — we have our own)
fly apps create bsts-nx        # or pick another name; update app + PHX_HOST in fly.toml

# 2. Secrets
fly secrets set SECRET_KEY_BASE=$(cd site && mix phx.gen.secret)

# 3. Deploy (remote builder; no local Docker needed)
fly deploy --remote-only
```

## What's already configured

- `fly.toml`: single `shared-cpu-2x` / 2 GB machine, `min_machines_running = 1`
  (the first impression should be instant — demos compute in-process),
  HTTP health check on `/`, PORT 8080.
- `Dockerfile`: two-stage build on `hexpm/elixir` 1.19.5 / OTP 28.3.3
  (trixie-slim), mirrors the repo layout (`/app` = library, `/app/site` = app),
  builds assets (tailwind + esbuild download during the build) and a release.
  Runtime is pure BinaryBackend — no NIFs, no XLA, nothing GPU.
- No database. Nothing is persisted; every demo is computed per-request from
  seeded scenarios. (If a DB is ever needed, the plan of record is SQLite.)
- The prod release was smoke-tested locally:
  `MIX_ENV=prod mix assets.deploy && mix release` then `bin/bsts_site start`
  serves all routes in ~15 ms.

## Post-deploy checklist

- Load `/speed` and press "Race the lanes" — the page measures the actual
  machine, so published latency talk stays honest on Fly hardware.
- MCMC demos (marketing, policy, counterfactual, demand, calibration,
  diagnostics, gibbs) run behind a 3-slot semaphore (`BstsSite.Demos.Limiter`);
  under load visitors get a polite busy note rather than a pegged machine.
  If the machine feels roomy, bump `@max_concurrent`.
- Two config switches in `site/config/config.exs` when the time comes:
  - `:github_url` — the public source URL, kept aligned with the package
    metadata in the root `mix.exs` (`treetopdevs/bsts_nx`).
  - `:hex_published` — flip to true after `mix hex.publish`; `/start` swaps
    the git-dependency install snippet for the Hex one automatically.
