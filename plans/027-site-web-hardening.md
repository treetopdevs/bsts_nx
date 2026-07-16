# Plan 027: Site web hardening (CSP, secure session cookie, force_ssl exclude)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat 9b7cb8d..HEAD -- site/lib/bsts_site_web/router.ex site/lib/bsts_site_web/endpoint.ex site/config/prod.exs site/assets/js/app.js`
> On mismatch with the excerpts below, STOP.

## Status

- **Priority**: P3
- **Effort**: M
- **Risk**: LOW-MED (an over-tight CSP breaks chart styling or the LiveView
  websocket; mitigated by explicit local verification steps)
- **Depends on**: none
- **Category**: security (defense-in-depth — no active exploit was found)
- **Planned at**: commit `9b7cb8d`, 2026-07-08

## Why this matters

The security audit found the site fundamentally sound (no XSS vector — the
one `raw/1` renders static Makeup output; params are clamped; LiveDashboard
is dev-gated). Three defense-in-depth gaps remain:

1. **No Content-Security-Policy.** `put_secure_browser_headers` runs with no
   custom headers, so the baseline set (x-frame-options,
   x-content-type-options, referrer-policy) is present but nothing constrains
   script/style/connect sources. A CSP contains any *future* XSS and is cheap
   for a site with a single self-hosted JS bundle and self-hosted fonts.
2. **Session cookie lacks `Secure`.** Effectively TLS-only today because Fly
   edge forces HTTPS, but the attribute should be on the cookie itself.
3. **`force_ssl` excludes `localhost`/`127.0.0.1` hosts** — a
   Host-header-controlled bypass of the HSTS/redirect layer. Also mitigated
   by the Fly edge, but it's a footgun if the edge config ever changes.

## Current state

- `site/lib/bsts_site_web/router.ex` (browser pipeline, ~line 10):

  ```elixir
  plug :protect_from_forgery
  plug :put_secure_browser_headers
  ```

- `site/lib/bsts_site_web/endpoint.ex` (~lines 7–12):

  ```elixir
  @session_options [
    store: :cookie,
    key: "_bsts_site_key",
    signing_salt: "ITO8FmJV",
    same_site: "Lax"
  ]
  ```

- `site/config/prod.exs` (~lines 13–21):

  ```elixir
  config :bsts_site, BstsSiteWeb.Endpoint,
    force_ssl: [
      rewrite_on: [:x_forwarded_proto],
      exclude: [
        # paths: ["/health"],
        hosts: ["localhost", "127.0.0.1"]
      ]
    ]
  ```

- CSP-relevant facts, verified: JS is one self-hosted bundle
  (`site/assets/js/app.js`, esbuilt to `/assets/`; no eval, live-reload hooks
  gated to development); fonts are self-hosted woff2 under
  `priv/static/fonts`; charts are inline SVG with **inline `style=`
  attributes** (e.g. `charts.ex:472 style={"max-width: ..."}`) so `style-src`
  needs `'unsafe-inline'`; LiveView needs a websocket to the same host
  (`connect-src` must allow `wss:`); images are local + inline SVG (`img-src
  'self' data:` to be safe).
- `fly.toml`: `force_https = true` at the edge (why 2 and 3 are currently
  mitigated).

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Site tests | `cd site && mix test` | 0 failures |
| Run locally | `cd site && mix phx.server` | serves http://localhost:4000 |
| Inspect headers | `curl -sI http://localhost:4000/ \| grep -i content-security` | shows the policy |
| Compile/format | `cd site && mix compile --warnings-as-errors && mix format --check-formatted` | exit 0 |

Tooling note: `mix` is a `mise` shim; prefix `mise exec -- ` if needed.

## Scope

**In scope**:
- `site/lib/bsts_site_web/router.ex`
- `site/lib/bsts_site_web/endpoint.ex`
- `site/config/prod.exs`
- `site/test/bsts_site_web/live/routes_smoke_test.exs` (one header assertion,
  if plan 023 landed; else a minimal new conn test)

**Out of scope**:
- Rewriting inline SVG styles to classes to enable a strict `style-src`
  (nice-to-have; large diff across charts for marginal gain — record as
  future work).
- CSP reporting endpoints/report-only rollout infrastructure — this site has
  no error-collection backend; the local verification below substitutes.
- `fly.toml`, `assets/js`.

## Git workflow

- Branch: `advisor/027-site-web-hardening` (from `execute-plans`).
- Commit style: `sec(site): CSP, secure session cookie, tighten force_ssl exclude`
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Add the CSP

In `router.ex`, pass the policy to the existing plug:

```elixir
plug :put_secure_browser_headers, %{
  "content-security-policy" =>
    "default-src 'self'; " <>
      "script-src 'self'; " <>
      "style-src 'self' 'unsafe-inline'; " <>
      "img-src 'self' data:; " <>
      "font-src 'self'; " <>
      "connect-src 'self' ws: wss:; " <>
      "object-src 'none'; " <>
      "frame-ancestors 'none'; " <>
      "base-uri 'self'"
}
```

`'unsafe-inline'` for styles is a deliberate, documented trade (inline SVG
chart styles); scripts stay strict.

**Verify**: `cd site && mix phx.server`, then browse `/`, one demo page, one
engine page, one trust page with the browser devtools console open →
**zero CSP violation messages**, charts styled correctly, LiveView connects
(interactions work — drag a slider). Also
`curl -sI http://localhost:4000/ | grep -i content-security` shows the policy.

### Step 2: Secure session cookie in prod

The `@session_options` module attribute is compile-time shared across envs;
make the secure flag config-driven:

```elixir
@session_options [
  store: :cookie,
  key: "_bsts_site_key",
  signing_salt: "ITO8FmJV",
  same_site: "Lax",
  secure: Application.compile_env(:bsts_site, :secure_session_cookie, false)
]
```

and in `site/config/prod.exs` add
`config :bsts_site, secure_session_cookie: true`.
(Dev stays `false` so http://localhost keeps its session/CSRF flow.)

**Verify**: `cd site && mix compile --warnings-as-errors` → exit 0;
`mix phx.server` + a page load still works (dev cookie unaffected).

### Step 3: Tighten the `force_ssl` exclude

In `site/config/prod.exs`, replace the hosts exclude with a path exclude that
matches how the deploy actually health-checks (fly.toml checks `GET /` over
the internal port — Fly's checks hit the app directly, not via the edge, so
they arrive as plain HTTP; excluding by path keeps them green without a
host-spoofable hole):

```elixir
force_ssl: [
  rewrite_on: [:x_forwarded_proto],
  exclude: [paths: ["/"]]
]
```

Wait — excluding `/` would exempt the landing page; that defeats the purpose.
Correct approach: Fly's HTTP health check *does* follow the app's redirect?
It does not need to: `rewrite_on: [:x_forwarded_proto]` makes requests that
arrived via the edge (which sets `x-forwarded-proto: https`) count as HTTPS
already, and Fly health checks tolerate 3xx only if configured. **Do this
instead**: keep the exclude list EMPTY (`exclude: []`), then verify the Fly
health check still passes after deploy — if the check fails on the 301, add a
dedicated health route (e.g. `get "/healthz"` returning 200 in the router)
and exclude `paths: ["/healthz"]`, updating `fly.toml`'s `[checks]` path in
the same commit. Choose based on evidence, not guesswork; both end states are
acceptable, `hosts: ["localhost", ...]` is not.

**Verify**: `cd site && mix compile --warnings-as-errors` → exit 0 (config is
compile-time for force_ssl). Record which end state you chose. The live
health-check behavior can only be confirmed at next deploy — flag it in your
report as the post-merge check.

### Step 4: Header regression test

Add to the smoke tests (plan 023's file, or a small new conn test): a `get`
on `/` asserts the `content-security-policy` response header is present and
contains `script-src 'self'`.

**Verify**: `cd site && mix test` → 0 failures, 1 new test.

## Test plan

Step 4's header assertion + plan 023's smoke suite (catches CSP-induced
render breakage indirectly via LiveView connect in tests? no — tests bypass
the browser, hence the mandatory manual browser pass in Step 1 with devtools
open; list the pages you checked in the report).

## Done criteria

- [ ] `curl -sI` on a running dev server shows the CSP header
- [ ] Browser pass over ≥4 pages: zero CSP violations, charts styled, sliders live
- [ ] Session cookie `secure: true` in prod config only
- [ ] `force_ssl` exclude no longer contains hosts; end state recorded
- [ ] `cd site && mix test` → 0 failures incl. the header test
- [ ] Compile + format clean; only in-scope files changed
- [ ] `plans/README.md` status row updated (note the post-deploy health-check verification)

## STOP conditions

Stop and report back (do not improvise) if:

- Any CSP violation appears in the browser pass that isn't solved by the
  documented `style-src 'unsafe-inline'` — report the directive + offending
  resource; do NOT loosen `script-src`.
- The LiveView websocket fails to connect under the policy (would mean
  `connect-src` needs a form this plan didn't anticipate).
- `put_secure_browser_headers` on this Phoenix version doesn't merge custom
  headers as documented — use a dedicated `put_resp_header` plug instead and
  say so.
- You cannot determine how the Fly health check interacts with force_ssl —
  pick the `/healthz` variant (it's deploy-safe by construction) and flag it.

## Maintenance notes

- Post-deploy check: Fly health checks green with the new force_ssl config;
  browse the prod site once with devtools open for CSP violations (prod
  serves digested assets — same origins, should be identical).
- Any future third-party embed (analytics, fonts CDN) must amend the CSP —
  that's the point of having it.
- The `style-src 'unsafe-inline'` exception disappears if chart inline styles
  ever migrate to classes/attributes — noted as optional future work.
