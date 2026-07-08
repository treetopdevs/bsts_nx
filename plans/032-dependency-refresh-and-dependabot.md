# Plan 032: Dependency refresh (nx lock alignment, Phoenix/LiveView patches, dev tooling) + dependabot coverage

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat 9b7cb8d..HEAD -- mix.exs mix.lock site/mix.exs site/mix.lock`
> Plan 021 adds `exla` to the site — expected drift; refresh whatever is
> live. Re-run `mix hex.outdated` in both projects rather than trusting the
> versions below.

## Status

- **Priority**: P3
- **Effort**: S
- **Risk**: LOW (patch/minor bumps + one new config file)
- **Depends on**: none (if plan 021 is in flight, land this after it to avoid
  lockfile collisions)
- **Category**: migration + dx
- **Planned at**: commit `9b7cb8d`, 2026-07-08

## Why this matters

No major-version lag exists anywhere (audited 2026-07-08), but three drifts
accumulated because nothing watches for them:

1. **nx lock skew**: the library locks `nx 0.12.0`, the site locks
   `nx 0.12.1` — the site *ships* library code against an nx patch the
   library's test suite never runs on.
2. **Public-facing patch lag**: the site is one patch behind on both
   `phoenix` (1.8.8→1.8.9) and `phoenix_live_view` (1.2.5→1.2.6);
   framework patches routinely carry fixes and there's no reason for the
   internet-facing app to lag.
3. **No update automation**: no dependabot/renovate config exists for either
   mix project, the Dockerfile base image (date-pinned
   `trixie-20260623`), or GitHub Actions (which plan 016 pins to SHAs —
   pins that then need something to bump them).

Also folded in: dev-only minors (`ex_doc` 0.40.1→0.40.3, `stream_data`
1.2.0→1.3.0) and an investigate-then-decide on the `emlx ~> 0.3.0` pin that
blocks emlx 0.4.0 (optional Apple-Silicon backend; hex reports "Update not
possible" under the current requirement).

## Current state

- `mix.lock`: `nx 0.12.0`; `site/mix.lock`: `nx 0.12.1` (both read directly
  at planning). Root requirement `{:nx, "~> 0.12.0"}` permits 0.12.1;
  `exla 0.12.0` / `xla 0.10.0` constraints permit it too (verified by
  `hex.outdated` during the audit).
- `site/mix.exs`: `{:phoenix, "~> 1.8.8"}`, `{:phoenix_live_view, "~> 1.2.0"}`
  — both requirements already allow the available patches.
- Root `mix.exs:42`: `{:emlx, "~> 0.3.0", optional: true}` — blocks 0.4.0 by
  requirement, not by ecosystem constraint (unverified whether emlx 0.4.0
  still targets nx ~> 0.12 — that's the Step 4 check).
- `.github/` contains only `workflows/ci.yml` — no `dependabot.yml`.
- Accuracy backstop available: the full library suite, plus (if plan 020
  landed) `mix test --only slow` for the statistical lane.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Outdated (root) | `mix hex.outdated` | table of current vs latest |
| Outdated (site) | `cd site && mix hex.outdated` | table |
| Update root | `mix deps.update nx ex_doc stream_data` | lock updated |
| Update site | `cd site && mix deps.update phoenix phoenix_live_view nx` | lock updated |
| emlx metadata | `mix hex.info emlx 0.4.0` | shows its nx requirement |
| Library verify | `bash scripts/ci.sh` | exit 0 |
| Statistical lane (if plan 020 landed) | `mix test --only slow` | 0 failures |
| Site verify | `cd site && mix test && mix compile --warnings-as-errors` | 0 failures |
| YAML check | `ruby -ryaml -e 'YAML.load_file(".github/dependabot.yml"); puts "ok"'` | `ok` |

Tooling note: `mix` is a `mise` shim; prefix `mise exec -- ` if needed.

## Scope

**In scope**:
- `mix.lock`, `site/mix.lock` (via `mix deps.update` only — never hand-edit)
- `mix.exs` ONLY if the Step 4 emlx decision is "widen" (one line)
- `.github/dependabot.yml` (create)

**Out of scope**:
- Any `mix.exs` requirement changes beyond the optional emlx line.
- Major or minor bumps of nx/exla/phoenix lines beyond what `hex.outdated`
  shows as compatible "Update possible" within current requirements.
- `Dockerfile` base-image bump (dependabot will propose those going forward).
- CI workflow edits (plan 016).

## Git workflow

- Branch: `advisor/032-dependency-refresh-and-dependabot` (from
  `execute-plans`, after 021 if it's in flight).
- Commit style: `chore(deps): align nx locks, take Phoenix/LV patches, add dependabot`
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Refresh and align the root project

`mix hex.outdated` (record the table), then
`mix deps.update nx ex_doc stream_data`.

**Verify**: `grep '"nx"' mix.lock` shows `0.12.1` (or the current 0.12.x the
site uses — the invariant is **both locks equal**); `bash scripts/ci.sh` →
exit 0. If plan 020 landed, also `mix test --only slow` → 0 failures (the
statistical suite is the accuracy gate for the nx patch bump).

### Step 2: Refresh the site

`cd site && mix hex.outdated` (record), then
`cd site && mix deps.update phoenix phoenix_live_view nx`.

**Verify**: `grep '"nx"' mix.lock site/mix.lock` → identical versions;
`cd site && mix test && mix compile --warnings-as-errors` → clean;
`cd site && mix format --check-formatted` → exit 0.

### Step 3: Add dependabot

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: mix
    directory: "/"
    schedule: {interval: weekly}
    groups:
      dev-deps:
        dependency-type: development
  - package-ecosystem: mix
    directory: "/site"
    schedule: {interval: weekly}
  - package-ecosystem: docker
    directory: "/"
    schedule: {interval: weekly}
  - package-ecosystem: github-actions
    directory: "/"
    schedule: {interval: weekly}
```

(Adjust grouping syntax to current dependabot schema if it rejects
`dependency-type` for mix — grouping is nice-to-have, the four ecosystems are
the requirement.)

**Verify**: the YAML parses (ruby one-liner). Live validation only happens
server-side after push — note that in the report.

### Step 4: emlx pin — investigate, then decide

`mix hex.info emlx 0.4.0` → read its `nx` requirement.

- Compatible with `~> 0.12` → widen root `mix.exs` to
  `{:emlx, "~> 0.3 or ~> 0.4", optional: true}`, `mix deps.update emlx`, and
  — ONLY if you are on Apple Silicon with a working EMLX toolchain — run
  `BSTS_NX_TEST_BACKEND=emlx mix test test/structured_performance_smoke_test.exs test/utils_safe_solve_test.exs`.
  If you cannot run the EMLX smoke locally, still widen (it's optional +
  opt-in) but say so explicitly in the report.
- Incompatible → leave the pin, add one comment line in `mix.exs` explaining
  why (`# emlx 0.4 requires nx >= X — revisit with the next nx line`), and
  record the finding.

**Verify**: whichever branch — `bash scripts/ci.sh` → exit 0.

## Test plan

No new tests. Gates: full library CI-parity run (+ statistical lane when
available) after the root bump; full site suite after the site bump; both
lock files agreeing on nx is the machine-checkable core invariant.

## Done criteria

- [ ] `grep '"nx"' mix.lock site/mix.lock` → same version in both
- [ ] Site phoenix + phoenix_live_view at the latest patch their requirements allow (per fresh `hex.outdated`: nothing "Update possible" remains for them)
- [ ] `.github/dependabot.yml` exists, parses, covers mix(×2)+docker+github-actions
- [ ] emlx decision recorded (widened or comment-documented)
- [ ] `bash scripts/ci.sh` exit 0; `cd site && mix test` 0 failures
- [ ] Only in-scope files changed
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- Any test fails after a bump — report which dep and the failure; do not pin
  around it silently.
- `mix deps.update nx` wants to move other locked deps in surprising ways
  (more than nx+its two deps `complex`/`telemetry`) — inspect the lock diff;
  if it cascades, report before committing.
- `hex.outdated` shows a new nx/exla/phoenix MINOR line since planning
  (e.g. nx 0.13) — that's a migration decision, not a refresh; take only the
  patch-level updates and flag the minor.

## Maintenance notes

- Dependabot PRs land against CI — plans 015/016 must be merged for those PRs
  to be meaningfully gated (site job + hardened workflow). Until then treat
  green dependabot PRs with suspicion for site-only changes.
- Keep the "both locks agree on nx" invariant when merging dependabot PRs —
  they arrive per-directory; merge the pair together.
- The emlx line is dev-hardware-specific; it never gates CI or prod.
