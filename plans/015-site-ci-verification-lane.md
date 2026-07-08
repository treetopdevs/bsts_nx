# Plan 015: Give the `site/` Phoenix app an automated verification lane (CI job + local verifier step)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat 9b7cb8d..HEAD -- .github/workflows/ci.yml scripts/ci.sh site/mix.exs site/.formatter.exs`
> If any of these files changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: dx
- **Planned at**: commit `9b7cb8d`, 2026-07-08

## Why this matters

The repo contains two Mix projects: the BstsNx library at the root, and a
public-facing Phoenix LiveView showcase app in `site/` (deployed to Fly.io via
the root `Dockerfile` + `fly.toml`). The library has full CI coverage; the site
has **none**: no CI job compiles or tests it, the local verifier
`scripts/ci.sh` never enters `site/`, and the site's formatter config is never
checked anywhere. A compile warning, formatting drift, or failing test in the
deployed app can reach `main` (and a deploy) completely undetected. This plan
is also the prerequisite for any future site test-coverage work — tests that
never run in CI provide no protection.

The fix is cheap because a one-command verifier already exists in the site:
`mix precommit` (defined in `site/mix.exs`) runs compile with
warnings-as-errors, unused-dep check, format, and tests. As of the planning
commit, all site gates are green: `mix test` → 7 tests, 0 failures (~0.1 s);
`mix format --check-formatted` → clean; `mix compile --warnings-as-errors` →
clean. Site tests need **no** asset toolchain (tailwind/esbuild are
`runtime: Mix.env() == :dev` deps and `site/config/test.exs` sets
`server: false`), so the CI job needs no asset steps.

## Current state

- `.github/workflows/ci.yml` — four jobs (`test`, `test_exla`, `quality`,
  `r_parity`), all operating on the root library only. Nothing references
  `site/`. Each existing job caches only `path: deps` keyed on
  `hashFiles('**/mix.lock')`.
- `scripts/ci.sh` — the local CI-parity verifier; full current content:

  ```bash
  #!/usr/bin/env bash
  set -euo pipefail

  step() {
    printf '\n==> %s\n' "$1"
    shift
    "$@"
  }

  # Local parity with the default GitHub Actions lanes in
  # .github/workflows/ci.yml. Optional backend, slow, external, and R parity
  # checks stay separate so this remains a practical pre-push verifier.
  step "Fetch test dependencies" env MIX_ENV=test mix deps.get
  step "Compile with warnings as errors" env MIX_ENV=test mix compile --warnings-as-errors
  step "Run non-external tests" env MIX_ENV=test mix test --exclude external
  
  step "Fetch dev dependencies" env MIX_ENV=dev mix deps.get
  step "Check formatting" env MIX_ENV=dev mix format --check-formatted
  step "Build docs" env MIX_ENV=dev mix docs
  ```

- `site/mix.exs` (aliases section, ~line 84) — the existing one-command site
  verifier:

  ```elixir
  precommit: ["compile --warnings-as-errors", "deps.unlock --unused", "format", "test"]
  ```

  Note the `format` step in `precommit` *rewrites* files rather than checking.
  For CI we want the non-mutating `mix format --check-formatted`, so the CI
  job and the `ci.sh` step run explicit commands instead of `mix precommit`.

- `site/.formatter.exs` — a separate formatter config (`import_deps:
  [:phoenix]`, `Phoenix.LiveView.HTMLFormatter`); root `mix format` never
  covers `site/` (root `.formatter.exs` inputs are
  `["*.{ex,exs}", "{config,lib,test}/**/*.{ex,exs}"]`).
- `site/mix.lock` — separate lockfile; the site depends on the library via
  `{:bsts_nx, path: ".."}` and on `{:heroicons, github: "tailwindlabs/heroicons",
  tag: "v2.2.0", ...}` (needs `git` at deps.get time — available on GitHub
  runners by default).
- Site tests as of planning: `site/test/bsts_site/demos/limiter_test.exs`,
  `site/test/bsts_site_web/controllers/error_html_test.exs`,
  `site/test/bsts_site_web/controllers/error_json_test.exs` (7 tests total).

## Commands you will need

| Purpose | Command (from repo root) | Expected on success |
|---------|--------------------------|---------------------|
| Site deps | `cd site && mix deps.get` | exit 0 |
| Site compile | `cd site && mix compile --warnings-as-errors` | exit 0, no warnings |
| Site format | `cd site && mix format --check-formatted` | exit 0 |
| Site tests | `cd site && mix test` | `7 tests, 0 failures` (or more tests, 0 failures) |
| Full local verify | `bash scripts/ci.sh` | all steps pass, exit 0 |
| Shell syntax | `bash -n scripts/ci.sh` | exit 0, no output |
| YAML validity | `ruby -ryaml -e 'YAML.load_file(".github/workflows/ci.yml"); puts "ok"'` | prints `ok` |

Tooling note: `mix` on this machine is a `mise` shim. If a local `mix`
invocation errors with `unexpected argument '-n'` or a `mise exec` usage
message, prefix the command with `mise exec -- ` (e.g. `mise exec -- mix test`).

## Scope

**In scope** (the only files you should modify):
- `.github/workflows/ci.yml`
- `scripts/ci.sh`

**Out of scope** (do NOT touch, even though they look related):
- `site/mix.exs` — do not change the `precommit` alias or deps.
- `site/.formatter.exs`, root `.formatter.exs` — formatter unification is a
  separate concern (root formatter changes are pending in plan 007's branch).
- Anything under `site/lib`, `site/test`, `site/assets` — if a site gate fails,
  that's a STOP condition, not something to fix here.
- `Dockerfile`, `fly.toml` — deploy pipeline unchanged.
- CI hardening (permissions, SHA pinning, `_build` caching, version pinning) —
  that is plan 016, which depends on this plan; don't do it here.

## Git workflow

- Branch: `advisor/015-site-ci-verification-lane` (branched from the current
  working branch, `execute-plans`).
- Commit style (match `git log`): imperative, optionally conventional-commit
  prefixed, e.g. `ci: add site verification job and local verifier step`.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Confirm the site gates are green before touching anything

From the repo root run the four site commands in the table above
(`deps.get`, `compile --warnings-as-errors`, `format --check-formatted`,
`test`).

**Verify**: all four exit 0; `mix test` reports `0 failures`. If any fails,
STOP — the baseline claim in this plan no longer holds.

### Step 2: Add a `site` job to `.github/workflows/ci.yml`

Insert this job after the `quality` job, matching the file's existing style
(same action versions currently used in the file — `actions/checkout@v4`,
`actions/cache@v4`, `erlef/setup-beam@v1`):

```yaml
  site:
    name: Site (Elixir 1.19 / OTP 28)
    runs-on: ubuntu-latest

    env:
      MIX_ENV: test

    defaults:
      run:
        working-directory: site

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Elixir + OTP
        uses: erlef/setup-beam@v1
        with:
          elixir-version: "1.19"
          otp-version: "28"

      - name: Cache site deps
        uses: actions/cache@v4
        with:
          path: site/deps
          key: ${{ runner.os }}-mix-site-1.19-28-${{ hashFiles('site/mix.lock') }}
          restore-keys: |
            ${{ runner.os }}-mix-site-1.19-28-

      - name: Install Hex/Rebar
        run: |
          mix local.hex --force
          mix local.rebar --force

      - name: Fetch dependencies
        run: mix deps.get

      - name: Compile
        run: mix compile --warnings-as-errors

      - name: Check formatting
        run: mix format --check-formatted

      - name: Run tests
        run: mix test
```

Notes that matter:
- `working-directory: site` via `defaults.run` keeps every `run:` step in the
  site project. `actions/cache` paths are repo-root-relative regardless, hence
  `site/deps`.
- The cache key hashes `site/mix.lock` specifically (the existing jobs hash
  `**/mix.lock`, which would also invalidate on root lock changes — fine for
  them, but the site key should track the site lock).
- No tailwind/esbuild/asset steps — tests don't need them (see "Why this
  matters").

**Verify**: `ruby -ryaml -e 'YAML.load_file(".github/workflows/ci.yml"); puts "ok"'`
→ prints `ok`.

### Step 3: Append site steps to `scripts/ci.sh`

Add to the end of `scripts/ci.sh` (after the "Build docs" step), preserving
the existing `step` helper style:

```bash
step "Fetch site dependencies" env -C site MIX_ENV=test mix deps.get
step "Compile site with warnings as errors" env -C site MIX_ENV=test mix compile --warnings-as-errors
step "Check site formatting" env -C site MIX_ENV=test mix format --check-formatted
step "Run site tests" env -C site MIX_ENV=test mix test
```

(`env -C <dir>` runs the command with that working directory — GNU/BSD env on
macOS 13+ and Linux both support `-C`. If `env -C` is unavailable on the
machine, use a `( cd site && ... )` subshell form instead; keep `set -euo
pipefail` semantics intact.)

Also update the comment block at the top of the script so it mentions the site
lane (it currently describes only the library lanes).

**Verify**: `bash -n scripts/ci.sh` → exit 0.

### Step 4: Run the full local verifier end-to-end

**Verify**: `bash scripts/ci.sh` → every step prints `==> ...` and the script
exits 0, with the four new site steps visible at the end and `0 failures` in
the site test output.

### Step 5: Confirm nothing else changed

**Verify**: `git status --porcelain` → only `.github/workflows/ci.yml` and
`scripts/ci.sh` modified.

## Test plan

No new ExUnit tests — this plan adds verification infrastructure. The
verification is the infrastructure itself:

- `bash scripts/ci.sh` passes end-to-end locally (Step 4).
- The YAML parses (Step 2). Actual GitHub-side execution can only be observed
  after push; note in your report that the `site` job is untested on a live
  runner until the branch gets a PR.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `grep -c "working-directory: site" .github/workflows/ci.yml` → `1`
- [ ] `grep -c "hashFiles('site/mix.lock')" .github/workflows/ci.yml` → `1`
- [ ] `grep -c "mix format --check-formatted" scripts/ci.sh` → `2` (root + site)
- [ ] `ruby -ryaml -e 'YAML.load_file(".github/workflows/ci.yml"); puts "ok"'` prints `ok`
- [ ] `bash scripts/ci.sh` exits 0
- [ ] `git status --porcelain` shows only the two in-scope files modified
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- Any site gate in Step 1 fails at baseline (test failure, format drift, or
  compile warning) — the site needs a fix first, which is out of scope here.
- `site/mix.exs` no longer contains the `precommit` alias or the deps
  described in "Current state" (site restructured since planning).
- `ci.yml` has materially changed shape since `9b7cb8d` (e.g. plan 016 landed
  first and the job layout differs) — reconcile with the reviewer instead of
  merging blindly.
- `mix deps.get` in `site/` fails (e.g. the heroicons GitHub dep unreachable).

## Maintenance notes

- Plan 016 (CI hardening) edits the same workflow file — execute it *after*
  this plan and rebase it on this job layout. The new `site` job should get
  the same hardening treatment (permissions, SHA pins, `_build` caching).
- If site tests ever start needing built assets (e.g. LiveView tests asserting
  on digested paths), the job will need `mix assets.setup && mix assets.build`
  before `mix test` — revisit then, not now.
- Future site test-coverage work (LiveView smoke tests, demo contract tests —
  see the unplanned findings in `plans/README.md`) lands inside this lane
  automatically once this job exists.
