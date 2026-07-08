# Plan 016: Harden the CI workflow (token scope, action pinning, `_build` caching, toolchain alignment, branch triggers)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat 9b7cb8d..HEAD -- .github/workflows/ci.yml Dockerfile .tool-versions`
> Plan 015 is EXPECTED to have changed `ci.yml` (it adds a `site` job) — that
> is not drift; apply this plan's changes to all jobs including `site`. Any
> other structural change to these files: compare the "Current state" excerpts
> against the live code; on a mismatch, treat it as a STOP condition.

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: MED (cache-key and pin mistakes can break CI in ways only visible on a live runner)
- **Depends on**: plans/015-site-ci-verification-lane.md (same file; execute 015 first)
- **Category**: security + dx
- **Planned at**: commit `9b7cb8d`, 2026-07-08

## Why this matters

Four independent audit findings converge on `.github/workflows/ci.yml`:

1. **No `permissions:` block** — every job runs with the repo's default
   `GITHUB_TOKEN`, which is often write-capable. A compromised transitive dep
   pulled by `mix deps.get`, or a tampered action, could push to the repo.
   The jobs only build/test/format/docs; they need `contents: read`.
2. **Actions pinned to mutable tags** (`actions/checkout@v4`,
   `actions/cache@v4`, `erlef/setup-beam@v1`, `r-lib/actions/setup-r@v2`) — a
   moved tag silently executes attacker-controlled code in CI.
3. **`_build` never cached** — each of the parallel jobs cold-compiles Nx and
   the whole dep tree on every run (minutes each, per job, per push).
4. **Three disagreeing toolchain sources**: `.tool-versions` pins Erlang
   28.3.1 / Elixir 1.19.5; the `Dockerfile` builds prod on OTP **28.3.3**; CI
   floats on `elixir: "1.19"` / `otp: "28"` (resolved to newest matching patch
   at run time). Dev, CI, and prod can all run different BEAM patch releases,
   and identical commits can produce different CI results over time.

Additionally, CI `push` triggers cover only `main`/`master`, while active work
happens on the long-lived `execute-plans` branch — direct pushes there get no
CI.

## Current state

- `.github/workflows/ci.yml` at `9b7cb8d`:
  - Lines 3–8: `on.push.branches: [main, master]`, plus `pull_request` and
    `workflow_dispatch` (with a `run_r_parity` boolean input).
  - No `permissions:` key anywhere in the file.
  - Jobs `test` (matrix elixir "1.19"/otp "28"), `test_exla`, `quality`,
    `r_parity`; after plan 015 there is also a `site` job.
  - Every cache block looks like:
    ```yaml
    - name: Cache deps
      uses: actions/cache@v4
      with:
        path: deps
        key: ${{ runner.os }}-mix-${{ matrix.elixir }}-${{ matrix.otp }}-${{ hashFiles('**/mix.lock') }}
    ```
    (`path: deps` only — no `_build`.)
  - Setup steps: `uses: erlef/setup-beam@v1` with literal
    `elixir-version: "1.19"` / `otp-version: "28"`.
- `.tool-versions` (repo root):
  ```
  erlang 28.3.1
  elixir 1.19.5-otp-28
  ```
- `Dockerfile` lines 6–8:
  ```dockerfile
  ARG ELIXIR_VERSION=1.19.5
  ARG OTP_VERSION=28.3.3
  ARG DEBIAN_VERSION=trixie-20260623
  ```

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| YAML validity | `ruby -ryaml -e 'YAML.load_file(".github/workflows/ci.yml"); puts "ok"'` | prints `ok` |
| Resolve a tag to a commit SHA | `git ls-remote https://github.com/actions/checkout 'refs/tags/v4*' \| tail -5` | tag list with SHAs |
| Deref an annotated tag | `git ls-remote https://github.com/actions/checkout 'refs/tags/v4.2.2^{}'` | one line: `<commit-sha> refs/tags/v4.2.2^{}` |
| Check a hexpm image tag exists | `docker manifest inspect hexpm/elixir:1.19.5-erlang-28.3.1-debian-trixie-20260623-slim >/dev/null && echo exists` | prints `exists` (needs docker; see Step 4 fallback) |
| Local library verify | `bash scripts/ci.sh` | exit 0 |

Tooling note: `mix` on this machine is a `mise` shim; prefix with
`mise exec -- ` if a plain invocation misbehaves.

## Scope

**In scope** (the only files you should modify):
- `.github/workflows/ci.yml`
- `Dockerfile` (one ARG line)

**Out of scope**:
- `.tool-versions` — it is the source of truth this plan aligns *to*; change
  it only via the Step 4 fallback, and say so in your report.
- `scripts/ci.sh`, `site/**`, `mix.exs`, `fly.toml`.
- Adding new CI jobs (the nightly statistical lane is plan 020).
- Dependabot/renovate config — recorded as a separate unplanned finding.

## Git workflow

- Branch: `advisor/016-ci-hardening-and-toolchain`, branched from the result
  of plan 015 (or from `execute-plans` if 015 already landed there).
- Commit message style: `ci: restrict token, pin actions, cache _build, align toolchain`
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Add a least-privilege permissions block

At the top level of `ci.yml` (after the `on:` block, before `jobs:`), add:

```yaml
permissions:
  contents: read
```

No job in this workflow writes to the repo, so no per-job elevation is needed.

**Verify**: `ruby -ryaml -e 'puts YAML.load_file(".github/workflows/ci.yml")["permissions"]["contents"]'`
→ prints `read`.

### Step 2: Pin every third-party action to a full commit SHA

For each `uses:` in the file (`actions/checkout@v4`, `actions/cache@v4`,
`erlef/setup-beam@v1`, `r-lib/actions/setup-r@v2` — including copies in the
plan-015 `site` job):

1. Find the latest release tag on that major line, e.g.
   `git ls-remote https://github.com/actions/checkout 'refs/tags/v4*'` and
   pick the highest `v4.x.y`.
2. Resolve the *commit* SHA (annotated tags need the `^{}` deref form shown in
   the commands table).
3. Replace the tag with the 40-char SHA and keep the version as a trailing
   comment, e.g.:
   ```yaml
   uses: actions/checkout@<40-char-sha> # v4.x.y
   ```

Do this for **every occurrence** (the file has multiple copies per action
across jobs — `grep -n 'uses:' .github/workflows/ci.yml` to enumerate).

**Verify**: `grep -n 'uses:' .github/workflows/ci.yml` → every line matches
`uses: <owner>/<repo>@[0-9a-f]{40} # v...` (no bare `@v` tags remain).

### Step 3: Cache `_build` alongside `deps`

In each mix-based job (`test`, `test_exla`, `quality`, `r_parity`, `site`),
extend the cache block's `path` to include the build dir, and add the job's
flavor to the key so different `MIX_ENV`/backend builds don't collide (the
existing keys already differ per job — keep that property):

```yaml
      - name: Cache deps and build
        uses: actions/cache@<sha> # v4.x.y
        with:
          path: |
            deps
            _build
          key: ${{ runner.os }}-mix-<jobflavor>-${{ hashFiles('**/mix.lock') }}
          restore-keys: |
            ${{ runner.os }}-mix-<jobflavor>-
```

For the `site` job the paths are `site/deps` and `site/_build`, keyed on
`hashFiles('site/mix.lock')`.

Keep each job's existing key prefix (e.g. `-mix-exla-`, `-mix-quality-`,
`-mix-r-parity-`) so restored `_build` artifacts always match the job that
produced them.

**Verify**: `grep -c '_build' .github/workflows/ci.yml` → equals the number of
mix jobs (5 with the site job).

### Step 4: Make `.tool-versions` the single toolchain source

1. **CI**: in every `erlef/setup-beam` step, replace the literal versions with:
   ```yaml
   with:
     version-file: .tool-versions
     version-type: strict
   ```
   In the `test` job, this replaces the matrix-provided
   `${{ matrix.elixir }}`/`${{ matrix.otp }}`; since the matrix currently has
   exactly one entry, either drop the matrix entirely or leave it and stop
   interpolating it into setup-beam — dropping it is cleaner. If you drop the
   matrix, also fix the job `name:` and cache keys that interpolate
   `matrix.elixir`/`matrix.otp` (replace with literals `1.19-28` or a
   flavor word).
2. **Dockerfile**: change `ARG OTP_VERSION=28.3.3` → `ARG OTP_VERSION=28.3.1`
   so prod matches `.tool-versions`.
3. **Existence check**: verify the builder image tag
   `hexpm/elixir:1.19.5-erlang-28.3.1-debian-trixie-20260623-slim` exists
   (docker command in the table, or check
   https://hub.docker.com/r/hexpm/elixir/tags?name=1.19.5-erlang-28.3.1 via
   WebFetch/browser).
   **Fallback**: if no such image exists for 28.3.1, align in the other
   direction instead — set `.tool-versions` erlang to `28.3.3` (one line) and
   leave the Dockerfile at 28.3.3 — and state in your report that you took the
   fallback. (This is the one sanctioned edit to `.tool-versions`.)

**Verify**:
- `grep -c 'version-file: .tool-versions' .github/workflows/ci.yml` → equals
  the number of setup-beam steps (5 with the site job).
- `grep 'OTP_VERSION' Dockerfile` and `grep erlang .tool-versions` show the
  same patch version.

### Step 5: Add the working branch to the push triggers

```yaml
on:
  push:
    branches:
      - main
      - master
      - execute-plans
```

**Verify**: `ruby -ryaml -e 'p YAML.load_file(".github/workflows/ci.yml")["on"]["push"]["branches"]'`
→ includes `execute-plans`. (Note: Ruby may render the `on:` key as `true` due
to YAML 1.1 boolean coercion — if `["on"]` returns nil, use
`.fetch(true)` instead; either way confirm the three branches are present.)

### Step 6: Full-file sanity pass

**Verify**:
- `ruby -ryaml -e 'YAML.load_file(".github/workflows/ci.yml"); puts "ok"'` → `ok`
- `git status --porcelain` → only `.github/workflows/ci.yml` and `Dockerfile`
  (plus `.tool-versions` only if the Step 4 fallback fired).
- `bash scripts/ci.sh` → exit 0 (proves the local toolchain still matches what
  CI will now pin to).

## Test plan

No ExUnit tests. The real test is a live CI run, which cannot happen until the
branch is pushed/PR'd — state this in your report. Local proxies: YAML parses,
`scripts/ci.sh` passes, all greps in the done criteria hold. A reviewer should
trigger `workflow_dispatch` (or open a draft PR) as the first post-merge
action and watch all jobs go green, including cache save/restore behavior on a
second run.

## Done criteria

- [ ] `permissions: contents: read` present at workflow top level
- [ ] Zero `uses: ...@v<digit>` mutable-tag references remain (`grep -En 'uses: .*@v[0-9]' .github/workflows/ci.yml` → no matches)
- [ ] Every mix job caches both `deps` and `_build` (site job: `site/deps` + `site/_build`)
- [ ] Every setup-beam step uses `version-file: .tool-versions` + `version-type: strict`
- [ ] `Dockerfile` OTP ARG equals `.tool-versions` erlang version
- [ ] `on.push.branches` includes `execute-plans`
- [ ] YAML parses; `bash scripts/ci.sh` exits 0
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- Plan 015's `site` job is absent from `ci.yml` AND plan 015 is not marked
  DONE in `plans/README.md` — execute 015 first.
- You cannot resolve a commit SHA for any action tag (network/auth failure) —
  do not guess SHAs from memory.
- Neither the 28.3.1 hexpm image exists nor the Step 4 fallback is acceptable
  (e.g. `.tool-versions` was changed since planning to something else
  entirely).
- The `r_parity` job's R setup conflicts with SHA pinning in a way you can't
  resolve mechanically (e.g. `r-lib/actions` subdirectory action pinning
  behaves differently — pin the root repo SHA; if unsure, stop).

## Maintenance notes

- SHA pins go stale: the natural companion is a `.github/dependabot.yml` with
  a `github-actions` ecosystem (recorded as unplanned finding DEP-04 in
  `plans/README.md`) — that automates pin bumps.
- Plan 020 (nightly statistical lane) adds another job to this file; it should
  copy the hardened patterns (pinned SHAs, `version-file`, `_build` cache).
- If a future job needs write scopes (e.g. publishing docs), elevate
  per-job (`permissions:` inside that job), never at the workflow level.
- When Erlang/Elixir are upgraded, `.tool-versions` is now the only file to
  edit for dev+CI; the Dockerfile ARG must be bumped in the same commit.
