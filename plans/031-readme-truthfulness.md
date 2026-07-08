# Plan 031: Make the README (and getting-started) truthful — install path, showcase link, Status section

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat 9b7cb8d..HEAD -- README.md docs/getting-started.md`
> On mismatch with the excerpts below, STOP. ALSO run
> `mix hex.info bsts_nx` — if the package now EXISTS on Hex, the install
> sections are already true and Steps 2–3 invert (see STOP conditions).

## Status

- **Priority**: P2
- **Effort**: S
- **Risk**: LOW (docs only)
- **Depends on**: none (rides alongside the D-LAUNCH direction decision — see
  Why)
- **Category**: docs
- **Planned at**: commit `9b7cb8d`, 2026-07-08

## Why this matters

The README's very first instruction fails for every reader:
`{:bsts_nx, "~> 0.1"}` is a Hex dependency, and the package is **not on Hex**
(verified 2026-07-08: `mix hex.info bsts_nx` → "No package with name
bsts_nx"). `docs/getting-started.md` repeats the same snippet. Meanwhile the
project's best asset — the live showcase site — is never mentioned, and the
"Status" section describes the shipped `Operational` pipeline ("High-level
pipeline APIs now default to an operational Elixir/Nx lane" per the
Quick-start section above it) as *future* direction, contradicting the rest
of the file and underselling maturity. This plan makes the docs match
reality **today**, in a way that degrades gracefully when the launch triad
(repo URL / Hex publish / site deploy) lands.

## Current state

- `README.md:10-18` — the failing install snippet:

  ```elixir
  def deps do
    [
      {:bsts_nx, "~> 0.1"}
    ]
  end
  ```

  (Same shape again in the `Mix.install/2` example around lines 30–37.)
- `docs/getting-started.md:9-15` — identical Hex-style snippet.
- `README.md:204-221` — the Status section ends: "The near-term direction is
  an Elixir-first operational pipeline with explicit execution metadata,
  plus optional R-backed offline parity/reporting." — but
  `BstsNx.Operational` (`prepare/4`, `run/4`, `run/6`) is shipped, public,
  and is the documented default lane earlier in the same README (~lines
  99–125). The limitations list (lines 209–213) doesn't mention Operational
  at all.
- No mention of the showcase site anywhere in README (grep verified). The
  site's canonical URL is `https://bsts-nx.fly.dev` (`fly.toml`), but the
  site **may not be deployed yet** (launch pending).
- The canonical repo URL is itself contested (mix.exs vs git remote —
  D-LAUNCH decision); this plan must NOT bake in either GitHub URL.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Hex status check | `mix hex.info bsts_nx` | "No package with name bsts_nx" (as of planning) |
| Site reachability | `curl -s -o /dev/null -w '%{http_code}' https://bsts-nx.fly.dev/robots.txt` | `200` if deployed; connection failure/404 if not |
| Docs build | `mix docs` | exit 0 |
| Full verify | `bash scripts/ci.sh` | exit 0 |

Tooling note: `mix` is a `mise` shim; prefix `mise exec -- ` if needed.

## Scope

**In scope**:
- `README.md`
- `docs/getting-started.md`

**Out of scope**:
- `mix.exs` URLs (D-LAUNCH decision; plan must not pick a repo URL).
- Publishing to Hex, deploying the site, or flipping site config flags.
- Any other doc under `docs/` (plan 030 handles doc hygiene).
- CHANGELOG.

## Git workflow

- Branch: `advisor/031-readme-truthfulness` (from `execute-plans`).
- Commit style: `docs: honest install instructions, showcase link, accurate Status`
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Establish current launch state

Run the Hex check and the site reachability check from the commands table;
record both results.

**Verify**: you know, with evidence, (a) package on Hex? (b) site live?

### Step 2: Fix the install sections (README + getting-started)

Replace each Hex-style snippet with a truthful two-state form:

```markdown
> **Note**: `bsts_nx` is not yet published to Hex. Until it is, install from
> source:

```elixir
def deps do
  [
    {:bsts_nx, git: "<REPO_GIT_URL>", tag: "..."}  # or a path dependency in an umbrella/monorepo
  ]
end
```

Once published, this becomes `{:bsts_nx, "~> 0.1"}`.
```

For `<REPO_GIT_URL>`: because the canonical public URL is unresolved (mix.exs
says `Cleveland-Software-LLC/bsts_elixir`, the remote is
`treetopdevs/bsts_nx`), do NOT choose — write the snippet with the remote URL
that `git remote get-url origin` reports **minus credentials** (strip any
`user@` prefix), and add an HTML comment
`<!-- TODO(D-LAUNCH): confirm canonical repo URL and Hex publish; then simplify this section -->`.
Apply the same treatment to the `Mix.install/2` example (README ~30–37) and
`docs/getting-started.md:9-15`.

**Verify**: `grep -n '"~> 0.1"' README.md docs/getting-started.md` matches
only inside the "once published" forward-looking sentence(s);
`grep -c "not yet published" README.md docs/getting-started.md` → ≥1 each;
`grep -n "treetopdevs@" README.md docs/getting-started.md` → no matches (no
credentials leaked into the URL).

### Step 3: Add the showcase-site section

Add a short section near the top of README (after the intro paragraph):

- If Step 1 found the site LIVE: a "Live showcase" section linking
  `https://bsts-nx.fly.dev` with one sentence ("every figure is computed
  live on the server by this library, with honest self-grading against
  planted ground truth").
- If NOT live: the same section text, but phrased as "A live showcase site
  ships in [`site/`](site/) (Phoenix LiveView; every demo computed live by
  this library)" linking the directory instead of the URL, plus the HTML
  comment `<!-- TODO(D-LAUNCH): swap to https://bsts-nx.fly.dev once deployed -->`.

**Verify**: `grep -n "showcase" README.md` → ≥1 match; the link target
matches the Step 1 evidence.

### Step 4: Rewrite the Status section

Edit `README.md:204-221` so that:

1. The "shipped" list explicitly includes the operational pipeline:
   `BstsNx.Operational` (prepare/run compiled-filter lane with execution
   metadata) alongside structured composition — moving it OUT of "near-term
   direction".
2. The limitations list (scalar-oriented defn paths, diagonal Q / scalar R,
   scalar observations in structured MCMC) stays as-is — it's accurate.
3. "Near-term direction" keeps only what is actually not built (richer
   component families, multivariate observations through structured MCMC —
   per the roadmap sentence at line 220).

Keep the section's tone; this is a truth fix, not a marketing rewrite.

**Verify**: `grep -n "Operational" README.md` shows it referenced in the
Status/shipped context; the phrase "near-term direction is an Elixir-first
operational pipeline" no longer appears.

### Step 5: Build + full pass

**Verify**: `mix docs` → exit 0 (README is the ExDoc main page — must still
render); `bash scripts/ci.sh` → exit 0.

## Test plan

Docs-only: the greps above plus `mix docs`. Reviewer reads the README diff
top-to-bottom — the whole deliverable is that a newcomer can follow it
without hitting a dead end.

## Done criteria

- [ ] Install sections in README + getting-started work as written today
      (git/source path) and note the future Hex form
- [ ] No credentials in any URL
- [ ] Showcase section present, link target matching deployment reality
- [ ] Status section lists Operational as shipped; stale direction sentence gone
- [ ] `mix docs` and `bash scripts/ci.sh` exit 0
- [ ] Only README.md and docs/getting-started.md changed
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- `mix hex.info bsts_nx` shows the package IS now published — then the
  correct edit is the inverse (keep the Hex snippet, drop nothing, still add
  the showcase + Status fixes); confirm which world you're in before editing.
- The git remote URL contains credentials you cannot cleanly strip, or the
  remote has changed to something that contradicts both known candidates.
- The Status rewrite would require you to judge whether an API is "shipped"
  beyond `Operational` (whose existence you can verify in
  `lib/bsts_nx/operational.ex`) — stick to the verified item; don't
  editorialize other capabilities.

## Maintenance notes

- The two `TODO(D-LAUNCH)` comments are the cleanup hooks: when the repo URL
  is settled + package published + site deployed, a follow-up simplifies both
  sections (and plan 032's dependabot won't touch docs — this stays manual).
- The site's `/start` page has its own install snippet driven by the
  `:hex_published` config flag — it flips independently; no coordination
  needed here.
