# Release Readiness Plan (No Public Publish Yet)

This plan prepares `bsts_nx` for Hex release while explicitly holding the final publish step.

## Current Decision

- Do **not** run `mix hex.publish` yet.
- Prepare everything so publishing later is a low-risk, one-command action.

## Phase 1: Stabilize and Validate (Now)

Goal: confidence in quality and compatibility.

- [ ] Keep CI green across supported Elixir/OTP matrix.
- [ ] Keep docs building cleanly (`mix docs`).
- [ ] Keep full test suite green (`mix test`).
- [ ] Confirm no known API-breaking changes are pending.

## Phase 2: Release Artifacts (Before First Public Version)

Goal: complete release documentation and metadata.

- [ ] Add `CHANGELOG.md` with an initial `0.1.0` entry.
- [ ] Finalize `mix.exs` package metadata (`description`, `keywords`, links, license).
- [ ] Final README pass (install snippet + realistic examples).
- [ ] Final guide pass (overview/getting-started/module-reference consistency).

## Phase 3: Dry-Run Packaging (No Publish)

Goal: verify package would publish cleanly.

Run locally:

```bash
mix format
mix test
mix docs
mix hex.build
```

Acceptance criteria:

- [ ] No formatting diffs.
- [ ] Tests pass.
- [ ] Docs generate with no warnings.
- [ ] Hex package tarball builds successfully.

## Phase 4: Pre-Release Gating

Goal: explicit go/no-go checkpoint.

- [ ] Decide release version (semver).
- [ ] Confirm release notes/changelog are complete.
- [ ] Tag candidate commit (`vX.Y.Z`) **only after approval**.
- [ ] Reconfirm whether release is public Hex or private org Hex.

## Phase 5: Publish (Later, Explicit Approval Required)

Only do this after a direct go-ahead.

```bash
mix hex.publish
mix hex.docs
```

## Optional Interim Distribution (No Hex Publish)

If you want users to consume the package before public Hex:

1. Git dependency from a tagged commit:

```elixir
{:bsts_nx, git: "https://github.com/treetopdevs/bsts_nx.git", tag: "vX.Y.Z"}
```

2. Private Hex org (when ready):
- publish to org repo first,
- validate consumer flow,
- promote to public only when policy/product are ready.

## Owner Notes

- This document is intentionally conservative.
- The final publish step is blocked until explicitly requested.
