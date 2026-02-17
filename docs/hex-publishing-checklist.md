# Hex Publishing Checklist

This checklist is for releasing `bsts_nx` to [Hex](https://hex.pm/) with clear, reproducible quality gates.

## 1. Decide the Release Scope

- [ ] Confirm the exact changes included in the release.
- [ ] Confirm semantic version bump type:
  - [ ] patch (`x.y.Z`) for bug fixes only.
  - [ ] minor (`x.Y.z`) for backward-compatible features.
  - [ ] major (`X.y.z`) for breaking changes.
- [ ] Capture any migration notes needed by downstream users.

## 2. Package Metadata (Hex Discoverability)

Check `mix.exs`:

- [ ] `description` clearly states value and use cases.
- [ ] `package/0` includes:
  - [ ] `licenses`
  - [ ] `maintainers`
  - [ ] `links`
  - [ ] `keywords`
  - [ ] `files`
- [ ] `source_url` and `homepage_url` are valid.
- [ ] `version` is updated to the intended release version.

## 3. Documentation Quality Gate

- [ ] API docs compile with no warnings:

```bash
mix docs
```

- [ ] Guide set reflects current API behavior:
  - [ ] `overview`
  - [ ] `getting-started`
  - [ ] `core-modeling`
  - [ ] `causal-inference-and-attribution`
  - [ ] `forecasting-and-applications`
  - [ ] `synthetic-data-and-validation`
  - [ ] `module-reference`
- [ ] Examples are copy/paste runnable and match current function signatures.
- [ ] `README.md` install snippet and quick starts are up to date.

## 4. Test and Build Gate

- [ ] Format check:

```bash
mix format
```

- [ ] Full test suite:

```bash
mix test
```

- [ ] Optional stricter pass before publish:

```bash
mix test --include external
```

- [ ] Ensure lockfile and dependency changes are intentional.

## 5. Changelog and Release Notes

- [ ] Add a changelog entry for the release version.
- [ ] Include at least:
  - [ ] Added
  - [ ] Changed
  - [ ] Fixed
  - [ ] Breaking (if any)
  - [ ] Upgrade notes
- [ ] Add short release highlights for Hex/GitHub release text.

## 6. Git Hygiene

- [ ] Ensure release commit contains only intended files.
- [ ] Tag release commit with `vX.Y.Z`.
- [ ] Push commit and tag to origin.

Example:

```bash
git tag v0.1.0
git push origin main --tags
```

## 7. Hex Authentication and Package Checks

- [ ] Confirm Hex user auth on release machine:

```bash
mix hex.user whoami
```

- [ ] If needed, authenticate:

```bash
mix hex.user auth
```

- [ ] Verify package metadata locally:

```bash
mix hex.build
```

## 8. Publish

Publish to Hex:

```bash
mix hex.publish
```

First release for a package name may prompt confirmation.

If publishing docs separately:

```bash
mix hex.docs
```

## 9. Post-Release Verification

- [ ] Verify package page: [https://hex.pm/packages/bsts_nx](https://hex.pm/packages/bsts_nx)
- [ ] Verify docs page: [https://hexdocs.pm/bsts_nx](https://hexdocs.pm/bsts_nx)
- [ ] Confirm install command works from clean project:

```elixir
{:bsts_nx, "~> X.Y"}
```

- [ ] Announce release notes to users.

## 10. Optional Release Automation

For repeatable releases, consider CI gating for:

- [ ] `mix format --check-formatted`
- [ ] `mix test`
- [ ] `mix docs`
- [ ] version/changelog consistency checks
- [ ] dry-run package build (`mix hex.build`)

This keeps release quality stable as the project grows.
