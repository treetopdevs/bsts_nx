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
