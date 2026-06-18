# Repository Guidelines

## Project Structure & Module Organization
- `lib/` holds the library source, with the main namespace in `lib/bsts_nx.ex` and feature modules under `lib/bsts_nx/` (e.g., `kalman_filter.ex`, `gibbs_sampler.ex`, `causal_impact.ex`).
- `test/` contains ExUnit tests, plus `test/test_helper.exs` for test bootstrapping.
- `mix.exs` defines the application, version, and dependencies (Nx with optional EMLX/EXLA backends).

## Build, Test, and Development Commands
- `mix deps.get`: fetch dependencies.
- `bash scripts/ci.sh`: run the local CI-parity verifier (compile with warnings as errors, non-external tests, format check, and docs).
- `mix compile`: compile the library.
- `iex -S mix`: start an interactive shell with the project loaded.
- `mix test`: run the full ExUnit suite.
- `mix test --exclude external`: skip tests tagged `@moduletag :external`.
- `mix test --only external`: run only the external tests.
- `mix test --include slow`: include tests tagged `:slow`.
- `BSTS_NX_TEST_BACKEND=exla mix test test/structured_performance_smoke_test.exs test/utils_safe_solve_test.exs`: run the optional EXLA smoke lane.
- `BSTS_NX_TEST_BACKEND=emlx mix test test/structured_performance_smoke_test.exs test/utils_safe_solve_test.exs`: run the optional EMLX smoke lane where available.
- `BSTS_NX_ENABLE_R_PARITY=1 mix test test/r_parity_test.exs`: run the optional R parity lane when R dependencies are installed.
- `mix format`: apply standard Elixir formatting (no custom formatter config is present).
- `mix bench.structured_backends`: compare structured sampler behavior across BinaryBackend, EXLA, and EMLX where available.

## Coding Style & Naming Conventions
- Use idiomatic Elixir style and rely on `mix format` for consistent 2-space indentation.
- Modules are CamelCase under the `BstsNx` namespace; files are snake_case in `lib/bsts_nx/`.
- Tests live in `test/` and use the `_test.exs` suffix.

## Testing Guidelines
- The project uses ExUnit; keep tests deterministic where possible.
- For stochastic behavior, seed RNGs (e.g., `:rand.seed/2`) and assert on stable invariants.
- Tag long-running or external comparisons with `@moduletag :external` so they can be skipped.

## Commit & Pull Request Guidelines
- The current Git history has a single commit (`first commit`), so no established message convention exists yet.
- Suggested practice: short, imperative commit subjects with optional detail in the body.
- PRs should include a concise summary, rationale, and the exact test command(s) run. Call out any changes to numerical tolerances or randomness handling.

## Configuration & Performance Notes
- EXLA is optional; enable it in your consuming application by adding the dependency and setting:
  ```elixir
  config :nx, :default_backend, EXLA.Backend
  ```
- EMLX is also optional. Structured EMLX GPU workflows are currently limited by missing linalg primitive support, so use EMLX CPU, EXLA, or BinaryBackend as practical fallback paths.
- This repository does not include a `config/` directory; application-level config lives in the host app.
