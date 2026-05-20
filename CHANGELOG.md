# Changelog

## Unreleased

- The scalar `sample_general/5` Gibbs path now uses the compiled scalar Kalman filter and smoother handoff while preserving f64 parity with the previous public behavior.
- The project now targets Elixir 1.19 with Nx 0.12, plus optional EMLX 0.3 and EXLA 0.12 backend dependencies.
- Added a structured backend benchmark exposed through `mix bench.structured_backends`; current docs clarify that EMLX GPU structured workflows are limited by missing linalg primitive support and should fall back to EMLX CPU, EXLA, or BinaryBackend.
- Causal impact summaries now use linearly interpolated percentile bounds for credible intervals, matching NumPy/R-style quantile behavior. This can shift published interval endpoints relative to nearest-rank bounds from earlier revisions.
- Review cleanup: R sidecar overrides now trim and validate executable `Rscript` paths, timeout strings are parsed safely, Shapley tie handling uses true relative tolerance, time-weighted Shapley docs match the global recency anchor, anomaly fallback scoring avoids repeated list indexing, and BCT AR fit/predict PRNG splitting is regression-tested.
