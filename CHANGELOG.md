# Changelog

## Unreleased

- Causal impact summaries now use linearly interpolated percentile bounds for credible intervals, matching NumPy/R-style quantile behavior. This can shift published interval endpoints relative to nearest-rank bounds from earlier revisions.
- Review cleanup: R sidecar overrides now trim and validate executable `Rscript` paths, timeout strings are parsed safely, Shapley tie handling uses true relative tolerance, time-weighted Shapley docs match the global recency anchor, anomaly fallback scoring avoids repeated list indexing, and BCT AR fit/predict PRNG splitting is regression-tested.
