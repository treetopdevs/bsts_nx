# BSTS Lift Estimation — Validation Methodology

This document describes the statistical validation tests applied to the Bayesian Structural Time Series (BSTS) lift estimation pipeline in Scripps Verify. These tests help analysts assess whether the model's lift estimates are trustworthy.

## Overview

The BSTS pipeline estimates the **causal lift** of TV advertising on GA4 web sessions by:

1. Fetching minute-level GA4 session data and TV spot air times
2. Building a **baseline** (counterfactual) using either a **seasonal regression** or **local level** state-space model
3. Comparing actual sessions during "on-air" minutes (when a spot recently aired) against the baseline to estimate lift

The validation suite runs six independent checks that test different assumptions. A well-specified model should pass all of them.

---

## 1. Placebo (In-Time) Test

### What it tests
Whether the model produces a false positive when there is no real intervention.

### Methodology
1. Identify the real on-air window — the set of minute indices flagged as "on-air" due to TV spots
2. Count the number of on-air minutes: `n_on_air`
3. Find the **pre-period** — all minute indices that are NOT in the on-air set
4. Place a fake ("placebo") intervention window of size `n_on_air` in the **middle** of the pre-period (to avoid edge effects near the start or end)
5. Re-run the same estimator (seasonal regression or local level) with the placebo window as if those minutes were on-air
6. Compute placebo lift, lift %, and 95% confidence interval

### Pass criteria
- **Placebo CI includes zero**: The 95% confidence interval for placebo lift sessions should contain zero — meaning the model does NOT detect a statistically significant effect where none exists
- **|Placebo Lift %| < 5%**: The magnitude of the false effect should be small

### Why it matters
If the placebo test fails, the baseline is **biased** — it systematically over- or under-predicts during non-intervention periods. Any lift estimate from this model cannot be trusted because the model would "find" an effect even when there is none.

### Remediation
- Adjust Fourier terms (fewer terms reduce overfitting; more terms capture intra-day patterns)
- Toggle the RTS smoother (smoothing stabilizes the baseline but can amplify drift)
- Try the alternative model type (seasonal regression vs local level)
- Reduce the attribution window
- Extend the date range to provide more pre-period training data

---

## 2. Prediction Error (RMSE & MAPE)

### What it tests
How well the baseline fits the **off-air** (training) data.

### Methodology
1. Compute the full baseline across all minutes using the fitted model
2. Select only **off-air minutes** (minutes where no spot attribution window is active)
3. Calculate two metrics:
   - **RMSE** (Root Mean Squared Error): `sqrt(mean((actual - baseline)²))` — penalizes large errors more heavily
   - **MAPE** (Mean Absolute Percentage Error): `mean(|actual - baseline| / max(|actual|, 1.0))` — scale-independent fit metric. The denominator is floored at 1.0 to avoid division by zero for minutes with zero sessions

### Pass criteria
- **MAPE ≤ 20%**: Acceptable baseline accuracy
- **MAPE ≤ 10%**: Excellent fit
- RMSE is shown for context but has no fixed threshold (it depends on session volume)

### Why it matters
If the baseline doesn't fit the training data well, it's unlikely to produce accurate counterfactual predictions during on-air periods. High MAPE suggests the model is missing important patterns in the data (time-of-day effects, weekly seasonality, etc.).

### Remediation
- Increase daily Fourier terms to capture more intra-day seasonality
- Enable the RTS smoother for a less noisy baseline
- Extend the date range for more training data
- Check for data quality issues (missing hours, timezone mismatches)

---

## 3. CI Coverage Calibration

### What it tests
Whether the model's claimed 95% confidence interval actually covers ~95% of observations.

### Methodology
1. For each off-air minute, compute the predicted value and its uncertainty:
   - Predicted = baseline value at that minute
   - Standard deviation = `sqrt(state_variance + observation_variance)` where state variance comes from the Kalman filter's posterior covariance and observation variance from the fitted noise model
2. Compute 95% CI: `[predicted ± 1.96 × SD]`
3. Count what fraction of actual off-air session values fall within their respective CIs

### Pass criteria
- **Coverage between 85% and 99%**: A well-calibrated model should cover close to 95%
- **Below 85%**: CI is too narrow — the model is overconfident (likely overfitting)
- **Above 99%**: CI is too wide — the model is underconfident (uncertainty is inflated)

### Why it matters
The lift CI is only meaningful if the underlying uncertainty estimates are calibrated. An overconfident model (low coverage) produces CIs that are too narrow, leading to false significance. An underconfident model produces CIs so wide that real effects are masked.

### Remediation
- **Under-covering (<85%)**: Reduce Fourier terms (overfitting narrows CI artificially); extend date range
- **Over-covering (>99%)**: Switch to seasonal regression for tighter pattern modeling; check for data quality issues

---

## 4. Durbin-Watson Autocorrelation Test

### What it tests
Whether the model's residuals exhibit serial correlation (autocorrelation).

### Methodology
1. Compute residuals: `actual - baseline` for all minutes
2. Select only off-air residuals (same minutes used for training)
3. Compute the Durbin-Watson statistic:
   ```
   DW = Σ(rₜ - rₜ₋₁)² / Σ(rₜ²)
   ```
   where the sum runs over consecutive off-air minutes

### Pass criteria
- **DW ≈ 2.0**: No autocorrelation (ideal)
- **DW between 1.5 and 2.5**: Acceptable range
- **DW < 1.5**: Positive autocorrelation — residuals tend to be followed by residuals of the same sign
- **DW > 2.5**: Negative autocorrelation — residuals oscillate excessively

### Why it matters
The Kalman filter assumes innovations (prediction errors) are **white noise** — independent from one time step to the next. If residuals are autocorrelated:
- **Positive autocorrelation** (DW < 1.5): The model is missing a systematic pattern. The baseline is "lagging" behind the true signal, creating runs of same-sign errors. This inflates lift CI width and may bias point estimates.
- **Negative autocorrelation** (DW > 2.5): The model is "over-differencing" — the baseline oscillates around the true signal. This often happens when too many Fourier terms cause the model to chase noise.

### Remediation
- **DW too low**: Increase daily Fourier terms; enable RTS smoother; extend date range
- **DW too high**: Reduce Fourier terms; try disabling the smoother

---

## 5. Effect Stability (Sensitivity Analysis)

### What it tests
Whether the lift estimate is robust to small changes in the attribution window.

### Methodology
1. Take the current window size (e.g., 30 minutes)
2. Re-run the full lift estimation with `window - 5` minutes and `window + 5` minutes
3. Compare the resulting lift (in sessions) to the original estimate
4. Compute `max_pct_change = max(|lift_low - lift_base|, |lift_high - lift_base|) / |lift_base|`

### Pass criteria
- **Max % change ≤ 10%**: Very stable — the exact window choice doesn't matter much
- **Max % change ≤ 25%**: Reasonably stable
- **Max % change > 25%**: Sensitive — the lift estimate depends heavily on the exact window width

### Why it matters
If a small change in the attribution window (±5 minutes) causes the lift estimate to change dramatically, the result is fragile. This can happen when:
- A few high-traffic spots are right at the window boundary
- The window is very short and captures/misses individual minutes with high variance
- The baseline is noisy and the on-air set is small

### Remediation
- Results should be interpreted with caution when stability is poor
- Try different window sizes and pick one where lift is most consistent
- Consider whether the attribution window makes physical sense for the medium (TV spots may have delayed effects)

---

## 6. MCMC Diagnostics (R-hat & ESS)

### What it tests
Whether the multi-chain MCMC sampler has converged (only available when "Include full CI diagnostics" is enabled).

### Methodology
The `estimate_causal_impact` function runs multiple independent Markov chains. After sampling:

- **R-hat (Gelman-Rubin statistic)**: Compares between-chain variance to within-chain variance. If chains have converged to the same posterior distribution, R-hat ≈ 1.0. Computed separately for process variance (Q) and observation variance (R) parameters.
- **ESS (Effective Sample Size)**: Accounts for autocorrelation within each chain. Even with 1000 samples, high autocorrelation means fewer independent draws.

### Pass criteria
- **R-hat ≤ 1.1**: Chains have converged
- **ESS ≥ 100**: Sufficient effective samples for reliable posterior estimates

### Why it matters
MCMC-based CIs are only valid if the sampler has converged. High R-hat means different chains found different modes — the posterior estimate is unreliable. Low ESS means the posterior is estimated from too few independent samples, making CI width estimates noisy.

### Remediation
- Increase the number of samples (try 1000+)
- Increase burn-in (try 500+)
- Add more chains (try 4)
- For stubborn non-convergence: 2000 samples / 1000 burn-in

---

## Auto-Tune (Grid Search)

When diagnostics show failures, the **Auto-Tune** feature runs a grid search over 192 parameter combinations:

| Parameter | Values |
|-----------|--------|
| Model | Seasonal Regression, Local Level |
| Daily Fourier terms | 3, 4, 6, 8 |
| Weekly Fourier terms | 1, 2, 3 |
| Window (minutes) | 10, 15, 20, 30 |
| RTS Smoother | on, off |

Each combination is scored by running a placebo test. The scoring function:
1. **Passes placebo** (CI includes zero): Score = `|placebo_lift_pct|` (lower is better)
2. **Fails placebo**: Score = `1.0 + |placebo_lift_pct|` (penalized by 1.0)

The top 5 candidates are presented with their placebo results. "Apply" sets those parameters in the sidebar and re-runs the analysis.

---

## Interpreting Results

### All green
The model is well-specified. The lift estimate and its CI can be taken at face value.

### Placebo fails, others pass
The baseline has a systematic bias. The model is finding effects where none exist. **Do not trust the lift estimate** — use Auto-Tune to find better parameters.

### MAPE high, DW low
Classic under-specification: the model is missing patterns. Increase Fourier terms or enable the smoother.

### Coverage too low
The model is overconfident. CIs are artificially narrow, so "significant" results may be false positives. Reduce model complexity.

### Stability poor
The lift number is fragile. Report it with appropriate caveats, and consider using a window size where the estimate is more consistent.

### MCMC fails (R-hat > 1.1)
The posterior samples haven't converged. Increase samples/burn-in before trusting any MCMC-based CI.
