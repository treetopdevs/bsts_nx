defmodule BstsNx do
  @moduledoc """
  Bayesian Structural Time Series for Elixir.

  BstsNx is a general-purpose causal inference and time series framework
  built on [Nx](https://hex.pm/packages/nx).  It provides Kalman filtering,
  Gibbs sampling, causal impact estimation, forecasting, and anomaly
  detection — all with proper Bayesian uncertainty quantification.

  ## Core Modules

  The statistical engine:

    * `BstsNx.KalmanFilter` — Linear Kalman filter with missing data support
    * `BstsNx.Smoother` — RTS smoother and Carter-Kohn simulation smoother
    * `BstsNx.GibbsSampler` — MCMC sampling for state-space models
    * `BstsNx.CausalImpact` — Counterfactual-based causal inference
    * `BstsNx.Components` — Model building blocks (trend, seasonal, regression)
    * `BstsNx.Diagnostics` — MCMC convergence diagnostics (R-hat, ESS)

  ## High-Level APIs

  Domain-neutral interfaces:

    * `BstsNx.InterventionAnalysis` — Measure the causal effect of any intervention
    * `BstsNx.Forecaster` — Time series forecasting with credible intervals
    * `BstsNx.BCT.ARForecaster` — Phase A BCT-AR forecasting scaffold

  ## Domain Applications

  Pre-built solutions for common use cases:

    * `BstsNx.Applications.MarketingLift` — Digital marketing incrementality
    * `BstsNx.Applications.DemandForecaster` — Demand forecasting with causal drivers
    * `BstsNx.Applications.AnomalyDetector` — Anomaly detection with calibrated uncertainty
    * `BstsNx.Applications.PolicyEvaluator` — Interrupted time series for policy evaluation
    * `BstsNx.Applications.TVAttribution` — TV-to-web ad attribution

  ## Documentation Map

  If you are new to the project, read the guides in this order:

    * `overview` — module map + workflow selection
    * `hex-publishing-checklist` — release quality gates and publish steps
    * `getting-started` — first end-to-end examples
    * `core-modeling` — Kalman/smoother/Gibbs/model composition
    * `causal-inference-and-attribution` — intervention and overlap attribution
    * `forecasting-and-applications` — forecasting and domain wrappers
    * `synthetic-data-and-validation` — synthetic scenarios and calibration
    * `module-reference` — module-by-module lookup with snippets

  ## Indexing Conventions

  Be explicit with index semantics:

    * `pre_period` and `post_period` are **1-based inclusive**
    * spot windows are **0-based half-open** (`[window_start, window_end)`)

  ## Quick Start

  ### Causal Impact Analysis

      # Did a marketing campaign cause additional conversions?
      result = BstsNx.InterventionAnalysis.analyze(
        daily_conversions,
        %{pre_period: {1, 30}, post_period: {31, 45}},
        seasonality: 7
      )

      result.summary.cumulative_effect.mean  # incremental conversions
      result.significant?                    # statistically significant?

  ### Forecasting

      # Forecast next 14 days with weekly seasonality
      forecast = BstsNx.Forecaster.fit_predict(
        historical_data, 14,
        seasonality: 7
      )

      forecast.mean   # point forecasts
      forecast.lower  # 2.5th percentile
      forecast.upper  # 97.5th percentile

  ### Anomaly Detection

      detector = BstsNx.Applications.AnomalyDetector.fit(normal_data,
        seasonality: 24
      )

      scores = BstsNx.Applications.AnomalyDetector.score(detector, new_data)

  ## Optional EXLA Backend

  By default, all numerical operations run on the Elixir/Erlang VM.  For
  improved performance, add the optional `:exla` dependency and configure:

      config :nx, :default_backend, EXLA.Backend

  This enables JIT compilation for the Kalman filter and smoother,
  providing significant speed-ups for large time series.
  """
end
