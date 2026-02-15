defmodule BstsNxDoctestTest do
  use ExUnit.Case, async: true

  # Core modules
  doctest BstsNx.Components
  doctest BstsNx.CausalImpact
  doctest BstsNx.KalmanFilter
  doctest BstsNx.StateSpace
  doctest BstsNx.ShapleyAllocator
  doctest BstsNx.SpotAttributor
  doctest BstsNx.CovariateSelection

  # High-level APIs
  doctest BstsNx.ModelBuilder
  doctest BstsNx.InterventionAnalysis
  doctest BstsNx.Forecaster

  # Application modules
  doctest BstsNx.Applications.AnomalyDetector
  doctest BstsNx.Applications.DemandForecaster
  doctest BstsNx.Applications.MarketingLift
  doctest BstsNx.Applications.PolicyEvaluator
  doctest BstsNx.Applications.TVAttribution
end
