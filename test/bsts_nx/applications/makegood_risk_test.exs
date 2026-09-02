defmodule BstsNx.Applications.MakegoodRiskTest do
  use ExUnit.Case, async: true

  alias BstsNx.Applications.MakegoodRisk
  alias BstsNx.Forecast

  test "computes delivery and shortfall from complete trajectories" do
    audience = Forecast.new([[100.0, 100.0], [80.0, 80.0], [120.0, 120.0]])

    risk =
      MakegoodRisk.evaluate(
        audience,
        [1.0, 1.0],
        200.0,
        reserve_quantile: 0.10
      )

    assert Forecast.draws_to_lists(risk.delivery_forecast) == [[200.0], [160.0], [240.0]]
    assert risk.expected_delivery == 200.0
    assert risk.median_delivery == 200.0
    assert risk.conservative_delivery == 160.0
    assert_in_delta risk.underdelivery_probability, 1.0 / 3.0, 1.0e-12
    assert_in_delta risk.expected_shortfall, 40.0 / 3.0, 1.0e-12
    assert risk.conditional_expected_shortfall == 40.0
  end

  test "converts expected shortfall into expected makegood units" do
    audience = Forecast.new([[50.0], [100.0]])
    risk = MakegoodRisk.evaluate(audience, [1.0], 100.0)

    assert MakegoodRisk.expected_makegood_units(risk, 25.0) == 1.0
  end

  test "rejects invalid guarantees and reserve quantiles" do
    audience = Forecast.new([[100.0]])
    nan = Nx.Constants.nan() |> Nx.to_number()

    assert_raise ArgumentError, ~r/guarantee must be non-negative/, fn ->
      MakegoodRisk.evaluate(audience, [1.0], -1.0)
    end

    assert_raise ArgumentError, ~r/guarantee must be finite/, fn ->
      MakegoodRisk.evaluate(audience, [1.0], nan)
    end

    assert_raise ArgumentError, ~r/reserve_quantile must be a finite value in/, fn ->
      MakegoodRisk.evaluate(audience, [1.0], 100.0, reserve_quantile: 1.1)
    end
  end

  test "rejects negative exposures and invalid metadata" do
    audience = Forecast.new([[100.0]])

    assert_raise ArgumentError, ~r/exposure_weights.*non-negative/s, fn ->
      MakegoodRisk.evaluate(audience, [-1.0], 100.0)
    end

    assert_raise ArgumentError, ~r/metadata must be a map/, fn ->
      MakegoodRisk.evaluate(audience, [1.0], 100.0, metadata: [])
    end
  end
end
