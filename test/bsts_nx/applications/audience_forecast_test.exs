defmodule BstsNx.Applications.AudienceForecastTest do
  use ExUnit.Case, async: true

  alias BstsNx.Applications.AudienceForecast
  alias BstsNx.Forecast

  test "multiplies universe, viewing, and share for each retained draw" do
    viewing = Forecast.new([[0.50, 0.60], [0.40, 0.50]])
    share = Forecast.new([[0.20, 0.25], [0.30, 0.20]])

    audience = AudienceForecast.combine(viewing, share, 1_000.0)

    assert Forecast.draws_to_lists(audience) == [
             [100.0, 150.0],
             [120.0, 100.0]
           ]

    assert audience.metadata.decomposition == :universe_viewing_share
  end

  test "accepts a horizon-specific universe" do
    viewing = Forecast.new([[0.50, 0.50], [0.50, 0.50]])
    share = Forecast.new([[0.20, 0.20], [0.20, 0.20]])

    audience = AudienceForecast.from_hut_share(viewing, share, [1_000.0, 2_000.0])

    assert Forecast.draws_to_lists(audience) == [
             [100.0, 200.0],
             [100.0, 200.0]
           ]
  end

  test "accepts an aligned universe forecast" do
    viewing = Forecast.new([[0.50], [0.40]])
    share = Forecast.new([[0.20], [0.25]])
    universe = Forecast.new([[1_000.0], [2_000.0]])

    audience = AudienceForecast.combine(viewing, share, universe)

    assert Forecast.draws_to_lists(audience) == [[100.0], [200.0]]
  end

  test "accepts a per-draw universe matrix" do
    viewing = Forecast.new([[0.50], [0.40]])
    share = Forecast.new([[0.20], [0.25]])

    audience = AudienceForecast.combine(viewing, share, [[1_000.0], [2_000.0]])

    assert Forecast.draws_to_lists(audience) == [[100.0], [200.0]]
  end

  test "rejects negative and misshaped universes" do
    viewing = Forecast.new([[0.50, 0.60], [0.40, 0.50]])
    share = Forecast.new([[0.20, 0.25], [0.30, 0.20]])

    assert_raise ArgumentError, ~r/non-negative values/, fn ->
      AudienceForecast.combine(viewing, share, [-1.0, 2.0])
    end

    assert_raise ArgumentError, ~r/universe must be scalar/, fn ->
      AudienceForecast.combine(viewing, share, [1.0, 2.0, 3.0])
    end
  end

  test "rejects forecasts with incompatible draw alignment" do
    viewing = Forecast.new([[0.50], [0.40]])
    share = Forecast.new([[0.20]])

    assert_raise ArgumentError, ~r/same draw and horizon dimensions/, fn ->
      AudienceForecast.combine(viewing, share, 1_000.0)
    end
  end
end
