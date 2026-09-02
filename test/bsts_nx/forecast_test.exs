defmodule BstsNx.ForecastTest do
  use ExUnit.Case, async: true

  alias BstsNx.Forecast

  test "retains joint draws and computes marginal summaries" do
    forecast = Forecast.new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], alpha: 0.20)

    assert forecast.num_draws == 3
    assert forecast.horizon == 2
    assert Forecast.draws_to_lists(forecast) == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    summary = Forecast.summary(forecast)
    assert summary.mean == [3.0, 4.0]
    assert summary.median == [3.0, 4.0]
    assert summary.lower == [1.0, 2.0]
    assert summary.upper == [5.0, 6.0]
    assert Forecast.quantile(forecast, 0.5) == [3.0, 4.0]
  end

  test "combines compatible forecasts draw by draw" do
    left = Forecast.new([[1.0, 2.0], [3.0, 4.0]])
    right = Forecast.new([[10.0, 20.0], [30.0, 40.0]])

    combined = Forecast.combine(left, right, &(&1 * &2))

    assert Forecast.draws_to_lists(combined) == [[10.0, 40.0], [90.0, 160.0]]
  end

  test "maps, scales, and aggregates complete trajectories" do
    forecast = Forecast.new([[10.0, 20.0], [30.0, 40.0]])

    mapped = Forecast.map(forecast, &(&1 + 1.0))
    assert Forecast.draws_to_lists(mapped) == [[11.0, 21.0], [31.0, 41.0]]

    scaled = Forecast.scale(forecast, [2.0, 0.5])
    assert Forecast.draws_to_lists(scaled) == [[20.0, 10.0], [60.0, 20.0]]

    total = Forecast.weighted_sum(forecast, [1.0, 2.0])
    assert Forecast.draws_to_lists(total) == [[50.0], [110.0]]
  end

  test "computes threshold probabilities and shortfall measures" do
    forecast = Forecast.new([[80.0], [100.0], [120.0]])

    assert [probability] = Forecast.probability_below(forecast, 100.0)
    assert_in_delta probability, 1.0 / 3.0, 1.0e-12

    assert [expected] = Forecast.expected_shortfall(forecast, 100.0)
    assert_in_delta expected, 20.0 / 3.0, 1.0e-12

    assert Forecast.conditional_expected_shortfall(forecast, 100.0) == [20.0]
  end

  test "rejects incompatible draw shapes" do
    left = Forecast.new([[1.0, 2.0], [3.0, 4.0]])
    right = Forecast.new([[1.0], [2.0]])

    assert_raise ArgumentError, ~r/same draw and horizon dimensions/, fn ->
      Forecast.combine(left, right, &Kernel.+/2)
    end
  end

  test "rejects invalid metadata, quantiles, and thresholds" do
    forecast = Forecast.new([[1.0], [2.0]])
    nan = Nx.Constants.nan() |> Nx.to_number()

    assert_raise ArgumentError, ~r/metadata must be a map/, fn ->
      Forecast.new([[1.0]], metadata: [])
    end

    assert_raise ArgumentError, ~r/probability must be a finite value in/, fn ->
      Forecast.quantile(forecast, nan)
    end

    assert_raise ArgumentError, ~r/threshold must be finite/, fn ->
      Forecast.expected_shortfall(forecast, nan)
    end
  end
end
