defmodule BstsNx.Applications.DemandForecasterTest do
  use ExUnit.Case, async: true

  alias BstsNx.Applications.DemandForecaster

  describe "forecast/2" do
    test "basic demand forecast without seasonality" do
      :rand.seed(:exsss, {100, 101, 102})
      data = Enum.map(1..30, fn i -> 100.0 + i * 0.5 + :rand.normal() * 3 end)

      forecast =
        DemandForecaster.forecast(data, horizon: 7, num_samples: 10, burn_in: 5, seed: 42)

      assert length(forecast.mean) == 7
      assert length(forecast.lower) == 7
      assert length(forecast.upper) == 7
      assert forecast.horizon == 7

      # Forecasts should show continuing trend (roughly in range)
      Enum.each(forecast.mean, fn m ->
        assert m > 50.0 and m < 200.0
      end)
    end

    test "demand forecast with weekly seasonality" do
      :rand.seed(:exsss, {200, 201, 202})

      data =
        Enum.map(1..28, fn i ->
          base = 100.0 + i * 0.3
          seasonal = :math.sin(2 * :math.pi() * i / 7) * 10
          base + seasonal + :rand.normal() * 2
        end)

      forecast =
        DemandForecaster.forecast(data,
          horizon: 7,
          seasonality: 7,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert length(forecast.mean) == 7
    end

    test "demand forecast with external regressors" do
      :rand.seed(:exsss, {300, 301, 302})

      # Simulated promotion effect
      promo_train = Enum.map(1..20, fn i -> if rem(i, 5) == 0, do: 1.0, else: 0.0 end)

      data =
        Enum.zip(1..20, promo_train)
        |> Enum.map(fn {i, p} -> 100.0 + i * 0.2 + p * 15.0 + :rand.normal() * 2 end)

      promo_future = Enum.map(1..5, fn i -> if rem(i, 5) == 0, do: 1.0, else: 0.0 end)

      # Reshape as {T, 1} tensors
      train_regressors = Enum.map(promo_train, fn v -> [v] end)
      future_regressors = Enum.map(promo_future, fn v -> [v] end)

      forecast =
        DemandForecaster.forecast(data,
          horizon: 5,
          regressors: %{training: train_regressors, future: future_regressors},
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert length(forecast.mean) == 5
    end
  end

  describe "safety_stock/2" do
    test "computes safety stock from forecast" do
      forecast = %{
        mean: [100.0, 110.0, 105.0],
        lower: [80.0, 90.0, 85.0],
        upper: [120.0, 130.0, 125.0],
        sd: [10.0, 10.0, 10.0],
        horizon: 3,
        alpha: 0.05
      }

      safety = DemandForecaster.safety_stock(forecast)

      assert safety.service_level == 0.95
      assert length(safety.per_period) == 3
      # Safety stock = z * sd, where z ≈ 1.645 for 95% one-sided
      Enum.each(safety.per_period, fn ss ->
        assert_in_delta ss, 1.644854 * 10.0, 0.01
      end)
    end

    test "safety stock with lead time" do
      forecast = %{
        mean: [100.0, 100.0, 100.0, 100.0, 100.0],
        lower: [90.0, 90.0, 90.0, 90.0, 90.0],
        upper: [110.0, 110.0, 110.0, 110.0, 110.0],
        sd: [5.0, 5.0, 5.0, 5.0, 5.0],
        horizon: 5,
        alpha: 0.05
      }

      safety = DemandForecaster.safety_stock(forecast, lead_time: 3)

      # Total safety stock over 3-period lead time: 3 × z × 5.0
      assert_in_delta safety.total, 3 * 1.644854 * 5.0, 0.01
    end
  end

  describe "promotion_impact/3" do
    test "measures promotion effect" do
      :rand.seed(:exsss, {400, 401, 402})
      pre = Enum.map(1..20, fn _ -> 100.0 + :rand.normal() * 3 end)
      promo = Enum.map(1..5, fn _ -> 115.0 + :rand.normal() * 3 end)
      obs = pre ++ promo

      promotion = %{
        baseline_start: 1,
        baseline_end: 20,
        promo_start: 21,
        promo_end: 25
      }

      result =
        DemandForecaster.promotion_impact(obs, promotion,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert result.summary.cumulative_effect.mean > 0
    end
  end

  describe "validation" do
    test "rejects empty observations" do
      assert_raise ArgumentError, ~r/non-empty/, fn ->
        DemandForecaster.forecast([], horizon: 5)
      end
    end

    test "rejects invalid horizon" do
      assert_raise ArgumentError, ~r/horizon/, fn ->
        DemandForecaster.forecast([1.0, 2.0], horizon: -1)
      end
    end

    test "rejects non-integer horizon" do
      assert_raise ArgumentError, ~r/horizon/, fn ->
        DemandForecaster.forecast([1.0, 2.0], horizon: 2.5)
      end
    end

    test "rejects invalid num_samples" do
      assert_raise ArgumentError, ~r/num_samples/, fn ->
        DemandForecaster.forecast([1.0, 2.0], horizon: 2, num_samples: 0)
      end
    end

    test "rejects invalid alpha" do
      assert_raise ArgumentError, ~r/alpha/, fn ->
        DemandForecaster.forecast([1.0, 2.0], horizon: 2, alpha: 1.0)
      end
    end

    test "rejects future regressors with wrong column count" do
      obs = Enum.map(1..20, fn _ -> 100.0 end)

      assert_raise ArgumentError, ~r/future regressors columns/, fn ->
        DemandForecaster.forecast(obs,
          horizon: 5,
          regressors: %{
            training: Nx.tensor(Enum.map(1..20, fn _ -> [1.0] end)),
            future: Nx.tensor(Enum.map(1..5, fn _ -> [1.0, 2.0] end))
          },
          num_samples: 5,
          seed: 42
        )
      end
    end

    test "rejects future regressors with wrong row count (vs horizon)" do
      obs = Enum.map(1..20, fn _ -> 100.0 end)

      assert_raise ArgumentError, ~r/future regressors rows/, fn ->
        DemandForecaster.forecast(obs,
          horizon: 5,
          regressors: %{
            training: Nx.tensor(Enum.map(1..20, fn _ -> [1.0] end)),
            future: Nx.tensor(Enum.map(1..3, fn _ -> [1.0] end))
          },
          num_samples: 5,
          seed: 42
        )
      end
    end

    test "rejects training regressors with wrong row count" do
      obs = Enum.map(1..20, fn _ -> 100.0 end)

      assert_raise ArgumentError, ~r/training regressors rows/, fn ->
        DemandForecaster.forecast(obs,
          horizon: 5,
          regressors: %{
            training: Nx.tensor(Enum.map(1..15, fn _ -> [1.0] end)),
            future: Nx.tensor(Enum.map(1..5, fn _ -> [1.0] end))
          },
          num_samples: 5,
          seed: 42
        )
      end
    end
  end

  describe "multi-column regressors" do
    test "forecast with temperature + promotions" do
      :rand.seed(:exsss, {600, 601, 602})
      n = 30
      horizon = 5

      temp = Enum.map(1..n, fn i -> 20.0 + :math.sin(i / 5.0) * 10.0 end)
      promo = Enum.map(1..n, fn i -> if rem(i, 7) == 0, do: 1.0, else: 0.0 end)

      obs =
        Enum.zip(temp, promo)
        |> Enum.map(fn {t, p} -> 100.0 + t * 0.5 + p * 30.0 + :rand.normal() * 3 end)

      training = Nx.tensor(Enum.zip(temp, promo) |> Enum.map(fn {t, p} -> [t, p] end))

      future_temp = Enum.map((n + 1)..(n + horizon), fn i -> 20.0 + :math.sin(i / 5.0) * 10.0 end)

      future_promo =
        Enum.map((n + 1)..(n + horizon), fn i -> if rem(i, 7) == 0, do: 1.0, else: 0.0 end)

      future = Nx.tensor(Enum.zip(future_temp, future_promo) |> Enum.map(fn {t, p} -> [t, p] end))

      forecast =
        DemandForecaster.forecast(obs,
          horizon: horizon,
          regressors: %{training: training, future: future},
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert length(forecast.mean) == horizon
      assert length(forecast.sd) == horizon

      # All predictions should have positive variance
      Enum.each(forecast.sd, fn s -> assert s > 0 end)
    end

    test "forecast_with_decomposition/2 with regressors" do
      :rand.seed(:exsss, {610, 611, 612})
      n = 30
      horizon = 5

      regressors = Nx.tensor(Enum.map(1..n, fn _ -> [:rand.normal() * 5] end))
      future = Nx.tensor(Enum.map(1..horizon, fn _ -> [:rand.normal() * 5] end))
      obs = Enum.map(1..n, fn _ -> 100.0 + :rand.normal() * 3 end)

      {forecast, decomp} =
        DemandForecaster.forecast_with_decomposition(obs,
          horizon: horizon,
          regressors: %{training: regressors, future: future},
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert length(forecast.mean) == horizon
      assert is_map(decomp)
      assert length(decomp.fitted) == n
    end
  end
end
