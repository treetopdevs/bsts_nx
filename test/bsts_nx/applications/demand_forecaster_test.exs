defmodule BstsNx.Applications.DemandForecasterTest do
  use ExUnit.Case, async: true
  import ExUnit.CaptureLog

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

    test "preserves missing observations through wrapper coercion" do
      observations = [nil, :nan, Nx.Constants.nan(), 100.0, 101.0, 102.0, 103.0]

      log =
        capture_log(fn ->
          forecast =
            DemandForecaster.forecast(observations,
              horizon: 2,
              num_samples: 2,
              burn_in: 0,
              seed: 42
            )

          assert forecast.horizon == 2
          assert length(forecast.mean) == 2

          Enum.each(forecast.mean ++ forecast.sd ++ forecast.lower ++ forecast.upper, fn value ->
            assert finite_number?(value)
          end)
        end)

      assert log =~ "missing observations"
    end
  end

  describe "Forecaster compatibility" do
    test "preserves demand priors, multiple seasonalities, and constant regressor columns" do
      observations = Enum.map(1..12, fn t -> 20.0 + t / 2 + rem(t, 3) end)

      for {seasonality, regressors, num_samples} <- [
            {nil, nil, 1},
            {[3, 4], nil, 4},
            {nil,
             %{
               training: Enum.map(1..12, fn t -> [t / 12, 1.0] end),
               future: [[2.0, 3.0], [3.0, 4.0]]
             }, 4}
          ] do
        opts = [
          horizon: 2,
          seasonality: seasonality,
          regressors: regressors,
          num_samples: num_samples,
          burn_in: 1,
          thin: 2,
          alpha: 0.2,
          seed: 42
        ]

        spec =
          BstsNx.Components.local_linear_trend_spec(
            initial_level: hd(observations),
            initial_cov_level: 10.0,
            var_level: 0.1,
            var_slope: 0.01
          )

        spec =
          Enum.reduce(List.wrap(seasonality), spec, fn period, spec ->
            BstsNx.Components.compose_specs(spec, BstsNx.Components.seasonal_spec(period))
          end)

        {spec, training, future} =
          case regressors do
            nil ->
              {spec, nil, nil}

            %{training: training, future: future} ->
              {BstsNx.Components.compose_specs(
                 spec,
                 BstsNx.Components.regression_spec(Nx.tensor(training))
               ), training, future}
          end

        keys = Nx.Random.split(Nx.Random.key(42), parts: 2)

        fit =
          BstsNx.Forecaster.fit(observations,
            model_spec: spec,
            regressors: training,
            num_samples: num_samples,
            burn_in: 1,
            thin: 2,
            key: BstsNx.Utils.split_key_at(keys, 0)
          )

        expected =
          BstsNx.Forecaster.predict(fit,
            horizon: 2,
            alpha: 0.2,
            future_regressors: future,
            key: BstsNx.Utils.split_key_at(keys, 1)
          )

        {actual, decomposition} = DemandForecaster.forecast_with_decomposition(observations, opts)

        assert Map.keys(actual) |> Enum.sort() ==
                 Enum.sort([:mean, :sd, :lower, :upper, :horizon, :alpha, :components])

        assert actual.components == nil
        assert actual.horizon == 2
        assert actual.alpha == 0.2

        for field <- [:mean, :sd, :lower, :upper] do
          assert length(actual[field]) == length(expected[field])

          for {actual_value, expected_value} <- Enum.zip(actual[field], expected[field]) do
            assert_in_delta actual_value, expected_value, 1.0e-9
          end
        end

        assert decomposition == BstsNx.Forecaster.decompose(fit)
      end
    end

    test "explicit key wins over seed and unrelated forecasting options stay ignored" do
      observations = Enum.map(1..8, &(&1 * 1.0))
      opts = [horizon: 2, num_samples: 2, burn_in: 0, seed: 42]
      expected = DemandForecaster.forecast(observations, opts)

      actual =
        DemandForecaster.forecast(
          observations,
          opts ++
            [
              key: Nx.Random.key(42),
              return: :draws,
              format: :tensors,
              observation_variance_weights: :invalid,
              future_observation_variance_weights: :invalid
            ]
        )

      assert actual == expected
      assert DemandForecaster.forecast(observations, Keyword.put(opts, :seed, 99)) != expected

      assert DemandForecaster.forecast(
               observations,
               opts |> Keyword.put(:seed, 99) |> Keyword.put(:key, Nx.Random.key(42))
             ) == expected
    end

    test "sample count and model errors retain their precedence" do
      assert_raise ArgumentError, ~r/num_samples/, fn ->
        DemandForecaster.forecast([1.0, 2.0], horizon: 1, num_samples: 0, seasonality: 1)
      end

      assert_raise ArgumentError, ~r/future regressors columns/, fn ->
        DemandForecaster.forecast([1.0, 2.0],
          horizon: 1,
          num_samples: 1,
          burn_in: -1,
          regressors: %{training: [[1.0], [2.0]], future: [[1.0, 2.0]]}
        )
      end
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

      # Total safety stock over 3 independent periods: z * sqrt(3 * sigma^2)
      assert_in_delta safety.total, 1.644854 * :math.sqrt(3.0 * 25.0), 0.01
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

  defp finite_number?(value) when is_number(value), do: value == value and abs(value) < 1.0e300
  defp finite_number?(_value), do: false
end
