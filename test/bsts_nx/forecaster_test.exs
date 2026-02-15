defmodule BstsNx.ForecasterTest do
  use ExUnit.Case, async: true

  alias BstsNx.Forecaster

  describe "fit/2 and predict/2" do
    test "scalar model fit and predict" do
      :rand.seed(:exsss, {100, 101, 102})
      data = Enum.map(1..30, fn _ -> 50.0 + :rand.normal() * 2 end)

      fit_result = Forecaster.fit(data, num_samples: 10, burn_in: 5, seed: 42)

      assert fit_result.method == :scalar
      assert fit_result.training_length == 30
      assert length(fit_result.posterior_samples) == 10

      forecast = Forecaster.predict(fit_result, horizon: 5, seed: 42)

      assert length(forecast.mean) == 5
      assert length(forecast.lower) == 5
      assert length(forecast.upper) == 5
      assert length(forecast.sd) == 5
      assert forecast.horizon == 5
      assert forecast.alpha == 0.05

      # Lower should be less than or equal to mean, mean less than or equal to upper
      Enum.zip([forecast.lower, forecast.mean, forecast.upper])
      |> Enum.each(fn {l, m, u} ->
        assert l <= m + 1.0e-6
        assert m <= u + 1.0e-6
      end)
    end

    test "structured model with seasonality" do
      :rand.seed(:exsss, {200, 201, 202})

      data =
        Enum.map(1..28, fn i ->
          50.0 + :math.sin(2 * :math.pi() * i / 7) * 5 + :rand.normal()
        end)

      fit_result = Forecaster.fit(data, seasonality: 7, num_samples: 10, burn_in: 5, seed: 42)

      assert fit_result.method == :structured
      assert fit_result.spec != nil

      forecast = Forecaster.predict(fit_result, horizon: 7, seed: 42)

      assert length(forecast.mean) == 7
    end
  end

  describe "fit_predict/3" do
    test "one-shot forecast" do
      :rand.seed(:exsss, {300, 301, 302})
      data = Enum.map(1..30, fn _ -> 100.0 + :rand.normal() * 5 end)

      forecast = Forecaster.fit_predict(data, 10, num_samples: 10, burn_in: 5, seed: 42)

      assert length(forecast.mean) == 10
      assert forecast.horizon == 10

      # Forecasts should be in a reasonable range
      Enum.each(forecast.mean, fn m ->
        assert m > 50.0 and m < 200.0
      end)
    end
  end

  describe "fit/2 and predict/2 with regressors" do
    test "fit/2 with regressors returns structured model" do
      :rand.seed(:exsss, {500, 501, 502})
      data = Enum.map(1..30, fn _ -> 50.0 + :rand.normal() * 2 end)
      regressors = Nx.tensor(Enum.map(1..30, fn _ -> [:rand.normal() * 5] end))

      fit_result =
        BstsNx.Forecaster.fit(data,
          regressors: regressors,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert fit_result.method == :structured
      assert fit_result.n_regression_dims == 1
      assert fit_result.spec != nil
      # Trend (2) + 1 regressor = 3
      assert Nx.axis_size(fit_result.spec.f, 0) == 3
    end

    test "predict/2 with future_regressors generates regressor-aware forecasts" do
      :rand.seed(:exsss, {510, 511, 512})
      data = Enum.map(1..30, fn _ -> 50.0 + :rand.normal() * 2 end)
      regressors = Nx.tensor(Enum.map(1..30, fn _ -> [:rand.normal() * 5] end))
      future_regressors = Nx.tensor(Enum.map(1..5, fn _ -> [:rand.normal() * 5] end))

      fit_result =
        BstsNx.Forecaster.fit(data,
          regressors: regressors,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      forecast =
        BstsNx.Forecaster.predict(fit_result,
          horizon: 5,
          future_regressors: future_regressors,
          seed: 42
        )

      assert length(forecast.mean) == 5
      assert length(forecast.lower) == 5
      assert length(forecast.upper) == 5
      assert forecast.horizon == 5
    end

    test "predict/2 without future_regressors uses last H (backward compat)" do
      :rand.seed(:exsss, {520, 521, 522})
      data = Enum.map(1..30, fn _ -> 50.0 + :rand.normal() * 2 end)
      regressors = Nx.tensor(Enum.map(1..30, fn _ -> [:rand.normal() * 5] end))

      fit_result =
        BstsNx.Forecaster.fit(data,
          regressors: regressors,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      # Should not crash without future_regressors
      forecast = BstsNx.Forecaster.predict(fit_result, horizon: 5, seed: 42)
      assert length(forecast.mean) == 5
    end

    test "rejects future_regressors with wrong row count" do
      :rand.seed(:exsss, {530, 531, 532})
      data = Enum.map(1..30, fn _ -> 50.0 + :rand.normal() * 2 end)
      regressors = Nx.tensor(Enum.map(1..30, fn _ -> [:rand.normal() * 5] end))

      fit_result =
        BstsNx.Forecaster.fit(data,
          regressors: regressors,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      # 3 rows for horizon=5 should fail
      assert_raise ArgumentError, ~r/future_regressors rows/, fn ->
        BstsNx.Forecaster.predict(fit_result,
          horizon: 5,
          future_regressors: Nx.tensor([[1.0], [2.0], [3.0]]),
          seed: 42
        )
      end
    end

    test "rejects future_regressors with wrong column count" do
      :rand.seed(:exsss, {540, 541, 542})
      data = Enum.map(1..30, fn _ -> 50.0 + :rand.normal() * 2 end)
      regressors = Nx.tensor(Enum.map(1..30, fn _ -> [:rand.normal() * 5] end))

      fit_result =
        BstsNx.Forecaster.fit(data,
          regressors: regressors,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      # 2 columns but only 1 training regressor
      assert_raise ArgumentError, ~r/future_regressors columns/, fn ->
        BstsNx.Forecaster.predict(fit_result,
          horizon: 5,
          future_regressors: Nx.tensor(Enum.map(1..5, fn _ -> [1.0, 2.0] end)),
          seed: 42
        )
      end
    end

    test "rejects future_regressors when model has no regressors" do
      fit_result =
        BstsNx.Forecaster.fit(Enum.map(1..20, fn _ -> 50.0 end),
          num_samples: 5,
          burn_in: 2,
          seed: 42
        )

      # Scalar model has n_regression_dims=0, so any columns will mismatch
      assert_raise ArgumentError, ~r/future_regressors columns/, fn ->
        BstsNx.Forecaster.predict(fit_result,
          horizon: 3,
          future_regressors: Nx.tensor([[1.0], [2.0], [3.0]]),
          seed: 42
        )
      end
    end
  end

  describe "decompose/1" do
    test "returns nil for scalar models" do
      data = Enum.map(1..20, fn _ -> 50.0 end)
      fit_result = Forecaster.fit(data, num_samples: 5, burn_in: 2, seed: 42)
      assert Forecaster.decompose(fit_result) == nil
    end

    test "returns decomposition for structured models" do
      :rand.seed(:exsss, {400, 401, 402})

      data =
        Enum.map(1..14, fn i ->
          50.0 + :math.sin(2 * :math.pi() * i / 7) * 5 + :rand.normal()
        end)

      fit_result = Forecaster.fit(data, seasonality: 7, num_samples: 5, burn_in: 2, seed: 42)
      decomp = Forecaster.decompose(fit_result)

      assert is_map(decomp)
      assert length(decomp.fitted) == 14
      assert length(decomp.states) == 14
    end
  end

  describe "validation" do
    test "rejects empty observations" do
      assert_raise ArgumentError, ~r/non-empty/, fn ->
        Forecaster.fit([])
      end
    end

    test "rejects invalid num_samples" do
      assert_raise ArgumentError, ~r/num_samples/, fn ->
        Forecaster.fit([1.0, 2.0], num_samples: 0)
      end
    end

    test "rejects non-integer num_samples" do
      assert_raise ArgumentError, ~r/num_samples/, fn ->
        Forecaster.fit([1.0, 2.0], num_samples: 5.5)
      end
    end

    test "rejects invalid forecast horizon" do
      fit_result = Forecaster.fit([1.0, 2.0, 3.0], num_samples: 5, burn_in: 2, seed: 42)

      assert_raise ArgumentError, ~r/horizon/, fn ->
        Forecaster.predict(fit_result, horizon: 0)
      end

      assert_raise ArgumentError, ~r/horizon/, fn ->
        Forecaster.predict(fit_result, horizon: 2.5)
      end
    end

    test "rejects invalid forecast alpha" do
      fit_result = Forecaster.fit([1.0, 2.0, 3.0], num_samples: 5, burn_in: 2, seed: 42)

      assert_raise ArgumentError, ~r/alpha/, fn ->
        Forecaster.predict(fit_result, horizon: 3, alpha: 0.0)
      end
    end
  end
end
