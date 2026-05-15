defmodule BstsNx.BCT.ARForecasterTest do
  use ExUnit.Case, async: true

  alias BstsNx.BCT.ARForecaster

  describe "fit/2" do
    test "returns scaffold fit result with BCT metadata" do
      data = synthetic_series(80)

      fit =
        ARForecaster.fit(data,
          order: 2,
          context_depth: 3,
          quantizer_bins: 16,
          num_samples: 50
        )

      assert fit.backend == :bct_ar
      assert fit.phase == :scaffold
      assert fit.order == 2
      assert fit.context_depth == 3
      assert fit.quantizer_bins == 16
      assert fit.context_tree == nil
      assert fit.quantizer == %{bins: 16, edges: nil}
      assert fit.training_length == 80
      assert length(fit.coefficients) == 3
      assert fit.innovation_sd > 0.0
      assert fit.in_sample_rmse >= 0.0
    end

    test "rejects invalid fit inputs" do
      assert_raise ArgumentError, ~r/non-empty/, fn ->
        ARForecaster.fit([])
      end

      assert_raise ArgumentError, ~r/order/, fn ->
        ARForecaster.fit([1.0, 2.0], order: 0)
      end

      assert_raise ArgumentError, ~r/greater than order/, fn ->
        ARForecaster.fit([1.0, 2.0], order: 2)
      end

      assert_raise ArgumentError, ~r/context_depth/, fn ->
        ARForecaster.fit([1.0, 2.0, 3.0], context_depth: 0)
      end

      assert_raise ArgumentError, ~r/quantizer_bins/, fn ->
        ARForecaster.fit([1.0, 2.0, 3.0], quantizer_bins: 1)
      end
    end

    test "accepts boundary case where order is length minus one" do
      data = [10.0, 11.0, 12.0]
      fit = ARForecaster.fit(data, order: 2)

      assert fit.order == 2
      assert fit.training_length == 3
      assert length(fit.coefficients) == 3
    end

    test "recovers AR(2) coefficients approximately on synthetic data" do
      data = synthetic_ar2_series(500, 0.5, 0.7, -0.2, 0.1)
      fit = ARForecaster.fit(data, order: 2, ridge: 1.0e-6)
      [b0, b1, b2] = fit.coefficients

      assert_in_delta b0, 0.5, 0.08
      assert_in_delta b1, 0.7, 0.06
      assert_in_delta b2, -0.2, 0.06
    end
  end

  describe "predict/2" do
    test "returns forecast with credible interval shape and ordering" do
      data = synthetic_series(100)
      fit = ARForecaster.fit(data, order: 2, num_samples: 80)
      forecast = ARForecaster.predict(fit, horizon: 12, alpha: 0.1, seed: 42)

      assert forecast.backend == :bct_ar
      assert forecast.phase == :scaffold
      assert forecast.horizon == 12
      assert forecast.alpha == 0.1
      assert forecast.num_samples == 80

      assert length(forecast.mean) == 12
      assert length(forecast.sd) == 12
      assert length(forecast.lower) == 12
      assert length(forecast.upper) == 12

      Enum.zip([forecast.lower, forecast.mean, forecast.upper, forecast.sd])
      |> Enum.each(fn {l, m, u, sd} ->
        assert l <= m + 1.0e-9
        assert m <= u + 1.0e-9
        assert sd >= 0.0
      end)
    end

    test "is deterministic with a fixed seed" do
      data = synthetic_series(90)
      fit = ARForecaster.fit(data, order: 2, num_samples: 60)

      a = ARForecaster.predict(fit, horizon: 8, seed: 1234)
      b = ARForecaster.predict(fit, horizon: 8, seed: 1234)

      assert a.mean == b.mean
      assert a.lower == b.lower
      assert a.upper == b.upper
      assert a.sd == b.sd
    end

    test "explicit key is deterministic and takes precedence over seed" do
      data = synthetic_series(90)
      fit = ARForecaster.fit(data, order: 2, num_samples: 60)
      key = Nx.Random.key(2026)

      a = ARForecaster.predict(fit, horizon: 8, key: key, seed: 123)
      b = ARForecaster.predict(fit, horizon: 8, key: key, seed: 987_654)

      assert a.mean == b.mean
      assert a.lower == b.lower
      assert a.upper == b.upper
      assert a.sd == b.sd
    end

    test "num_samples override is honored in predict output" do
      fit = ARForecaster.fit(synthetic_series(80), order: 2, num_samples: 10)
      forecast = ARForecaster.predict(fit, horizon: 7, num_samples: 37, seed: 42)

      assert forecast.num_samples == 37
      assert length(forecast.mean) == 7
      assert length(forecast.sd) == 7
      assert length(forecast.lower) == 7
      assert length(forecast.upper) == 7
    end

    test "long horizon forecast stays finite" do
      fit = ARForecaster.fit(synthetic_series(120), order: 2, num_samples: 50)
      forecast = ARForecaster.predict(fit, horizon: 180, num_samples: 50, seed: 77)

      Enum.each(forecast.mean, &assert_finite/1)
      Enum.each(forecast.lower, &assert_finite/1)
      Enum.each(forecast.upper, &assert_finite/1)
      Enum.each(forecast.sd, &assert_finite/1)
    end

    test "rejects invalid prediction inputs" do
      fit = ARForecaster.fit(synthetic_series(60), order: 2)

      assert_raise ArgumentError, ~r/horizon/, fn ->
        ARForecaster.predict(fit, horizon: 0)
      end

      assert_raise ArgumentError, ~r/alpha/, fn ->
        ARForecaster.predict(fit, horizon: 5, alpha: 1.0)
      end

      assert_raise ArgumentError, ~r/num_samples/, fn ->
        ARForecaster.predict(fit, horizon: 5, num_samples: 0)
      end
    end
  end

  describe "fit_predict/3" do
    test "fits and predicts in one call" do
      forecast =
        ARForecaster.fit_predict(
          synthetic_series(75),
          6,
          order: 2,
          seed: 7,
          num_samples: 30
        )

      assert forecast.backend == :bct_ar
      assert forecast.phase == :scaffold
      assert forecast.horizon == 6
      assert length(forecast.mean) == 6
    end

    test "matches separate fit and predict when prediction key is split" do
      data = synthetic_series(85)
      opts = [order: 2, num_samples: 35, alpha: 0.1, seed: 123]

      predict_key =
        123
        |> Nx.Random.key()
        |> Nx.Random.split(parts: 2)
        |> BstsNx.Utils.split_key_at(1)

      predict_opts =
        opts
        |> Keyword.put(:horizon, 9)
        |> Keyword.put(:key, predict_key)
        |> Keyword.delete(:seed)

      separate =
        data
        |> ARForecaster.fit(opts)
        |> ARForecaster.predict(predict_opts)

      combined = ARForecaster.fit_predict(data, 9, opts)

      assert separate.mean == combined.mean
      assert separate.lower == combined.lower
      assert separate.upper == combined.upper
      assert separate.sd == combined.sd
      assert separate.alpha == combined.alpha
      assert separate.horizon == combined.horizon
      assert separate.num_samples == combined.num_samples
    end
  end

  defp synthetic_series(n) do
    _ = :rand.seed(:exsss, {101, 102, 103})

    {data_rev, _prev1, _prev2} =
      Enum.reduce(1..n, {[], 80.0, 79.5}, fn idx, {acc, prev1, prev2} ->
        noise = :rand.normal() * 0.4
        value = 0.4 + 0.65 * prev1 - 0.15 * prev2 + 0.01 * idx + noise
        {[value | acc], value, prev1}
      end)

    Enum.reverse(data_rev)
  end

  defp synthetic_ar2_series(n, intercept, phi1, phi2, noise_sd) do
    _ = :rand.seed(:exsss, {212, 213, 214})

    {data_rev, _x1, _x2} =
      Enum.reduce(1..n, {[], 0.0, 0.0}, fn _idx, {acc, x1, x2} ->
        noise = :rand.normal() * noise_sd
        x = intercept + phi1 * x1 + phi2 * x2 + noise
        {[x | acc], x, x1}
      end)

    Enum.reverse(data_rev)
  end

  defp assert_finite(x) when is_number(x) do
    assert x == x
    assert abs(x) < 1.0e300
  end
end
