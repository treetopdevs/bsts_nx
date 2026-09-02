defmodule BstsNx.ForecasterObservationWeightsTest do
  use ExUnit.Case, async: true

  alias BstsNx.Components
  alias BstsNx.Forecaster
  alias BstsNx.ObservationWeights

  describe "known observation-variance weights" do
    test "prewhitens observations and observation rows exactly" do
      spec = Components.local_level_spec()

      {observations, weighted_spec} =
        ObservationWeights.prewhiten([10.0, 20.0], spec, [1.0, 4.0])

      assert observations == [10.0, 10.0]
      assert Enum.map(weighted_spec.h, &Nx.to_flat_list/1) == [[1.0], [0.5]]
    end

    test "all-one weights preserve structured posterior forecasts" do
      observations =
        Enum.map(1..21, fn index ->
          50.0 + rem(index, 7) * 0.5
        end)

      fit_options = [seasonality: 7, num_samples: 8, burn_in: 4, seed: 101]

      unweighted_fit = Forecaster.fit(observations, fit_options)

      weighted_fit =
        Forecaster.fit(
          observations,
          Keyword.put(
            fit_options,
            :observation_variance_weights,
            List.duplicate(1.0, length(observations))
          )
        )

      unweighted = Forecaster.predict(unweighted_fit, horizon: 4, seed: 202)

      weighted =
        Forecaster.predict(weighted_fit,
          horizon: 4,
          future_observation_variance_weights: List.duplicate(1.0, 4),
          seed: 202
        )

      assert weighted_fit.observation_variance_mode == :scaled_known
      assert_all_close(unweighted.mean, weighted.mean)
      assert_all_close(unweighted.sd, weighted.sd)
      assert_all_close(unweighted.lower, weighted.lower)
      assert_all_close(unweighted.upper, weighted.upper)
    end

    test "promotes a weighted scalar model to the structured sampler" do
      fit =
        Forecaster.fit([10.0, 10.5, 11.0, 10.8],
          observation_variance_weights: [1.0, 1.0, 4.0, 1.0],
          num_samples: 4,
          burn_in: 2,
          seed: 303
        )

      assert fit.method == :structured
      assert fit.observation_variance_mode == :scaled_known
      assert %BstsNx.ModelSpec{} = fit.spec

      forecast = Forecaster.predict(fit, horizon: 2, seed: 404)
      assert length(forecast.mean) == 2
    end

    test "larger future variance weights widen posterior forecast dispersion" do
      observations =
        Enum.map(1..28, fn index ->
          100.0 + :math.sin(2.0 * :math.pi() * index / 7.0) * 4.0
        end)

      fit =
        Forecaster.fit(observations,
          seasonality: 7,
          num_samples: 24,
          burn_in: 12,
          seed: 505
        )

      reference =
        Forecaster.predict(fit,
          horizon: 5,
          future_observation_variance_weights: List.duplicate(1.0, 5),
          seed: 606
        )

      less_reliable =
        Forecaster.predict(fit,
          horizon: 5,
          future_observation_variance_weights: List.duplicate(25.0, 5),
          seed: 606
        )

      assert Enum.sum(less_reliable.sd) > Enum.sum(reference.sd)
    end

    test "validates training and future weights" do
      assert_raise ArgumentError, ~r/observation_variance_weights length/, fn ->
        Forecaster.fit([1.0, 2.0, 3.0],
          seasonality: 2,
          observation_variance_weights: [1.0, 1.0],
          num_samples: 1,
          seed: 707
        )
      end

      assert_raise ArgumentError, ~r/greater than zero/, fn ->
        Forecaster.fit([1.0, 2.0, 3.0],
          seasonality: 2,
          observation_variance_weights: [1.0, 0.0, 1.0],
          num_samples: 1,
          seed: 707
        )
      end

      fit =
        Forecaster.fit([1.0, 2.0, 3.0, 4.0],
          seasonality: 2,
          num_samples: 2,
          burn_in: 1,
          seed: 707
        )

      assert_raise ArgumentError, ~r/future_observation_variance_weights length/, fn ->
        Forecaster.predict(fit,
          horizon: 3,
          future_observation_variance_weights: [1.0, 1.0],
          seed: 808
        )
      end
    end
  end

  describe "posterior forecast draws" do
    test "returns list draws alongside the backward-compatible summary" do
      fit =
        Forecaster.fit(Enum.map(1..14, &(&1 * 1.0)),
          seasonality: 7,
          num_samples: 6,
          burn_in: 3,
          seed: 909
        )

      forecast =
        Forecaster.predict(fit,
          horizon: 3,
          return: :both,
          seed: 1_010
        )

      assert forecast.num_draws == 6
      assert length(forecast.draws) == 6
      assert Enum.all?(forecast.draws, &(length(&1) == 3))
      assert length(forecast.mean) == 3
    end

    test "returns tensor summaries and draws for Nx pipelines" do
      fit =
        Forecaster.fit(Enum.map(1..14, &(&1 * 1.0)),
          seasonality: 7,
          num_samples: 6,
          burn_in: 3,
          seed: 1_111
        )

      forecast =
        Forecaster.predict(fit,
          horizon: 4,
          return: :both,
          format: :tensors,
          seed: 1_212
        )

      assert Nx.shape(forecast.draws) == {6, 4}
      assert Nx.shape(forecast.mean) == {4}
      assert Nx.shape(forecast.sd) == {4}
      assert Nx.shape(forecast.lower) == {4}
      assert Nx.shape(forecast.upper) == {4}
    end

    test "supports draw-only output and rejects unknown output modes" do
      fit =
        Forecaster.fit(Enum.map(1..10, &(&1 * 1.0)),
          num_samples: 4,
          burn_in: 2,
          seed: 1_313
        )

      result = Forecaster.predict(fit, horizon: 2, return: :draws, seed: 1_414)

      assert result.horizon == 2
      assert result.num_draws == 4
      assert length(result.draws) == 4

      assert_raise ArgumentError, ~r/return must be/, fn ->
        Forecaster.predict(fit, horizon: 2, return: :unknown, seed: 1_414)
      end

      assert_raise ArgumentError, ~r/format must be/, fn ->
        Forecaster.predict(fit, horizon: 2, format: :unknown, seed: 1_414)
      end
    end
  end

  defp assert_all_close(left, right) do
    assert length(left) == length(right)

    left
    |> Enum.zip(right)
    |> Enum.each(fn {left_value, right_value} ->
      assert_in_delta left_value, right_value, 1.0e-8
    end)
  end
end
