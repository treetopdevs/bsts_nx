defmodule BstsNx.Synthetic.GeneratorTest do
  use ExUnit.Case, async: true

  alias BstsNx.Synthetic.Generator
  alias BstsNx.Synthetic.Scenarios

  @minimal_config %{
    name: "Test",
    total_minutes: 1440,
    baseline: %{
      intercept: 50.0,
      trend: 0.0,
      daily_fourier: [{5.0, 3.0}],
      weekly_fourier: []
    },
    spots: [
      %{id: "s1", window_start: 480, window_end: 510}
    ],
    noise_sd: 1.0,
    effect: %{lambda: 0.5, ec: 1.0, slope: 2.0, coefficient: 10.0},
    controls: %{count: 2, correlation: 0.3},
    seed: 42
  }

  describe "generate/1" do
    test "returns all required keys" do
      result = Generator.generate(@minimal_config)

      assert is_list(result.observations)
      assert is_list(result.ground_truth.baseline)
      assert is_list(result.ground_truth.tv_contribution)
      assert is_map(result.ground_truth.spot_attributions)
      assert is_map(result.ground_truth.coalition_values)
      assert is_float(result.ground_truth.total_lift)
      assert is_list(result.spots)
      assert is_list(result.controls)
      assert is_map(result.config)
    end

    test "observations length matches total_minutes" do
      result = Generator.generate(@minimal_config)
      assert length(result.observations) == @minimal_config.total_minutes
    end

    test "ground truth series lengths match total_minutes" do
      result = Generator.generate(@minimal_config)
      assert length(result.ground_truth.baseline) == @minimal_config.total_minutes
      assert length(result.ground_truth.tv_contribution) == @minimal_config.total_minutes
    end

    test "decomposition is additive (within noise tolerance)" do
      config = %{@minimal_config | noise_sd: 0.0}
      result = Generator.generate(config)

      # With zero noise, observations = baseline + tv_contribution exactly
      Enum.zip_with(
        [result.observations, result.ground_truth.baseline, result.ground_truth.tv_contribution],
        fn [obs, base, tv] ->
          assert_in_delta obs, base + tv, 1.0e-10
        end
      )
    end

    test "total_lift is positive when spots are present" do
      result = Generator.generate(@minimal_config)
      assert result.ground_truth.total_lift > 0.0
    end

    test "total_lift includes carryover effect outside on-air windows" do
      config = %{
        @minimal_config
        | noise_sd: 0.0,
          effect: %{lambda: 0.85, ec: 1.0, slope: 2.0, coefficient: 10.0}
      }

      result = Generator.generate(config)
      tv = result.ground_truth.tv_contribution

      on_air_sum =
        tv
        |> Enum.with_index()
        |> Enum.reduce(0.0, fn {val, idx}, acc ->
          if idx >= 480 and idx < 510, do: acc + val, else: acc
        end)

      assert_in_delta result.ground_truth.total_lift, Enum.sum(tv), 1.0e-10
      assert result.ground_truth.total_lift > on_air_sum
    end

    test "spot attributions are positive and consistent with total lift" do
      config = %{@minimal_config | noise_sd: 0.0}
      result = Generator.generate(config)

      # Each spot should have a positive attribution
      Enum.each(result.ground_truth.spot_attributions, fn {_id, lift} ->
        assert lift > 0.0
      end)

      # Individual attributions should not exceed total lift (with adstock
      # spillover, individual sums can exceed total due to non-additivity,
      # but each individual attribution should be bounded by total lift)
      Enum.each(result.ground_truth.spot_attributions, fn {_id, lift} ->
        assert lift <= result.ground_truth.total_lift
      end)
    end

    test "controls count matches config" do
      result = Generator.generate(@minimal_config)
      assert length(result.controls) == @minimal_config.controls.count
    end

    test "each control series has correct length" do
      result = Generator.generate(@minimal_config)

      Enum.each(result.controls, fn control ->
        assert length(control) == @minimal_config.total_minutes
      end)
    end

    test "tv_contribution is zero outside effect reach" do
      # With lambda=0 and a single spot, effect should only be in the window
      config = %{
        @minimal_config
        | effect: %{lambda: 0.0, ec: 1.0, slope: 2.0, coefficient: 10.0}
      }

      result = Generator.generate(config)

      # Before window_start, contribution should be 0
      Enum.each(0..479, fn t ->
        assert Enum.at(result.ground_truth.tv_contribution, t) == 0.0
      end)
    end

    test "deterministic with same seed" do
      result1 = Generator.generate(@minimal_config)
      result2 = Generator.generate(@minimal_config)
      assert result1.observations == result2.observations
    end

    test "different seeds produce different results" do
      config2 = %{@minimal_config | seed: 99}
      result1 = Generator.generate(@minimal_config)
      result2 = Generator.generate(config2)
      refute result1.observations == result2.observations
    end

    test "works with no controls" do
      config = %{@minimal_config | controls: %{count: 0, correlation: 0.0}}
      result = Generator.generate(config)
      assert result.controls == []
    end

    test "works with scenario_a config" do
      config = Scenarios.scenario_a()
      result = Generator.generate(config)
      assert length(result.observations) == config.total_minutes
      assert result.ground_truth.total_lift > 0.0
    end

    test "spot_attributions are Shapley-consistent for overlapping spots" do
      config = %{
        @minimal_config
        | noise_sd: 0.0,
          spots: [
            %{id: "a", window_start: 500, window_end: 560},
            %{id: "b", window_start: 530, window_end: 590}
          ]
      }

      result = Generator.generate(config)

      shapley_sum = result.ground_truth.spot_attributions |> Map.values() |> Enum.sum()
      coalition_key = Enum.sort(["a", "b"])
      coalition_lift = Map.fetch!(result.ground_truth.coalition_values, coalition_key)

      assert_in_delta shapley_sum, coalition_lift, 1.0e-8
    end

    test "generated controls track configured correlation direction" do
      config = %{
        @minimal_config
        | controls: %{count: 1, correlation: 0.6},
          total_minutes: 3000
      }

      result = Generator.generate(config)
      control = hd(result.controls)
      corr = pearson(control, result.ground_truth.tv_contribution)

      assert corr > 0.35
    end

    test "invalid controls.count raises" do
      config = %{@minimal_config | controls: %{count: -1, correlation: 0.2}}

      assert_raise ArgumentError, ~r/controls.count/, fn ->
        Generator.generate(config)
      end
    end

    test "negative noise_sd raises" do
      config = %{@minimal_config | noise_sd: -1.0}

      assert_raise ArgumentError, ~r/noise_sd/, fn ->
        Generator.generate(config)
      end
    end
  end

  describe "generate_baseline/2" do
    test "flat baseline with no Fourier terms" do
      baseline_config = %{
        intercept: 100.0,
        trend: 0.0,
        daily_fourier: [],
        weekly_fourier: []
      }

      result = Generator.generate_baseline(100, baseline_config)
      assert length(result) == 100
      assert Enum.all?(result, &(&1 == 100.0))
    end

    test "linear trend increases over time" do
      baseline_config = %{
        intercept: 0.0,
        trend: 1.0,
        daily_fourier: [],
        weekly_fourier: []
      }

      result = Generator.generate_baseline(1440, baseline_config)
      # After 1 day (1440 min), trend contribution = 1.0 * (1439/1440)
      assert Enum.at(result, 0) == 0.0
      assert_in_delta Enum.at(result, 1439), 1439 / 1440.0, 1.0e-10
    end

    test "daily Fourier has period 1440" do
      baseline_config = %{
        intercept: 0.0,
        trend: 0.0,
        daily_fourier: [{1.0, 0.0}],
        weekly_fourier: []
      }

      result = Generator.generate_baseline(2880, baseline_config)
      # Value at t=0 should equal value at t=1440
      assert_in_delta Enum.at(result, 0), Enum.at(result, 1440), 1.0e-10
    end
  end

  describe "apply_effect/2" do
    test "zero impulse produces zero effect" do
      impulse = [0.0, 0.0, 0.0]
      effect = %{lambda: 0.5, ec: 1.0, slope: 2.0, coefficient: 10.0}
      result = Generator.apply_effect(impulse, effect)
      assert Enum.all?(result, &(&1 == 0.0))
    end

    test "coefficient scales the output" do
      impulse = [1.0, 0.0, 0.0]
      effect1 = %{lambda: 0.0, ec: 1.0, slope: 2.0, coefficient: 5.0}
      effect2 = %{lambda: 0.0, ec: 1.0, slope: 2.0, coefficient: 10.0}

      r1 = Generator.apply_effect(impulse, effect1)
      r2 = Generator.apply_effect(impulse, effect2)

      # effect2 has 2x coefficient, so should be 2x
      assert_in_delta Enum.at(r2, 0) / Enum.at(r1, 0), 2.0, 1.0e-10
    end
  end

  describe "generate_controls/4" do
    test "zero-variance tv signal rejects non-zero requested correlation" do
      rng = :rand.seed_s(:exsss, {1, 2, 3})
      tv_signal = List.duplicate(0.0, 20)
      controls = %{count: 1, correlation: 0.4}

      assert_raise ArgumentError, ~r/near-zero variance/, fn ->
        Generator.generate_controls(20, tv_signal, controls, rng)
      end
    end
  end

  defp pearson(xs, ys) do
    n = length(xs)
    mean_x = Enum.sum(xs) / n
    mean_y = Enum.sum(ys) / n

    {cov, var_x, var_y} =
      Enum.zip(xs, ys)
      |> Enum.reduce({0.0, 0.0, 0.0}, fn {x, y}, {c, vx, vy} ->
        dx = x - mean_x
        dy = y - mean_y
        {c + dx * dy, vx + dx * dx, vy + dy * dy}
      end)

    denom = :math.sqrt(var_x * var_y)
    if denom <= 1.0e-12, do: 0.0, else: cov / denom
  end
end
