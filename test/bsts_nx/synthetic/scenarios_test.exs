defmodule BstsNx.Synthetic.ScenariosTest do
  use ExUnit.Case, async: true

  alias BstsNx.Synthetic.Scenarios
  alias BstsNx.Synthetic.Generator

  describe "all/0" do
    test "returns 5 scenarios" do
      assert length(Scenarios.all()) == 5
    end

    test "each entry has {id, name, config} shape" do
      Enum.each(Scenarios.all(), fn {id, name, config} ->
        assert is_atom(id)
        assert is_binary(name)
        assert is_map(config)
      end)
    end

    test "all scenarios have unique IDs" do
      ids = Enum.map(Scenarios.all(), fn {id, _, _} -> id end)
      assert length(ids) == length(Enum.uniq(ids))
    end
  end

  for {id, name, _config} <- [
        {:scenario_a, "Single Campaign", nil},
        {:scenario_b, "Overlapping Campaigns", nil},
        {:scenario_c, "Varying Effect Sizes", nil},
        {:scenario_d, "Confounded", nil},
        {:scenario_e, "Diminishing Returns", nil}
      ] do
    describe "#{id} (#{name})" do
      test "has all required config keys" do
        config = apply(Scenarios, unquote(id), [])

        assert is_binary(config.name)
        assert is_integer(config.total_minutes)
        assert config.total_minutes > 0
        assert is_map(config.baseline)
        assert is_list(config.spots)
        assert is_float(config.noise_sd)
        assert is_map(config.effect)
        assert is_map(config.controls)
        assert is_integer(config.seed)
      end

      test "baseline config has all keys" do
        config = apply(Scenarios, unquote(id), [])
        b = config.baseline

        assert is_float(b.intercept)
        assert is_float(b.trend)
        assert is_list(b.daily_fourier)
        assert is_list(b.weekly_fourier)
      end

      test "effect config has all keys" do
        config = apply(Scenarios, unquote(id), [])
        e = config.effect

        assert is_float(e.lambda) and e.lambda >= 0.0 and e.lambda <= 1.0
        assert is_float(e.ec) and e.ec > 0.0
        assert is_float(e.slope) and e.slope > 0.0
        assert is_float(e.coefficient) and e.coefficient > 0.0
      end

      test "spots have valid windows within total_minutes" do
        config = apply(Scenarios, unquote(id), [])

        Enum.each(config.spots, fn spot ->
          assert is_binary(spot.id)
          assert spot.window_start >= 0
          assert spot.window_end > spot.window_start
          assert spot.window_end <= config.total_minutes
        end)
      end

      test "can generate synthetic data successfully" do
        config = apply(Scenarios, unquote(id), [])
        result = Generator.generate(config)

        assert length(result.observations) == config.total_minutes
        assert result.ground_truth.total_lift > 0.0
        assert length(result.spots) == length(config.spots)
      end
    end
  end

  describe "scenario_b (overlap-specific)" do
    test "has overlapping spots from two campaigns" do
      config = Scenarios.scenario_b()

      b1_spots = Enum.filter(config.spots, &String.starts_with?(&1.id, "b1"))
      b2_spots = Enum.filter(config.spots, &String.starts_with?(&1.id, "b2"))

      assert length(b1_spots) > 0
      assert length(b2_spots) > 0

      # Verify overlap: b1 runs days 5-12, b2 runs days 8-16
      b1_range =
        MapSet.new(Enum.flat_map(b1_spots, fn s -> s.window_start..(s.window_end - 1) end))

      b2_range =
        MapSet.new(Enum.flat_map(b2_spots, fn s -> s.window_start..(s.window_end - 1) end))

      overlap = MapSet.intersection(b1_range, b2_range)
      assert MapSet.size(overlap) > 0
    end
  end

  describe "scenario_d (confounded)" do
    test "spots are only on weekdays" do
      config = Scenarios.scenario_d()

      Enum.each(config.spots, fn spot ->
        day = div(spot.window_start, 1440)
        day_of_week = rem(day, 7)
        assert day_of_week not in [5, 6], "Spot #{spot.id} is on weekend day #{day_of_week}"
      end)
    end
  end

  describe "scenario_e (diminishing returns)" do
    test "has steep saturation curve (low ec, high slope)" do
      config = Scenarios.scenario_e()
      assert config.effect.ec < 1.0
      assert config.effect.slope >= 3.0
    end

    test "has high spot frequency" do
      config = Scenarios.scenario_e()
      assert length(config.spots) > 50
    end
  end
end
