defmodule BstsNx.SpotAttributorTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias BstsNx.SpotAttributor

  # ---------------------------------------------------------------
  # Helpers
  # ---------------------------------------------------------------

  defp make_spot(id, ws, we), do: %{id: id, window_start: ws, window_end: we}

  defp flat_counterfactual(mean_val, var_val, obs_var, t) do
    %{
      mean: List.duplicate(mean_val, t),
      variance: List.duplicate(var_val, t),
      obs_variance: obs_var
    }
  end

  # ---------------------------------------------------------------
  # 1. compute_window_lift
  # ---------------------------------------------------------------

  describe "compute_window_lift/4" do
    test "basic lift computation" do
      obs = [100.0, 102.0, 150.0, 148.0, 101.0]

      cf = %{
        mean: [100.0, 101.0, 102.0, 103.0, 104.0],
        variance: [1.0, 1.0, 1.0, 1.0, 1.0],
        obs_variance: 0.5
      }

      result = SpotAttributor.compute_window_lift(obs, cf, 2, 4)

      expected_lift = 150.0 - 102.0 + (148.0 - 103.0)
      expected_var = 1.0 + 0.5 + (1.0 + 0.5)

      assert_in_delta result.lift, expected_lift, 1.0e-10
      assert_in_delta result.variance, expected_var, 1.0e-10
    end

    test "zero lift when observations equal baseline" do
      obs = [100.0, 101.0, 102.0, 103.0, 104.0]

      cf = %{
        mean: [100.0, 101.0, 102.0, 103.0, 104.0],
        variance: [1.0, 1.0, 1.0, 1.0, 1.0],
        obs_variance: 0.5
      }

      result = SpotAttributor.compute_window_lift(obs, cf, 1, 3)

      assert_in_delta result.lift, 0.0, 1.0e-10
      assert result.variance > 0.0
    end

    test "negative lift" do
      obs = [100.0, 90.0, 85.0, 100.0]

      cf = %{
        mean: [100.0, 100.0, 100.0, 100.0],
        variance: [1.0, 1.0, 1.0, 1.0],
        obs_variance: 0.5
      }

      result = SpotAttributor.compute_window_lift(obs, cf, 1, 3)

      expected_lift = 90.0 - 100.0 + (85.0 - 100.0)
      assert_in_delta result.lift, expected_lift, 1.0e-10
      assert result.lift < 0.0
    end

    test "window at start of series" do
      obs = [150.0, 140.0, 100.0, 100.0]

      cf = %{
        mean: [100.0, 100.0, 100.0, 100.0],
        variance: [2.0, 2.0, 2.0, 2.0],
        obs_variance: 1.0
      }

      result = SpotAttributor.compute_window_lift(obs, cf, 0, 2)

      expected_lift = 150.0 - 100.0 + (140.0 - 100.0)
      assert_in_delta result.lift, expected_lift, 1.0e-10
    end

    test "window at end of series" do
      obs = [100.0, 100.0, 150.0, 160.0]

      cf = %{
        mean: [100.0, 100.0, 100.0, 100.0],
        variance: [1.0, 1.0, 1.0, 1.0],
        obs_variance: 0.5
      }

      result = SpotAttributor.compute_window_lift(obs, cf, 2, 4)

      expected_lift = 150.0 - 100.0 + (160.0 - 100.0)
      assert_in_delta result.lift, expected_lift, 1.0e-10
    end

    test "empty window (start == end) returns zero" do
      obs = [100.0, 200.0]
      cf = %{mean: [100.0, 100.0], variance: [1.0, 1.0], obs_variance: 0.5}

      result = SpotAttributor.compute_window_lift(obs, cf, 1, 1)

      assert_in_delta result.lift, 0.0, 1.0e-10
      assert_in_delta result.variance, 0.0, 1.0e-10
    end
  end

  # ---------------------------------------------------------------
  # 2. compute_coalition_lift
  # ---------------------------------------------------------------

  describe "compute_coalition_lift/3" do
    test "single spot same as window_lift" do
      obs = [100.0, 102.0, 150.0, 148.0, 101.0]

      cf = %{
        mean: [100.0, 101.0, 102.0, 103.0, 104.0],
        variance: [1.0, 1.0, 1.0, 1.0, 1.0],
        obs_variance: 0.5
      }

      spot = make_spot("a", 2, 4)
      coalition = SpotAttributor.compute_coalition_lift(obs, cf, [spot])
      window = SpotAttributor.compute_window_lift(obs, cf, 2, 4)

      assert_in_delta coalition.lift, window.lift, 1.0e-10
      assert_in_delta coalition.variance, window.variance, 1.0e-10
    end

    test "two non-overlapping spots" do
      obs = [100.0, 150.0, 100.0, 160.0, 100.0]

      cf = %{
        mean: [100.0, 100.0, 100.0, 100.0, 100.0],
        variance: [1.0, 1.0, 1.0, 1.0, 1.0],
        obs_variance: 0.5
      }

      spots = [make_spot("a", 1, 2), make_spot("b", 3, 4)]
      result = SpotAttributor.compute_coalition_lift(obs, cf, spots)

      # Individual lifts: a = 50, b = 60, union = 50 + 60 = 110
      assert_in_delta result.lift, 110.0, 1.0e-10
    end

    test "two overlapping spots deduplicates indices" do
      obs = [100.0, 150.0, 160.0, 170.0, 100.0]

      cf = %{
        mean: [100.0, 100.0, 100.0, 100.0, 100.0],
        variance: [1.0, 1.0, 1.0, 1.0, 1.0],
        obs_variance: 0.5
      }

      # a covers [1,3), b covers [2,4) — overlap at index 2
      spots = [make_spot("a", 1, 3), make_spot("b", 2, 4)]
      result = SpotAttributor.compute_coalition_lift(obs, cf, spots)

      # Union is indices {1, 2, 3}: lifts = 50 + 60 + 70 = 180
      expected_lift = 150.0 - 100.0 + (160.0 - 100.0) + (170.0 - 100.0)
      assert_in_delta result.lift, expected_lift, 1.0e-10

      # Variance: 3 indices × (1.0 + 0.5) = 4.5
      assert_in_delta result.variance, 4.5, 1.0e-10
    end

    test "fully overlapping spots same as single window" do
      obs = [100.0, 150.0, 160.0, 100.0]

      cf = %{
        mean: [100.0, 100.0, 100.0, 100.0],
        variance: [1.0, 1.0, 1.0, 1.0],
        obs_variance: 0.5
      }

      # Both cover [1,3)
      spots = [make_spot("a", 1, 3), make_spot("b", 1, 3)]
      result = SpotAttributor.compute_coalition_lift(obs, cf, spots)

      # Same as single window [1,3): lifts = 50 + 60 = 110
      single = SpotAttributor.compute_window_lift(obs, cf, 1, 3)
      assert_in_delta result.lift, single.lift, 1.0e-10
      assert_in_delta result.variance, single.variance, 1.0e-10
    end
  end

  # ---------------------------------------------------------------
  # 3. p_positive
  # ---------------------------------------------------------------

  describe "p_positive/2" do
    test "positive mean gives probability > 0.5" do
      assert SpotAttributor.p_positive(5.0, 1.0) > 0.5
    end

    test "negative mean gives probability < 0.5" do
      assert SpotAttributor.p_positive(-5.0, 1.0) < 0.5
    end

    test "zero mean gives probability 0.5" do
      assert_in_delta SpotAttributor.p_positive(0.0, 1.0), 0.5, 1.0e-10
    end

    test "near-zero variance with positive mean returns 1.0" do
      assert SpotAttributor.p_positive(10.0, 1.0e-20) == 1.0
    end

    test "near-zero variance with negative mean returns 0.0" do
      assert SpotAttributor.p_positive(-10.0, 1.0e-20) == 0.0
    end

    test "large positive mean approaches 1.0" do
      p = SpotAttributor.p_positive(100.0, 1.0)
      assert p > 0.999
    end

    test "large negative mean approaches 0.0" do
      p = SpotAttributor.p_positive(-100.0, 1.0)
      assert p < 0.001
    end
  end

  # ---------------------------------------------------------------
  # 4. attribute (non-overlapping)
  # ---------------------------------------------------------------

  describe "attribute (non-overlapping)" do
    test "single spot" do
      obs = [100.0, 102.0, 150.0, 148.0, 101.0]

      cf = %{
        mean: [100.0, 101.0, 102.0, 103.0, 104.0],
        variance: [1.0, 1.0, 1.0, 1.0, 1.0],
        obs_variance: 0.5
      }

      spots = [make_spot("s1", 2, 4)]
      result = SpotAttributor.attribute(obs, spots, cf)

      assert length(result.attributions) == 1
      [attr] = result.attributions

      assert attr.spot_id == "s1"
      expected_lift = 150.0 - 102.0 + (148.0 - 103.0)
      assert_in_delta attr.lift, expected_lift, 1.0e-10
      assert attr.lift_sd > 0.0
      assert attr.lift_lower < attr.lift
      assert attr.lift_upper > attr.lift
      assert attr.p_positive > 0.5

      assert_in_delta result.total_lift, expected_lift, 1.0e-10
    end

    test "multiple non-overlapping spots" do
      obs = [100.0, 150.0, 100.0, 160.0, 100.0]
      cf = flat_counterfactual(100.0, 1.0, 0.5, 5)

      spots = [make_spot("a", 1, 2), make_spot("b", 3, 4)]
      result = SpotAttributor.attribute(obs, spots, cf)

      assert length(result.attributions) == 2
      assert result.overlap_groups == []

      lifts = Map.new(result.attributions, fn a -> {a.spot_id, a.lift} end)
      assert_in_delta lifts["a"], 50.0, 1.0e-10
      assert_in_delta lifts["b"], 60.0, 1.0e-10
    end

    test "result sums: total_lift equals sum of individual lifts" do
      obs = [100.0, 150.0, 100.0, 160.0, 100.0]
      cf = flat_counterfactual(100.0, 1.0, 0.5, 5)

      spots = [make_spot("a", 1, 2), make_spot("b", 3, 4)]
      result = SpotAttributor.attribute(obs, spots, cf)

      sum_of_lifts = Enum.reduce(result.attributions, 0.0, fn a, acc -> acc + a.lift end)
      assert_in_delta result.total_lift, sum_of_lifts, 1.0e-10
    end

    test "CI contains lift" do
      obs = [100.0, 150.0, 100.0]
      cf = flat_counterfactual(100.0, 1.0, 0.5, 3)

      spots = [make_spot("a", 1, 2)]
      result = SpotAttributor.attribute(obs, spots, cf)

      [attr] = result.attributions
      assert attr.lift_lower <= attr.lift
      assert attr.lift_upper >= attr.lift
    end
  end

  # ---------------------------------------------------------------
  # 5. attribute (overlapping)
  # ---------------------------------------------------------------

  describe "attribute (overlapping)" do
    test "two overlapping spots use Shapley split" do
      obs = [100.0, 150.0, 160.0, 170.0, 100.0]
      cf = flat_counterfactual(100.0, 1.0, 0.5, 5)

      # a covers [1,3), b covers [2,4) — overlap at index 2
      spots = [make_spot("a", 1, 3), make_spot("b", 2, 4)]
      result = SpotAttributor.attribute(obs, spots, cf)

      assert length(result.attributions) == 2

      # Coalition lift = sum over union {1,2,3}: 50 + 60 + 70 = 180
      coalition_lift = 150.0 - 100.0 + (160.0 - 100.0) + (170.0 - 100.0)
      sum_of_allocated = Enum.reduce(result.attributions, 0.0, fn a, acc -> acc + a.lift end)

      # Shapley + normalization ensures allocations sum to coalition lift
      assert_in_delta sum_of_allocated, coalition_lift, 1.0e-6

      # overlap_groups should contain the group
      assert length(result.overlap_groups) == 1
    end

    test "three-way overlap" do
      obs = List.duplicate(100.0, 3) ++ [200.0, 200.0, 200.0] ++ List.duplicate(100.0, 3)
      cf = flat_counterfactual(100.0, 1.0, 0.5, 9)

      # All three overlap in [3,6)
      spots = [
        make_spot("a", 3, 6),
        make_spot("b", 3, 6),
        make_spot("c", 3, 6)
      ]

      result = SpotAttributor.attribute(obs, spots, cf)

      assert length(result.attributions) == 3

      # All spots have identical windows → symmetric → should get equal shares
      lifts = Enum.map(result.attributions, & &1.lift)
      expected_per_spot = 300.0 / 3.0

      Enum.each(lifts, fn l ->
        assert_in_delta l, expected_per_spot, 1.0e-6
      end)
    end

    test "overlap_groups correctly reported" do
      obs = List.duplicate(100.0, 10)
      cf = flat_counterfactual(100.0, 1.0, 0.5, 10)

      spots = [
        make_spot("a", 1, 4),
        make_spot("b", 3, 6),
        make_spot("c", 8, 10)
      ]

      result = SpotAttributor.attribute(obs, spots, cf)

      # a and b overlap; c is standalone
      assert length(result.overlap_groups) == 1
      assert Enum.sort(hd(result.overlap_groups)) == ["a", "b"]
    end
  end

  # ---------------------------------------------------------------
  # 6. attribute (edge cases)
  # ---------------------------------------------------------------

  describe "attribute (edge cases)" do
    test "empty spots list" do
      obs = [100.0, 200.0, 100.0]
      cf = flat_counterfactual(100.0, 1.0, 0.5, 3)

      result = SpotAttributor.attribute(obs, [], cf)

      assert result.attributions == []
      assert result.total_lift == 0.0
      assert result.total_lift_sd == 0.0
      assert result.overlap_groups == []
    end

    test "zero-length windows" do
      obs = [100.0, 200.0, 100.0]
      cf = flat_counterfactual(100.0, 1.0, 0.5, 3)

      spots = [make_spot("a", 1, 1)]
      result = SpotAttributor.attribute(obs, spots, cf)

      [attr] = result.attributions
      assert_in_delta attr.lift, 0.0, 1.0e-10
    end

    test "all-zero observations with non-zero baseline" do
      obs = [0.0, 0.0, 0.0, 0.0]
      cf = flat_counterfactual(100.0, 1.0, 0.5, 4)

      spots = [make_spot("a", 1, 3)]
      result = SpotAttributor.attribute(obs, spots, cf)

      [attr] = result.attributions
      # lift = (0 - 100) + (0 - 100) = -200
      assert_in_delta attr.lift, -200.0, 1.0e-10
      assert attr.p_positive < 0.01
    end

    test "negative lift produces p_positive < 0.5" do
      obs = [100.0, 50.0, 60.0, 100.0]
      cf = flat_counterfactual(100.0, 1.0, 0.5, 4)

      spots = [make_spot("a", 1, 3)]
      result = SpotAttributor.attribute(obs, spots, cf)

      [attr] = result.attributions
      assert attr.lift < 0.0
      assert attr.p_positive < 0.5
    end
  end

  # ---------------------------------------------------------------
  # 7. default_spot_value_function
  # ---------------------------------------------------------------

  describe "default_spot_value_function/4" do
    test "returns callable" do
      obs = [100.0, 150.0, 100.0]
      cf = flat_counterfactual(100.0, 1.0, 0.5, 3)
      spots = [make_spot("a", 1, 2)]

      vf = SpotAttributor.default_spot_value_function(obs, cf, spots)
      assert is_function(vf, 1)
    end

    test "empty coalition returns 0.0" do
      obs = [100.0, 150.0, 100.0]
      cf = flat_counterfactual(100.0, 1.0, 0.5, 3)
      spots = [make_spot("a", 1, 2)]

      vf = SpotAttributor.default_spot_value_function(obs, cf, spots)
      assert vf.([]) == 0.0
    end

    test "single spot returns individual lift" do
      obs = [100.0, 150.0, 100.0]
      cf = flat_counterfactual(100.0, 1.0, 0.5, 3)
      spots = [make_spot("a", 1, 2)]

      vf = SpotAttributor.default_spot_value_function(obs, cf, spots)
      individual = SpotAttributor.compute_window_lift(obs, cf, 1, 2)

      assert_in_delta vf.(["a"]), individual.lift, 1.0e-10
    end

    test "coalition returns union lift" do
      obs = [100.0, 150.0, 160.0, 170.0, 100.0]
      cf = flat_counterfactual(100.0, 1.0, 0.5, 5)

      spots = [make_spot("a", 1, 3), make_spot("b", 2, 4)]

      vf = SpotAttributor.default_spot_value_function(obs, cf, spots)
      coalition = SpotAttributor.compute_coalition_lift(obs, cf, spots)

      assert_in_delta vf.(["a", "b"]), coalition.lift, 1.0e-10
    end
  end

  # ---------------------------------------------------------------
  # Property-based tests
  # ---------------------------------------------------------------

  describe "property: total lift = sum of per-spot lifts (efficiency)" do
    property "allocations always sum to total_lift" do
      check all(
              n <- StreamData.integer(1..4),
              max_runs: 20
            ) do
        t = 20

        # Non-overlapping spots
        spots =
          Enum.map(0..(n - 1), fn i ->
            start = i * 4
            make_spot("s#{i}", start, min(start + 3, t))
          end)

        obs = Enum.map(1..t, fn i -> 100.0 + i * 1.0 end)
        cf = flat_counterfactual(100.0, 1.0, 0.5, t)

        result = SpotAttributor.attribute(obs, spots, cf)
        sum_of_lifts = Enum.reduce(result.attributions, 0.0, fn a, acc -> acc + a.lift end)

        assert_in_delta result.total_lift, sum_of_lifts, 1.0e-6
      end
    end
  end

  describe "property: p_positive in [0, 1]" do
    property "always returns value in [0, 1]" do
      check all(
              mean <- StreamData.float(min: -1000.0, max: 1000.0),
              variance <- StreamData.float(min: 0.001, max: 1000.0),
              max_runs: 50
            ) do
        p = SpotAttributor.p_positive(mean, variance)
        assert p >= 0.0 and p <= 1.0
      end
    end
  end

  describe "property: symmetric spots get equal allocations" do
    property "identical spots receive equal lift" do
      check all(
              n <- StreamData.integer(2..4),
              max_runs: 10
            ) do
        obs = List.duplicate(200.0, 10)
        cf = flat_counterfactual(100.0, 1.0, 0.5, 10)

        # All spots have identical windows
        spots = Enum.map(1..n, fn i -> make_spot("s#{i}", 2, 5) end)
        result = SpotAttributor.attribute(obs, spots, cf)

        lifts = Enum.map(result.attributions, & &1.lift)
        expected = hd(lifts)

        Enum.each(lifts, fn l ->
          assert_in_delta l, expected, 1.0e-6
        end)
      end
    end
  end

  describe "property: window lift variance is non-negative" do
    property "variance is always >= 0" do
      check all(
              ws <- StreamData.integer(0..8),
              we <- StreamData.integer(0..10),
              max_runs: 30
            ) do
        # Ensure we >= ws for a valid window
        {ws, we} = {min(ws, we), max(ws, we)}
        t = max(we, 1)
        obs = List.duplicate(100.0, t)
        cf = flat_counterfactual(100.0, 1.0, 0.5, t)

        result = SpotAttributor.compute_window_lift(obs, cf, ws, we)
        assert result.variance >= 0.0
      end
    end
  end

  # ---------------------------------------------------------------
  # Integration test
  # ---------------------------------------------------------------

  describe "integration: synthetic spike detection" do
    test "detects injected spike during spot window" do
      # Build a synthetic time series: flat baseline with a known spike
      t = 100
      baseline_mean = 100.0
      spike_magnitude = 50.0
      spike_start = 40
      spike_end = 60

      # Observations: baseline + spike during window
      observations =
        Enum.map(0..(t - 1), fn i ->
          if i >= spike_start and i < spike_end do
            baseline_mean + spike_magnitude
          else
            baseline_mean
          end
        end)

      counterfactual = %{
        mean: List.duplicate(baseline_mean, t),
        variance: List.duplicate(0.1, t),
        obs_variance: 0.1
      }

      spots = [make_spot("spike", spike_start, spike_end)]
      result = SpotAttributor.attribute(observations, spots, counterfactual)

      [attr] = result.attributions

      # Lift should be close to spike_magnitude × window_length = 50 × 20 = 1000
      expected_total_lift = spike_magnitude * (spike_end - spike_start)
      assert_in_delta attr.lift, expected_total_lift, 1.0e-6

      # CI should cover the true value
      assert attr.lift_lower <= expected_total_lift
      assert attr.lift_upper >= expected_total_lift

      # P(positive) should be very high for a clear spike
      assert attr.p_positive > 0.999
    end

    test "two spots with different spike magnitudes" do
      t = 100
      baseline_mean = 100.0

      observations =
        Enum.map(0..(t - 1), fn i ->
          cond do
            i >= 10 and i < 20 -> baseline_mean + 30.0
            i >= 50 and i < 60 -> baseline_mean + 80.0
            true -> baseline_mean
          end
        end)

      counterfactual = %{
        mean: List.duplicate(baseline_mean, t),
        variance: List.duplicate(0.1, t),
        obs_variance: 0.1
      }

      spots = [make_spot("small", 10, 20), make_spot("large", 50, 60)]
      result = SpotAttributor.attribute(observations, spots, counterfactual)

      lifts = Map.new(result.attributions, fn a -> {a.spot_id, a.lift} end)

      # small: 30 × 10 = 300, large: 80 × 10 = 800
      assert_in_delta lifts["small"], 300.0, 1.0e-6
      assert_in_delta lifts["large"], 800.0, 1.0e-6
      assert_in_delta result.total_lift, 1100.0, 1.0e-6
    end
  end
end
