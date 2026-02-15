defmodule BstsNx.ShapleyTimeDecayTest do
  use ExUnit.Case, async: true

  alias BstsNx.ShapleyAllocator

  # ---------------------------------------------------------------
  # Helpers
  # ---------------------------------------------------------------

  defp make_spot(id, ws, we), do: %{id: id, window_start: ws, window_end: we}

  # ---------------------------------------------------------------
  # time_weighted_value_function/3
  # ---------------------------------------------------------------

  describe "time_weighted_value_function/3" do
    test "empty coalition returns 0.0" do
      lifts = %{"a" => 100.0, "b" => 80.0}
      times = %{"a" => 0, "b" => 7}
      vf = ShapleyAllocator.time_weighted_value_function(lifts, times)

      assert vf.([]) == 0.0
    end

    test "single spot returns its full lift (no decay relative to self)" do
      lifts = %{"a" => 100.0, "b" => 80.0}
      times = %{"a" => 0, "b" => 7}
      vf = ShapleyAllocator.time_weighted_value_function(lifts, times)

      assert_in_delta vf.(["a"]), 100.0, 1.0e-10
      assert_in_delta vf.(["b"]), 80.0, 1.0e-10
    end

    test "two spots at different times — recent spot gets higher weight" do
      lifts = %{"old" => 100.0, "new" => 100.0}
      times = %{"old" => 0, "new" => 14}
      vf = ShapleyAllocator.time_weighted_value_function(lifts, times, half_life: 7.0)

      # t_max = 14, old is at t=0 so decay = exp(-ln(2)/7 * 14) = exp(-2*ln2) = 0.25
      # new is at t=14 so decay = exp(0) = 1.0
      # v = 100*0.25 + 100*1.0 = 125.0
      val = vf.(["new", "old"])
      assert_in_delta val, 125.0, 1.0e-10

      # The old spot contributes less due to decay
      old_contribution = 100.0 * :math.exp(-:math.log(2) / 7.0 * 14.0)
      new_contribution = 100.0 * 1.0
      assert new_contribution > old_contribution
    end

    test "half-life decay: spot exactly one half-life back gets ~0.5x weight" do
      lifts = %{"recent" => 100.0, "older" => 100.0}
      times = %{"recent" => 14, "older" => 7}
      vf = ShapleyAllocator.time_weighted_value_function(lifts, times, half_life: 7.0)

      # t_max = 14, older at t=7, delta = 7 (one half-life)
      # weight for older = exp(-ln(2)/7 * 7) = exp(-ln(2)) = 0.5
      # v = 100*1.0 + 100*0.5 = 150.0
      val = vf.(["older", "recent"])
      assert_in_delta val, 150.0, 1.0e-10

      # Verify the older spot contributes exactly half
      single_recent = vf.(["recent"])
      single_older = vf.(["older"])
      assert_in_delta single_recent, 100.0, 1.0e-10
      assert_in_delta single_older, 100.0, 1.0e-10

      # In coalition, older is discounted by 0.5
      expected_older_contribution = 100.0 * 0.5
      expected_total = 100.0 + expected_older_contribution
      assert_in_delta val, expected_total, 1.0e-10
    end

    test "two spots at same time get equal weights (no decay)" do
      lifts = %{"a" => 100.0, "b" => 80.0}
      times = %{"a" => 5, "b" => 5}
      vf = ShapleyAllocator.time_weighted_value_function(lifts, times)

      # Both at same time => both have weight = exp(0) = 1.0
      val = vf.(["a", "b"])
      assert_in_delta val, 180.0, 1.0e-10
    end

    test "custom half_life option works" do
      lifts = %{"a" => 100.0, "b" => 100.0}
      times = %{"a" => 0, "b" => 3}

      vf_short = ShapleyAllocator.time_weighted_value_function(lifts, times, half_life: 3.0)
      vf_long = ShapleyAllocator.time_weighted_value_function(lifts, times, half_life: 30.0)

      val_short = vf_short.(["a", "b"])
      val_long = vf_long.(["a", "b"])

      # With half_life=3, spot "a" is exactly one half-life back: weight = 0.5
      # v = 100*0.5 + 100*1.0 = 150.0
      assert_in_delta val_short, 150.0, 1.0e-10

      # With half_life=30, spot "a" is 3/30 = 0.1 half-lives back
      # weight = exp(-ln(2)/30 * 3) = exp(-ln(2)*0.1) ≈ 0.9330
      expected_weight = :math.exp(-:math.log(2) / 30.0 * 3.0)
      assert_in_delta val_long, 100.0 * expected_weight + 100.0, 1.0e-10

      # Longer half-life means less decay, so val_long > val_short
      assert val_long > val_short
    end

    test "integration with ShapleyAllocator.allocate/4 — allocations sum to coalition lift" do
      lifts = %{"a" => 120.0, "b" => 80.0, "c" => 60.0}
      times = %{"a" => 0, "b" => 3, "c" => 7}
      spots = [make_spot("a", 0, 10), make_spot("b", 3, 13), make_spot("c", 7, 17)]

      vf = ShapleyAllocator.time_weighted_value_function(lifts, times, half_life: 7.0)
      coalition_lift = 200.0

      result = ShapleyAllocator.allocate(spots, coalition_lift, vf)
      total = result |> Map.values() |> Enum.sum()

      assert_in_delta total, coalition_lift, 1.0e-10
      assert map_size(result) == 3
      assert Map.has_key?(result, "a")
      assert Map.has_key?(result, "b")
      assert Map.has_key?(result, "c")
    end

    test "works with ShapleyAllocator.exact_shapley/2" do
      lifts = %{"a" => 100.0, "b" => 100.0}
      times = %{"a" => 0, "b" => 7}
      spots = [make_spot("a", 0, 10), make_spot("b", 5, 15)]

      vf = ShapleyAllocator.time_weighted_value_function(lifts, times, half_life: 7.0)

      result = ShapleyAllocator.exact_shapley(spots, vf)

      # Efficiency: Shapley values sum to v(N)
      grand_coalition_value = vf.(["a", "b"])
      total = result |> Map.values() |> Enum.sum()
      assert_in_delta total, grand_coalition_value, 1.0e-10

      # With equal lifts, both single-spot values are 100.0 and
      # both marginals of adding either to the other are 50.0,
      # so Shapley values are equal: phi_a = phi_b = 75.0
      assert_in_delta result["a"], 75.0, 1.0e-10
      assert_in_delta result["b"], 75.0, 1.0e-10
    end

    test "unknown spot ID contributes zero lift" do
      lifts = %{"a" => 100.0}
      times = %{"a" => 5}
      vf = ShapleyAllocator.time_weighted_value_function(lifts, times)

      # "unknown" has default lift 0.0 and default time 0
      val = vf.(["a", "unknown"])
      assert_in_delta val, 100.0, 1.0e-10
    end

    test "default half_life is 7.0" do
      lifts = %{"a" => 100.0, "b" => 100.0}
      times = %{"a" => 0, "b" => 7}

      vf_default = ShapleyAllocator.time_weighted_value_function(lifts, times)
      vf_explicit = ShapleyAllocator.time_weighted_value_function(lifts, times, half_life: 7.0)

      val_default = vf_default.(["a", "b"])
      val_explicit = vf_explicit.(["a", "b"])

      assert_in_delta val_default, val_explicit, 1.0e-10
    end
  end
end
