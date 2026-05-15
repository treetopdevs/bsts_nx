defmodule BstsNx.ShapleyAllocatorTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias BstsNx.ShapleyAllocator

  # ---------------------------------------------------------------
  # Helpers
  # ---------------------------------------------------------------

  defp make_spot(id, ws, we), do: %{id: id, window_start: ws, window_end: we}

  defp symmetric_value_fn do
    fn ids -> length(ids) * 50.0 end
  end

  # Classic 3-player game:
  #   v({}) = 0, v({i}) = 0 for all i
  #   v({1,2}) = 90, v({1,3}) = 80, v({2,3}) = 70
  #   v({1,2,3}) = 120
  # Hand-computed Shapley values: φ₁ = 45.0, φ₂ = 40.0, φ₃ = 35.0
  defp three_player_value_fn do
    fn
      [] -> 0.0
      ["1"] -> 0.0
      ["2"] -> 0.0
      ["3"] -> 0.0
      ["1", "2"] -> 90.0
      ["1", "3"] -> 80.0
      ["2", "3"] -> 70.0
      ["1", "2", "3"] -> 120.0
    end
  end

  # ---------------------------------------------------------------
  # 1. detect_overlaps
  # ---------------------------------------------------------------

  describe "detect_overlaps/1" do
    test "empty input returns empty list" do
      assert ShapleyAllocator.detect_overlaps([]) == []
    end

    test "non-overlapping spots become singletons" do
      spots = [
        make_spot("a", 0, 10),
        make_spot("b", 20, 30),
        make_spot("c", 40, 50)
      ]

      groups = ShapleyAllocator.detect_overlaps(spots)

      assert length(groups) == 3
      assert Enum.all?(groups, &(length(&1) == 1))
    end

    test "fully overlapping spots form one group" do
      spots = [
        make_spot("a", 0, 100),
        make_spot("b", 10, 90),
        make_spot("c", 20, 80)
      ]

      groups = ShapleyAllocator.detect_overlaps(spots)

      assert length(groups) == 1
      assert length(hd(groups)) == 3
    end

    test "partial overlaps split correctly" do
      spots = [
        make_spot("a", 0, 10),
        make_spot("b", 5, 15),
        make_spot("c", 30, 40),
        make_spot("d", 35, 45)
      ]

      groups = ShapleyAllocator.detect_overlaps(spots)

      assert length(groups) == 2
      group_ids = Enum.map(groups, fn g -> Enum.map(g, & &1.id) |> Enum.sort() end)
      assert Enum.sort(group_ids) == [["a", "b"], ["c", "d"]]
    end

    test "transitive overlap merges A↔B, B↔C into one group" do
      spots = [
        make_spot("a", 0, 10),
        make_spot("b", 5, 20),
        make_spot("c", 15, 30)
      ]

      groups = ShapleyAllocator.detect_overlaps(spots)

      assert length(groups) == 1
      ids = hd(groups) |> Enum.map(& &1.id) |> Enum.sort()
      assert ids == ["a", "b", "c"]
    end

    test "touching boundaries are separate groups (strict < comparison)" do
      spots = [
        make_spot("a", 0, 10),
        make_spot("b", 10, 20)
      ]

      groups = ShapleyAllocator.detect_overlaps(spots)

      assert length(groups) == 2
    end

    test "unsorted input is handled correctly" do
      spots = [
        make_spot("c", 20, 30),
        make_spot("a", 0, 10),
        make_spot("b", 5, 15)
      ]

      groups = ShapleyAllocator.detect_overlaps(spots)

      assert length(groups) == 2
      # a and b overlap, c is separate
      group_ids = Enum.map(groups, fn g -> Enum.map(g, & &1.id) |> Enum.sort() end)
      assert Enum.sort(group_ids) == [["a", "b"], ["c"]]
    end

    test "identical timestamps form one group" do
      spots = [
        make_spot("a", 5, 10),
        make_spot("b", 5, 10),
        make_spot("c", 5, 10)
      ]

      groups = ShapleyAllocator.detect_overlaps(spots)

      assert length(groups) == 1
      assert length(hd(groups)) == 3
    end
  end

  # ---------------------------------------------------------------
  # 2. exact_shapley
  # ---------------------------------------------------------------

  describe "exact_shapley/2" do
    test "single player gets full value" do
      spots = [make_spot("a", 0, 10)]

      value_fn = fn
        ["a"] -> 100.0
        [] -> 0.0
      end

      result = ShapleyAllocator.exact_shapley(spots, value_fn)

      assert_in_delta result["a"], 100.0, 1.0e-10
    end

    test "two symmetric players get equal shares" do
      spots = [make_spot("a", 0, 10), make_spot("b", 0, 10)]

      result = ShapleyAllocator.exact_shapley(spots, symmetric_value_fn())

      assert_in_delta result["a"], result["b"], 1.0e-10
    end

    test "dummy player gets zero marginal contribution" do
      spots = [make_spot("a", 0, 10), make_spot("dummy", 0, 10)]

      # dummy contributes nothing — value only depends on whether "a" is present
      value_fn = fn
        ids -> if "a" in ids, do: 100.0, else: 0.0
      end

      result = ShapleyAllocator.exact_shapley(spots, value_fn)

      assert_in_delta result["dummy"], 0.0, 1.0e-10
      assert_in_delta result["a"], 100.0, 1.0e-10
    end

    test "known 3-player hand-computed game" do
      spots = [make_spot("1", 0, 10), make_spot("2", 0, 10), make_spot("3", 0, 10)]

      result = ShapleyAllocator.exact_shapley(spots, three_player_value_fn())

      assert_in_delta result["1"], 45.0, 1.0e-10
      assert_in_delta result["2"], 40.0, 1.0e-10
      assert_in_delta result["3"], 35.0, 1.0e-10
    end

    test "efficiency axiom: sum equals v(N)" do
      spots = [make_spot("1", 0, 10), make_spot("2", 0, 10), make_spot("3", 0, 10)]
      vf = three_player_value_fn()

      result = ShapleyAllocator.exact_shapley(spots, vf)
      total = result |> Map.values() |> Enum.sum()

      # v(N) = v({1,2,3}) = 120.0
      assert_in_delta total, 120.0, 1.0e-10
    end

    test "empty spots returns empty map" do
      assert ShapleyAllocator.exact_shapley([], fn _ -> 0.0 end) == %{}
    end
  end

  # ---------------------------------------------------------------
  # 3. monte_carlo_shapley
  # ---------------------------------------------------------------

  describe "monte_carlo_shapley/4" do
    test "converges to exact for small n" do
      spots = [make_spot("1", 0, 10), make_spot("2", 0, 10), make_spot("3", 0, 10)]
      vf = three_player_value_fn()
      key = Nx.Random.key(123)

      exact = ShapleyAllocator.exact_shapley(spots, vf)
      mc = ShapleyAllocator.monte_carlo_shapley(spots, vf, 50_000, key)

      for id <- ["1", "2", "3"] do
        assert_in_delta mc[id], exact[id], 1.0, "MC for #{id} should converge to exact"
      end
    end

    test "deterministic with same key" do
      spots = [make_spot("a", 0, 10), make_spot("b", 5, 15)]
      vf = symmetric_value_fn()
      key = Nx.Random.key(99)

      r1 = ShapleyAllocator.monte_carlo_shapley(spots, vf, 500, key)
      r2 = ShapleyAllocator.monte_carlo_shapley(spots, vf, 500, key)

      assert r1 == r2
    end

    test "different keys produce different results" do
      spots = [make_spot("a", 0, 10), make_spot("b", 5, 15)]

      # Use an asymmetric value function so permutation order matters
      vf = fn
        [] -> 0.0
        ["a"] -> 80.0
        ["b"] -> 30.0
        ["a", "b"] -> 100.0
      end

      r1 = ShapleyAllocator.monte_carlo_shapley(spots, vf, 100, Nx.Random.key(1))
      r2 = ShapleyAllocator.monte_carlo_shapley(spots, vf, 100, Nx.Random.key(999))

      # With different keys, asymmetric game, and only 100 samples, results should differ
      refute r1 == r2
    end

    test "empty spots returns empty map" do
      assert ShapleyAllocator.monte_carlo_shapley([], fn _ -> 0.0 end, 100, Nx.Random.key(1)) ==
               %{}
    end
  end

  # ---------------------------------------------------------------
  # 4. allocate
  # ---------------------------------------------------------------

  describe "allocate/4" do
    test "empty spots returns empty map" do
      assert ShapleyAllocator.allocate([], 100.0, fn _ -> 0.0 end) == %{}
    end

    test "single spot gets full coalition_lift" do
      spots = [make_spot("solo", 0, 10)]
      result = ShapleyAllocator.allocate(spots, 75.0, fn _ -> 1.0 end)
      assert result == %{"solo" => 75.0}
    end

    test "result sums to coalition_lift" do
      spots = [make_spot("1", 0, 10), make_spot("2", 0, 10), make_spot("3", 0, 10)]
      vf = three_player_value_fn()

      result = ShapleyAllocator.allocate(spots, 200.0, vf)
      total = result |> Map.values() |> Enum.sum()

      assert_in_delta total, 200.0, 1.0e-10
    end

    test "uses exact for n <= 12" do
      spots = Enum.map(1..12, fn i -> make_spot("s#{i}", 0, 10) end)
      vf = symmetric_value_fn()

      # Should not raise and should produce deterministic results
      result = ShapleyAllocator.allocate(spots, 120.0, vf)

      assert map_size(result) == 12
      total = result |> Map.values() |> Enum.sum()
      assert_in_delta total, 120.0, 1.0e-10
    end

    test "uses MC for n > 12" do
      spots = Enum.map(1..13, fn i -> make_spot("s#{i}", 0, 10) end)
      vf = symmetric_value_fn()

      result = ShapleyAllocator.allocate(spots, 130.0, vf, n_samples: 1_000)

      assert map_size(result) == 13
      total = result |> Map.values() |> Enum.sum()
      assert_in_delta total, 130.0, 1.0e-10
    end

    test "negative coalition_lift produces negative allocations" do
      spots = [make_spot("a", 0, 10), make_spot("b", 5, 15)]
      vf = symmetric_value_fn()

      result = ShapleyAllocator.allocate(spots, -100.0, vf)
      total = result |> Map.values() |> Enum.sum()

      assert_in_delta total, -100.0, 1.0e-10
    end

    test "zero coalition_lift produces zero allocations" do
      spots = [make_spot("a", 0, 10), make_spot("b", 5, 15)]
      vf = symmetric_value_fn()

      result = ShapleyAllocator.allocate(spots, 0.0, vf)
      total = result |> Map.values() |> Enum.sum()

      assert_in_delta total, 0.0, 1.0e-10
    end
  end

  # ---------------------------------------------------------------
  # 5. default_value_function
  # ---------------------------------------------------------------

  describe "default_value_function/2" do
    test "empty coalition returns 0.0" do
      vf = ShapleyAllocator.default_value_function(%{"a" => 100.0})
      assert vf.([]) == 0.0
    end

    test "single player returns individual lift" do
      vf = ShapleyAllocator.default_value_function(%{"a" => 100.0, "b" => 80.0})
      assert_in_delta vf.(["a"]), 100.0, 1.0e-10
      assert_in_delta vf.(["b"]), 80.0, 1.0e-10
    end

    test "diminishing returns with default decay" do
      vf = ShapleyAllocator.default_value_function(%{"a" => 100.0, "b" => 100.0})

      # With decay=0.7: v(["a","b"]) = 100*0.7^0 + 100*0.7^1 = 100 + 70 = 170
      assert_in_delta vf.(["a", "b"]), 170.0, 1.0e-10

      # Coalition of both should be less than sum of individuals (200)
      assert vf.(["a", "b"]) < vf.(["a"]) + vf.(["b"])
    end

    test "custom decay" do
      vf = ShapleyAllocator.default_value_function(%{"a" => 100.0, "b" => 100.0}, decay: 0.5)

      # v(["a","b"]) = 100*0.5^0 + 100*0.5^1 = 100 + 50 = 150
      assert_in_delta vf.(["a", "b"]), 150.0, 1.0e-10
    end

    test "decay of 1.0 means no diminishing returns" do
      vf = ShapleyAllocator.default_value_function(%{"a" => 100.0, "b" => 80.0}, decay: 1.0)

      assert_in_delta vf.(["a", "b"]), 180.0, 1.0e-10
    end

    test "near-equal large lifts share tie weights by relative tolerance" do
      vf =
        ShapleyAllocator.default_value_function(
          %{"a" => 1.0e15, "b" => 1.0e15 + 500.0},
          decay: 0.0
        )

      assert_in_delta vf.(["a", "b"]), 1.0e15 + 250.0, 1.0
    end

    test "tiny lifts only tie when they are relatively close" do
      vf =
        ShapleyAllocator.default_value_function(
          %{"a" => 1.0e-15, "b" => 1.0e-13},
          decay: 0.0
        )

      assert_in_delta vf.(["a", "b"]), 1.0e-13, 1.0e-20
    end

    test "unknown ID contributes zero lift" do
      vf = ShapleyAllocator.default_value_function(%{"a" => 100.0})
      assert_in_delta vf.(["a", "unknown"]), 100.0 + 0.0 * 0.7, 1.0e-10
    end
  end

  describe "time_weighted_value_function/3" do
    test "anchors coalition weights to the full attribution window" do
      vf =
        ShapleyAllocator.time_weighted_value_function(
          %{"old" => 100.0, "new" => 100.0},
          %{"old" => 0, "new" => 10},
          half_life: 10.0
        )

      assert_in_delta vf.(["old"]), 50.0, 1.0e-10
      assert_in_delta vf.(["new"]), 100.0, 1.0e-10
    end
  end

  # ---------------------------------------------------------------
  # 6. normalize_to_total
  # ---------------------------------------------------------------

  describe "normalize_to_total/2" do
    test "proportional scaling" do
      result = ShapleyAllocator.normalize_to_total(%{"a" => 30.0, "b" => 70.0}, 200.0)

      assert_in_delta result["a"], 60.0, 1.0e-10
      assert_in_delta result["b"], 140.0, 1.0e-10
    end

    test "preserves relative proportions" do
      input = %{"a" => 10.0, "b" => 20.0, "c" => 30.0}
      result = ShapleyAllocator.normalize_to_total(input, 120.0)

      # Ratios should be 1:2:3
      assert_in_delta result["a"] / result["b"], 0.5, 1.0e-10
      assert_in_delta result["b"] / result["c"], 2.0 / 3.0, 1.0e-10
    end

    test "near-zero total distributes equally" do
      result =
        ShapleyAllocator.normalize_to_total(%{"a" => 1.0e-15, "b" => -1.0e-15}, 100.0)

      assert_in_delta result["a"], 50.0, 1.0e-10
      assert_in_delta result["b"], 50.0, 1.0e-10
    end

    test "empty map returns empty map" do
      assert ShapleyAllocator.normalize_to_total(%{}, 100.0) == %{}
    end

    test "negative target_total" do
      result = ShapleyAllocator.normalize_to_total(%{"a" => 60.0, "b" => 40.0}, -100.0)

      assert_in_delta result["a"], -60.0, 1.0e-10
      assert_in_delta result["b"], -40.0, 1.0e-10
    end
  end

  # ---------------------------------------------------------------
  # Property-based tests
  # ---------------------------------------------------------------

  describe "property: efficiency" do
    property "allocations always sum to coalition_lift" do
      check all(
              n <- StreamData.integer(1..6),
              lift <- StreamData.float(min: -1000.0, max: 1000.0),
              max_runs: 30
            ) do
        spots = Enum.map(1..n, fn i -> make_spot("p#{i}", 0, 10) end)
        vf = symmetric_value_fn()

        result = ShapleyAllocator.allocate(spots, lift, vf)
        total = result |> Map.values() |> Enum.sum()

        assert_in_delta total,
                        lift,
                        1.0e-6,
                        "Allocations should sum to coalition_lift #{lift}, got #{total}"
      end
    end
  end

  describe "property: symmetry" do
    property "identical spots get equal allocations" do
      check all(
              n <- StreamData.integer(2..5),
              lift <- StreamData.float(min: 1.0, max: 1000.0),
              max_runs: 30
            ) do
        spots = Enum.map(1..n, fn i -> make_spot("p#{i}", 0, 10) end)
        vf = symmetric_value_fn()

        result = ShapleyAllocator.allocate(spots, lift, vf)
        values = Map.values(result)
        expected = lift / n

        Enum.each(values, fn v ->
          assert_in_delta v,
                          expected,
                          1.0e-6,
                          "Symmetric players should each get #{expected}, got #{v}"
        end)
      end
    end
  end

  describe "property: exact and MC convergence" do
    property "MC converges to exact for small n" do
      check all(
              n <- StreamData.integer(2..5),
              max_runs: 10
            ) do
        spots = Enum.map(1..n, fn i -> make_spot("p#{i}", 0, 10) end)
        vf = symmetric_value_fn()
        key = Nx.Random.key(42)

        exact = ShapleyAllocator.exact_shapley(spots, vf)
        mc = ShapleyAllocator.monte_carlo_shapley(spots, vf, 20_000, key)

        for {id, exact_val} <- exact do
          assert_in_delta mc[id],
                          exact_val,
                          2.0,
                          "MC for #{id} should converge to exact (#{exact_val}), got #{mc[id]}"
        end
      end
    end
  end

  describe "property: detect_overlaps coverage" do
    property "groups are disjoint and cover all input spots" do
      check all(
              n <- StreamData.integer(1..20),
              max_runs: 30
            ) do
        # Generate random spots
        spots =
          Enum.map(1..n, fn i ->
            start = :rand.uniform(100)
            len = :rand.uniform(20)
            make_spot("s#{i}", start, start + len)
          end)

        groups = ShapleyAllocator.detect_overlaps(spots)

        # All spots accounted for
        all_ids_from_groups =
          groups
          |> Enum.flat_map(fn g -> Enum.map(g, & &1.id) end)
          |> Enum.sort()

        original_ids = spots |> Enum.map(& &1.id) |> Enum.sort()
        assert all_ids_from_groups == original_ids

        # No ID appears in multiple groups
        assert length(all_ids_from_groups) == length(Enum.uniq(all_ids_from_groups))
      end
    end
  end

  # ---------------------------------------------------------------
  # Edge cases
  # ---------------------------------------------------------------

  describe "edge cases" do
    test "value function always returns 0 produces equal allocation" do
      spots = [make_spot("a", 0, 10), make_spot("b", 5, 15), make_spot("c", 3, 8)]
      zero_fn = fn _ids -> 0.0 end

      result = ShapleyAllocator.allocate(spots, 90.0, zero_fn)
      total = result |> Map.values() |> Enum.sum()

      assert_in_delta total, 90.0, 1.0e-10
      # All Shapley values are 0, so normalize_to_total distributes equally
      Enum.each(Map.values(result), fn v ->
        assert_in_delta v, 30.0, 1.0e-10
      end)
    end

    test "n=12 boundary uses exact" do
      spots = Enum.map(1..12, fn i -> make_spot("s#{i}", 0, 10) end)
      vf = symmetric_value_fn()

      result = ShapleyAllocator.allocate(spots, 120.0, vf)

      # Each symmetric player should get 120/12 = 10.0
      Enum.each(Map.values(result), fn v ->
        assert_in_delta v, 10.0, 1.0e-6
      end)
    end

    test "n=13 boundary uses MC" do
      spots = Enum.map(1..13, fn i -> make_spot("s#{i}", 0, 10) end)
      vf = symmetric_value_fn()

      result = ShapleyAllocator.allocate(spots, 130.0, vf, n_samples: 5_000)

      # Each symmetric player should get ~130/13 = 10.0
      Enum.each(Map.values(result), fn v ->
        assert_in_delta v, 10.0, 2.0, "MC allocation should be near 10.0, got #{v}"
      end)
    end

    test "spots with identical timestamps" do
      spots = [
        make_spot("a", 5, 5),
        make_spot("b", 5, 5),
        make_spot("c", 5, 5)
      ]

      # All spots have same start and end — but start < end is false (start == end),
      # so each is its own group
      groups = ShapleyAllocator.detect_overlaps(spots)
      # window_start (5) is NOT < group_end (5), so separate groups
      assert length(groups) == 3
    end

    test "single spot in detect_overlaps" do
      spots = [make_spot("solo", 0, 10)]
      groups = ShapleyAllocator.detect_overlaps(spots)

      assert groups == [[make_spot("solo", 0, 10)]]
    end
  end
end
