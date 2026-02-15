defmodule BstsNx.SeasonalComponentTest do
  use ExUnit.Case, async: true

  alias BstsNx.Components
  alias BstsNx.GibbsSampler
  alias BstsNx.ModelSpec

  # ── Raw component: seasonal/2 ──────────────────────────────────────────

  describe "seasonal/2" do
    test "S=7 produces correct dimensions" do
      c = Components.seasonal(7, 0.1)
      assert Nx.shape(c.f) == {6, 6}
      assert Nx.shape(c.q) == {6, 6}
      assert Nx.shape(c.h) == {1, 6}
    end

    test "S=4 produces correct dimensions" do
      c = Components.seasonal(4, 0.5)
      assert Nx.shape(c.f) == {3, 3}
      assert Nx.shape(c.q) == {3, 3}
      assert Nx.shape(c.h) == {1, 3}
    end

    test "S=2 (minimum) produces 1×1 matrices" do
      c = Components.seasonal(2, 1.0)
      assert Nx.shape(c.f) == {1, 1}
      assert Nx.shape(c.q) == {1, 1}
      assert Nx.shape(c.h) == {1, 1}
    end

    test "transition matrix has -1 in first row" do
      c = Components.seasonal(7, 0.1)
      first_row = Nx.to_flat_list(c.f[0])
      assert Enum.all?(first_row, fn v -> v == -1.0 end)
    end

    test "transition matrix has identity shift below first row" do
      c = Components.seasonal(5, 0.1)
      # dim = 4, shift is 3×4 with I_3 in columns 0..2
      f = c.f

      # Row 1: [1, 0, 0, 0]
      assert Nx.to_number(f[1][0]) == 1.0
      assert Nx.to_number(f[1][1]) == 0.0
      assert Nx.to_number(f[1][2]) == 0.0
      assert Nx.to_number(f[1][3]) == 0.0

      # Row 2: [0, 1, 0, 0]
      assert Nx.to_number(f[2][0]) == 0.0
      assert Nx.to_number(f[2][1]) == 1.0
      assert Nx.to_number(f[2][2]) == 0.0
      assert Nx.to_number(f[2][3]) == 0.0

      # Row 3: [0, 0, 1, 0]
      assert Nx.to_number(f[3][0]) == 0.0
      assert Nx.to_number(f[3][1]) == 0.0
      assert Nx.to_number(f[3][2]) == 1.0
      assert Nx.to_number(f[3][3]) == 0.0
    end

    test "Q has process variance only on first diagonal" do
      c = Components.seasonal(7, 0.3)
      q = c.q
      # Q[0,0] = 0.3
      assert_in_delta Nx.to_number(q[0][0]), 0.3, 1.0e-6
      # All other diagonal entries = 0
      for i <- 1..5 do
        assert Nx.to_number(q[i][i]) == 0.0
      end
    end

    test "H observes only the first state" do
      c = Components.seasonal(7, 0.1)
      h = c.h
      assert Nx.to_number(h[0][0]) == 1.0

      for j <- 1..5 do
        assert Nx.to_number(h[0][j]) == 0.0
      end
    end

    test "S=2 transition is [[-1]]" do
      c = Components.seasonal(2, 0.5)
      assert Nx.to_number(c.f[0][0]) == -1.0
    end

    test "raises for num_seasons < 2" do
      assert_raise ArgumentError, ~r/num_seasons must be an integer >= 2/, fn ->
        Components.seasonal(1, 0.1)
      end
    end
  end

  # ── ModelSpec: seasonal_spec/2 ──────────────────────────────────────────

  describe "seasonal_spec/2" do
    test "returns a ModelSpec struct with correct dimensions" do
      spec = Components.seasonal_spec(7)
      assert %ModelSpec{} = spec
      assert Nx.shape(spec.f) == {6, 6}
      assert Nx.shape(spec.h) == {1, 6}
      assert Nx.shape(spec.x0) == {6}
      assert Nx.shape(spec.p0) == {6, 6}
    end

    test "has exactly one q_spec for the seasonal innovation" do
      spec = Components.seasonal_spec(7)
      assert length(spec.q_specs) == 1
      qs = hd(spec.q_specs)
      assert qs.dim_index == 0
    end

    test "respects custom options" do
      spec = Components.seasonal_spec(4, process_var: 0.5, obs_var: 2.0, initial_cov: 3.0)
      assert hd(spec.q_specs).initial == 0.5
      assert spec.obs_var == 2.0
      # p0 diagonal should be 3.0
      assert_in_delta Nx.to_number(spec.p0[0][0]), 3.0, 1.0e-6
      assert_in_delta Nx.to_number(spec.p0[1][1]), 3.0, 1.0e-6
    end

    test "raises for num_seasons < 2" do
      assert_raise ArgumentError, ~r/num_seasons must be an integer >= 2/, fn ->
        Components.seasonal_spec(1)
      end
    end

    test "x0 is all zeros" do
      spec = Components.seasonal_spec(7)
      zeros = Nx.broadcast(0.0, {6})
      assert Nx.to_flat_list(spec.x0) == Nx.to_flat_list(zeros)
    end
  end

  # ── Composition with other components ──────────────────────────────────

  describe "compose_specs with seasonal" do
    test "local_level + seasonal produces combined spec" do
      ll = Components.local_level_spec()
      ss = Components.seasonal_spec(7)
      combined = Components.compose_specs(ll, ss)

      # 1 (level) + 6 (seasonal) = 7 state dims
      assert Nx.shape(combined.f) == {7, 7}
      assert Nx.shape(combined.x0) == {7}
      assert Nx.shape(combined.p0) == {7, 7}

      # H: {1, 7}
      assert Nx.shape(combined.h) == {1, 7}

      # q_specs: 1 (level) + 1 (seasonal) = 2
      assert length(combined.q_specs) == 2
      # Level at dim 0, seasonal at dim 1
      assert Enum.at(combined.q_specs, 0).dim_index == 0
      assert Enum.at(combined.q_specs, 1).dim_index == 1
    end

    test "local_linear_trend + seasonal produces combined spec" do
      llt = Components.local_linear_trend_spec(var_level: 0.5, var_slope: 0.05)
      ss = Components.seasonal_spec(7, process_var: 0.2)
      combined = Components.compose_specs(llt, ss)

      # 2 (trend) + 6 (seasonal) = 8 state dims
      assert Nx.shape(combined.f) == {8, 8}
      assert Nx.shape(combined.x0) == {8}

      # q_specs: 2 (trend) + 1 (seasonal) = 3
      assert length(combined.q_specs) == 3
      # Seasonal dim_index should be shifted by 2
      seasonal_qs = Enum.at(combined.q_specs, 2)
      assert seasonal_qs.dim_index == 2
      assert seasonal_qs.initial == 0.2
    end
  end

  # ── Seasonal constraint: effects sum to ~0 ─────────────────────────────

  describe "seasonal effect constraint" do
    test "repeated transition application cycles through seasons summing to ~0" do
      # With no noise (process_var = 0), seasonal effects must exactly sum to 0
      # Start with known seasonal effects for S=4 (dim=3)
      # e.g., effects [2.0, -1.0, 0.0] → next = -(2 + -1 + 0) = -1.0
      # Full cycle: [2.0, -1.0, 0.0, -1.0] → sum = 0.0
      c = Components.seasonal(4, 0.0)
      f = c.f

      # Initial state: [2.0, -1.0, 0.0]
      x = Nx.tensor([2.0, -1.0, 0.0])

      # Evolve through a full cycle (S=4 steps)
      effects = [Nx.to_number(x[0])]

      {_, all_effects} =
        Enum.reduce(1..3, {x, effects}, fn _step, {state, effs} ->
          next_state = Nx.dot(f, state)
          {next_state, effs ++ [Nx.to_number(next_state[0])]}
        end)

      # Full cycle should sum to 0
      cycle_sum = Enum.sum(all_effects)
      assert_in_delta cycle_sum, 0.0, 1.0e-6
    end
  end

  # ── Raw component: compose with StateSpace ─────────────────────────────

  describe "seasonal/2 with StateSpace.compose" do
    test "composes with local_level via StateSpace" do
      ll = Components.local_level(0.5)
      ss = Components.seasonal(7, 0.1)
      combined = BstsNx.StateSpace.compose(ll, ss)

      # 1 + 6 = 7
      assert Nx.shape(combined.f) == {7, 7}
      assert Nx.shape(combined.q) == {7, 7}
      assert Nx.shape(combined.h) == {1, 7}
    end
  end

  # ── Integration with Kalman filter ─────────────────────────────────────

  describe "Kalman filter integration" do
    test "can run Kalman filter on seasonal data" do
      # Generate synthetic data with S=4 quarterly pattern
      :rand.seed(:exsss, {400, 401, 402})
      pattern = [10.0, 5.0, -3.0, -12.0]

      obs =
        Enum.flat_map(1..5, fn _cycle ->
          Enum.map(pattern, fn p -> p + :rand.normal() * 0.5 end)
        end)

      spec = Components.seasonal_spec(4, process_var: 0.1, obs_var: 0.5)

      {filtered, _predicted} =
        BstsNx.KalmanFilter.filter_with_pred(
          obs,
          spec.f,
          spec.h,
          build_q(spec),
          spec.obs_var,
          spec.x0,
          spec.p0
        )

      assert length(filtered) == 20

      # Filtered states should be vectors of dimension 3 (S-1)
      {state, _cov} = hd(filtered)
      assert Nx.size(Nx.flatten(state)) == 3
    end
  end

  # ── Integration with structured Gibbs sampler ──────────────────────────

  describe "Gibbs sampler integration" do
    test "seasonal_spec runs through sample_structured" do
      :rand.seed(:exsss, {500, 501, 502})
      pattern = [8.0, 2.0, -4.0, -6.0]

      obs =
        Enum.flat_map(1..8, fn _cycle ->
          Enum.map(pattern, fn p -> p + :rand.normal() * 0.5 end)
        end)

      spec = Components.seasonal_spec(4, process_var: 0.1, obs_var: 0.5, initial_cov: 10.0)

      samples = GibbsSampler.sample_structured(obs, spec, 5, burn_in: 3, seed: 42)

      assert length(samples) == 5

      Enum.each(samples, fn s ->
        assert is_list(s.states)
        assert length(s.states) == 32

        # Each state should be a vector of dim 3
        Enum.each(s.states, fn st ->
          assert Nx.size(Nx.flatten(st)) == 3
        end)

        # Q matrix: {3, 3}
        assert Nx.shape(s.q_matrix) == {3, 3}
        # Only Q[0,0] should be positive (resampled); others should stay 0
        assert Nx.to_number(s.q_matrix[0][0]) > 0.0
        assert Nx.to_number(s.q_matrix[1][1]) == 0.0
        assert Nx.to_number(s.q_matrix[2][2]) == 0.0

        assert Nx.to_number(s.obs_var) > 0.0
      end)
    end

    test "composed local_level + seasonal runs through sample_structured" do
      :rand.seed(:exsss, {600, 601, 602})
      # Generate: level ~100, quarterly seasonal pattern
      pattern = [5.0, 2.0, -3.0, -4.0]

      obs =
        Enum.flat_map(1..6, fn _cycle ->
          Enum.map(pattern, fn p -> 100.0 + p + :rand.normal() * 1.0 end)
        end)

      ll = Components.local_level_spec(process_var: 1.0, obs_var: 1.0, initial_state: 100.0)
      ss = Components.seasonal_spec(4, process_var: 0.1)
      combined = Components.compose_specs(ll, ss)

      samples = GibbsSampler.sample_structured(obs, combined, 3, burn_in: 2, seed: 99)

      assert length(samples) == 3

      Enum.each(samples, fn s ->
        assert length(s.states) == 24

        # State dimension: 1 (level) + 3 (seasonal) = 4
        Enum.each(s.states, fn st ->
          assert Nx.size(Nx.flatten(st)) == 4
        end)

        # Q matrix: {4, 4}
        assert Nx.shape(s.q_matrix) == {4, 4}
      end)
    end
  end

  # ── Helpers ─────────────────────────────────────────────────────────────

  # Build Q matrix from spec for Kalman filter (which expects a matrix, not q_specs)
  defp build_q(spec) do
    n = Nx.axis_size(spec.f, 0)
    diag = List.duplicate(0.0, n)

    diag =
      Enum.reduce(spec.q_specs, diag, fn qs, d ->
        List.replace_at(d, qs.dim_index, qs.initial)
      end)

    Nx.tensor(diag) |> Nx.make_diagonal()
  end
end
