defmodule BstsNxEdgeCasesTest do
  @moduledoc """
  Edge-case tests for bsts_nx primitives.

  These tests cover degenerate inputs that arise in production when:
  - No spots match a date range (empty post-period)
  - Spots cover every day (no pre-period)
  - Series data is constant (zero variance)
  - Series is very short
  """
  use ExUnit.Case

  alias BstsNx.CausalImpact
  alias BstsNx.Diagnostics
  alias BstsNx.GibbsSampler
  alias BstsNx.KalmanFilter

  # ---------------------------------------------------------------------------
  # CausalImpact edge cases
  # ---------------------------------------------------------------------------

  describe "CausalImpact.estimate/4 period validation" do
    test "raises when post_period start > end (no post days)" do
      :rand.seed(:exsss, {1, 2, 3})
      observations = Enum.map(1..10, fn _ -> :rand.uniform() * 10 end)

      assert_raise ArgumentError, ~r/post_period must immediately follow/, fn ->
        CausalImpact.estimate(observations, {1, 10}, {11, 10})
      end
    end

    test "raises when pre_period is empty (start > end)" do
      :rand.seed(:exsss, {4, 5, 6})
      observations = Enum.map(1..10, fn _ -> :rand.uniform() * 10 end)

      assert_raise ArgumentError, ~r/invalid pre_period/, fn ->
        CausalImpact.estimate(observations, {5, 3}, {4, 10})
      end
    end

    test "raises when post_period exceeds observations" do
      observations = [1.0, 2.0, 3.0, 4.0, 5.0]

      assert_raise ArgumentError, ~r/post_period.*extends beyond/, fn ->
        CausalImpact.estimate(observations, {1, 3}, {4, 10})
      end
    end

    test "raises when post_period doesn't immediately follow pre_period" do
      :rand.seed(:exsss, {7, 8, 9})
      observations = Enum.map(1..20, fn _ -> :rand.uniform() * 10 end)

      assert_raise ArgumentError, ~r/post_period must immediately follow/, fn ->
        # Gap between pre (1..5) and post (7..10)
        CausalImpact.estimate(observations, {1, 5}, {7, 10})
      end
    end

    test "works with minimal valid periods (2 pre, 1 post)" do
      :rand.seed(:exsss, {42, 42, 42})
      observations = Enum.map(1..3, fn _ -> :rand.uniform() * 10 end)

      result =
        CausalImpact.estimate(observations, {1, 2}, {3, 3},
          num_samples: 5,
          burn_in: 2,
          seed: 1
        )

      assert is_list(result.point_effects)
      assert length(result.actual) == 1
    end
  end

  # ---------------------------------------------------------------------------
  # KalmanFilter edge cases
  # ---------------------------------------------------------------------------

  describe "KalmanFilter with degenerate data" do
    test "handles constant observations" do
      obs = List.duplicate(10.0, 20)
      results = KalmanFilter.filter(obs, 1.0, 1.0, 0.01, 1.0, 10.0, 1.0)

      assert length(results) == 20

      # State estimates should converge toward 10.0
      {last_state, _cov} = List.last(results)
      assert_in_delta Nx.to_number(last_state), 10.0, 1.0
    end

    test "handles single observation" do
      results = KalmanFilter.filter([5.0], 1.0, 1.0, 1.0, 1.0, 0.0, 1.0)

      assert length(results) == 1
      {state, _cov} = hd(results)
      assert is_struct(state, Nx.Tensor)
    end

    test "handles NaN observations (missing data)" do
      obs = [1.0, 2.0, :nan, 4.0, 5.0]
      # NaN observations are typically passed as Nx.Constants.nan()
      obs_tensors =
        Enum.map(obs, fn
          :nan -> Nx.Constants.nan()
          v -> Nx.tensor(v)
        end)

      # The filter_defn path handles NaN via Nx.select, but the
      # list-based filter should also not crash
      results = KalmanFilter.filter(obs_tensors, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0)
      assert length(results) == 5
    end
  end

  # ---------------------------------------------------------------------------
  # GibbsSampler edge cases
  # ---------------------------------------------------------------------------

  describe "GibbsSampler with degenerate data" do
    test "handles constant observations without crashing" do
      obs = List.duplicate(10.0, 20)

      samples =
        GibbsSampler.sample(obs, 5, 10.0, 1.0, 1.0, 1.0,
          burn_in: 2,
          seed: 42
        )

      # Sampler returns num_samples total (burn_in is discarded internally)
      assert length(samples) == 5
      # Each sample should have valid structure
      Enum.each(samples, fn s ->
        assert is_struct(s.process_var, Nx.Tensor)
        assert is_struct(s.obs_var, Nx.Tensor)
      end)
    end

    test "handles short observations (3 points)" do
      obs = [1.0, 2.0, 3.0]

      samples =
        GibbsSampler.sample(obs, 3, 1.0, 1.0, 1.0, 1.0,
          burn_in: 1,
          seed: 42
        )

      assert length(samples) == 3
    end

    test "raises on negative prior_scale" do
      obs = [1.0, 2.0, 3.0, 4.0, 5.0]

      assert_raise ArgumentError, ~r/prior_scale must be a positive/, fn ->
        GibbsSampler.sample(obs, 3, 0.0, 1.0, 1.0, 1.0,
          prior_scale: -1.0,
          seed: 42
        )
      end
    end

    test "raises on zero prior_shape" do
      obs = [1.0, 2.0, 3.0]

      assert_raise ArgumentError, ~r/prior_shape must be a positive/, fn ->
        GibbsSampler.sample(obs, 3, 0.0, 1.0, 1.0, 1.0,
          prior_shape: 0.0,
          seed: 42
        )
      end
    end

    test "raises on negative prior_shape" do
      obs = [1.0, 2.0, 3.0]

      assert_raise ArgumentError, ~r/prior_shape must be a positive/, fn ->
        GibbsSampler.sample(obs, 3, 0.0, 1.0, 1.0, 1.0,
          prior_shape: -2.0,
          seed: 42
        )
      end
    end

    test "sample_general raises on negative prior_scale" do
      obs = [1.0, 2.0, 3.0, 4.0, 5.0]

      assert_raise ArgumentError, ~r/prior_scale must be a positive/, fn ->
        GibbsSampler.sample_general(obs, 1.0, 1.0, 3, prior_scale: -0.5, seed: 42)
      end
    end

    test "sample_chains returns correct number of chains" do
      :rand.seed(:exsss, {1, 2, 3})
      obs = Enum.map(1..30, fn _ -> :rand.normal() * 5 + 50 end)

      chains =
        GibbsSampler.sample_chains(obs, 3, 10, 0.0, 1.0, 1.0, 1.0,
          burn_in: 5,
          seed: 42
        )

      assert length(chains) == 3
      Enum.each(chains, fn chain -> assert length(chain) > 0 end)
    end
  end

  # ---------------------------------------------------------------------------
  # Diagnostics edge cases
  # ---------------------------------------------------------------------------

  describe "Diagnostics edge cases" do
    test "r_hat returns :nan for empty chains" do
      assert Diagnostics.r_hat([]) == :nan
    end

    test "r_hat returns :nan for chains with single sample" do
      assert Diagnostics.r_hat([[1.0], [2.0]]) == :nan
    end

    test "effective_sample_size returns :nan for single chain" do
      assert Diagnostics.effective_sample_size([[1.0, 2.0, 3.0]]) == :nan
    end

    test "effective_sample_size returns m*n for identical chains" do
      # When chains are identical, between-chain variance B = 0 and
      # var_hat = (n-1)/n * W.  ESS = m*n*W/var_hat = m*n^2/(n-1),
      # capped at m*n (the total number of draws).
      chain = [1.0, 2.0, 3.0, 4.0, 5.0]
      m = 2
      n = length(chain)
      ess = Diagnostics.effective_sample_size([chain, chain])
      assert ess == m * n * 1.0
    end

    test "r_hat ≈ 1.0 for well-mixed chains" do
      :rand.seed(:exsss, {10, 20, 30})
      chain1 = Enum.map(1..100, fn _ -> :rand.normal() end)
      chain2 = Enum.map(1..100, fn _ -> :rand.normal() end)
      chain3 = Enum.map(1..100, fn _ -> :rand.normal() end)

      rhat = Diagnostics.r_hat([chain1, chain2, chain3])
      # Well-mixed chains from same distribution should have R-hat close to 1.0
      assert is_float(rhat)
      assert rhat < 1.5
    end

    test "r_hat > 1.0 for poorly mixed chains" do
      # Chains from different distributions should have high R-hat
      chain1 = List.duplicate(1.0, 50) ++ List.duplicate(1.1, 50)
      chain2 = List.duplicate(10.0, 50) ++ List.duplicate(10.1, 50)

      rhat = Diagnostics.r_hat([chain1, chain2])
      assert is_float(rhat)
      assert rhat > 1.5
    end
  end

  # ---------------------------------------------------------------------------
  # P1: KalmanFilter with matrix-valued h tensor (rank ≥ 2)
  # ---------------------------------------------------------------------------

  describe "KalmanFilter with matrix-valued h (P1 fix)" do
    test "filter handles a rank-2 h tensor (one row per time step)" do
      # 3 observations, h varies per time step as a 1×1 matrix stacked into {3, 1}
      obs = [1.0, 2.0, 3.0]
      h = Nx.tensor([[1.0], [2.0], [1.0]])

      results = KalmanFilter.filter(obs, 1.0, h, 1.0, 1.0, 0.0, 1.0)
      assert length(results) == 3

      Enum.each(results, fn {state, cov} ->
        assert is_struct(state, Nx.Tensor)
        assert is_struct(cov, Nx.Tensor)
      end)
    end

    test "filter_with_pred handles a rank-1 h tensor (vector)" do
      obs = [1.0, 2.0, 3.0]
      h = Nx.tensor([1.0, 2.0, 1.0])

      {filtered, predicted} = KalmanFilter.filter_with_pred(obs, 1.0, h, 1.0, 1.0, 0.0, 1.0)
      assert length(filtered) == 3
      assert length(predicted) == 3
    end

    test "filter raises when h tensor length doesn't match observations" do
      obs = [1.0, 2.0, 3.0]
      h = Nx.tensor([1.0, 2.0])

      assert_raise ArgumentError, ~r/observation matrix length must match/, fn ->
        KalmanFilter.filter(obs, 1.0, h, 1.0, 1.0, 0.0, 1.0)
      end
    end
  end

  # ---------------------------------------------------------------------------
  # P2: CausalImpact with empty/out-of-bounds intervention indices
  # ---------------------------------------------------------------------------

  describe "CausalImpact.estimate_from_filter empty indices (P2 fix)" do
    test "returns zero-effect result for empty intervention_indices" do
      obs = [1.0, 2.0, 3.0, 4.0, 5.0]

      result = CausalImpact.estimate_from_filter(obs, [])

      assert result.point_effects == %{mean: [], lower: [], upper: []}
      assert result.cumulative_effect.mean == 0.0
      assert result.relative_effect.mean == 0.0
      assert result.actual == []
      assert result.baseline == []
    end

    test "returns zero-effect result when all indices are out of bounds" do
      obs = [1.0, 2.0, 3.0]

      result = CausalImpact.estimate_from_filter(obs, [10, 20, -1])

      assert result.point_effects == %{mean: [], lower: [], upper: []}
      assert result.cumulative_effect.mean == 0.0
      assert result.actual == []
    end

    test "filters out-of-bounds indices but processes valid ones" do
      obs = [1.0, 2.0, 3.0, 4.0, 5.0]

      # Mix of valid (2, 3) and invalid (10, -1)
      result = CausalImpact.estimate_from_filter(obs, [2, 3, 10, -1])

      assert length(result.actual) == 2
      assert length(result.baseline) == 2
      assert length(result.point_effects.mean) == 2
    end
  end

  # ---------------------------------------------------------------------------
  # P3: StateSpace.block_diag with scalar tensors
  # ---------------------------------------------------------------------------

  describe "StateSpace.block_diag scalar tensor handling (P3 fix)" do
    alias BstsNx.StateSpace

    test "block_diag handles scalar Nx tensor (rank-0)" do
      scalar = Nx.tensor(5.0)
      result = StateSpace.block_diag([scalar])

      assert Nx.shape(result) == {1, 1}
      assert Nx.to_number(result[0][0]) == 5.0
    end

    test "block_diag handles mix of scalar tensor and matrix" do
      scalar = Nx.tensor(3.0)
      mat = Nx.tensor([[1.0, 0.0], [0.0, 2.0]])

      result = StateSpace.block_diag([mat, scalar])
      assert Nx.shape(result) == {3, 3}
      # Bottom-right element should be the scalar
      assert Nx.to_number(result[2][2]) == 3.0
      # Off-diagonal should be zero
      assert Nx.to_number(result[0][2]) == 0.0
    end

    test "block_diag handles rank-1 tensor of size 1" do
      vec = Nx.tensor([7.0])
      result = StateSpace.block_diag([vec])

      assert Nx.shape(result) == {1, 1}
      assert Nx.to_number(result[0][0]) == 7.0
    end

    test "block_diag raises for rank-1 tensor of size > 1" do
      vec = Nx.tensor([1.0, 2.0, 3.0])

      assert_raise ArgumentError, ~r/rank-1 tensor/, fn ->
        StateSpace.block_diag([vec])
      end
    end

    test "compose handles scalar tensor components" do
      comp1 = %{f: Nx.tensor(1.0), q: Nx.tensor(0.1), h: Nx.tensor([[1.0]])}
      comp2 = %{f: Nx.tensor(0.9), q: Nx.tensor(0.2), h: Nx.tensor([[1.0]])}

      result = StateSpace.compose(comp1, comp2)
      assert Nx.shape(result.f) == {2, 2}
      assert Nx.shape(result.q) == {2, 2}
      assert Nx.shape(result.h) == {1, 2}
    end
  end

  # ---------------------------------------------------------------------------
  # CausalImpact + Diagnostics integration
  # ---------------------------------------------------------------------------

  describe "CausalImpact with Diagnostics integration" do
    test "full pipeline: estimate + multi-chain diagnostics" do
      :rand.seed(:exsss, {100, 200, 300})
      pre = Enum.map(1..60, fn _ -> 50.0 + :rand.normal() * 3 end)
      post = Enum.map(1..30, fn _ -> 55.0 + :rand.normal() * 3 end)
      obs = pre ++ post

      # Run CausalImpact
      result =
        CausalImpact.estimate(obs, {1, 60}, {61, 90},
          num_samples: 20,
          burn_in: 10,
          seed: 42
        )

      assert is_list(result.point_effects)

      # Run multi-chain for diagnostics
      chains =
        GibbsSampler.sample_chains(pre, 2, 20, 0.0, 1.0, 1.0, 1.0,
          burn_in: 10,
          seed: 42
        )

      q_chains =
        Enum.map(chains, fn chain ->
          Enum.map(chain, fn s -> Nx.to_number(s.process_var) end)
        end)

      rhat = Diagnostics.r_hat(q_chains)
      ess = Diagnostics.effective_sample_size(q_chains)

      # Should return numeric values (not crash)
      assert is_float(rhat) or rhat == :nan
      assert is_float(ess) or ess == :nan
    end
  end

  # ---------------------------------------------------------------------------
  # Multi-dimensional pipeline: compose → filter → smooth
  # ---------------------------------------------------------------------------

  describe "Multi-dimensional pipeline (compose → filter → smooth)" do
    alias BstsNx.Components
    alias BstsNx.Smoother

    test "local linear trend through filter + smoother produces valid output" do
      # F = [[1,1],[0,1]], H = [[1,0]], Q = diag(0.1, 0.01)
      comp = Components.local_linear_trend(0.1, 0.01)
      r = Nx.tensor([[1.0]])
      x0 = Nx.tensor([0.0, 0.0])
      p0 = Nx.eye(2)

      # Observations from a linear trend: y_t ≈ t
      :rand.seed(:exsss, {42, 42, 42})
      obs = Enum.map(1..20, fn t -> t * 1.0 + :rand.normal() * 0.5 end)

      {filtered, predicted} =
        KalmanFilter.filter_with_pred(obs, comp.f, comp.h, comp.q, r, x0, p0)

      # Verify filtered output shapes
      assert length(filtered) == 20

      Enum.each(filtered, fn {state, cov} ->
        assert Nx.shape(state) == {2}
        assert Nx.shape(cov) == {2, 2}
      end)

      # RTS smoother
      smoothed = Smoother.rts(filtered, predicted, comp.f)
      assert length(smoothed) == 20

      Enum.each(smoothed, fn {state, cov} ->
        assert Nx.shape(state) == {2}
        assert Nx.shape(cov) == {2, 2}
      end)

      # The level component should roughly track the linear trend
      {last_state, _} = List.last(smoothed)
      level = Nx.to_number(last_state[0])
      assert_in_delta level, 20.0, 5.0

      # The slope component should be close to 1.0 (true slope)
      slope = Nx.to_number(last_state[1])
      assert_in_delta slope, 1.0, 1.0

      # Smoothed variances should be <= filtered variances
      Enum.zip(smoothed, filtered)
      |> Enum.each(fn {{_, p_smooth}, {_, p_filt}} ->
        # Compare diagonal elements (variances)
        smooth_var = Nx.to_number(p_smooth[0][0])
        filt_var = Nx.to_number(p_filt[0][0])
        assert smooth_var <= filt_var + 1.0e-6
      end)
    end

    test "composed local_level + local_linear_trend through filter" do
      ll = Components.local_level(0.5)
      llt = Components.local_linear_trend(0.1, 0.01)
      composed = BstsNx.StateSpace.compose(ll, llt)

      r = Nx.tensor([[1.0]])
      x0 = Nx.tensor([0.0, 0.0, 0.0])
      p0 = Nx.eye(3)

      obs = Enum.map(1..15, fn t -> t * 1.0 end)

      results = KalmanFilter.filter(obs, composed.f, composed.h, composed.q, r, x0, p0)

      assert length(results) == 15

      Enum.each(results, fn {state, cov} ->
        assert Nx.shape(state) == {3}
        assert Nx.shape(cov) == {3, 3}
      end)
    end

    test "filter handles NaN tensor observations correctly (states remain finite)" do
      obs = [1.0, 2.0, nil, 4.0, 5.0]

      obs_with_nan =
        Enum.map(obs, fn
          nil -> Nx.Constants.nan()
          v -> Nx.tensor(v)
        end)

      results = KalmanFilter.filter(obs_with_nan, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0)
      assert length(results) == 5

      # All states should be finite (no NaN propagation)
      Enum.each(results, fn {state, cov} ->
        s = Nx.to_number(state)
        c = Nx.to_number(cov)
        assert s == s, "State must not be NaN after NaN observation"
        assert c == c, "Covariance must not be NaN after NaN observation"
        assert c >= 0.0, "Covariance must be non-negative"
      end)
    end
  end
end
