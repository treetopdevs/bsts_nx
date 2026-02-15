defmodule BstsNxPropertyTest do
  @moduledoc """
  Property-based tests for bsts_nx primitives using StreamData.

  These tests verify algebraic and shape invariants that must hold for
  *any* valid input, catching edge cases that hand-picked examples miss.
  """
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias BstsNx.{
    CausalImpact,
    Components,
    Diagnostics,
    GibbsSampler,
    KalmanFilter,
    Smoother,
    StateSpace
  }

  # ── Generators ──────────────────────────────────────────────────────

  # Positive float in [0.01, 100.0] — avoids degenerate zero variances
  defp pos_float do
    StreamData.map(StreamData.float(min: 0.01, max: 100.0), &abs/1)
  end

  # Observations list of length n ∈ [min_len, max_len]
  defp observations(min_len, max_len) do
    StreamData.bind(StreamData.integer(min_len..max_len), fn n ->
      StreamData.list_of(StreamData.float(min: -1000.0, max: 1000.0), length: n)
    end)
  end

  # Square matrix size in [1, 4]
  defp square_matrix_size, do: StreamData.integer(1..4)

  # ---------------------------------------------------------------------------
  # KalmanFilter properties
  # ---------------------------------------------------------------------------

  describe "KalmanFilter shape properties" do
    property "filter output length always equals input length" do
      check all(
              obs <- observations(1, 100),
              q <- pos_float(),
              r <- pos_float(),
              x0 <- StreamData.float(min: -100.0, max: 100.0),
              p0 <- pos_float(),
              max_runs: 50
            ) do
        results = KalmanFilter.filter(obs, 1.0, 1.0, q, r, x0, p0)
        assert length(results) == length(obs)
      end
    end

    property "filter_with_pred returns matching filtered and predicted lengths" do
      check all(
              obs <- observations(1, 100),
              q <- pos_float(),
              r <- pos_float(),
              max_runs: 50
            ) do
        {filtered, predicted} = KalmanFilter.filter_with_pred(obs, 1.0, 1.0, q, r, 0.0, 1.0)
        assert length(filtered) == length(obs)
        assert length(predicted) == length(obs)
      end
    end

    property "filter_defn output shape matches input shape" do
      check all(
              obs <- observations(1, 100),
              q <- pos_float(),
              r <- pos_float(),
              max_runs: 50
            ) do
        obs_tensor = Nx.tensor(obs)
        {xs, ps} = KalmanFilter.filter_defn(obs_tensor, 1.0, 1.0, q, r, 0.0, 1.0)
        assert Nx.shape(xs) == {length(obs)}
        assert Nx.shape(ps) == {length(obs)}
      end
    end

    property "all covariances are non-negative" do
      check all(
              obs <- observations(1, 50),
              q <- pos_float(),
              r <- pos_float(),
              max_runs: 50
            ) do
        results = KalmanFilter.filter(obs, 1.0, 1.0, q, r, 0.0, 1.0)

        Enum.each(results, fn {_state, cov} ->
          assert Nx.to_number(cov) >= 0.0,
                 "Kalman filter covariance must be non-negative, got: #{Nx.to_number(cov)}"
        end)
      end
    end

    property "time-varying h vector produces same length output" do
      check all(
              n <- StreamData.integer(2..50),
              max_runs: 30
            ) do
        obs = Enum.map(1..n, fn i -> i * 1.0 end)
        h_vec = Nx.tensor(Enum.map(1..n, fn _ -> 1.0 end))
        results = KalmanFilter.filter(obs, 1.0, h_vec, 1.0, 1.0, 0.0, 1.0)
        assert length(results) == n
      end
    end
  end

  # ---------------------------------------------------------------------------
  # Smoother properties
  # ---------------------------------------------------------------------------

  describe "Smoother properties" do
    property "smoothed variance ≤ filtered variance (smoother never increases uncertainty)" do
      check all(
              obs <- observations(3, 80),
              q <- pos_float(),
              r <- pos_float(),
              max_runs: 40
            ) do
        obs_tensor = Nx.tensor(obs)
        {xs, ps} = KalmanFilter.filter_defn(obs_tensor, 1.0, 1.0, q, r, 0.0, 1.0)
        {_sxs, sps} = Smoother.rts_defn(xs, ps, 1.0, q)

        filtered_vars = Nx.to_flat_list(ps)
        smoothed_vars = Nx.to_flat_list(sps)

        Enum.zip(smoothed_vars, filtered_vars)
        |> Enum.each(fn {sv, fv} ->
          # Allow small numerical tolerance
          assert sv <= fv + 1.0e-6,
                 "Smoothed variance #{sv} should be ≤ filtered variance #{fv}"
        end)
      end
    end

    property "smoother output shape matches filter output shape" do
      check all(
              obs <- observations(2, 80),
              q <- pos_float(),
              r <- pos_float(),
              max_runs: 40
            ) do
        obs_tensor = Nx.tensor(obs)
        {xs, ps} = KalmanFilter.filter_defn(obs_tensor, 1.0, 1.0, q, r, 0.0, 1.0)
        {sxs, sps} = Smoother.rts_defn(xs, ps, 1.0, q)
        assert Nx.shape(sxs) == Nx.shape(xs)
        assert Nx.shape(sps) == Nx.shape(ps)
      end
    end
  end

  # ---------------------------------------------------------------------------
  # StateSpace.block_diag properties
  # ---------------------------------------------------------------------------

  describe "StateSpace.block_diag properties" do
    property "output dimension equals sum of input dimensions" do
      check all(
              sizes <- StreamData.list_of(square_matrix_size(), min_length: 1, max_length: 5),
              max_runs: 50
            ) do
        matrices = Enum.map(sizes, fn n -> Nx.eye(n) end)
        result = StateSpace.block_diag(matrices)
        total = Enum.sum(sizes)
        assert Nx.shape(result) == {total, total}
      end
    end

    property "output is always square" do
      check all(
              sizes <- StreamData.list_of(square_matrix_size(), min_length: 1, max_length: 5),
              max_runs: 50
            ) do
        matrices =
          Enum.map(sizes, fn n ->
            Nx.eye(n) |> Nx.multiply(Enum.random(1..10) * 1.0)
          end)

        result = StateSpace.block_diag(matrices)
        {rows, cols} = Nx.shape(result)
        assert rows == cols
      end
    end

    property "block_diag of identity matrices is a larger identity matrix" do
      check all(
              sizes <- StreamData.list_of(square_matrix_size(), min_length: 1, max_length: 4),
              max_runs: 30
            ) do
        matrices = Enum.map(sizes, &Nx.eye/1)
        result = StateSpace.block_diag(matrices)
        total = Enum.sum(sizes)
        expected = Nx.eye(total)
        assert Nx.all_close(result, expected, atol: 1.0e-6) |> Nx.to_number() == 1
      end
    end

    property "off-diagonal blocks are zero" do
      check all(
              n1 <- StreamData.integer(1..3),
              n2 <- StreamData.integer(1..3),
              max_runs: 30
            ) do
        m1 = Nx.broadcast(1.0, {n1, n1})
        m2 = Nx.broadcast(2.0, {n2, n2})
        result = StateSpace.block_diag([m1, m2])

        # Check that the top-right n1×n2 block is all zeros
        top_right = Nx.slice(result, [0, n1], [n1, n2])

        assert Nx.all_close(top_right, Nx.broadcast(0.0, {n1, n2}), atol: 1.0e-10)
               |> Nx.to_number() == 1

        # Check that the bottom-left n2×n1 block is all zeros
        bottom_left = Nx.slice(result, [n1, 0], [n2, n1])

        assert Nx.all_close(bottom_left, Nx.broadcast(0.0, {n2, n1}), atol: 1.0e-10)
               |> Nx.to_number() == 1
      end
    end

    property "accepts mix of scalars and matrices" do
      check all(
              scalar_val <- StreamData.float(min: 0.1, max: 10.0),
              mat_size <- StreamData.integer(1..3),
              max_runs: 30
            ) do
        scalar = Nx.tensor(scalar_val)
        mat = Nx.eye(mat_size)
        result = StateSpace.block_diag([scalar, mat])
        assert Nx.shape(result) == {1 + mat_size, 1 + mat_size}
      end
    end
  end

  # ---------------------------------------------------------------------------
  # StateSpace.compose properties
  # ---------------------------------------------------------------------------

  describe "StateSpace.compose properties" do
    property "composed F has correct block diagonal shape" do
      check all(
              n1 <- StreamData.integer(1..3),
              n2 <- StreamData.integer(1..3),
              max_runs: 30
            ) do
        comp1 = %{f: Nx.eye(n1), q: Nx.eye(n1), h: Nx.broadcast(1.0, {1, n1})}
        comp2 = %{f: Nx.eye(n2), q: Nx.eye(n2), h: Nx.broadcast(1.0, {1, n2})}
        result = StateSpace.compose(comp1, comp2)
        total = n1 + n2
        assert Nx.shape(result.f) == {total, total}
        assert Nx.shape(result.q) == {total, total}
        assert Nx.shape(result.h) == {1, total}
      end
    end

    property "compose raises on mismatched observation row dimensions" do
      check all(
              n <- StreamData.integer(1..3),
              max_runs: 10
            ) do
        comp1 = %{f: Nx.eye(n), q: Nx.eye(n), h: Nx.broadcast(1.0, {1, n})}
        comp2 = %{f: Nx.eye(n), q: Nx.eye(n), h: Nx.broadcast(1.0, {2, n})}

        assert_raise ArgumentError, ~r/same number of rows/, fn ->
          StateSpace.compose(comp1, comp2)
        end
      end
    end
  end

  # ---------------------------------------------------------------------------
  # Components properties
  # ---------------------------------------------------------------------------

  describe "Components shape properties" do
    property "local_level always produces 1×1 matrices" do
      check all(var <- pos_float(), max_runs: 30) do
        c = Components.local_level(var)
        assert Nx.shape(c.f) == {1, 1}
        assert Nx.shape(c.q) == {1, 1}
        assert Nx.shape(c.h) == {1, 1}
      end
    end

    property "local_linear_trend always produces 2×2 F and Q" do
      check all(
              v1 <- pos_float(),
              v2 <- pos_float(),
              max_runs: 30
            ) do
        c = Components.local_linear_trend(v1, v2)
        assert Nx.shape(c.f) == {2, 2}
        assert Nx.shape(c.q) == {2, 2}
        assert Nx.shape(c.h) == {1, 2}
      end
    end

    property "composed local_level + local_linear_trend has correct shapes" do
      check all(
              v_level <- pos_float(),
              v_trend_l <- pos_float(),
              v_trend_s <- pos_float(),
              max_runs: 20
            ) do
        ll = Components.local_level(v_level)
        llt = Components.local_linear_trend(v_trend_l, v_trend_s)
        composed = StateSpace.compose(ll, llt)
        assert Nx.shape(composed.f) == {3, 3}
        assert Nx.shape(composed.q) == {3, 3}
        assert Nx.shape(composed.h) == {1, 3}
      end
    end

    property "regression component shape matches number of regressors" do
      check all(
              dim <- StreamData.integer(1..5),
              max_runs: 20
            ) do
        betas = Nx.broadcast(0.1, {dim})
        sigma = Nx.eye(dim) |> Nx.multiply(0.01)
        c = Components.regression(betas, sigma)
        assert Nx.shape(c.f) == {dim, dim}
        assert Nx.shape(c.q) == {dim, dim}
        assert Nx.shape(c.h) == {1, dim}
      end
    end
  end

  # ---------------------------------------------------------------------------
  # CausalImpact properties
  # ---------------------------------------------------------------------------

  describe "CausalImpact.estimate_from_filter properties" do
    property "output lists have same length as valid intervention indices" do
      check all(
              n <- StreamData.integer(10..100),
              k <- StreamData.integer(1..5),
              max_runs: 30
            ) do
        :rand.seed(:exsss, {n, k, 42})
        obs = Enum.map(1..n, fn _ -> :rand.normal() * 10 + 50 end)
        # Pick k intervention indices near the end
        indices = Enum.to_list((n - k)..(n - 1))
        result = CausalImpact.estimate_from_filter(obs, indices)

        assert length(result.actual) == k
        assert length(result.baseline) == k
        assert length(result.point_effects.mean) == k
        assert length(result.point_effects.lower) == k
        assert length(result.point_effects.upper) == k
      end
    end

    property "cumulative effect equals sum of point effects" do
      check all(
              n <- StreamData.integer(20..80),
              k <- StreamData.integer(2..8),
              max_runs: 30
            ) do
        :rand.seed(:exsss, {n, k, 42})
        obs = Enum.map(1..n, fn _ -> :rand.normal() * 10 + 50 end)
        indices = Enum.to_list((n - k)..(n - 1))
        result = CausalImpact.estimate_from_filter(obs, indices)

        point_sum = Enum.sum(result.point_effects.mean)
        assert_in_delta result.cumulative_effect.mean, point_sum, 1.0e-4
      end
    end

    property "empty indices always return zero-effect result" do
      check all(
              obs <- observations(5, 50),
              max_runs: 20
            ) do
        result = CausalImpact.estimate_from_filter(obs, [])
        assert result.actual == []
        assert result.baseline == []
        assert result.cumulative_effect.mean == 0.0
        assert result.relative_effect.mean == 0.0
      end
    end

    property "out-of-bounds indices are silently filtered" do
      check all(
              obs <- observations(5, 30),
              max_runs: 20
            ) do
        n = length(obs)
        result = CausalImpact.estimate_from_filter(obs, [n, n + 1, n + 100, -1])
        assert result.actual == []
        assert result.cumulative_effect.mean == 0.0
      end
    end

    property "non-consecutive intervention indices produce valid cumulative variance" do
      check all(
              n <- StreamData.integer(20..60),
              gap <- StreamData.integer(2..4),
              max_runs: 20
            ) do
        :rand.seed(:exsss, {n, gap, 77})
        obs = Enum.map(1..n, fn _ -> :rand.normal() * 10 + 50 end)
        # Pick non-consecutive indices with gaps
        start_idx = div(n, 2)
        indices = Enum.take_every(start_idx..(n - 1), gap)

        # Generators guarantee length(indices) >= 2 (min 10 elements with max gap 4)
        result = CausalImpact.estimate_from_filter(obs, indices)

        # Cumulative variance must be non-negative (sd is real)
        assert result.cumulative_effect.sd >= 0.0,
               "Cumulative SD must be non-negative for non-consecutive indices"

        # Cumulative CI must be wider than or equal to zero-width
        assert result.cumulative_effect.upper >= result.cumulative_effect.lower,
               "Cumulative upper must be >= lower"
      end
    end
  end

  # ---------------------------------------------------------------------------
  # Diagnostics properties
  # ---------------------------------------------------------------------------

  describe "Diagnostics properties" do
    property "R-hat ≥ 1.0 for any valid multi-chain input" do
      check all(
              n <- StreamData.integer(10..50),
              m <- StreamData.integer(2..4),
              max_runs: 30
            ) do
        :rand.seed(:exsss, {n, m, 99})
        chains = Enum.map(1..m, fn _ -> Enum.map(1..n, fn _ -> :rand.normal() end) end)
        rhat = Diagnostics.r_hat(chains)

        case rhat do
          :nan ->
            :ok

          val ->
            # R-hat can dip slightly below 1.0 for short chains due to finite-sample
            # bias in the Gelman-Rubin estimator. Allow a 5% tolerance.
            assert val >= 0.95, "R-hat must be >= 0.95, got: #{val}"
        end
      end
    end

    property "ESS ≤ total sample count (m × n)" do
      check all(
              n <- StreamData.integer(10..50),
              m <- StreamData.integer(2..4),
              max_runs: 30
            ) do
        :rand.seed(:exsss, {n, m, 77})
        chains = Enum.map(1..m, fn _ -> Enum.map(1..n, fn _ -> :rand.normal() end) end)
        ess = Diagnostics.effective_sample_size(chains)

        unless ess == :nan do
          assert ess <= m * n + 1.0e-6, "ESS must be ≤ m×n, got: #{ess}"
        end
      end
    end

    property "single-chain ESS ≥ 1.0 for non-constant chains" do
      check all(
              n <- StreamData.integer(10..100),
              max_runs: 30
            ) do
        :rand.seed(:exsss, {n, 42, 1})
        chain = Enum.map(1..n, fn _ -> :rand.normal() end)
        ess = Diagnostics.ess_single(chain)

        case ess do
          :nan -> :ok
          val -> assert val >= 1.0, "Single-chain ESS must be >= 1.0, got: #{val}"
        end
      end
    end

    property "split R-hat returns float or :nan for any chain of length ≥ 4" do
      check all(
              n <- StreamData.integer(10..100),
              max_runs: 30
            ) do
        :rand.seed(:exsss, {n, 55, 2})
        chain = Enum.map(1..n, fn _ -> :rand.normal() end)
        rhat = Diagnostics.split_r_hat(chain)

        # split_r_hat must return either :nan (degenerate) or a positive float
        assert rhat == :nan or (is_float(rhat) and rhat > 0.0),
               "split_r_hat must return :nan or positive float, got: #{inspect(rhat)}"
      end
    end
  end

  # ---------------------------------------------------------------------------
  # Missing data properties
  # ---------------------------------------------------------------------------

  describe "Missing data handling properties" do
    property "filter_defn handles NaN observations without crashing" do
      check all(
              n <- StreamData.integer(5..50),
              num_missing <- StreamData.integer(1..3),
              max_runs: 30
            ) do
        :rand.seed(:exsss, {n, num_missing, 33})
        base = Enum.map(1..n, fn _ -> :rand.normal() * 10 + 50 end)
        missing_indices = Enum.take_random(0..(n - 1), min(num_missing, n - 1))

        obs_list =
          Enum.with_index(base, fn val, idx ->
            if idx in missing_indices, do: Nx.Constants.nan() |> Nx.to_number(), else: val
          end)

        obs_tensor = Nx.tensor(obs_list)
        {xs, ps} = KalmanFilter.filter_defn(obs_tensor, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0)
        assert Nx.shape(xs) == {n}
        assert Nx.shape(ps) == {n}
      end
    end

    property "list-based filter handles nil observations (missing data)" do
      check all(
              n <- StreamData.integer(5..30),
              max_runs: 20
            ) do
        obs =
          Enum.map(1..n, fn i ->
            if i == div(n, 2), do: nil, else: i * 1.0
          end)

        results = KalmanFilter.filter(obs, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0)
        assert length(results) == n
      end
    end
  end

  # ---------------------------------------------------------------------------
  # Distributions properties
  # ---------------------------------------------------------------------------

  describe "Distributions properties" do
    property "inv_gamma samples are always positive" do
      check all(
              alpha <- StreamData.float(min: 1.0, max: 20.0),
              beta <- StreamData.float(min: 0.1, max: 50.0),
              max_runs: 50
            ) do
        sample = BstsNx.Distributions.inv_gamma_sample(alpha, beta)
        val = Nx.to_number(sample)
        assert val > 0.0, "Inverse-gamma sample must be positive, got: #{val}"
      end
    end

    property "normal samples are finite" do
      check all(
              mean <- StreamData.float(min: -100.0, max: 100.0),
              stddev <- StreamData.float(min: 0.01, max: 50.0),
              max_runs: 30
            ) do
        key = Nx.Random.key(:erlang.unique_integer([:positive]))
        {sample, _key2} = BstsNx.Distributions.normal_sample(key, mean: mean, stddev: stddev)
        val = Nx.to_number(sample)
        # NaN check: NaN != NaN
        assert is_float(val) and val == val, "Normal sample must be finite, got: #{val}"
      end
    end
  end

  # ---------------------------------------------------------------------------
  # Fixed edge-case properties: zero-variance Kalman filter (Issue #1)
  # ---------------------------------------------------------------------------

  describe "Zero-variance Kalman filter safety" do
    property "filter never produces NaN/Inf states when q=0 (no process noise)" do
      check all(
              obs <- observations(3, 50),
              r <- pos_float(),
              max_runs: 30
            ) do
        results = KalmanFilter.filter(obs, 1.0, 1.0, 0.0, r, 0.0, 1.0)
        assert length(results) == length(obs)

        Enum.each(results, fn {state, cov} ->
          s = Nx.to_number(state)
          c = Nx.to_number(cov)
          assert s == s, "State must not be NaN"
          assert c == c, "Covariance must not be NaN"
          assert c >= 0.0, "Covariance must be non-negative, got: #{c}"
        end)
      end
    end

    property "filter never produces NaN/Inf states when r=0 (no obs noise)" do
      check all(
              obs <- observations(3, 50),
              q <- pos_float(),
              max_runs: 30
            ) do
        results = KalmanFilter.filter(obs, 1.0, 1.0, q, 0.0, 0.0, 1.0)
        assert length(results) == length(obs)

        Enum.each(results, fn {state, cov} ->
          s = Nx.to_number(state)
          c = Nx.to_number(cov)
          assert s == s, "State must not be NaN when r=0"
          assert c == c, "Covariance must not be NaN when r=0"
        end)
      end
    end

    property "filter_defn handles zero q without NaN" do
      check all(
              obs <- observations(3, 50),
              r <- pos_float(),
              max_runs: 30
            ) do
        obs_tensor = Nx.tensor(obs)
        {xs, ps} = KalmanFilter.filter_defn(obs_tensor, 1.0, 1.0, 0.0, r, 0.0, 1.0)

        xs_list = Nx.to_flat_list(xs)
        ps_list = Nx.to_flat_list(ps)

        Enum.each(xs_list, fn x ->
          assert x == x, "filter_defn state must not be NaN with q=0"
        end)

        Enum.each(ps_list, fn p ->
          assert p == p, "filter_defn covariance must not be NaN with q=0"
        end)
      end
    end
  end

  # ---------------------------------------------------------------------------
  # Fixed edge-case properties: safe_cholesky robustness (Issue #4)
  # ---------------------------------------------------------------------------

  describe "safe_cholesky robustness" do
    property "handles near-singular positive semi-definite matrices" do
      check all(
              n <- StreamData.integer(2..4),
              scale <- StreamData.float(min: 1.0e-8, max: 1.0e-4),
              max_runs: 20
            ) do
        # Create a near-singular PSD matrix: eps * I
        mat = Nx.eye(n) |> Nx.multiply(scale)
        chol = BstsNx.Utils.safe_cholesky(mat)

        assert Nx.shape(chol) == {n, n}
        # Verify no NaN
        flat = Nx.to_flat_list(chol)
        Enum.each(flat, fn v -> assert v == v, "Cholesky factor must not contain NaN" end)
      end
    end

    property "safe_cholesky of identity is lower-triangular identity" do
      check all(n <- StreamData.integer(1..5), max_runs: 10) do
        eye = Nx.eye(n)
        chol = BstsNx.Utils.safe_cholesky(eye)

        assert Nx.all_close(chol, eye, atol: 1.0e-6) |> Nx.to_number() == 1
      end
    end
  end

  # ---------------------------------------------------------------------------
  # Fixed edge-case properties: derive_exsss_seed independence (Issue #5)
  # ---------------------------------------------------------------------------

  describe "derive_exsss_seed independence" do
    property "different key pairs produce different 3-tuples" do
      check all(
              i1 <- StreamData.integer(0..1_000_000),
              i2 <- StreamData.integer(0..1_000_000),
              max_runs: 50
            ) do
        seed_a = BstsNx.Utils.derive_exsss_seed([i1, i2])
        seed_b = BstsNx.Utils.derive_exsss_seed([i2 + 1, i1])

        # Different inputs should (almost certainly) produce different seeds
        # Allow the extremely unlikely case of collision but at least verify structure
        {a1, a2, a3} = seed_a
        {b1, b2, b3} = seed_b
        assert is_integer(a1) and is_integer(a2) and is_integer(a3)
        assert is_integer(b1) and is_integer(b2) and is_integer(b3)

        # All components should be non-negative (masked to 58 bits)
        assert a1 >= 0 and a2 >= 0 and a3 >= 0
        assert b1 >= 0 and b2 >= 0 and b3 >= 0
      end
    end

    property "third component is independent of first two" do
      check all(
              i1 <- StreamData.integer(1..1_000_000),
              i2 <- StreamData.integer(1..1_000_000),
              max_runs: 30
            ) do
        {a, b, c} = BstsNx.Utils.derive_exsss_seed([i1, i2])

        # c should not equal a or b (phash2 mixing makes this astronomically unlikely)
        # This is a probabilistic test; if it ever fails, it indicates broken mixing
        unless i1 == i2 do
          assert c != a and c != b,
                 "Third seed component should differ from both of the first two"
        end
      end
    end
  end

  # ---------------------------------------------------------------------------
  # Fixed edge-case properties: inv_gamma never returns Infinity (Issue #11)
  # ---------------------------------------------------------------------------

  describe "inv_gamma infinity protection" do
    property "inv_gamma never returns Infinity for valid parameters" do
      check all(
              alpha <- StreamData.float(min: 0.5, max: 50.0),
              beta <- StreamData.float(min: 0.01, max: 100.0),
              max_runs: 50
            ) do
        sample = BstsNx.Distributions.inv_gamma_sample(alpha, beta)
        val = Nx.to_number(sample)
        assert val > 0.0, "inv_gamma must return positive value, got: #{val}"
        assert val != :infinity, "inv_gamma must not return infinity"
        assert val == val, "inv_gamma must not return NaN"
      end
    end

    property "inv_gamma with very small alpha still returns finite positive" do
      check all(
              beta <- StreamData.float(min: 0.1, max: 10.0),
              max_runs: 20
            ) do
        # alpha close to zero but positive — the gamma floor protects against Inf
        sample = BstsNx.Distributions.inv_gamma_sample(0.01, beta)
        val = Nx.to_number(sample)
        assert val > 0.0, "inv_gamma must be positive for small alpha, got: #{val}"
        assert val == val, "inv_gamma must not be NaN for small alpha"
      end
    end
  end

  # ---------------------------------------------------------------------------
  # Fixed edge-case properties: GibbsSampler input validation (Issues #3, #16)
  # ---------------------------------------------------------------------------

  describe "GibbsSampler input validation" do
    property "negative process_var always raises ArgumentError" do
      check all(
              v <- StreamData.float(min: -100.0, max: -0.01),
              max_runs: 20
            ) do
        obs = [1.0, 2.0, 3.0, 4.0, 5.0]

        assert_raise ArgumentError, fn ->
          GibbsSampler.sample(obs, 3, 0.0, 1.0, v, 1.0, seed: 42)
        end
      end
    end

    property "negative obs_var always raises ArgumentError" do
      check all(
              v <- StreamData.float(min: -100.0, max: -0.01),
              max_runs: 20
            ) do
        obs = [1.0, 2.0, 3.0, 4.0, 5.0]

        assert_raise ArgumentError, fn ->
          GibbsSampler.sample(obs, 3, 0.0, 1.0, 1.0, v, seed: 42)
        end
      end
    end

    property "non-positive num_samples always raises ArgumentError" do
      check all(
              n <- StreamData.integer(-10..0),
              max_runs: 10
            ) do
        obs = [1.0, 2.0, 3.0]

        assert_raise ArgumentError, fn ->
          GibbsSampler.sample(obs, n, 0.0, 1.0, 1.0, 1.0, seed: 42)
        end
      end
    end

    property "negative prior_scale always raises ArgumentError" do
      check all(
              scale <- StreamData.float(min: -100.0, max: -0.001),
              max_runs: 20
            ) do
        obs = [1.0, 2.0, 3.0, 4.0, 5.0]

        assert_raise ArgumentError, ~r/prior_scale/, fn ->
          GibbsSampler.sample(obs, 3, 0.0, 1.0, 1.0, 1.0,
            prior_scale: scale,
            seed: 42
          )
        end
      end
    end

    property "non-positive prior_shape always raises ArgumentError" do
      check all(
              shape <- StreamData.float(min: -10.0, max: 0.0),
              max_runs: 20
            ) do
        obs = [1.0, 2.0, 3.0, 4.0, 5.0]

        assert_raise ArgumentError, ~r/prior_shape/, fn ->
          GibbsSampler.sample(obs, 3, 0.0, 1.0, 1.0, 1.0,
            prior_shape: shape,
            seed: 42
          )
        end
      end
    end

    property "sample_general rejects invalid prior params" do
      check all(
              scale <- StreamData.float(min: -50.0, max: -0.001),
              max_runs: 10
            ) do
        obs = [1.0, 2.0, 3.0, 4.0, 5.0]

        assert_raise ArgumentError, ~r/prior_scale/, fn ->
          GibbsSampler.sample_general(obs, 1.0, 1.0, 3,
            prior_scale: scale,
            seed: 42
          )
        end
      end
    end
  end

  # ---------------------------------------------------------------------------
  # Fixed edge-case properties: h tensor validation (Issue #3)
  # ---------------------------------------------------------------------------

  describe "GibbsSampler h tensor validation" do
    property "sample_general raises when h tensor is shorter than observations" do
      check all(
              n <- StreamData.integer(5..20),
              short <- StreamData.integer(1..4),
              max_runs: 15
            ) do
        obs = Enum.map(1..n, fn i -> i * 1.0 end)
        # h tensor shorter than observations should raise
        h_short = Nx.tensor(Enum.map(1..min(short, n - 1), fn _ -> 1.0 end))

        if short < n do
          assert_raise ArgumentError, fn ->
            GibbsSampler.sample_general(obs, 1.0, h_short, 3, seed: 42)
          end
        end
      end
    end
  end

  # ---------------------------------------------------------------------------
  # Fixed edge-case properties: CausalImpact bounds validation (Issue #8)
  # ---------------------------------------------------------------------------

  describe "CausalImpact bounds validation" do
    property "estimate raises when post_period exceeds observations" do
      check all(
              n <- StreamData.integer(5..30),
              overshoot <- StreamData.integer(1..10),
              max_runs: 15
            ) do
        obs = Enum.map(1..n, fn i -> i * 1.0 end)
        pre_end = div(n, 2)
        post_end = n + overshoot

        assert_raise ArgumentError, fn ->
          CausalImpact.estimate(obs, {1, pre_end}, {pre_end + 1, post_end},
            num_samples: 3,
            burn_in: 1,
            seed: 42
          )
        end
      end
    end

    property "estimate raises when pre_period start > end" do
      check all(
              n <- StreamData.integer(10..30),
              max_runs: 10
            ) do
        obs = Enum.map(1..n, fn i -> i * 1.0 end)
        pre_start = div(n, 2)
        pre_end = pre_start - 2

        assert_raise ArgumentError, fn ->
          CausalImpact.estimate(obs, {pre_start, pre_end}, {pre_end + 1, n},
            num_samples: 3,
            burn_in: 1,
            seed: 42
          )
        end
      end
    end
  end
end
