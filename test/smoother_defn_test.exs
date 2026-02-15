defmodule BstsNx.SmootherDefnTest do
  use ExUnit.Case, async: true

  alias BstsNx.KalmanFilter
  alias BstsNx.Smoother

  test "rts_defn matches rts on a simple constant series" do
    obs = [1.0, 2.0, 3.0, 2.5, 2.0]
    f = 1.0
    h = 1.0
    q = 0.5
    r = 1.0
    x0 = 0.0
    p0 = 1.0

    # Run filter_defn for the compiled path
    obs_tensor = Nx.tensor(obs, type: {:f, 32})
    {xs, ps} = KalmanFilter.filter_defn(obs_tensor, f, h, q, r, x0, p0)
    {sxs, sps} = Smoother.rts_defn(xs, ps, f, q)

    # Run the list-based path
    {filtered, predicted} = KalmanFilter.filter_with_pred(obs, f, h, q, r, x0, p0)
    smoothed_list = Smoother.rts(filtered, predicted, f)

    smoothed_xs = Enum.map(smoothed_list, fn {x, _p} -> Nx.to_number(x) end)
    smoothed_ps = Enum.map(smoothed_list, fn {_x, p} -> Nx.to_number(p) end)

    defn_xs = Nx.to_flat_list(sxs)
    defn_ps = Nx.to_flat_list(sps)

    Enum.zip(smoothed_xs, defn_xs)
    |> Enum.each(fn {expected, actual} ->
      assert_in_delta(
        expected,
        actual,
        1.0e-4,
        "smoothed state mismatch: expected #{expected}, got #{actual}"
      )
    end)

    Enum.zip(smoothed_ps, defn_ps)
    |> Enum.each(fn {expected, actual} ->
      assert_in_delta(
        expected,
        actual,
        1.0e-4,
        "smoothed variance mismatch: expected #{expected}, got #{actual}"
      )
    end)
  end

  test "rts_defn matches rts on a longer random walk series" do
    # Generate a longer series to exercise the backward loop
    :rand.seed(:exsss, {42, 43, 44})
    obs = Enum.map(1..50, fn _ -> :rand.normal() * 2.0 + 10.0 end)
    f = 1.0
    q = 0.3
    r = 1.0
    x0 = 10.0
    p0 = 1.0

    obs_tensor = Nx.tensor(obs, type: {:f, 32})
    {xs, ps} = KalmanFilter.filter_defn(obs_tensor, f, 1.0, q, r, x0, p0)
    {sxs, _sps} = Smoother.rts_defn(xs, ps, f, q)

    {filtered, predicted} = KalmanFilter.filter_with_pred(obs, f, 1.0, q, r, x0, p0)
    smoothed_list = Smoother.rts(filtered, predicted, f)

    smoothed_xs = Enum.map(smoothed_list, fn {x, _p} -> Nx.to_number(x) end)
    defn_xs = Nx.to_flat_list(sxs)

    Enum.zip(smoothed_xs, defn_xs)
    |> Enum.each(fn {expected, actual} ->
      assert_in_delta(expected, actual, 1.0e-3)
    end)
  end

  test "rts_defn with missing data (NaN)" do
    nan = Nx.Constants.nan() |> Nx.to_number()
    obs = [1.0, 2.0, nan, 4.0, 5.0]
    obs_tensor = Nx.tensor(obs, type: {:f, 32})

    {xs, ps} = KalmanFilter.filter_defn(obs_tensor, 1.0, 1.0, 0.5, 1.0, 0.0, 1.0)
    {sxs, sps} = Smoother.rts_defn(xs, ps, 1.0, 0.5)

    # Verify outputs are finite and have correct shape
    assert Nx.shape(sxs) == {5}
    assert Nx.shape(sps) == {5}

    # All smoothed values should be finite
    sxs_list = Nx.to_flat_list(sxs)
    sps_list = Nx.to_flat_list(sps)

    Enum.each(sxs_list, fn x ->
      refute is_float(x) and x != x, "smoothed state should not be NaN"
    end)

    Enum.each(sps_list, fn p ->
      assert p > 0.0, "smoothed variance should be positive"
    end)
  end

  test "rts_defn on single observation returns filtered value" do
    obs_tensor = Nx.tensor([5.0], type: {:f, 32})
    {xs, ps} = KalmanFilter.filter_defn(obs_tensor, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0)
    {sxs, sps} = Smoother.rts_defn(xs, ps, 1.0, 1.0)

    # For a single observation, smoothed == filtered
    assert_in_delta(Nx.to_number(sxs[0]), Nx.to_number(xs[0]), 1.0e-6)
    assert_in_delta(Nx.to_number(sps[0]), Nx.to_number(ps[0]), 1.0e-6)
  end

  test "smoothed variances are less than or equal to filtered variances" do
    obs = Enum.map(1..20, fn i -> i * 1.0 + :math.sin(i) end)
    obs_tensor = Nx.tensor(obs, type: {:f, 32})

    {xs, ps} = KalmanFilter.filter_defn(obs_tensor, 1.0, 1.0, 0.5, 1.0, 0.0, 1.0)
    {_sxs, sps} = Smoother.rts_defn(xs, ps, 1.0, 0.5)

    filtered_ps = Nx.to_flat_list(ps)
    smoothed_ps = Nx.to_flat_list(sps)

    # Smoothed variance should be <= filtered variance at each time step
    Enum.zip(filtered_ps, smoothed_ps)
    |> Enum.each(fn {fp, sp} ->
      assert sp <= fp + 1.0e-6,
             "smoothed var (#{sp}) should be <= filtered var (#{fp})"
    end)
  end

  # --- rts_defn_with_lag1 tests ---

  test "rts_defn_with_lag1 returns consistent smoothed states and variances" do
    obs = [1.0, 2.0, 3.0, 2.5, 2.0]
    f = 1.0
    q = 0.5
    r = 1.0
    x0 = 0.0
    p0 = 1.0

    obs_tensor = Nx.tensor(obs, type: {:f, 32})
    {xs, ps} = KalmanFilter.filter_defn(obs_tensor, f, 1.0, q, r, x0, p0)

    # Compare standard rts_defn with rts_defn_with_lag1
    {sxs, sps} = Smoother.rts_defn(xs, ps, f, q)
    {sxs_lag, sps_lag, lag1} = Smoother.rts_defn_with_lag1(xs, ps, f, q)

    # Smoothed states and variances should match
    Enum.zip(Nx.to_flat_list(sxs), Nx.to_flat_list(sxs_lag))
    |> Enum.each(fn {expected, actual} ->
      assert_in_delta(expected, actual, 1.0e-4,
        "smoothed state mismatch with lag1 variant")
    end)

    Enum.zip(Nx.to_flat_list(sps), Nx.to_flat_list(sps_lag))
    |> Enum.each(fn {expected, actual} ->
      assert_in_delta(expected, actual, 1.0e-4,
        "smoothed variance mismatch with lag1 variant")
    end)

    # lag1 should have t-1 entries
    assert Nx.shape(lag1) == {4}
  end

  test "lag-one covariances are non-negative for f=1 local-level model" do
    obs = Enum.map(1..20, fn i -> i * 1.0 + :math.sin(i) end)
    obs_tensor = Nx.tensor(obs, type: {:f, 32})

    {xs, ps} = KalmanFilter.filter_defn(obs_tensor, 1.0, 1.0, 0.5, 1.0, 0.0, 1.0)
    {_sxs, _sps, lag1} = Smoother.rts_defn_with_lag1(xs, ps, 1.0, 0.5)

    # For f=1 (random walk), lag-one covariances should be non-negative
    # because smoother gain c >= 0 and smoothed variance P >= 0
    lag1_list = Nx.to_flat_list(lag1)

    Enum.each(lag1_list, fn cov ->
      assert cov >= -1.0e-6,
             "lag-one covariance should be non-negative for f=1, got #{cov}"
    end)
  end

  test "lag-one covariances satisfy P_{i+1,i|T} = P_{i+1|T} * G_i" do
    obs = [1.0, 3.0, 2.0, 4.0, 5.0]
    f = 1.0
    q = 0.5
    r = 1.0

    obs_tensor = Nx.tensor(obs, type: {:f, 32})
    {xs, ps} = KalmanFilter.filter_defn(obs_tensor, f, 1.0, q, r, 0.0, 1.0)
    {_sxs, sps, lag1} = Smoother.rts_defn_with_lag1(xs, ps, f, q)

    # Manually verify: lag1[i] = sps[i+1] * gain[i]
    # gain[i] = ps_filt[i] * f / (f * ps_filt[i] * f + q)
    ps_filt_list = Nx.to_flat_list(ps)
    sps_list = Nx.to_flat_list(sps)
    lag1_list = Nx.to_flat_list(lag1)

    Enum.each(0..3, fn i ->
      p_filt = Enum.at(ps_filt_list, i)
      p_pred = f * p_filt * f + q
      gain = if abs(p_pred) < 1.0e-15, do: 0.0, else: p_filt * f / p_pred
      expected_lag1 = Enum.at(sps_list, i + 1) * gain
      actual_lag1 = Enum.at(lag1_list, i)

      assert_in_delta(expected_lag1, actual_lag1, 1.0e-4,
        "lag1[#{i}] mismatch: expected #{expected_lag1}, got #{actual_lag1}")
    end)
  end

  test "block-sum variance >= marginal-sum variance" do
    # Generate a correlated series where block-sum variance should be larger
    obs = Enum.map(1..30, fn i -> i * 0.5 + :math.sin(i * 0.3) * 2.0 end)
    obs_tensor = Nx.tensor(obs, type: {:f, 32})

    {xs, ps} = KalmanFilter.filter_defn(obs_tensor, 1.0, 1.0, 0.5, 1.0, 0.0, 1.0)
    {_sxs, sps, lag1} = Smoother.rts_defn_with_lag1(xs, ps, 1.0, 0.5)

    r = 1.0
    # Use a block of indices (simulating on-air minutes)
    block_indices = Enum.to_list(10..19)

    # Marginal-sum variance: Σ (P_t + r)
    sps_list = Nx.to_flat_list(sps)

    marginal_var =
      Enum.reduce(block_indices, 0.0, fn i, acc ->
        acc + Enum.at(sps_list, i) + r
      end)

    # Block-sum variance includes cross-covariances
    lag1_list = Nx.to_flat_list(lag1)
    ps_filt_list = Nx.to_flat_list(ps)

    # Compute smoother gains for recursive cross-covariance calculation
    gains =
      Enum.map(0..28, fn i ->
        p_filt = Enum.at(ps_filt_list, i)
        p_pred = 1.0 * p_filt * 1.0 + 0.5
        if abs(p_pred) < 1.0e-15, do: 0.0, else: p_filt * 1.0 / p_pred
      end)

    # Compute cross-covariance Cov(x_i, x_j) for i < j using recursion:
    # Cov(x_t, x_{t+1}) = lag1[t]
    # Cov(x_t, x_{t+k}) = gain[t] * Cov(x_{t+1}, x_{t+k})
    cross_cov_sum =
      for i <- block_indices, j <- block_indices, i < j, reduce: 0.0 do
        acc ->
          # Compute Cov(x_i, x_j) recursively
          cov = compute_cross_cov(i, j, lag1_list, gains)
          acc + 2.0 * cov
      end

    block_var = marginal_var + cross_cov_sum

    assert block_var >= marginal_var - 1.0e-6,
           "block-sum variance (#{block_var}) should be >= marginal-sum variance (#{marginal_var})"

    # The cross-covariance sum should be positive for a random walk
    assert cross_cov_sum > 0.0,
           "cross-covariance sum should be positive for correlated states"
  end

  # Recursively compute Cov(x_i, x_j) for i < j
  defp compute_cross_cov(i, j, lag1_list, _gains) when j == i + 1 do
    Enum.at(lag1_list, i)
  end

  defp compute_cross_cov(i, j, lag1_list, gains) when j > i + 1 do
    Enum.at(gains, i) * compute_cross_cov(i + 1, j, lag1_list, gains)
  end

  test "rts_defn_with_lag1 on single observation returns trivial lag1" do
    obs_tensor = Nx.tensor([5.0], type: {:f, 32})
    {xs, ps} = KalmanFilter.filter_defn(obs_tensor, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0)
    {sxs, sps, lag1} = Smoother.rts_defn_with_lag1(xs, ps, 1.0, 1.0)

    # For a single observation, smoothed == filtered
    assert_in_delta(Nx.to_number(sxs[0]), Nx.to_number(xs[0]), 1.0e-6)
    assert_in_delta(Nx.to_number(sps[0]), Nx.to_number(ps[0]), 1.0e-6)
    # lag1 is {1} with a zero element (no meaningful cross-covariances)
    assert Nx.shape(lag1) == {1}
    assert_in_delta(Nx.to_number(lag1[0]), 0.0, 1.0e-6)
  end
end
