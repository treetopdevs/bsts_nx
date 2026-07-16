defmodule BstsNxSmootherRtsDedupTest do
  use ExUnit.Case, async: true

  alias BstsNx.KalmanFilter
  alias BstsNx.Smoother

  test "rts_defn matches the means/covariances from rts_defn_with_lag1" do
    obs = Nx.tensor([1.0, 2.0, 1.5, 3.0, 2.5, 4.0], type: {:f, 64})
    {xs, ps} = KalmanFilter.filter_defn(obs, 1.0, 1.0, 0.2, 0.6, 0.0, 1.0)

    {sxs_a, sps_a} = Smoother.rts_defn(xs, ps, 1.0, 0.2)
    {sxs_b, sps_b, _lag1} = Smoother.rts_defn_with_lag1(xs, ps, 1.0, 0.2)

    assert Nx.to_flat_list(sxs_a) == Nx.to_flat_list(sxs_b)
    assert Nx.to_flat_list(sps_a) == Nx.to_flat_list(sps_b)
  end

  test "rts_defn preserves f64 output dtype" do
    obs = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: {:f, 64})
    {xs, ps} = KalmanFilter.filter_defn(obs, 1.0, 1.0, 0.1, 0.5, 0.0, 1.0)
    {sxs, sps} = Smoother.rts_defn(xs, ps, 1.0, 0.1)

    assert Nx.type(sxs) == {:f, 64}
    assert Nx.type(sps) == {:f, 64}
  end
end
