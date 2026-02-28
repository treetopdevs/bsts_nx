defmodule BstsNx.SmootherDefnMatrixTest do
  use ExUnit.Case, async: true

  alias BstsNx.{KalmanFilter, Smoother}

  @emlx_backend? match?(EMLX.Backend, Nx.default_backend()) or
                   match?({EMLX.Backend, _}, Nx.default_backend())

  if @emlx_backend? do
    @tag skip: "EMLX backend does not implement Nx.Backend.lu/3 required by matrix RTS defn path"
  end

  test "rts_defn_matrix matches eager rts for matrix state models" do
    nan = Nx.Constants.nan() |> Nx.to_number()
    observations = [1.0, 1.4, nan, 2.1, 2.6]
    obs_t = Nx.tensor(observations, type: {:f, 32})

    f = Nx.tensor([[1.0, 1.0], [0.0, 1.0]])
    h = Nx.tensor([[1.0, 0.0]])
    q = Nx.tensor([[0.1, 0.0], [0.0, 0.02]])
    r = Nx.tensor(0.15)
    x0 = Nx.tensor([0.0, 0.0])
    p0 = Nx.eye(2)

    {filtered_eager, predicted_eager} =
      KalmanFilter.filter_with_pred(observations, f, h, q, r, x0, p0)

    smoothed_eager = Smoother.rts(filtered_eager, predicted_eager, f)

    sxs_eager = smoothed_eager |> Enum.map(&elem(&1, 0)) |> Nx.stack()
    sps_eager = smoothed_eager |> Enum.map(&elem(&1, 1)) |> Nx.stack()

    {xs, ps} = KalmanFilter.filter_defn_multi(obs_t, f, h, q, r, x0, p0)
    {sxs_defn, sps_defn} = Smoother.rts_defn_matrix(xs, ps, f, q)

    assert Nx.shape(sxs_defn) == Nx.shape(sxs_eager)
    assert Nx.shape(sps_defn) == Nx.shape(sps_eager)
    assert Nx.all_close(sxs_defn, sxs_eager, atol: 1.0e-4, rtol: 1.0e-4) |> Nx.to_number() == 1
    assert Nx.all_close(sps_defn, sps_eager, atol: 1.0e-4, rtol: 1.0e-4) |> Nx.to_number() == 1
  end

  test "rts_and_simulate_defn returns smoothing outputs plus one sampled path" do
    observations = Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32})
    {xs, ps} = KalmanFilter.filter_defn(observations, 1.0, 1.0, 0.2, 0.3, 0.0, 1.0)

    {sxs, sps, path, key_out} =
      Smoother.rts_and_simulate_defn(xs, ps, 1.0, 0.2, Nx.Random.key(77))

    assert Nx.shape(sxs) == {3}
    assert Nx.shape(sps) == {3}
    assert Nx.shape(path) == {3}
    assert Nx.shape(key_out) == {2}
  end
end
