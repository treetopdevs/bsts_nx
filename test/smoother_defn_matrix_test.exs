defmodule BstsNx.SmootherDefnMatrixTest do
  use ExUnit.Case, async: true

  alias BstsNx.{KalmanFilter, Smoother}
  alias BstsNx.Utils

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

  test "matrix RTS solve and pinv strategy implementations agree on stable and near-singular inputs" do
    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      for {label, xs, ps, f, q, tolerance} <- matrix_strategy_cases() do
        {sxs_solve, sps_solve} = Smoother.rts_defn_matrix_impl(xs, ps, f, q)
        {sxs_pinv, sps_pinv} = Smoother.rts_defn_matrix_impl_pinv(xs, ps, f, q)

        assert_all_close(sxs_solve, sxs_pinv, label, tolerance)
        assert_all_close(sps_solve, sps_pinv, label, tolerance)
        refute Utils.has_non_finite?(sxs_solve)
        refute Utils.has_non_finite?(sxs_pinv)
        refute Utils.has_non_finite?(sps_solve)
        refute Utils.has_non_finite?(sps_pinv)
      end
    end)
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

  test "compiled scalar smoother raises clearly for empty filtered inputs" do
    assert_raise ArgumentError, ~r/filtered_xs must contain at least one time step/, fn ->
      Smoother.rts_and_simulate_defn([], [], 1.0, 0.2, Nx.Random.key(77))
    end
  end

  test "compiled matrix smoother raises clearly for empty filtered inputs" do
    f = Nx.eye(2)
    q = Nx.eye(2)

    assert_raise ArgumentError, ~r/filtered_xs must contain at least one time step/, fn ->
      Smoother.rts_defn_matrix([], [], f, q)
    end

    assert_raise ArgumentError, ~r/filtered_xs must contain at least one time step/, fn ->
      Smoother.simulate_from_filtered_defn_matrix([], [], f, q, Nx.Random.key(77))
    end
  end

  if @emlx_backend? do
    @tag skip:
           "EMLX backend does not implement Nx.Backend.lu/3 required by matrix simulation path"
  end

  test "simulate_from_filtered_defn_matrix returns deterministic samples for a fixed key" do
    observations = Nx.tensor([1.0, 1.4, 1.9, 2.2], type: {:f, 32})

    f = Nx.tensor([[1.0, 1.0], [0.0, 1.0]])
    h = Nx.tensor([[1.0, 0.0]])
    q = Nx.tensor([[0.1, 0.0], [0.0, 0.02]])
    r = Nx.tensor(0.15)
    x0 = Nx.tensor([0.0, 0.0])
    p0 = Nx.eye(2)

    {xs, ps} = KalmanFilter.filter_defn_multi(observations, f, h, q, r, x0, p0)

    key = Nx.Random.key(404)
    {states_a, key_a} = Smoother.simulate_from_filtered_defn_matrix(xs, ps, f, q, key)
    {states_b, key_b} = Smoother.simulate_from_filtered_defn_matrix(xs, ps, f, q, key)

    assert Nx.shape(states_a) == {4, 2}
    assert Nx.all_close(states_a, states_b, atol: 1.0e-6, rtol: 1.0e-6) |> Nx.to_number() == 1
    assert Nx.to_flat_list(key_a) == Nx.to_flat_list(key_b)
  end

  test "matrix simulation solve and pinv strategy implementations agree for a fixed key" do
    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      {_label, xs, ps, f, q, tolerance} = stable_matrix_strategy_case()
      key = Nx.Random.key(909)

      {states_solve, key_solve} =
        Smoother.simulate_from_filtered_defn_matrix_impl_solve(xs, ps, f, q, key)

      {states_pinv, key_pinv} =
        Smoother.simulate_from_filtered_defn_matrix_impl_pinv(xs, ps, f, q, key)

      assert Nx.shape(states_solve) == Nx.shape(states_pinv)
      assert Nx.shape(key_solve) == Nx.shape(key_pinv)
      assert_all_close(states_solve, states_pinv, "simulation strategy parity", tolerance)
      assert Nx.to_flat_list(key_solve) == Nx.to_flat_list(key_pinv)
      refute Utils.has_non_finite?(states_solve)
      refute Utils.has_non_finite?(states_pinv)
    end)
  end

  test "simulate_from_filtered_defn_matrix uses all-or-nothing fallback for partial-NaN cholesky retry" do
    filtered_xs = Nx.tensor([[0.0, 0.0]], type: {:f, 64})
    filtered_ps = Nx.tensor([[[1.0, :nan], [:nan, 1.0]]], type: {:f, 64})
    f = Nx.eye(2, type: {:f, 64})
    q = Nx.eye(2, type: {:f, 64})
    key = Nx.Random.key(123)

    {states, _key_out} =
      Smoother.simulate_from_filtered_defn_matrix(filtered_xs, filtered_ps, f, q, key)

    {eps, _} = Nx.Random.normal(key, 0.0, 1.0, shape: {1, 2})
    fallback_scale = :math.sqrt(0.001001000001)
    expected = Nx.multiply(eps, fallback_scale)

    assert Nx.all_close(states, expected, atol: 1.0e-8, rtol: 1.0e-8) |> Nx.to_number() == 1
  end

  defp assert_all_close(left, right, label, tolerance) do
    assert Nx.all_close(left, right, atol: tolerance, rtol: tolerance) |> Nx.to_number() == 1,
           "#{label} tensors differ beyond #{tolerance}"
  end

  defp matrix_strategy_cases do
    [
      stable_matrix_strategy_case(),
      near_singular_matrix_strategy_case()
    ]
  end

  defp stable_matrix_strategy_case do
    {
      "stable",
      Nx.tensor(
        [
          [0.2, -0.1],
          [0.8, 0.1],
          [1.1, 0.25],
          [1.4, 0.2],
          [1.9, 0.15]
        ],
        type: {:f, 64}
      ),
      Nx.tensor(
        [
          [[1.4, 0.1], [0.1, 0.8]],
          [[1.1, 0.08], [0.08, 0.6]],
          [[0.9, 0.04], [0.04, 0.45]],
          [[0.7, 0.03], [0.03, 0.35]],
          [[0.55, 0.02], [0.02, 0.25]]
        ],
        type: {:f, 64}
      ),
      Nx.tensor([[1.0, 0.2], [0.0, 0.9]], type: {:f, 64}),
      Nx.tensor([[0.15, 0.01], [0.01, 0.08]], type: {:f, 64}),
      1.0e-8
    }
  end

  defp near_singular_matrix_strategy_case do
    {
      "near-singular",
      Nx.tensor(
        [
          [0.01, 0.0],
          [0.015, 0.001],
          [0.018, 0.0015],
          [0.02, 0.0017]
        ],
        type: {:f, 64}
      ),
      Nx.tensor(
        [
          [[1.0e-7, 2.0e-10], [2.0e-10, 2.0e-7]],
          [[1.2e-7, 1.0e-10], [1.0e-10, 1.8e-7]],
          [[1.1e-7, 0.0], [0.0, 1.6e-7]],
          [[9.0e-8, 0.0], [0.0, 1.4e-7]]
        ],
        type: {:f, 64}
      ),
      Nx.tensor([[0.98, 0.03], [0.0, 0.95]], type: {:f, 64}),
      Nx.tensor([[1.0e-8, 0.0], [0.0, 1.0e-8]], type: {:f, 64}),
      1.0e-3
    }
  end
end
