defmodule BstsNx.KalmanFilterMultivariateObservationTest do
  use ExUnit.Case, async: true

  alias BstsNx.KalmanFilter

  test "rank-1 h is rejected for multivariate observations" do
    obs = [Nx.tensor([1.0, 2.0]), Nx.tensor([1.1, 1.9])]
    h = Nx.tensor([1.0, 1.0])

    assert_raise ArgumentError, ~r/rank-1 observation matrix h implies scalar observations/, fn ->
      KalmanFilter.filter_with_pred(
        obs,
        Nx.eye(2),
        h,
        Nx.eye(2),
        Nx.eye(2),
        Nx.tensor([0.0, 0.0]),
        Nx.eye(2)
      )
    end
  end

  test "static rank-2 h row count must match multivariate observation dimension" do
    obs = [Nx.tensor([1.0, 2.0]), Nx.tensor([1.1, 1.9])]
    bad_h = Nx.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])

    assert_raise ArgumentError,
                 ~r/static observation matrix row count 3 must match observation dimension 2/,
                 fn ->
                   KalmanFilter.filter_with_pred(
                     obs,
                     Nx.eye(2),
                     bad_h,
                     Nx.eye(2),
                     Nx.eye(2),
                     Nx.tensor([0.0, 0.0]),
                     Nx.eye(2)
                   )
                 end
  end

  test "rank-3 time-varying multivariate h works" do
    obs = [Nx.tensor([1.0, 2.0]), Nx.tensor([1.1, 1.9])]

    h_tv =
      Nx.tensor([
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.1], [0.0, 1.0]]
      ])

    {filtered, predicted} =
      KalmanFilter.filter_with_pred(
        obs,
        Nx.eye(2),
        h_tv,
        Nx.eye(2) |> Nx.multiply(0.1),
        Nx.eye(2) |> Nx.multiply(0.1),
        Nx.tensor([0.0, 0.0]),
        Nx.eye(2)
      )

    assert length(filtered) == 2
    assert length(predicted) == 2

    Enum.each(filtered, fn {x, p} ->
      assert Nx.shape(x) == {2}
      assert Nx.shape(p) == {2, 2}
    end)
  end

  test "invalid h type raises" do
    assert_raise ArgumentError, ~r/invalid h/, fn ->
      KalmanFilter.filter_with_pred([1.0, 2.0], 1.0, :bad_h, 1.0, 1.0, 0.0, 1.0)
    end
  end

  test "list observations with multivariate values infer observation dimension correctly" do
    obs = [[1.0, 2.0], [1.1, 1.9]]

    {filtered, _predicted} =
      KalmanFilter.filter_with_pred(
        obs,
        Nx.eye(2),
        Nx.eye(2),
        Nx.eye(2) |> Nx.multiply(0.1),
        Nx.eye(2) |> Nx.multiply(0.1),
        Nx.tensor([0.0, 0.0]),
        Nx.eye(2)
      )

    assert length(filtered) == 2
  end

  test "all-missing observations default observation dimension to scalar path" do
    obs = [nil, nil, nil]
    h_tv = Nx.tensor([[1.0], [0.9], [1.1]])

    {filtered, predicted} = KalmanFilter.filter_with_pred(obs, 1.0, h_tv, 0.1, 0.5, 0.0, 1.0)

    assert length(filtered) == 3
    assert length(predicted) == 3
  end
end
