defmodule BstsNx.KalmanFilterRShapeTest do
  use ExUnit.Case, async: true

  alias BstsNx.KalmanFilter

  test "filter_with_pred accepts single-element list r" do
    observations = [1.0, 1.2, 0.9]

    {filtered, predicted} =
      KalmanFilter.filter_with_pred(
        observations,
        1.0,
        1.0,
        0.1,
        [0.2],
        0.0,
        1.0
      )

    assert length(filtered) == length(observations)
    assert length(predicted) == length(observations)
  end

  test "filter_defn accepts rank-2 h column vector with shape {t, 1}" do
    observations = Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32})
    h = Nx.tensor([[1.0], [2.0], [1.0]], type: {:f, 32})

    {xs, ps} = KalmanFilter.filter_defn(observations, 1.0, h, 0.1, 0.5, 0.0, 1.0)

    assert Nx.shape(xs) == {3}
    assert Nx.shape(ps) == {3}
  end

  test "filter_defn raises clearly on unsupported rank-2 h shape" do
    observations = Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32})
    bad_h = Nx.tensor([[1.0, 0.0], [2.0, 0.0], [1.0, 0.0]], type: {:f, 32})

    assert_raise ArgumentError, ~r/rank-2 shape \{3, 1\}/, fn ->
      KalmanFilter.filter_defn(observations, 1.0, bad_h, 0.1, 0.5, 0.0, 1.0)
    end
  end
end
