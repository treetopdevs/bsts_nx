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
end
