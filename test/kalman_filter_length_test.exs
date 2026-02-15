defmodule BstsNxKalmanFilterLengthTest do
  use ExUnit.Case
  alias BstsNx.KalmanFilter

  test "raises when h list length mismatches observations" do
    observations = [1.0, 2.0, 3.0]
    h = [1.0, 1.0]

    assert_raise ArgumentError, ~r/observation matrix length/, fn ->
      KalmanFilter.filter_with_pred(observations, 1.0, h, 1.0, 1.0, 0.0, 1.0)
    end
  end
end
