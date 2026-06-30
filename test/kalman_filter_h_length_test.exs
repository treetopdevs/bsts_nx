defmodule BstsNxKalmanFilterHLengthTest do
  use ExUnit.Case, async: true

  alias BstsNx.KalmanFilter

  test "rejects a rank-2 H whose row count is neither 1 nor the series length (scalar obs)" do
    obs = [1.0, 2.0, 3.0]
    # {4, 1}: 4 rows, but the series has 3 scalar observations and a static H
    # would be {1, 1}. 4 is ambiguous.
    h = Nx.tensor([[1.0], [1.0], [1.0], [1.0]])

    assert_raise ArgumentError, ~r/ambiguous/, fn ->
      KalmanFilter.filter(obs, 1.0, h, 0.1, 0.5, 0.0, 1.0)
    end
  end

  test "still accepts a {1, n} static row for scalar observations" do
    obs = [1.0, 2.0, 3.0]
    h = Nx.tensor([[1.0]])
    result = KalmanFilter.filter(obs, 1.0, h, 0.1, 0.5, 0.0, 1.0)
    assert length(result) == 3
  end

  test "still accepts a {T, n} time-varying H for scalar observations" do
    obs = [1.0, 2.0, 3.0]
    h = Nx.tensor([[1.0], [1.0], [1.0]])
    result = KalmanFilter.filter(obs, 1.0, h, 0.1, 0.5, 0.0, 1.0)
    assert length(result) == 3
  end
end
