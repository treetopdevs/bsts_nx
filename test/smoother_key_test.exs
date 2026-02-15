defmodule BstsNxSmootherKeyTest do
  use ExUnit.Case
  alias BstsNx.KalmanFilter
  alias BstsNx.Smoother

  test "simulate_with_key returns a new key when none is provided" do
    observations = [1.0, 2.0, 3.0]

    {filtered, predicted} =
      KalmanFilter.filter_with_pred(observations, 1.0, 1.0, 0.1, 0.5, 0.0, 1.0)

    smoothed = Smoother.rts(filtered, predicted, 1.0)

    {_states1, key1} = Smoother.simulate_with_key(smoothed, filtered, predicted, 1.0)
    {_states2, key2} = Smoother.simulate_with_key(smoothed, filtered, predicted, 1.0)

    assert match?(%Nx.Tensor{}, key1)
    assert match?(%Nx.Tensor{}, key2)
    refute Nx.to_flat_list(key1) == Nx.to_flat_list(key2)
  end
end
