defmodule BstsNx.SmootherRobustnessTest do
  use ExUnit.Case, async: true
  import ExUnit.CaptureLog

  alias BstsNx.Smoother

  test "simulate_with_key projects terminal covariance to PSD before sampling" do
    bad_cov = Nx.tensor([[1.0, 2.0], [2.0, 1.0]])
    smoothed = [{Nx.tensor([0.0, 0.0]), bad_cov}]
    filtered = smoothed
    predicted = smoothed
    noise = [Nx.tensor([1.0, 0.0])]

    log =
      capture_log(fn ->
        {states, _key} =
          Smoother.simulate_with_key(smoothed, filtered, predicted, Nx.eye(2), noise_list: noise)

        assert length(states) == 1
        [x0] = states
        assert Nx.shape(x0) == {2}
        assert Enum.any?(Nx.to_flat_list(x0), &(abs(&1) > 1.0e-6))
      end)

    assert log =~ "terminal covariance not positive-definite"
    assert log =~ "projecting to nearest PSD covariance"
  end

  test "simulate_with_key projects non-PD conditional covariance instead of zeroing noise" do
    x0 = Nx.tensor([0.0, 0.0])
    x1 = Nx.tensor([0.0, 0.0])
    identity = Nx.eye(2)
    small = Nx.eye(2) |> Nx.multiply(0.1)

    # p_pred at t=1 is intentionally much smaller than p_filt at t=0,
    # which makes the conditional covariance at t=0 negative.
    smoothed = [{x0, identity}, {x1, identity}]
    filtered = [{x0, identity}, {x1, identity}]
    predicted = [{x0, identity}, {x1, small}]
    noise = [Nx.tensor([1.0, 0.0]), Nx.tensor([0.0, 0.0])]

    log =
      capture_log(fn ->
        {states, _key} =
          Smoother.simulate_with_key(smoothed, filtered, predicted, identity, noise_list: noise)

        assert length(states) == 2
        [xk, _x_next] = states
        assert Enum.any?(Nx.to_flat_list(xk), &(abs(&1) > 1.0e-8))
      end)

    assert log =~ "conditional covariance at step 0 not positive-definite"
    assert log =~ "projecting to nearest PSD covariance"
  end
end
