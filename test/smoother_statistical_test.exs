defmodule BstsNx.SmootherStatisticalTest do
  use ExUnit.Case, async: true

  alias BstsNx.Smoother

  test "simulate_with_key matches terminal Gaussian moments for one-step model" do
    mean = Nx.tensor([1.5, -0.7])
    cov = Nx.tensor([[1.2, 0.35], [0.35, 0.9]])
    f = Nx.eye(2)
    smoothed = [{mean, cov}]
    filtered = smoothed
    predicted = smoothed

    n = 3_000
    key0 = Nx.Random.key(2026)

    {draws, _final_key} =
      Enum.map_reduce(1..n, key0, fn _, key_acc ->
        {[x_t], key_next} =
          Smoother.simulate_with_key(smoothed, filtered, predicted, f, key: key_acc)

        {Nx.to_flat_list(x_t), key_next}
      end)

    {sum_x1, sum_x2, sum_x1_sq, sum_x2_sq, sum_x1_x2} =
      Enum.reduce(draws, {0.0, 0.0, 0.0, 0.0, 0.0}, fn [x1, x2], {sx1, sx2, sx1s, sx2s, sx1x2} ->
        {sx1 + x1, sx2 + x2, sx1s + x1 * x1, sx2s + x2 * x2, sx1x2 + x1 * x2}
      end)

    mean_x1 = sum_x1 / n
    mean_x2 = sum_x2 / n
    cov_x11 = sum_x1_sq / n - mean_x1 * mean_x1
    cov_x22 = sum_x2_sq / n - mean_x2 * mean_x2
    cov_x12 = sum_x1_x2 / n - mean_x1 * mean_x2

    [target_mean_x1, target_mean_x2] = Nx.to_flat_list(mean)
    [target_cov_x11, target_cov_x12, _target_cov_x21, target_cov_x22] = Nx.to_flat_list(cov)

    assert_in_delta mean_x1, target_mean_x1, 0.08
    assert_in_delta mean_x2, target_mean_x2, 0.08
    assert_in_delta cov_x11, target_cov_x11, 0.12
    assert_in_delta cov_x22, target_cov_x22, 0.12
    assert_in_delta cov_x12, target_cov_x12, 0.1
  end
end
