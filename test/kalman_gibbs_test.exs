defmodule BstsNxKalmanGibbsTest do
  use ExUnit.Case, async: true
  import Nx, only: [to_number: 1]
  alias BstsNx.KalmanFilter
  alias BstsNx.GibbsSampler

  describe "KalmanFilter.filter/7" do
    test "simple scalar system" do
      observations = [1.0, 2.0, 3.0]
      # scalar parameters
      f = 1.0
      h = 1.0
      q = 1.0
      r = 1.0
      x0 = 0.0
      p0 = 1.0
      estimates = KalmanFilter.filter(observations, f, h, q, r, x0, p0)

      expected = [
        {2.0 / 3.0, 2.0 / 3.0},
        {1.5, 0.625},
        {2.428571428571429, 0.6190476190476191}
      ]

      Enum.zip(estimates, expected)
      |> Enum.each(fn {{x_t, p_t}, {x_exp, p_exp}} ->
        assert_in_delta(to_number(x_t), x_exp, 1.0e-6)
        assert_in_delta(to_number(p_t), p_exp, 1.0e-6)
      end)
    end

    test "empty observation list" do
      estimates = KalmanFilter.filter([], 1.0, 1.0, 1.0, 1.0, 0.0, 1.0)
      assert estimates == []
    end
  end

  describe "GibbsSampler.sample/6" do
    test "returns correct number of samples and state length" do
      observations = [1.0, 2.0, 3.0]
      samples = GibbsSampler.sample(observations, 3, 0.0, 1.0, 1.0, 1.0, seed: 42)
      assert length(samples) == 3

      Enum.each(samples, fn sample ->
        # Each sample should contain a state for each observation
        assert length(sample.states) == length(observations)
        assert length(sample.state_covs) == length(observations)
        # Variance placeholders should be tensors/scalars
        assert is_number(Nx.to_number(sample.process_var))
        assert is_number(Nx.to_number(sample.obs_var))
      end)
    end
  end
end
