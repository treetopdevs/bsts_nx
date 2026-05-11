defmodule BstsNx.GibbsSamplerDefnVarianceTest do
  use ExUnit.Case, async: true

  alias BstsNx.Components
  alias BstsNx.GibbsSampler

  test "scalar sampler resamples process and observation variances with defn inverse-gamma" do
    observations = [1.0, 1.1, 0.9, 1.2, 1.0, 1.15]

    samples =
      GibbsSampler.sample_general(observations, 1.0, 1.0, 3,
        initial_state: 1.0,
        initial_cov: 1.0,
        process_var: 0.2,
        obs_var: 0.2,
        prior_shape: 0.25,
        prior_scale: 0.75,
        seed: 2027
      )

    assert length(samples) == 3

    Enum.each(samples, fn sample ->
      assert finite_positive_tensor?(sample.process_var)
      assert finite_positive_tensor?(sample.obs_var)
    end)
  end

  test "structured sampler batches Q component inverse-gamma draws" do
    observations = Enum.map(1..8, &(&1 * 0.2))

    spec =
      Components.local_linear_trend_spec(
        var_level: 0.2,
        var_slope: 0.05,
        obs_var: 0.2,
        initial_cov_level: 1.0,
        initial_cov_slope: 1.0,
        prior_shape: 0.25,
        prior_scale: 0.75
      )

    samples = GibbsSampler.sample_structured(observations, spec, 2, burn_in: 1, seed: 3031)

    assert length(samples) == 2

    Enum.each(samples, fn sample ->
      assert Nx.shape(sample.q_matrix) == {2, 2}
      assert finite_positive_tensor?(sample.q_matrix[0][0])
      assert finite_positive_tensor?(sample.q_matrix[1][1])
      assert abs(Nx.to_number(sample.q_matrix[0][1])) < 1.0e-10
      assert abs(Nx.to_number(sample.q_matrix[1][0])) < 1.0e-10
      assert finite_positive_tensor?(sample.obs_var)
    end)
  end

  defp finite_positive_tensor?(tensor) do
    value = Nx.to_number(tensor)
    is_number(value) and value > 0.0 and value == value and value != :infinity
  end
end
