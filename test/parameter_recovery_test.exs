defmodule BstsNx.ParameterRecoveryTest do
  use ExUnit.Case

  alias BstsNx.Components
  alias BstsNx.Diagnostics
  alias BstsNx.GibbsSampler

  @moduletag :external
  @moduletag timeout: 180_000

  test "local linear trend sampler covers true variances with 95% HPD intervals" do
    :rand.seed(:exsss, {1_111, 1_112, 1_113})

    n = 160
    true_obs_var = 2.0
    true_level_var = 0.5
    true_slope_var = 0.12

    {_level, _slope, observations} =
      Enum.reduce(1..n, {100.0, 0.2, []}, fn _i, {level, slope, acc} ->
        slope_new = slope + :rand.normal() * :math.sqrt(true_slope_var)
        level_new = level + slope + :rand.normal() * :math.sqrt(true_level_var)
        y = level_new + :rand.normal() * :math.sqrt(true_obs_var)
        {level_new, slope_new, [y | acc]}
      end)

    observations = Enum.reverse(observations)

    spec =
      Components.local_linear_trend_spec(
        var_level: 1.0,
        var_slope: 0.2,
        obs_var: 1.0,
        initial_level: 100.0,
        initial_slope: 0.0,
        initial_cov_level: 20.0,
        initial_cov_slope: 2.0,
        prior_shape: 2.0,
        prior_scale: 1.0
      )

    samples = GibbsSampler.sample_structured(observations, spec, 160, burn_in: 160, seed: 1234)

    level_trace = Enum.map(samples, fn s -> Nx.to_number(s.q_matrix[0][0]) end)
    slope_trace = Enum.map(samples, fn s -> Nx.to_number(s.q_matrix[1][1]) end)
    obs_trace = Enum.map(samples, fn s -> Nx.to_number(s.obs_var) end)

    {level_low, level_high} = Diagnostics.hpd_interval(level_trace, 0.95)
    {slope_low, slope_high} = Diagnostics.hpd_interval(slope_trace, 0.95)
    {obs_low, obs_high} = Diagnostics.hpd_interval(obs_trace, 0.95)

    assert true_level_var >= level_low and true_level_var <= level_high
    assert true_slope_var >= slope_low and true_slope_var <= slope_high
    assert true_obs_var >= obs_low and true_obs_var <= obs_high
  end
end
