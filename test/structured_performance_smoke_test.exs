defmodule BstsNx.StructuredPerformanceSmokeTest do
  use ExUnit.Case, async: false
  @moduletag timeout: 180_000

  alias BstsNx.{Components, GibbsSampler}

  test "small structured sampler run completes within a smoke-test budget" do
    observations =
      Enum.map(0..47, fn t ->
        100.0 + 0.15 * t + 2.5 * :math.sin(2.0 * :math.pi() * t / 12.0)
      end)

    regressors =
      Enum.map(0..47, fn t ->
        [1.0 + :math.cos(t / 8.0)]
      end)
      |> Nx.tensor()

    trend_spec =
      Components.local_linear_trend_spec(
        initial_level: hd(observations),
        var_level: 0.2,
        var_slope: 0.02,
        obs_var: 2.0
      )

    seasonal_spec = Components.seasonal_spec(6, process_var: 0.05, obs_var: 2.0)
    regression_spec = Components.regression_spec(regressors, var_beta: 0.02, obs_var: 2.0)

    spec =
      trend_spec
      |> Components.compose_specs(seasonal_spec)
      |> Components.compose_specs(regression_spec)

    {elapsed_us, samples} =
      :timer.tc(fn ->
        GibbsSampler.sample_structured(observations, spec, 3, burn_in: 1, seed: 123)
      end)

    budget_us =
      case Nx.default_backend() do
        EXLA.Backend -> 25_000_000
        {EXLA.Backend, _opts} -> 25_000_000
        EMLX.Backend -> 90_000_000
        {EMLX.Backend, _opts} -> 90_000_000
        _ -> 8_000_000
      end

    assert length(samples) == 3
    assert elapsed_us < budget_us
  end
end
