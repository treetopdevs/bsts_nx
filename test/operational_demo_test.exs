defmodule BstsNx.OperationalDemoTest do
  use ExUnit.Case, async: true

  alias BstsNx.{CausalImpact, Components, Pipeline}

  test "fast filter estimate stays within demo latency budget" do
    observations = demo_observations()
    intervention_indices = Enum.to_list(96..143)

    {elapsed_us, result} =
      :timer.tc(fn ->
        CausalImpact.estimate_from_filter(observations, intervention_indices,
          q: 0.8,
          r: 4.0,
          x0: List.first(observations),
          p0: 10.0
        )
      end)

    assert result.cumulative_effect.mean > 0
    assert elapsed_us < 1_000_000
  end

  test "operational pipeline stays within useful batch-demo latency budget" do
    observations = demo_observations()
    spec = Components.local_level_spec(initial_state: List.first(observations), initial_cov: 10.0)

    spots = [
      %{id: "spot_a", window_start: 0, window_end: 12},
      %{id: "spot_b", window_start: 12, window_end: 24}
    ]

    {elapsed_us, result} =
      :timer.tc(fn ->
        Pipeline.run(observations, {1, 96}, {97, 144}, spots, spec,
          mode: :operational,
          alpha: 0.05,
          shapley_samples: 250
        )
      end)

    assert result.execution.mode == :operational
    assert result.execution.baseline == :forecast
    assert result.execution.method_used == :scalar_forecast_filter
    assert length(result.attributions.attributions) == 2
    assert elapsed_us < 100_000
  end

  defp demo_observations do
    Enum.map(0..143, fn t ->
      baseline = 80.0 + 0.05 * t + 7.0 * :math.sin(2.0 * :math.pi() * t / 24.0)
      lift = if t >= 96 and t <= 143, do: 12.0, else: 0.0
      baseline + lift
    end)
  end
end
