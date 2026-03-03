defmodule BstsNxTest do
  alias BstsNx.{Components, InterventionAnalysis}

  def run do
    Nx.global_default_backend(EXLA.Backend)
    # Fake observations
    obs =
      Enum.map(1..100, fn i ->
        base = if i > 80, do: 50.0, else: 10.0
        base + :math.sin(i * :math.pi() * 2 / 7) * 5 + :rand.normal() * 2
      end)

    # Composed structured model: local linear + weekly seasonality
    spec =
      Components.compose_specs(
        Components.local_linear_trend_spec(var_level: 1.0, var_slope: 0.1),
        Components.seasonal_spec(7)
      )

    opts = [
      model_spec: spec,
      method: :filter
    ]

    config = %{
      pre_period: {1, 80},
      post_period: {81, 100}
    }

    {time_compile, result} = :timer.tc(fn -> InterventionAnalysis.analyze(obs, config, opts) end)
    {time_exec, _result2} = :timer.tc(fn -> InterventionAnalysis.analyze(obs, config, opts) end)

    IO.puts("JIT run completed in #{time_compile / 1_000.0} ms")
    IO.puts("Cached run completed in #{time_exec / 1_000.0} ms")
    IO.puts("Significant? #{result.significant?}")
    IO.puts("Cum Effect: #{result.summary.cumulative_effect.mean}")
  end
end

BstsNxTest.run()
