Mix.Task.run("app.start")

alias BstsNx.{Components, GibbsSampler, Operational}

defmodule BstsNx.Bench.OptimizePlan do
  @moduledoc false

  def run do
    seed = 123
    scalar_obs = scalar_observations(200)
    structured_obs = structured_observations(96)
    structured_spec = structured_spec(structured_obs)
    operational_obs = operational_observations()

    scalar_operational_spec =
      Components.local_level_spec(initial_state: hd(operational_obs), initial_cov: 10.0)

    structured_operational_spec =
      Components.local_linear_trend_spec(
        initial_level: hd(operational_obs),
        initial_slope: 0.0,
        initial_cov_level: 10.0,
        initial_cov_slope: 1.0,
        var_level: 0.8,
        var_slope: 0.02,
        obs_var: 4.0
      )

    operational_spots = [
      %{id: "spot_a", window_start: 0, window_end: 12},
      %{id: "spot_b", window_start: 12, window_end: 24}
    ]

    prepared_scalar = Operational.prepare(scalar_operational_spec, {1, 96}, {97, 144})
    prepared_structured = Operational.prepare(structured_operational_spec, {1, 96}, {97, 144})

    scalar_forecast =
      benchmark_warm(fn ->
        Operational.forecast(prepared_scalar, operational_obs)
      end)

    structured_forecast =
      benchmark_warm(fn ->
        Operational.forecast(prepared_structured, operational_obs)
      end)

    guide_pipeline =
      benchmark_warm(fn ->
        Operational.run(prepared_scalar, operational_obs, operational_spots,
          alpha: 0.05,
          shapley_samples: 250
        )
      end)

    batch_obs = batch_operational_observations(2_000)
    batch_spec = structured_spec(batch_obs)
    batch_prepared = Operational.prepare(batch_spec, {1, 1_500}, {1_501, 2_000})

    batch_pipeline =
      benchmark_warm(fn ->
        Operational.run(batch_prepared, batch_obs, operational_spots, shapley_samples: 250)
      end)

    scalar = benchmark(fn -> run_scalar(scalar_obs, seed) end)
    structured = benchmark(fn -> run_structured(structured_obs, structured_spec, seed) end)

    results = %{
      generated_at: DateTime.utc_now() |> DateTime.to_iso8601(),
      git_sha: git_sha(),
      backend: inspect(Nx.default_backend()),
      budgets_us: %{
        prepared_scalar_forecast: 25_000,
        prepared_structured_forecast: 100_000,
        guide_operational_pipeline: 100_000,
        batch_operational_pipeline: 1_000_000
      },
      scenarios: %{
        prepared_scalar_forecast: scalar_forecast,
        prepared_structured_forecast: structured_forecast,
        guide_operational_pipeline: guide_pipeline,
        batch_operational_pipeline: batch_pipeline,
        scalar_gibbs: scalar,
        structured_gibbs: structured
      }
    }

    output_path =
      System.get_env("OPTIMIZE_PLAN_OUTPUT") || Path.join(["bench", "results", "current.json"])

    File.mkdir_p!(Path.dirname(output_path))
    json = encode_json(results)
    File.write!(output_path, json <> "\n")

    IO.puts("Wrote benchmark results to #{output_path}")
    IO.puts(json)
    enforce_operational_budgets!(results)
  end

  defp run_scalar(observations, seed) do
    samples = GibbsSampler.sample(observations, 200, 0.0, 1.0, 0.1, 0.1, seed: seed)
    %{num_samples: length(samples)}
  end

  defp run_structured(observations, spec, seed) do
    samples = GibbsSampler.sample_structured(observations, spec, 120, burn_in: 40, seed: seed)
    %{num_samples: length(samples)}
  end

  defp structured_spec(observations) do
    Components.local_linear_trend_spec(
      initial_level: hd(observations),
      var_level: 0.1,
      var_slope: 0.01,
      obs_var: 1.0
    )
  end

  defp scalar_observations(n) do
    Enum.map(0..(n - 1), fn t ->
      trend = 10.0 + 0.03 * t
      seasonal = 0.8 * :math.sin(2.0 * :math.pi() * t / 24.0)
      trend + seasonal
    end)
  end

  defp structured_observations(n) do
    Enum.map(0..(n - 1), fn t ->
      100.0 + 0.2 * t + 2.0 * :math.sin(2.0 * :math.pi() * t / 12.0)
    end)
  end

  defp operational_observations do
    Enum.map(0..143, fn t ->
      baseline = 80.0 + 0.05 * t + 7.0 * :math.sin(2.0 * :math.pi() * t / 24.0)
      lift = if t >= 96, do: 12.0, else: 0.0
      baseline + lift
    end)
  end

  defp batch_operational_observations(n) do
    Enum.map(0..(n - 1), fn t ->
      baseline = 100.0 + 0.02 * t + 8.0 * :math.sin(2.0 * :math.pi() * t / 24.0)
      lift = if t >= 1_500, do: 9.0, else: 0.0
      baseline + lift
    end)
  end

  defp benchmark(fun) do
    {elapsed_us, result} = :timer.tc(fun)
    Map.put(result, :elapsed_us, elapsed_us)
  end

  defp benchmark_warm(fun) do
    {cold_us, _result} = :timer.tc(fun)

    warm_runs =
      Enum.map(1..5, fn _ ->
        {elapsed_us, result} = :timer.tc(fun)
        Map.put(normalize_result(result), :elapsed_us, elapsed_us)
      end)

    best = Enum.min_by(warm_runs, & &1.elapsed_us)
    best |> Map.put(:cold_elapsed_us, cold_us) |> Map.put(:warm_runs, length(warm_runs))
  end

  defp normalize_result(%{attributions: attributions, execution: execution}) do
    %{
      total_lift: attributions.total_lift,
      method_used: Atom.to_string(execution.method_used)
    }
  end

  defp normalize_result(%{summary: summary, execution: execution}) do
    %{
      cumulative_effect: summary.cumulative_effect.mean,
      method_used: Atom.to_string(execution.method_used)
    }
  end

  defp git_sha do
    case System.cmd("git", ["rev-parse", "--short", "HEAD"], stderr_to_stdout: true) do
      {sha, 0} -> String.trim(sha)
      _ -> "unknown"
    end
  end

  defp encode_json(value, level \\ 0)

  defp encode_json(map, level) when is_map(map) do
    entries =
      map
      |> Enum.map(fn {k, v} ->
        {to_string(k), encode_json(v, level + 1)}
      end)
      |> Enum.sort_by(&elem(&1, 0))
      |> Enum.map(fn {k, v} ->
        "#{indent(level + 1)}\"#{escape(k)}\": #{v}"
      end)
      |> Enum.join(",\n")

    "{\n#{entries}\n#{indent(level)}}"
  end

  defp encode_json(list, level) when is_list(list) do
    body =
      list
      |> Enum.map(fn item -> "#{indent(level + 1)}#{encode_json(item, level + 1)}" end)
      |> Enum.join(",\n")

    "[\n#{body}\n#{indent(level)}]"
  end

  defp encode_json(value, _level) when is_binary(value), do: "\"#{escape(value)}\""
  defp encode_json(value, _level) when is_integer(value) or is_float(value), do: to_string(value)
  defp encode_json(true, _level), do: "true"
  defp encode_json(false, _level), do: "false"
  defp encode_json(nil, _level), do: "null"

  defp indent(level), do: String.duplicate("  ", level)

  defp escape(str) do
    str
    |> String.replace("\\", "\\\\")
    |> String.replace("\"", "\\\"")
  end

  defp enforce_operational_budgets!(results) do
    failures =
      results.budgets_us
      |> Enum.flat_map(fn {scenario, budget_us} ->
        elapsed_us = results.scenarios[scenario].elapsed_us

        if elapsed_us > budget_us do
          ["#{scenario} elapsed #{elapsed_us}us exceeds #{budget_us}us"]
        else
          []
        end
      end)

    if failures != [] do
      raise "Operational benchmark budget exceeded:\n" <> Enum.join(failures, "\n")
    end
  end
end

BstsNx.Bench.OptimizePlan.run()
