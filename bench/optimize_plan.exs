Mix.Task.run("app.start")

alias BstsNx.{Components, GibbsSampler}

defmodule BstsNx.Bench.OptimizePlan do
  @moduledoc false

  def run do
    seed = 123
    scalar_obs = scalar_observations(200)
    structured_obs = structured_observations(96)
    structured_spec = structured_spec(structured_obs)

    scalar = benchmark(fn -> run_scalar(scalar_obs, seed) end)
    structured = benchmark(fn -> run_structured(structured_obs, structured_spec, seed) end)

    results = %{
      generated_at: DateTime.utc_now() |> DateTime.to_iso8601(),
      git_sha: git_sha(),
      backend: inspect(Nx.default_backend()),
      scenarios: %{
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

  defp benchmark(fun) do
    {elapsed_us, result} = :timer.tc(fun)
    Map.put(result, :elapsed_us, elapsed_us)
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
end

BstsNx.Bench.OptimizePlan.run()
