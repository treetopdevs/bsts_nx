Mix.Task.run("app.start")

alias BstsNx.{Components, GibbsSampler, ModelSpec}

defmodule BstsNx.Bench.StructuredBackendBenchmark do
  @moduledoc false

  @default_backends "binary,exla,emlx_cpu,emlx_gpu"
  @default_t 500
  @default_num_samples 1_000
  @default_burn_in 100
  @default_controls 3
  @default_warm_runs 1

  def run do
    opts = %{
      t: env_int("BSTS_NX_BENCH_T", @default_t),
      num_samples: env_int("BSTS_NX_BENCH_SAMPLES", @default_num_samples),
      burn_in: env_int("BSTS_NX_BENCH_BURN_IN", @default_burn_in),
      controls: env_int("BSTS_NX_BENCH_CONTROLS", @default_controls),
      warm_runs: env_int("BSTS_NX_BENCH_WARM_RUNS", @default_warm_runs),
      dtype: env_dtype("BSTS_NX_BENCH_DTYPE", {:f, 32}),
      seed: env_int("BSTS_NX_BENCH_SEED", 123)
    }

    validate_opts!(opts)

    backends =
      "BSTS_NX_BENCH_BACKENDS"
      |> System.get_env(@default_backends)
      |> String.split(",", trim: true)
      |> Enum.map(&String.trim/1)

    results = Enum.map(backends, &run_backend(&1, opts))

    output = %{
      generated_at: DateTime.utc_now() |> DateTime.to_iso8601(),
      git_sha: git_sha(),
      workload: opts,
      results: results
    }

    json = encode_json(output)
    maybe_write_output(json)
    IO.puts(json)
  end

  defp run_backend(name, opts) do
    case configure_backend(name, opts.dtype) do
      {:ok, backend_meta} ->
        benchmark_backend(name, backend_meta, opts)

      {:skip, reason} ->
        %{
          backend: name,
          status: "skipped",
          reason: reason
        }
    end
  end

  defp benchmark_backend(name, backend_meta, opts) do
    {observations, spec} = workload(opts)

    {cold_us, cold_result} =
      :timer.tc(fn ->
        GibbsSampler.sample_structured(observations, spec, opts.num_samples,
          burn_in: opts.burn_in,
          seed: opts.seed
        )
      end)

    warm_run_indices = if opts.warm_runs == 0, do: [], else: 1..opts.warm_runs

    warm_runs =
      warm_run_indices
      |> Enum.map(fn run_idx ->
        {elapsed_us, samples} =
          :timer.tc(fn ->
            GibbsSampler.sample_structured(observations, spec, opts.num_samples,
              burn_in: opts.burn_in,
              seed: opts.seed + run_idx
            )
          end)

        %{
          run: run_idx,
          elapsed_us: elapsed_us,
          elapsed_ms: div(elapsed_us + 999, 1_000),
          samples: length(samples)
        }
      end)

    best_warm_us =
      case warm_runs do
        [] -> nil
        runs -> runs |> Enum.map(& &1.elapsed_us) |> Enum.min()
      end

    %{
      backend: name,
      status: "ok",
      compiler: backend_meta.compiler,
      default_backend: backend_meta.default_backend,
      probe_backend: backend_meta.probe_backend,
      emlx_gpu?: backend_meta.emlx_gpu?,
      dtype: inspect(opts.dtype),
      state_dim: Nx.axis_size(spec.f, 0),
      cold_elapsed_us: cold_us,
      cold_elapsed_ms: div(cold_us + 999, 1_000),
      warm_best_us: best_warm_us,
      warm_best_ms: if(best_warm_us, do: div(best_warm_us + 999, 1_000), else: nil),
      warm_runs: warm_runs,
      cold_samples: length(cold_result),
      final_obs_var: final_obs_var(cold_result)
    }
  rescue
    error ->
      %{
        backend: name,
        status: "failed",
        error: Exception.message(error),
        default_backend: inspect(Nx.default_backend())
      }
  end

  defp configure_backend("binary", dtype) do
    Nx.global_default_backend(Nx.BinaryBackend)
    Nx.Defn.global_default_options(compiler: Nx.Defn.Evaluator)
    {:ok, backend_meta(:evaluator, dtype)}
  end

  defp configure_backend("exla", dtype) do
    if Code.ensure_loaded?(EXLA.Backend) do
      Nx.global_default_backend(Nx.BinaryBackend)
      Nx.Defn.global_default_options(compiler: EXLA)
      _ = Nx.backend_transfer(Nx.tensor([0.0], type: dtype), EXLA.Backend)
      {:ok, backend_meta(:exla, dtype)}
    else
      {:skip, "EXLA.Backend is unavailable"}
    end
  rescue
    error -> {:skip, "EXLA setup failed: #{Exception.message(error)}"}
  end

  defp configure_backend("emlx_cpu", dtype) do
    configure_emlx({EMLX.Backend, device: :cpu}, :emlx_cpu, dtype)
  end

  defp configure_backend("emlx_gpu", dtype) do
    configure_emlx({EMLX.Backend, device: :gpu}, :emlx_gpu, dtype)
  end

  defp configure_backend(other, _dtype), do: {:skip, "unknown backend #{inspect(other)}"}

  defp configure_emlx(backend, compiler, dtype) do
    if Code.ensure_loaded?(EMLX.Backend) do
      Nx.global_default_backend(backend)
      Nx.Defn.global_default_options(compiler: EMLX)
      _ = Nx.backend_transfer(Nx.tensor([0.0], type: dtype), backend)
      {:ok, backend_meta(compiler, dtype)}
    else
      {:skip, "EMLX.Backend is unavailable"}
    end
  rescue
    error -> {:skip, "#{compiler} setup failed: #{Exception.message(error)}"}
  end

  defp backend_meta(compiler, dtype) do
    probe = Nx.tensor([0.0], type: dtype)
    probe_backend = inspect(probe)

    %{
      compiler: Atom.to_string(compiler),
      default_backend: inspect(Nx.default_backend()),
      probe_backend: probe_backend,
      emlx_gpu?: String.contains?(probe_backend, "EMLX.Backend<gpu")
    }
  end

  defp workload(opts) do
    observations =
      Enum.map(0..(opts.t - 1), fn t ->
        trend = 100.0 + 0.025 * t
        daily = 8.0 * :math.sin(2.0 * :math.pi() * t / 24.0)
        weekly = 3.0 * :math.cos(2.0 * :math.pi() * t / 168.0)
        control_effect = 2.0 * :math.sin(t / 19.0) - 1.5 * :math.cos(t / 37.0)
        trend + daily + weekly + control_effect
      end)

    regressors =
      Enum.map(0..(opts.t - 1), fn t ->
        Enum.map(0..(opts.controls - 1), fn j ->
          phase = (j + 1) * 0.31
          1.0 + :math.sin((t + 1) * phase / 8.0) + 0.5 * :math.cos((t + j + 3) / 17.0)
        end)
      end)
      |> Nx.tensor(type: opts.dtype)

    trend_spec =
      Components.local_linear_trend_spec(
        initial_level: hd(observations),
        initial_slope: 0.0,
        initial_cov_level: 10.0,
        initial_cov_slope: 1.0,
        var_level: 0.2,
        var_slope: 0.02,
        obs_var: 4.0
      )

    seasonal_spec = Components.seasonal_spec(24, process_var: 0.05, obs_var: 4.0)
    regression_spec = Components.regression_spec(regressors, var_beta: 0.02, obs_var: 4.0)

    spec =
      trend_spec
      |> Components.compose_specs(seasonal_spec)
      |> Components.compose_specs(regression_spec)
      |> cast_spec(opts.dtype)

    {observations, spec}
  end

  defp cast_spec(%ModelSpec{} = spec, dtype) do
    %{
      spec
      | f: Nx.as_type(spec.f, dtype),
        h: cast_h(spec.h, dtype),
        x0: Nx.as_type(spec.x0, dtype),
        p0: Nx.as_type(spec.p0, dtype)
    }
  end

  defp cast_h(h, dtype) when is_list(h), do: Enum.map(h, &Nx.as_type(&1, dtype))
  defp cast_h(%Nx.Tensor{} = h, dtype), do: Nx.as_type(h, dtype)

  defp final_obs_var([]), do: nil

  defp final_obs_var(samples),
    do: samples |> List.last() |> Map.fetch!(:obs_var) |> Nx.to_number()

  defp validate_opts!(opts) do
    if opts.t < 500 do
      IO.warn("BSTS_NX_BENCH_T=#{opts.t} is below the recommended realistic floor of 500")
    end

    if opts.num_samples < 1_000 do
      IO.warn(
        "BSTS_NX_BENCH_SAMPLES=#{opts.num_samples} is below the recommended realistic floor of 1000"
      )
    end

    if opts.controls < 1 do
      raise ArgumentError, "BSTS_NX_BENCH_CONTROLS must be >= 1"
    end

    if opts.warm_runs < 0 do
      raise ArgumentError, "BSTS_NX_BENCH_WARM_RUNS must be >= 0"
    end
  end

  defp env_int(name, default) do
    case System.get_env(name) do
      nil ->
        default

      value ->
        case Integer.parse(value) do
          {parsed, ""} -> parsed
          _ -> raise ArgumentError, "#{name} must be an integer, got: #{inspect(value)}"
        end
    end
  end

  defp env_dtype(name, default) do
    case System.get_env(name) do
      nil -> default
      "f32" -> {:f, 32}
      "f64" -> {:f, 64}
      other -> raise ArgumentError, "#{name} must be f32 or f64, got: #{inspect(other)}"
    end
  end

  defp maybe_write_output(json) do
    case System.get_env("BSTS_NX_BENCH_OUTPUT") do
      nil ->
        :ok

      path ->
        File.mkdir_p!(Path.dirname(path))
        File.write!(path, json <> "\n")
        IO.puts("Wrote benchmark results to #{path}")
    end
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
      |> Enum.map(fn {k, v} -> {to_string(k), encode_json(v, level + 1)} end)
      |> Enum.sort_by(&elem(&1, 0))
      |> Enum.map(fn {k, v} -> "#{indent(level + 1)}\"#{escape(k)}\": #{v}" end)
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
  defp encode_json(value, _level) when is_integer(value), do: Integer.to_string(value)
  defp encode_json(value, _level) when is_float(value), do: Float.to_string(value)
  defp encode_json(true, _level), do: "true"
  defp encode_json(false, _level), do: "false"
  defp encode_json(nil, _level), do: "null"
  defp encode_json(value, _level), do: "\"#{escape(inspect(value))}\""

  defp indent(level), do: String.duplicate("  ", level)

  defp escape(str) do
    str
    |> String.replace("\\", "\\\\")
    |> String.replace("\"", "\\\"")
  end
end

BstsNx.Bench.StructuredBackendBenchmark.run()
