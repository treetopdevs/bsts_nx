Mix.Task.run("app.start")

defmodule BstsNx.Bench.GibbsFusion do
  @moduledoc """
  Compares the stepwise (per-iteration defn dispatch) and fused
  (single-defn whole-chain) Gibbs sampler paths.

  Usage:

      mix run bench/gibbs_fusion_benchmark.exs
      BENCH_BACKEND=exla mix run bench/gibbs_fusion_benchmark.exs
      BENCH_BACKEND=binary mix run bench/gibbs_fusion_benchmark.exs

  `BENCH_BACKEND=exla` requires the optional EXLA dependency. The fused
  path is expected to dominate there because the entire MCMC chain runs
  as one compiled XLA program.
  """

  alias BstsNx.{Components, GibbsSampler}

  def run do
    backend = configure_backend!(System.get_env("BENCH_BACKEND", "binary"))
    seed = 123
    scalar_obs = scalar_observations(200)
    structured_obs = structured_observations(96)

    structured_spec =
      Components.local_linear_trend_spec(
        initial_level: hd(structured_obs),
        var_level: 0.1,
        var_slope: 0.01,
        obs_var: 1.0
      )

    IO.puts("backend: #{backend}")

    report("scalar_gibbs (200 obs x 200 samples)", fn fused? ->
      GibbsSampler.sample(scalar_obs, 200, 0.0, 1.0, 0.1, 0.1, seed: seed, fused: fused?)
    end)

    report("structured_gibbs (96 obs x 120 samples + 40 burn-in)", fn fused? ->
      GibbsSampler.sample_structured(structured_obs, structured_spec, 120,
        burn_in: 40,
        seed: seed,
        fused: fused?
      )
    end)
  end

  defp report(label, fun) do
    # Warm both paths once so JIT compilation cost is excluded from timings.
    fun.(false)
    fun.(true)

    {stepwise_us, _} = :timer.tc(fn -> fun.(false) end)
    {fused_us, _} = :timer.tc(fn -> fun.(true) end)

    speedup = Float.round(stepwise_us / max(fused_us, 1), 2)

    IO.puts("#{label}")
    IO.puts("  stepwise: #{format_us(stepwise_us)}")
    IO.puts("  fused:    #{format_us(fused_us)}  (#{speedup}x)")
  end

  defp format_us(us) when us >= 1_000_000, do: "#{Float.round(us / 1_000_000, 2)}s"
  defp format_us(us), do: "#{Float.round(us / 1_000, 1)}ms"

  defp configure_backend!("binary") do
    Nx.global_default_backend(Nx.BinaryBackend)
    "Nx.BinaryBackend"
  end

  defp configure_backend!("exla") do
    Nx.global_default_backend(EXLA.Backend)
    Nx.Defn.global_default_options(compiler: EXLA)
    "EXLA"
  end

  defp configure_backend!(other) do
    raise ArgumentError, "BENCH_BACKEND must be binary or exla, got: #{inspect(other)}"
  end

  defp scalar_observations(n) do
    Enum.map(0..(n - 1), fn t ->
      10.0 + 0.03 * t + 0.8 * :math.sin(2.0 * :math.pi() * t / 24.0)
    end)
  end

  defp structured_observations(n) do
    Enum.map(0..(n - 1), fn t ->
      100.0 + 0.2 * t + 2.0 * :math.sin(2.0 * :math.pi() * t / 12.0)
    end)
  end
end

BstsNx.Bench.GibbsFusion.run()
