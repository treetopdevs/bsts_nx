defmodule BstsNx.OperationalPerformanceTest do
  use ExUnit.Case, async: false

  @moduletag :performance

  alias BstsNx.{Components, Operational}

  setup_all do
    previous = Nx.default_backend()
    Nx.global_default_backend(Nx.BinaryBackend)

    on_exit(fn ->
      case previous do
        {backend, opts} -> Nx.global_default_backend({backend, opts})
        backend -> Nx.global_default_backend(backend)
      end
    end)

    :ok
  end

  test "prepared warm scalar forecast stays under interactive budget" do
    observations = guide_observations()

    spec =
      Components.local_level_spec(
        initial_state: hd(observations),
        initial_cov: 10.0,
        process_var: 0.8,
        obs_var: 4.0
      )

    prepared = Operational.prepare(spec, {1, 96}, {97, 144})
    {_cold, warm_min} = cold_and_warm_min(fn -> Operational.forecast(prepared, observations) end)

    assert warm_min < 25_000
  end

  test "prepared warm structured forecast stays under interactive budget" do
    observations = guide_observations()

    spec =
      Components.local_linear_trend_spec(
        initial_level: hd(observations),
        initial_slope: 0.0,
        initial_cov_level: 10.0,
        initial_cov_slope: 1.0,
        var_level: 0.8,
        var_slope: 0.02,
        obs_var: 4.0
      )

    prepared = Operational.prepare(spec, {1, 96}, {97, 144})
    {_cold, warm_min} = cold_and_warm_min(fn -> Operational.forecast(prepared, observations) end)

    assert warm_min < 100_000
  end

  test "guide operational pipeline stays under interactive budget" do
    observations = guide_observations()

    spec =
      Components.local_level_spec(
        initial_state: hd(observations),
        initial_cov: 10.0,
        process_var: 0.8,
        obs_var: 4.0
      )

    spots = [
      %{id: "spot_a", window_start: 0, window_end: 12},
      %{id: "spot_b", window_start: 12, window_end: 24}
    ]

    prepared = Operational.prepare(spec, {1, 96}, {97, 144})

    {_cold, warm_min} =
      cold_and_warm_min(fn ->
        Operational.run(prepared, observations, spots, shapley_samples: 250)
      end)

    assert warm_min < 100_000
  end

  test "larger batch operational scenario stays under release budget" do
    observations = batch_observations(2_000)

    spec =
      Components.local_linear_trend_spec(
        initial_level: hd(observations),
        initial_slope: 0.02,
        initial_cov_level: 20.0,
        initial_cov_slope: 2.0,
        var_level: 0.5,
        var_slope: 0.01,
        obs_var: 4.0
      )

    spots = [
      %{id: "early", window_start: 0, window_end: 120},
      %{id: "middle", window_start: 100, window_end: 260},
      %{id: "late", window_start: 300, window_end: 500}
    ]

    prepared = Operational.prepare(spec, {1, 1_500}, {1_501, 2_000})

    {_cold, warm_min} =
      cold_and_warm_min(fn ->
        Operational.run(prepared, observations, spots, shapley_samples: 250)
      end)

    assert warm_min < 1_000_000
  end

  defp cold_and_warm_min(fun) do
    {cold, _} = :timer.tc(fun)

    warm_min =
      1..5
      |> Enum.map(fn _ -> :timer.tc(fun) |> elem(0) end)
      |> Enum.min()

    {cold, warm_min}
  end

  defp guide_observations do
    Enum.map(0..143, fn t ->
      baseline = 80.0 + 0.05 * t + 7.0 * :math.sin(2.0 * :math.pi() * t / 24.0)
      lift = if t >= 96, do: 12.0, else: 0.0
      baseline + lift
    end)
  end

  defp batch_observations(n) do
    Enum.map(0..(n - 1), fn t ->
      baseline = 100.0 + 0.02 * t + 8.0 * :math.sin(2.0 * :math.pi() * t / 24.0)
      lift = if t >= 1_500, do: 9.0, else: 0.0
      baseline + lift
    end)
  end
end
