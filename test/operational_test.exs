defmodule BstsNx.OperationalTest do
  use ExUnit.Case, async: true

  alias BstsNx.{Components, Operational, Pipeline}

  test "forecast baseline does not change when post-period actuals change" do
    pre = List.duplicate(10.0, 20)
    post_a = List.duplicate(10.0, 5)
    post_b = List.duplicate(50.0, 5)

    spec =
      Components.local_level_spec(
        initial_state: 10.0,
        initial_cov: 1.0,
        process_var: 0.01,
        obs_var: 0.1
      )

    prepared = Operational.prepare(spec, {1, 20}, {21, 25})

    a = Operational.forecast(prepared, pre ++ post_a).summary
    b = Operational.forecast(prepared, pre ++ post_b).summary

    assert a.baseline == b.baseline
    assert a.baseline_variance == b.baseline_variance
    refute a.actual == b.actual
  end

  test "forecast path uses ModelSpec initial state and covariance" do
    observations = [:nan, 50.0, 50.0, 50.0]

    anchored =
      Components.local_level_spec(
        initial_state: 50.0,
        initial_cov: 1.0e-9,
        process_var: 0.0,
        obs_var: 1.0e-6
      )

    wrong =
      Components.local_level_spec(
        initial_state: 0.0,
        initial_cov: 1.0e-9,
        process_var: 0.0,
        obs_var: 1.0e-6
      )

    anchored_summary =
      anchored
      |> Operational.prepare({1, 1}, {2, 4})
      |> Operational.forecast(observations)
      |> Map.fetch!(:summary)

    wrong_summary =
      wrong
      |> Operational.prepare({1, 1}, {2, 4})
      |> Operational.forecast(observations)
      |> Map.fetch!(:summary)

    assert_in_delta hd(anchored_summary.baseline), 50.0, 1.0e-6
    assert_in_delta hd(wrong_summary.baseline), 0.0, 1.0e-6
  end

  test "prepared and one-shot runs are numerically equivalent" do
    observations = demo_observations()

    spec =
      Components.local_level_spec(initial_state: hd(observations), process_var: 0.4, obs_var: 2.0)

    spots = [%{id: "campaign", window_start: 0, window_end: 12}]

    prepared = Operational.prepare(spec, {1, 48}, {49, 72})
    prepared_result = Operational.run(prepared, observations, spots)
    one_shot_result = Operational.run(observations, {1, 48}, {49, 72}, spots, spec)

    assert prepared_result.execution.prepared? == true
    assert one_shot_result.execution.prepared? == false
    assert prepared_result.summary.baseline == one_shot_result.summary.baseline

    assert prepared_result.summary.point_effects.mean ==
             one_shot_result.summary.point_effects.mean

    assert_in_delta prepared_result.attributions.total_lift,
                    one_shot_result.attributions.total_lift,
                    1.0e-9
  end

  test "tensor return keeps vector fields as tensors with list-equivalent values" do
    observations = demo_observations()
    spec = Components.local_linear_trend_spec(initial_level: hd(observations), obs_var: 2.0)
    prepared = Operational.prepare(spec, {1, 48}, {49, 72})

    list_summary = Operational.forecast(prepared, observations, return: :lists).summary
    tensor_summary = Operational.forecast(prepared, observations, return: :tensors).summary

    assert %Nx.Tensor{} = tensor_summary.actual
    assert %Nx.Tensor{} = tensor_summary.baseline
    assert %Nx.Tensor{} = tensor_summary.point_effects.mean
    assert Nx.shape(tensor_summary.actual) == {24}
    assert Nx.to_flat_list(tensor_summary.baseline) == list_summary.baseline
    assert Nx.to_flat_list(tensor_summary.point_effects.mean) == list_summary.point_effects.mean
  end

  test "time-varying regressors must cover the post period" do
    spec = Components.regression_spec(Nx.tensor([[1.0], [2.0], [3.0]]))

    assert_raise ArgumentError, ~r/must cover period ending at 4/, fn ->
      Operational.prepare(spec, {1, 2}, {3, 4})
    end
  end

  test "Pipeline operational mode delegates to explicit forecast engine metadata" do
    observations = demo_observations()

    spec =
      Components.local_level_spec(initial_state: hd(observations), process_var: 0.4, obs_var: 2.0)

    spots = [%{id: "campaign", window_start: 0, window_end: 12}]

    result = Pipeline.run(observations, {1, 48}, {49, 72}, spots, spec, mode: :operational)

    assert result.execution.mode == :operational
    assert result.execution.baseline == :forecast
    assert result.execution.method_used == :scalar_forecast_filter
    assert result.execution.prepared? == false
  end

  test "smooth baseline remains available as explicit compatibility mode" do
    observations = demo_observations()

    spec =
      Components.local_level_spec(initial_state: hd(observations), process_var: 0.4, obs_var: 2.0)

    prepared = Operational.prepare(spec, {1, 48}, {49, 72}, baseline: :smooth)

    result = Operational.forecast(prepared, observations)

    assert result.execution.baseline == :smooth
    assert result.execution.method_used == :scalar_smooth_filter
    assert length(result.summary.actual) == 24
  end

  defp demo_observations do
    Enum.map(0..71, fn t ->
      baseline = 80.0 + 0.05 * t + 4.0 * :math.sin(2.0 * :math.pi() * t / 12.0)
      lift = if t >= 48, do: 8.0, else: 0.0
      baseline + lift
    end)
  end
end
