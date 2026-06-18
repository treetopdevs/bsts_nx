defmodule BstsNxCausalImpactSummaryTest do
  use ExUnit.Case, async: true
  alias BstsNx.CausalImpact

  test "summary returns :nan bounds with a single sample" do
    result = %{
      point_effects: [[1.0, 2.0]],
      cumulative_effects: [3.0],
      relative_effects: [0.5],
      actual: [1.0, 2.0],
      counterfactual: [[0.0, 0.0]],
      pre_period: {1, 2},
      post_period: {3, 4}
    }

    summary = CausalImpact.summary(result)
    assert summary.cumulative_effect.sd == :nan
    assert summary.cumulative_effect.lower == :nan
    assert summary.cumulative_effect.upper == :nan
    assert summary.point_effects |> Enum.all?(fn s -> s.sd == :nan end)
  end

  test "summary uses linearly interpolated percentile bounds" do
    result = %{
      point_effects: [
        [0.0, 100.0],
        [10.0, 110.0],
        [20.0, 120.0],
        [30.0, 130.0]
      ],
      cumulative_effects: [0.0, 100.0, 200.0, 300.0],
      relative_effects: [0.0, 1.0, 2.0, 3.0],
      actual: [1.0, 2.0],
      counterfactual: [
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0]
      ],
      pre_period: {1, 2},
      post_period: {3, 4}
    }

    summary = CausalImpact.summary(result)

    assert_in_delta Enum.at(summary.point_effects, 0).lower, 0.75, 1.0e-12
    assert_in_delta Enum.at(summary.point_effects, 0).upper, 29.25, 1.0e-12
    assert_in_delta Enum.at(summary.point_effects, 1).lower, 100.75, 1.0e-12
    assert_in_delta Enum.at(summary.point_effects, 1).upper, 129.25, 1.0e-12
    assert_in_delta summary.cumulative_effect.lower, 7.5, 1.0e-12
    assert_in_delta summary.cumulative_effect.upper, 292.5, 1.0e-12
    assert_in_delta summary.relative_effect.lower, 0.075, 1.0e-12
    assert_in_delta summary.relative_effect.upper, 2.925, 1.0e-12
  end

  test "summary honors alpha when computing interval bounds" do
    result = %{
      point_effects: [
        [0.0, 100.0],
        [10.0, 110.0],
        [20.0, 120.0],
        [30.0, 130.0]
      ],
      cumulative_effects: [0.0, 100.0, 200.0, 300.0],
      relative_effects: [0.0, 1.0, 2.0, 3.0],
      actual: [1.0, 2.0],
      counterfactual: [
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0]
      ],
      pre_period: {1, 2},
      post_period: {3, 4}
    }

    summary_90 = CausalImpact.summary(result, alpha: 0.10)
    summary_95 = CausalImpact.summary(result)
    summary_99 = CausalImpact.summary(result, alpha: 0.01)

    width_90 = summary_90.cumulative_effect.upper - summary_90.cumulative_effect.lower
    width_95 = summary_95.cumulative_effect.upper - summary_95.cumulative_effect.lower
    width_99 = summary_99.cumulative_effect.upper - summary_99.cumulative_effect.lower

    assert width_90 < width_95
    assert width_95 < width_99
    assert_in_delta summary_90.cumulative_effect.lower, 15.0, 1.0e-12
    assert_in_delta summary_90.cumulative_effect.upper, 285.0, 1.0e-12
  end

  test "summary rejects invalid alpha values" do
    result = %{
      point_effects: [[1.0, 2.0], [2.0, 4.0]],
      cumulative_effects: [3.0, 6.0],
      relative_effects: [0.5, 1.0],
      actual: [1.0, 2.0],
      counterfactual: [[0.0, 0.0], [0.0, 0.0]],
      pre_period: {1, 2},
      post_period: {3, 4}
    }

    assert_raise ArgumentError, ~r/alpha/, fn ->
      CausalImpact.summary(result, alpha: 1.0)
    end
  end

  test "estimate rejects malformed PRNG keys instead of silently falling back to seed" do
    assert_raise ArgumentError, ~r/key must be an Nx.Random key/, fn ->
      CausalImpact.estimate([1.0, 2.0, 3.0, 4.0], {1, 2}, {3, 4},
        key: :not_a_key,
        seed: 123,
        num_samples: 1,
        burn_in: 0
      )
    end
  end
end
