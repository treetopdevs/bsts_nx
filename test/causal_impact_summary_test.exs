defmodule BstsNxCausalImpactSummaryTest do
  use ExUnit.Case
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
end
