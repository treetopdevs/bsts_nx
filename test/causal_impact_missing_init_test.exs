defmodule BstsNxCausalImpactMissingInitTest do
  use ExUnit.Case, async: true

  alias BstsNx.CausalImpact

  defp finite?(v) when is_number(v), do: v == v and v not in [:infinity, :neg_infinity]
  defp finite?(_), do: false

  test "estimate/4 tolerates a missing (NaN) first pre-period observation" do
    nan = Nx.Constants.nan()
    # First pre-period point is missing; the rest carry the real ~50 level.
    pre = [nan | Enum.map(1..29, fn _ -> 50.0 end)]
    post = Enum.map(1..15, fn _ -> 60.0 end)
    obs = pre ++ post

    result =
      CausalImpact.estimate(obs, {1, 30}, {31, 45},
        num_samples: 40,
        burn_in: 20,
        seed: 7
      )

    summary = CausalImpact.summary(result)

    assert finite?(summary.cumulative_effect.mean)
    assert finite?(summary.relative_effect.mean)
    assert Enum.all?(summary.point_effects, fn pe -> finite?(pe.mean) end)
  end
end
