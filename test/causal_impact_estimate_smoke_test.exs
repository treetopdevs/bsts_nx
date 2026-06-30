defmodule BstsNxCausalImpactEstimateSmokeTest do
  use ExUnit.Case, async: false

  alias BstsNx.CausalImpact

  # A value is finite if it equals itself (rules out the codebase's :nan atom and
  # float NaN) and is not an infinity atom.
  defp finite?(v) when is_number(v), do: v == v and v not in [:infinity, :neg_infinity]
  defp finite?(_), do: false

  test "estimate/4 recovers a large positive effect with finite, ordered summary" do
    :rand.seed(:exsss, {11, 22, 33})
    pre = Enum.map(1..60, fn _ -> 50.0 + :rand.normal() * 4.0 end)
    # +30 intervention — unambiguous for a local-level baseline.
    post = Enum.map(1..30, fn _ -> 80.0 + :rand.normal() * 4.0 end)
    obs = pre ++ post

    result =
      CausalImpact.estimate(obs, {1, 60}, {61, 90},
        num_samples: 60,
        burn_in: 30,
        seed: 4242
      )

    # Result shape (locks the public contract).
    assert length(result.actual) == 30
    assert length(result.cumulative_effects) == 60

    summary = CausalImpact.summary(result)

    # Finite summary (regression guard: a NaN init/posterior would poison these).
    assert finite?(summary.cumulative_effect.mean)
    assert finite?(summary.cumulative_effect.lower)
    assert finite?(summary.cumulative_effect.upper)
    assert finite?(summary.relative_effect.mean)

    # Credible interval is ordered.
    assert summary.cumulative_effect.lower <= summary.cumulative_effect.mean
    assert summary.cumulative_effect.mean <= summary.cumulative_effect.upper

    # A large positive intervention is detected as positive — both the cumulative
    # effect and the average pointwise effect are above zero.
    assert summary.cumulative_effect.mean > 0

    avg_point_mean =
      summary.point_effects
      |> Enum.map(& &1.mean)
      |> then(fn means -> Enum.sum(means) / length(means) end)

    assert avg_point_mean > 0
  end

  test "estimate/4 is deterministic for a fixed seed" do
    obs = List.duplicate(10.0, 40) ++ List.duplicate(18.0, 20)

    run = fn ->
      CausalImpact.estimate(obs, {1, 40}, {41, 60},
        num_samples: 40,
        burn_in: 20,
        seed: 7
      )
      |> CausalImpact.summary()
      |> Map.get(:cumulative_effect)
      |> Map.get(:mean)
    end

    assert run.() == run.()
  end
end
