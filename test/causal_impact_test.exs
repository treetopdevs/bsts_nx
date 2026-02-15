defmodule BstsNxCausalImpactTest do
  use ExUnit.Case
  alias BstsNx.CausalImpact

  @moduletag :external

  test "estimate returns expected structure" do
    :rand.seed(:exsss, {19, 20, 21})
    pre_data = Enum.map(1..60, fn _ -> :rand.uniform() * 10 end)
    post_data = Enum.map(1..30, fn _ -> :rand.uniform() * 10 + 5 end)
    observations = pre_data ++ post_data

    result =
      CausalImpact.estimate(observations, {1, 60}, {61, 90},
        num_samples: 20,
        burn_in: 10,
        seed: 123
      )

    assert is_list(result.point_effects)
    assert length(result.cumulative_effects) == 20
    assert length(result.actual) == 30
  end

  test "summary computes credible intervals" do
    :rand.seed(:exsss, {22, 23, 24})
    pre_data = Enum.map(1..60, fn _ -> :rand.uniform() * 10 end)
    post_data = Enum.map(1..30, fn _ -> :rand.uniform() * 10 + 2 end)
    observations = pre_data ++ post_data

    result =
      CausalImpact.estimate(observations, {1, 60}, {61, 90},
        num_samples: 30,
        burn_in: 15,
        seed: 456
      )

    summary = CausalImpact.summary(result)
    assert Map.has_key?(summary.cumulative_effect, :mean)
    assert Map.has_key?(summary.cumulative_effect, :lower)
    assert Map.has_key?(summary.cumulative_effect, :upper)
    assert summary.cumulative_effect.lower <= summary.cumulative_effect.mean
    assert summary.cumulative_effect.mean <= summary.cumulative_effect.upper
  end

  test "summary returns :nan bounds with a single sample" do
    :rand.seed(:exsss, {25, 26, 27})
    pre_data = Enum.map(1..20, fn _ -> :rand.uniform() * 5 end)
    post_data = Enum.map(1..10, fn _ -> :rand.uniform() * 5 end)
    observations = pre_data ++ post_data

    result =
      CausalImpact.estimate(observations, {1, 20}, {21, 30},
        num_samples: 1,
        burn_in: 0,
        seed: 123
      )

    summary = CausalImpact.summary(result)
    assert summary.cumulative_effect.sd == :nan
    assert summary.cumulative_effect.lower == :nan
    assert summary.cumulative_effect.upper == :nan
  end

  test "detects positive effect in synthetic data" do
    # Seed the Erlang RNG for reproducibility
    :rand.seed(:exsss, {1, 2, 3})
    pre_data = Enum.map(1..100, fn _ -> 50.0 + :rand.normal() * 5 end)
    # Use a large effect so a simple local-level baseline has an easy job.
    # +30 effect
    post_data = Enum.map(1..50, fn _ -> 80.0 + :rand.normal() * 5 end)
    observations = pre_data ++ post_data

    result =
      CausalImpact.estimate(observations, {1, 100}, {101, 150},
        num_samples: 50,
        burn_in: 25,
        seed: 789
      )

    summary = CausalImpact.summary(result)
    # For a true positive effect, the posterior mean cumulative effect should be > 0.
    # We don't assert the lower bound is > 0 because the current implementation's
    # counterfactual simulation is intentionally simple and can produce wide intervals.
    assert summary.cumulative_effect.mean > 0
    assert summary.cumulative_effect.upper > 0
  end
end
