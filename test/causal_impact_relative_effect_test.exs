defmodule BstsNx.CausalImpactRelativeEffectTest do
  use ExUnit.Case, async: true

  alias BstsNx.CausalImpact
  alias BstsNx.Components

  test "scalar estimate computes each relative draw from its own baseline denominator" do
    :rand.seed(:exsss, {1, 2, 3})

    observations =
      Enum.map(1..40, fn i ->
        baseline = if i <= 30, do: 10.0, else: 12.0
        baseline + :rand.normal() * 3.0
      end)

    result =
      CausalImpact.estimate(observations, {1, 30}, {31, 40},
        num_samples: 40,
        burn_in: 20,
        thin: 1,
        seed: 123,
        process_var: 0.5,
        obs_var: 1.0
      )

    summary = CausalImpact.summary(result)
    baseline_sums = Enum.map(result.counterfactual, &Enum.sum/1)

    expected_relative_effects =
      Enum.zip(result.cumulative_effects, baseline_sums)
      |> Enum.map(fn {cumulative_effect, baseline_sum} ->
        cumulative_effect / baseline_sum
      end)

    expected_mean = Enum.sum(expected_relative_effects) / length(expected_relative_effects)

    assert_in_delta(
      summary.relative_effect.mean,
      expected_mean,
      1.0e-6
    )

    Enum.zip(result.relative_effects, expected_relative_effects)
    |> Enum.each(fn {actual, expected} -> assert_in_delta(actual, expected, 1.0e-12) end)
  end

  test "structured estimate computes each relative draw from its own baseline denominator" do
    :rand.seed(:exsss, {1, 2, 3})

    observations =
      Enum.map(1..40, fn i ->
        baseline = if i <= 30, do: 10.0, else: 12.0
        baseline + :rand.normal() * 3.0
      end)

    spec =
      Components.local_level_spec(
        initial_state: 10.0,
        initial_cov: 5.0,
        process_var: 0.5,
        obs_var: 1.0
      )

    result =
      CausalImpact.estimate_structured(observations, {1, 30}, {31, 40}, spec,
        num_samples: 30,
        burn_in: 15,
        thin: 1,
        seed: 123
      )

    summary = CausalImpact.summary(result)
    baseline_sums = Enum.map(result.counterfactual, &Enum.sum/1)

    expected_relative_effects =
      Enum.zip(result.cumulative_effects, baseline_sums)
      |> Enum.map(fn {cumulative_effect, baseline_sum} ->
        cumulative_effect / baseline_sum
      end)

    expected_mean = Enum.sum(expected_relative_effects) / length(expected_relative_effects)

    assert_in_delta(
      summary.relative_effect.mean,
      expected_mean,
      1.0e-6
    )

    Enum.zip(result.relative_effects, expected_relative_effects)
    |> Enum.each(fn {actual, expected} -> assert_in_delta(actual, expected, 1.0e-12) end)
  end
end
