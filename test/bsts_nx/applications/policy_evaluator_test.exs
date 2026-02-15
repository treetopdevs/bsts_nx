defmodule BstsNx.Applications.PolicyEvaluatorTest do
  use ExUnit.Case, async: true

  alias BstsNx.Applications.PolicyEvaluator
  alias BstsNx.Components

  describe "evaluate/3" do
    test "detects positive policy effect" do
      :rand.seed(:exsss, {100, 101, 102})
      pre = Enum.map(1..30, fn _ -> 50.0 + :rand.normal() * 3 end)
      post = Enum.map(1..10, fn _ -> 40.0 + :rand.normal() * 3 end)
      obs = pre ++ post

      intervention = %{
        intervention_name: "speed_limit_reduction",
        intervention_date_index: 31,
        pre_period_start: 1,
        outcome_metric: "accidents_per_100k"
      }

      result =
        PolicyEvaluator.evaluate(obs, intervention,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert result.intervention.intervention_name == "speed_limit_reduction"
      assert result.n_pre == 30
      assert result.n_post == 10
      assert is_float(result.effect.cumulative)
      assert is_float(result.effect.per_period)
      assert is_float(result.effect.relative)
      assert is_binary(result.report)
      assert String.contains?(result.report, "speed_limit_reduction")
    end

    test "handles lag parameter" do
      :rand.seed(:exsss, {200, 201, 202})
      pre = Enum.map(1..20, fn _ -> 50.0 + :rand.normal() * 2 end)
      transition = Enum.map(1..3, fn _ -> 50.0 + :rand.normal() * 2 end)
      post = Enum.map(1..7, fn _ -> 40.0 + :rand.normal() * 2 end)
      obs = pre ++ transition ++ post

      intervention = %{
        intervention_name: "policy_with_lag",
        intervention_date_index: 21,
        pre_period_start: 1,
        outcome_metric: "metric"
      }

      result =
        PolicyEvaluator.evaluate(obs, intervention,
          lag: 3,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      # Post period should be 24..30 (skipping 3 lag periods)
      assert result.n_post == 7
    end

    test "report contains interpretation" do
      :rand.seed(:exsss, {300, 301, 302})
      pre = Enum.map(1..20, fn _ -> 50.0 + :rand.normal() end)
      post = Enum.map(1..5, fn _ -> 55.0 + :rand.normal() end)
      obs = pre ++ post

      intervention = %{
        intervention_name: "test_policy",
        intervention_date_index: 21,
        pre_period_start: 1,
        outcome_metric: "outcome"
      }

      result =
        PolicyEvaluator.evaluate(obs, intervention,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert String.contains?(result.report, "Interpretation")
    end

    test "evaluate/3 with control series from unaffected region" do
      :rand.seed(:exsss, {340, 341, 342})
      # Control region not affected by the policy
      control = Enum.map(1..25, fn _ -> 50.0 + :rand.normal() * 2 end)
      pre = Enum.map(Enum.take(control, 20), fn c -> c + :rand.normal() end)
      post = Enum.map(Enum.drop(control, 20), fn c -> c - 10.0 + :rand.normal() end)
      obs = pre ++ post

      intervention = %{
        intervention_name: "control_series_policy",
        intervention_date_index: 21,
        pre_period_start: 1,
        control_series: [control]
      }

      result =
        PolicyEvaluator.evaluate(obs, intervention,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert result.analysis.model_spec != nil
      # Model should include regression: trend (2) + 1 regressor = 3
      assert Nx.axis_size(result.analysis.model_spec.f, 0) >= 3
      assert is_float(result.effect.cumulative)
      assert String.contains?(result.report, "Cumulative effect")
    end

    test "pre_trend_check/3 with control series returns diagnostics" do
      :rand.seed(:exsss, {345, 346, 347})
      control = Enum.map(1..25, fn _ -> 50.0 + :rand.normal() * 2 end)
      obs = Enum.map(1..25, fn i -> Enum.at(control, i - 1) + :rand.normal() end)

      intervention = %{
        intervention_name: "trend_check_policy",
        intervention_date_index: 21,
        pre_period_start: 1,
        control_series: [control]
      }

      result = PolicyEvaluator.pre_trend_check(obs, intervention)

      assert result.valid == true
      assert is_float(result.slope)
      assert is_list(result.control_diagnostics)
      assert length(result.control_diagnostics) == 1
      diag = hd(result.control_diagnostics)
      assert is_float(diag.control_mean)
      assert is_float(diag.mean_difference)
    end

    test "preserves model_spec when controls are not provided" do
      :rand.seed(:exsss, {350, 351, 352})
      pre = Enum.map(1..20, fn _ -> 50.0 + :rand.normal() * 2 end)
      post = Enum.map(1..5, fn _ -> 45.0 + :rand.normal() * 2 end)
      obs = pre ++ post

      intervention = %{
        intervention_name: "custom_spec_policy",
        intervention_date_index: 21,
        pre_period_start: 1
      }

      spec = Components.local_level_spec(initial_state: 50.0, initial_cov: 10.0)

      result =
        PolicyEvaluator.evaluate(obs, intervention,
          model_spec: spec,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert result.analysis.model_spec != nil
      assert Nx.all_close(result.analysis.model_spec.f, spec.f)
      assert Nx.all_close(result.analysis.model_spec.h, spec.h)
    end

    test "composes control_series into provided model_spec" do
      :rand.seed(:exsss, {355, 356, 357})
      control = Enum.map(1..25, fn _ -> 50.0 + :rand.normal() * 2 end)
      pre = Enum.map(Enum.take(control, 20), fn c -> c + :rand.normal() end)
      post = Enum.map(Enum.drop(control, 20), fn c -> c - 8.0 + :rand.normal() end)
      obs = pre ++ post

      intervention = %{
        intervention_name: "custom_spec_with_controls_policy",
        intervention_date_index: 21,
        pre_period_start: 1,
        control_series: [control]
      }

      base_spec = Components.local_level_spec(initial_state: 50.0, initial_cov: 10.0)

      result =
        PolicyEvaluator.evaluate(obs, intervention,
          model_spec: base_spec,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      # local_level (1 state) + one control regressor = 2 states
      assert Nx.axis_size(result.analysis.model_spec.f, 0) == 2
      assert is_float(result.effect.cumulative)
    end
  end

  describe "evaluate_multi/3" do
    test "evaluates multiple outcome metrics" do
      :rand.seed(:exsss, {400, 401, 402})

      intervention = %{
        intervention_name: "multi_outcome_policy",
        intervention_date_index: 21,
        pre_period_start: 1
      }

      outcomes = %{
        "accidents" =>
          Enum.map(1..20, fn _ -> 50.0 + :rand.normal() * 2 end) ++
            Enum.map(1..5, fn _ -> 40.0 + :rand.normal() * 2 end),
        "speeding_tickets" =>
          Enum.map(1..20, fn _ -> 100.0 + :rand.normal() * 5 end) ++
            Enum.map(1..5, fn _ -> 80.0 + :rand.normal() * 5 end)
      }

      results =
        PolicyEvaluator.evaluate_multi(outcomes, intervention,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert Map.has_key?(results, "accidents")
      assert Map.has_key?(results, "speeding_tickets")
      assert results["accidents"].intervention.outcome_metric == "accidents"
    end
  end

  describe "pre_trend_check/3" do
    test "validates pre-intervention trends" do
      :rand.seed(:exsss, {500, 501, 502})
      data = Enum.map(1..30, fn _ -> 50.0 + :rand.normal() * 2 end)

      intervention = %{
        intervention_name: "test",
        intervention_date_index: 21,
        pre_period_start: 1
      }

      result = PolicyEvaluator.pre_trend_check(data, intervention)

      assert result.valid == true
      assert is_float(result.slope)
      assert is_float(result.t_statistic)
      assert is_boolean(result.significant_trend?)
      assert result.n_pre == 20
    end

    test "handles insufficient data" do
      intervention = %{
        intervention_name: "test",
        intervention_date_index: 3,
        pre_period_start: 1
      }

      result = PolicyEvaluator.pre_trend_check([1.0, 2.0, 3.0], intervention)

      # Only 2 pre-period points (indices 1..2) — insufficient for trend check
      assert result.valid == false
      assert String.contains?(result.reason, "insufficient")
    end
  end

  describe "validation" do
    test "rejects empty observations" do
      intervention = %{
        intervention_name: "test",
        intervention_date_index: 5,
        pre_period_start: 1
      }

      assert_raise ArgumentError, ~r/non-empty/, fn ->
        PolicyEvaluator.evaluate([], intervention)
      end
    end

    test "rejects invalid lag" do
      intervention = %{
        intervention_name: "test",
        intervention_date_index: 5,
        pre_period_start: 1
      }

      assert_raise ArgumentError, ~r/lag/, fn ->
        PolicyEvaluator.evaluate([1.0, 2.0, 3.0, 4.0, 5.0], intervention, lag: -1)
      end
    end
  end
end
