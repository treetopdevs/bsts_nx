defmodule BstsNx.Applications.MarketingLiftTest do
  use ExUnit.Case, async: true

  alias BstsNx.Applications.MarketingLift
  alias BstsNx.Components

  describe "measure_lift/3" do
    test "detects positive campaign lift" do
      :rand.seed(:exsss, {100, 101, 102})
      pre = Enum.map(1..30, fn _ -> 100.0 + :rand.normal() * 5 end)
      post = Enum.map(1..10, fn _ -> 120.0 + :rand.normal() * 5 end)
      obs = pre ++ post

      campaign = %{
        name: "summer_email",
        channel: :email,
        start_index: 31,
        end_index: 40,
        baseline_start: 1,
        baseline_end: 30
      }

      result = MarketingLift.measure_lift(obs, campaign, num_samples: 10, burn_in: 5, seed: 42)

      assert result.campaign.name == "summer_email"
      assert result.effect.cumulative > 0
      assert is_float(result.effect.cumulative_sd)
      assert is_float(result.effect.cumulative_lower)
      assert is_float(result.effect.cumulative_upper)
      assert is_float(result.effect.relative)
      assert is_boolean(result.significant?)
      assert result.baseline_mean > 0
      assert result.actual_mean > result.baseline_mean
    end

    test "handles zero-effect campaign" do
      :rand.seed(:exsss, {200, 201, 202})
      obs = Enum.map(1..40, fn _ -> 100.0 + :rand.normal() * 5 end)

      campaign = %{
        name: "no_effect_campaign",
        channel: :paid_search,
        start_index: 31,
        end_index: 40,
        baseline_start: 1,
        baseline_end: 30
      }

      result = MarketingLift.measure_lift(obs, campaign, num_samples: 10, burn_in: 5, seed: 42)

      # Effect should be near zero
      assert abs(result.effect.cumulative) < 200.0
    end
  end

  describe "measure_multi/3" do
    test "handles multiple non-overlapping campaigns" do
      :rand.seed(:exsss, {300, 301, 302})
      baseline = Enum.map(1..20, fn _ -> 100.0 + :rand.normal() * 3 end)
      camp1 = Enum.map(1..5, fn _ -> 115.0 + :rand.normal() * 3 end)
      gap = Enum.map(1..10, fn _ -> 100.0 + :rand.normal() * 3 end)
      camp2 = Enum.map(1..5, fn _ -> 110.0 + :rand.normal() * 3 end)
      obs = baseline ++ camp1 ++ gap ++ camp2

      campaigns = [
        %{
          name: "campaign_a",
          channel: :email,
          start_index: 21,
          end_index: 25,
          baseline_start: 1,
          baseline_end: 20
        },
        %{
          name: "campaign_b",
          channel: :push,
          start_index: 36,
          end_index: 40,
          baseline_start: 26,
          baseline_end: 35
        }
      ]

      result =
        MarketingLift.measure_multi(obs, campaigns, num_samples: 10, burn_in: 5, seed: 42)

      assert length(result.campaigns) == 2
      assert is_float(result.total_lift)
      assert is_float(result.total_lift_sd)
    end

    test "handles empty campaign list" do
      result = MarketingLift.measure_multi([1.0, 2.0, 3.0], [])

      assert result.campaigns == []
      assert result.total_lift == 0.0
    end
  end

  describe "measure_lift_by_date/4" do
    test "converts dates to indices and measures lift" do
      :rand.seed(:exsss, {500, 501, 502})
      pre = Enum.map(1..20, fn _ -> 100.0 + :rand.normal() * 5 end)
      post = Enum.map(1..5, fn _ -> 120.0 + :rand.normal() * 5 end)
      obs = pre ++ post

      # Generate date series: 25 consecutive days
      base_date = ~D[2024-01-01]
      dates = Enum.map(0..24, fn i -> Date.add(base_date, i) end)

      campaign = %{
        name: "date_campaign",
        channel: :email,
        start_date: ~D[2024-01-21],
        end_date: ~D[2024-01-25],
        baseline_start_date: ~D[2024-01-01],
        baseline_end_date: ~D[2024-01-20]
      }

      result =
        MarketingLift.measure_lift_by_date(dates, obs, campaign,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert result.campaign.name == "date_campaign"
      assert result.campaign.start_index == 21
      assert result.campaign.end_index == 25
      assert result.campaign.baseline_start == 1
      assert result.campaign.baseline_end == 20
      assert is_float(result.effect.cumulative)
    end
  end

  describe "measure_multi/3 with overlapping campaigns" do
    test "uses Shapley allocation for overlapping campaigns" do
      :rand.seed(:exsss, {600, 601, 602})
      baseline = Enum.map(1..20, fn _ -> 100.0 + :rand.normal() * 3 end)
      overlap = Enum.map(1..10, fn _ -> 120.0 + :rand.normal() * 3 end)
      obs = baseline ++ overlap

      # Both campaigns share the same end_index so the window_end
      # stays within the SpotAttributor's post-period length.
      campaigns = [
        %{
          name: "campaign_x",
          channel: :podcast,
          start_index: 21,
          end_index: 27,
          baseline_start: 1,
          baseline_end: 20
        },
        %{
          name: "campaign_y",
          channel: :radio,
          start_index: 24,
          end_index: 27,
          baseline_start: 1,
          baseline_end: 20
        }
      ]

      result =
        MarketingLift.measure_multi(obs, campaigns, num_samples: 10, burn_in: 5, seed: 42)

      assert length(result.campaigns) == 2
      assert length(result.overlap_groups) > 0

      # Verify overlapping campaigns have Shapley-allocated lift values
      Enum.each(result.campaigns, fn r ->
        assert is_float(r.effect.cumulative)
        assert is_float(r.effect.cumulative_sd)
        assert is_float(r.effect.relative)

        n_post = r.campaign.end_index - r.campaign.start_index + 1
        denom = r.baseline_mean * n_post
        expected_relative = if abs(denom) > 1.0e-10, do: r.effect.cumulative / denom, else: 0.0
        assert_in_delta r.effect.relative, expected_relative, 1.0e-6
      end)
    end

    test "filter method does not duplicate one group-level effect across campaigns" do
      :rand.seed(:exsss, {610, 611, 612})
      baseline = Enum.map(1..20, fn _ -> 100.0 + :rand.normal() * 2 end)
      overlap = Enum.map(1..10, fn _ -> 120.0 + :rand.normal() * 2 end)
      obs = baseline ++ overlap

      campaigns = [
        %{
          name: "campaign_long",
          channel: :podcast,
          start_index: 21,
          end_index: 28,
          baseline_start: 1,
          baseline_end: 20
        },
        %{
          name: "campaign_short",
          channel: :radio,
          start_index: 24,
          end_index: 27,
          baseline_start: 1,
          baseline_end: 20
        }
      ]

      result = MarketingLift.measure_multi(obs, campaigns, method: :filter)
      [r1, r2] = Enum.sort_by(result.campaigns, & &1.campaign.name)

      Enum.each(result.campaigns, fn campaign_result ->
        assert campaign_result.analysis.impact == nil
      end)

      # Distinct windows should not receive an identical copied group effect.
      assert abs(r1.effect.cumulative - r2.effect.cumulative) > 1.0e-6
    end
  end

  describe "model options" do
    test "preserves model_spec when controls are not provided" do
      :rand.seed(:exsss, {700, 701, 702})
      pre = Enum.map(1..30, fn _ -> 100.0 + :rand.normal() * 2 end)
      post = Enum.map(1..10, fn _ -> 110.0 + :rand.normal() * 2 end)
      obs = pre ++ post

      campaign = %{
        name: "custom_spec_campaign",
        channel: :email,
        start_index: 31,
        end_index: 40,
        baseline_start: 1,
        baseline_end: 30
      }

      spec = Components.local_level_spec(initial_state: 100.0, initial_cov: 5.0)

      result =
        MarketingLift.measure_lift(obs, campaign,
          model_spec: spec,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert result.analysis.model_spec != nil
      assert Nx.all_close(result.analysis.model_spec.f, spec.f)
      assert Nx.all_close(result.analysis.model_spec.h, spec.h)
    end

    test "single control series detects campaign lift" do
      :rand.seed(:exsss, {720, 721, 722})
      # Control series from a region without the campaign
      control = Enum.map(1..40, fn _ -> 100.0 + :rand.normal() * 3 end)
      pre = Enum.map(Enum.take(control, 30), fn c -> c + :rand.normal() * 2 end)
      post = Enum.map(Enum.drop(control, 30), fn c -> c + 15.0 + :rand.normal() * 2 end)
      obs = pre ++ post

      campaign = %{
        name: "control_series_campaign",
        channel: :email,
        start_index: 31,
        end_index: 40,
        baseline_start: 1,
        baseline_end: 30
      }

      result =
        MarketingLift.measure_lift(obs, campaign,
          control_series: [control],
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert result.analysis.model_spec != nil
      # Model should include regression dimensions
      assert Nx.axis_size(result.analysis.model_spec.f, 0) >= 3
      assert is_float(result.effect.cumulative)
    end

    test "multiple control series (p > 1)" do
      :rand.seed(:exsss, {730, 731, 732})
      control1 = Enum.map(1..40, fn _ -> 100.0 + :rand.normal() * 3 end)
      control2 = Enum.map(1..40, fn _ -> 50.0 + :rand.normal() * 5 end)
      pre = Enum.map(1..30, fn i -> Enum.at(control1, i - 1) + Enum.at(control2, i - 1) * 0.5 end)
      post = Enum.map(31..40, fn i -> Enum.at(control1, i - 1) + 20.0 end)
      obs = pre ++ post

      campaign = %{
        name: "multi_control_campaign",
        channel: :paid_search,
        start_index: 31,
        end_index: 40,
        baseline_start: 1,
        baseline_end: 30
      }

      result =
        MarketingLift.measure_lift(obs, campaign,
          control_series: [control1, control2],
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      # Trend (2) + 2 regressors = 4 state dims
      assert Nx.axis_size(result.analysis.model_spec.f, 0) == 4
      assert is_float(result.effect.cumulative)
    end

    test "composes control_series into provided model_spec" do
      :rand.seed(:exsss, {735, 736, 737})
      control = Enum.map(1..40, fn _ -> 100.0 + :rand.normal() * 3 end)
      pre = Enum.map(Enum.take(control, 30), fn c -> c + :rand.normal() * 2 end)
      post = Enum.map(Enum.drop(control, 30), fn c -> c + 12.0 + :rand.normal() * 2 end)
      obs = pre ++ post

      campaign = %{
        name: "custom_spec_plus_controls",
        channel: :email,
        start_index: 31,
        end_index: 40,
        baseline_start: 1,
        baseline_end: 30
      }

      base_spec = Components.local_level_spec(initial_state: 100.0, initial_cov: 5.0)

      result =
        MarketingLift.measure_lift(obs, campaign,
          model_spec: base_spec,
          control_series: [control],
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      # local_level (1 state) + one control regressor = 2 states
      assert Nx.axis_size(result.analysis.model_spec.f, 0) == 2
      assert is_float(result.effect.cumulative)
    end

    test "mismatched control_series length raises" do
      campaign = %{
        name: "bad_length",
        channel: :email,
        start_index: 4,
        end_index: 5,
        baseline_start: 1,
        baseline_end: 3
      }

      assert_raise ArgumentError, ~r/control series length/, fn ->
        MarketingLift.measure_lift([1.0, 2.0, 3.0, 4.0, 5.0], campaign,
          control_series: [[1.0, 2.0]]
        )
      end
    end

    test "treats empty control_series as no controls" do
      :rand.seed(:exsss, {710, 711, 712})
      pre = Enum.map(1..30, fn _ -> 100.0 + :rand.normal() * 2 end)
      post = Enum.map(1..10, fn _ -> 110.0 + :rand.normal() * 2 end)
      obs = pre ++ post

      campaign = %{
        name: "empty_controls_campaign",
        channel: :email,
        start_index: 31,
        end_index: 40,
        baseline_start: 1,
        baseline_end: 30
      }

      result =
        MarketingLift.measure_lift(obs, campaign,
          control_series: [],
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert is_float(result.effect.cumulative)
    end
  end

  describe "validation" do
    test "rejects empty observations" do
      campaign = %{
        name: "test",
        channel: :email,
        start_index: 1,
        end_index: 5,
        baseline_start: 1,
        baseline_end: 5
      }

      assert_raise ArgumentError, ~r/non-empty/, fn ->
        MarketingLift.measure_lift([], campaign)
      end
    end

    test "rejects invalid alpha" do
      campaign = %{
        name: "test",
        channel: :email,
        start_index: 31,
        end_index: 35,
        baseline_start: 1,
        baseline_end: 30
      }

      assert_raise ArgumentError, ~r/alpha/, fn ->
        MarketingLift.measure_lift([1.0, 2.0, 3.0], campaign, alpha: 2.0)
      end
    end

    test "rejects invalid seasonality" do
      campaign = %{
        name: "test",
        channel: :email,
        start_index: 31,
        end_index: 35,
        baseline_start: 1,
        baseline_end: 30
      }

      obs = Enum.map(1..40, fn _ -> 10.0 end)

      assert_raise ArgumentError, ~r/seasonality/, fn ->
        MarketingLift.measure_lift(obs, campaign, seasonality: 1)
      end
    end
  end
end
