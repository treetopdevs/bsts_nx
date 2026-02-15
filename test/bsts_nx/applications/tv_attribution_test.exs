defmodule BstsNx.Applications.TVAttributionTest do
  use ExUnit.Case, async: true

  alias BstsNx.Applications.TVAttribution

  describe "attribute/3" do
    test "runs full TV attribution pipeline" do
      :rand.seed(:exsss, {100, 101, 102})
      pre = Enum.map(1..30, fn _ -> 1000.0 + :rand.normal() * 50 end)
      post = Enum.map(1..10, fn _ -> 1200.0 + :rand.normal() * 50 end)
      obs = pre ++ post

      spots = [
        %{id: "spot_1", window_start: 0, window_end: 5},
        %{id: "spot_2", window_start: 5, window_end: 10}
      ]

      result =
        TVAttribution.attribute(obs, spots,
          pre_period: {1, 30},
          post_period: {31, 40},
          seasonality: 7,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert length(result.spots) == 2
      assert is_float(result.total_lift)
      assert is_float(result.total_lift_sd)
      assert is_list(result.overlaps)
      assert is_map(result.causal_impact_summary)
    end
  end

  describe "attribute/3 with regressors" do
    test "runs attribution with regressor covariates" do
      :rand.seed(:exsss, {110, 111, 112})
      pre = Enum.map(1..30, fn _ -> 1000.0 + :rand.normal() * 50 end)
      post = Enum.map(1..10, fn _ -> 1200.0 + :rand.normal() * 50 end)
      obs = pre ++ post

      # Competitor ad spending as a regressor
      regressors = Nx.tensor(Enum.map(1..40, fn _ -> [:rand.normal() * 10] end))

      spots = [
        %{id: "spot_1", window_start: 0, window_end: 5},
        %{id: "spot_2", window_start: 5, window_end: 10}
      ]

      result =
        TVAttribution.attribute(obs, spots,
          pre_period: {1, 30},
          post_period: {31, 40},
          seasonality: 7,
          regressors: regressors,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert length(result.spots) == 2
      assert is_float(result.total_lift)
    end
  end

  describe "rolling_baseline/2" do
    test "produces counterfactual baseline" do
      :rand.seed(:exsss, {200, 201, 202})
      data = Enum.map(1..50, fn _ -> 1000.0 + :rand.normal() * 50 end)

      {baseline, fit} =
        TVAttribution.rolling_baseline(data,
          horizon: 5,
          num_seasons: 7,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert length(baseline.mean) == 5
      assert length(baseline.variance) == 5
      assert is_float(baseline.obs_variance)
      assert is_map(fit)
    end
  end

  describe "rolling_baseline/2 with regressors" do
    test "passes regressors through to RollingBaseline" do
      :rand.seed(:exsss, {210, 211, 212})
      data = Enum.map(1..50, fn _ -> 1000.0 + :rand.normal() * 50 end)
      # Regressors must cover training (50) + horizon (5) = 55 rows
      regressors = Nx.tensor(Enum.map(1..55, fn _ -> [:rand.normal() * 10] end))

      {baseline, fit} =
        TVAttribution.rolling_baseline(data,
          horizon: 5,
          num_seasons: 7,
          num_samples: 10,
          burn_in: 5,
          seed: 42,
          regressors: regressors
        )

      assert length(baseline.mean) == 5
      assert length(baseline.variance) == 5
      assert is_float(baseline.obs_variance)
      assert is_map(fit)
    end
  end

  describe "attribute_from_baseline/4" do
    test "attributes spots from pre-computed baseline" do
      obs = [100.0, 150.0, 140.0, 130.0, 100.0]

      cf = %{
        mean: [100.0, 100.0, 100.0, 100.0, 100.0],
        variance: [4.0, 4.0, 4.0, 4.0, 4.0],
        obs_variance: 1.0
      }

      spots = [%{id: "spot_1", window_start: 1, window_end: 4}]

      result = TVAttribution.attribute_from_baseline(obs, spots, cf)

      assert length(result.attributions) == 1
      attr = hd(result.attributions)
      assert attr.spot_id == "spot_1"
      # Lift should be sum of (obs - baseline) for indices 1,2,3
      expected_lift = 150.0 - 100.0 + (140.0 - 100.0) + (130.0 - 100.0)
      assert_in_delta attr.lift, expected_lift, 0.01
    end
  end
end
