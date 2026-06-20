defmodule BstsNx.PipelineTest do
  use ExUnit.Case, async: true

  alias BstsNx.Pipeline
  alias BstsNx.Components

  # ── Result structure ──────────────────────────────────────────────────

  describe "run/6 result structure" do
    test "returns expected top-level keys" do
      {obs, spec, spots} = basic_setup()

      result =
        Pipeline.run(obs, {1, 30}, {31, 40}, spots, spec,
          num_samples: 5,
          burn_in: 2,
          seed: 42
        )

      assert Map.has_key?(result, :attributions)
      assert Map.has_key?(result, :causal_impact)
      assert Map.has_key?(result, :summary)
      assert Map.has_key?(result, :counterfactual)
      assert Map.has_key?(result, :execution)
      assert result.execution.mode == :operational
    end

    test "attributions has SpotAttributor structure" do
      {obs, spec, spots} = basic_setup()

      result =
        Pipeline.run(obs, {1, 30}, {31, 40}, spots, spec,
          num_samples: 5,
          burn_in: 2,
          seed: 42
        )

      attr = result.attributions
      assert is_list(attr.attributions)
      assert is_number(attr.total_lift)
      assert is_number(attr.total_lift_sd)
      assert is_list(attr.overlap_groups)
    end

    test "causal_impact has impact_result structure" do
      {obs, spec, spots} = basic_setup()

      result =
        Pipeline.run(obs, {1, 30}, {31, 40}, spots, spec,
          mode: :bayesian,
          num_samples: 5,
          burn_in: 2,
          seed: 42
        )

      ci = result.causal_impact
      assert is_list(ci.point_effects)
      assert is_list(ci.cumulative_effects)
      assert is_list(ci.actual)
      assert is_list(ci.counterfactual)
      assert ci.pre_period == {1, 30}
      assert ci.post_period == {31, 40}
    end

    test "summary has CausalImpact summary structure" do
      {obs, spec, spots} = basic_setup()

      result =
        Pipeline.run(obs, {1, 30}, {31, 40}, spots, spec,
          mode: :bayesian,
          num_samples: 5,
          burn_in: 2,
          seed: 42
        )

      summary = result.summary
      assert is_list(summary.point_effects)
      assert is_map(summary.cumulative_effect)
      assert is_map(summary.relative_effect)
      assert result.execution.mode == :bayesian
    end
  end

  # ── Single spot ───────────────────────────────────────────────────────

  describe "single non-overlapping spot" do
    test "detects positive lift with one spot covering full post period" do
      :rand.seed(:exsss, {100, 101, 102})
      pre = Enum.map(1..40, fn _ -> 50.0 + :rand.normal() * 2 end)
      post = Enum.map(1..10, fn _ -> 60.0 + :rand.normal() * 2 end)
      obs = pre ++ post

      spec = Components.local_level_spec(initial_state: 50.0, initial_cov: 10.0)
      spots = [%{id: "s1", window_start: 0, window_end: 10}]

      result =
        Pipeline.run(obs, {1, 40}, {41, 50}, spots, spec,
          num_samples: 15,
          burn_in: 5,
          seed: 42
        )

      assert length(result.attributions.attributions) == 1
      [attr] = result.attributions.attributions
      assert attr.spot_id == "s1"
      assert attr.lift > 0
    end

    test "spot covering partial post period" do
      :rand.seed(:exsss, {200, 201, 202})
      pre = Enum.map(1..30, fn _ -> 50.0 + :rand.normal() * 2 end)
      post = Enum.map(1..10, fn _ -> 55.0 + :rand.normal() * 2 end)
      obs = pre ++ post

      spec = Components.local_level_spec(initial_state: 50.0, initial_cov: 10.0)
      spots = [%{id: "s1", window_start: 2, window_end: 7}]

      result =
        Pipeline.run(obs, {1, 30}, {31, 40}, spots, spec,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert length(result.attributions.attributions) == 1
      [attr] = result.attributions.attributions
      assert attr.window_start == 2
      assert attr.window_end == 7
    end

    test "allows a gap between pre_period and post_period" do
      :rand.seed(:exsss, {210, 211, 212})
      pre = Enum.map(1..30, fn _ -> 50.0 + :rand.normal() * 2 end)
      washout = Enum.map(1..4, fn _ -> 52.0 + :rand.normal() * 2 end)
      post = Enum.map(1..6, fn _ -> 55.0 + :rand.normal() * 2 end)
      obs = pre ++ washout ++ post

      spec = Components.local_level_spec(initial_state: 50.0, initial_cov: 10.0)
      spots = [%{id: "s1", window_start: 0, window_end: 6}]

      result =
        Pipeline.run(obs, {1, 30}, {35, 40}, spots, spec,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert length(result.summary.actual) == 6
      assert length(result.attributions.attributions) == 1
    end
  end

  # ── Multiple non-overlapping spots ────────────────────────────────────

  describe "multiple non-overlapping spots" do
    test "each spot gets independent attribution" do
      :rand.seed(:exsss, {300, 301, 302})
      pre = Enum.map(1..40, fn _ -> 50.0 + :rand.normal() * 2 end)
      post = Enum.map(1..10, fn _ -> 60.0 + :rand.normal() * 2 end)
      obs = pre ++ post

      spec = Components.local_level_spec(initial_state: 50.0, initial_cov: 10.0)

      spots = [
        %{id: "early", window_start: 0, window_end: 4},
        %{id: "late", window_start: 6, window_end: 10}
      ]

      result =
        Pipeline.run(obs, {1, 40}, {41, 50}, spots, spec,
          num_samples: 15,
          burn_in: 5,
          seed: 42
        )

      assert length(result.attributions.attributions) == 2
      assert result.attributions.overlap_groups == []

      ids = Enum.map(result.attributions.attributions, & &1.spot_id)
      assert "early" in ids
      assert "late" in ids
    end
  end

  # ── Overlapping spots ─────────────────────────────────────────────────

  describe "overlapping spots" do
    test "uses Shapley allocation for overlapping windows" do
      :rand.seed(:exsss, {400, 401, 402})
      pre = Enum.map(1..40, fn _ -> 50.0 + :rand.normal() * 2 end)
      post = Enum.map(1..10, fn _ -> 60.0 + :rand.normal() * 2 end)
      obs = pre ++ post

      spec = Components.local_level_spec(initial_state: 50.0, initial_cov: 10.0)

      spots = [
        %{id: "a", window_start: 0, window_end: 6},
        %{id: "b", window_start: 4, window_end: 10}
      ]

      result =
        Pipeline.run(obs, {1, 40}, {41, 50}, spots, spec,
          num_samples: 15,
          burn_in: 5,
          seed: 42
        )

      assert length(result.attributions.attributions) == 2
      assert length(result.attributions.overlap_groups) == 1
      assert ["a", "b"] in result.attributions.overlap_groups
    end
  end

  # ── Empty spots ───────────────────────────────────────────────────────

  describe "empty spots" do
    test "returns empty attributions when no spots provided" do
      :rand.seed(:exsss, {500, 501, 502})
      pre = Enum.map(1..30, fn _ -> 50.0 + :rand.normal() * 2 end)
      post = Enum.map(1..10, fn _ -> 55.0 + :rand.normal() * 2 end)
      obs = pre ++ post

      spec = Components.local_level_spec(initial_state: 50.0, initial_cov: 10.0)

      result =
        Pipeline.run(obs, {1, 30}, {31, 40}, [], spec,
          num_samples: 5,
          burn_in: 2,
          seed: 42
        )

      assert result.attributions.attributions == []
      assert result.attributions.total_lift == 0.0
      # Operational filter still runs.
      assert length(result.summary.actual) == 10
    end
  end

  # ── Spot window validation ────────────────────────────────────────────

  describe "spot window validation" do
    setup do
      :rand.seed(:exsss, {600, 601, 602})
      obs = Enum.map(1..20, fn _ -> 50.0 + :rand.normal() end)
      spec = Components.local_level_spec()
      %{obs: obs, spec: spec}
    end

    test "raises when spot window exceeds post period", %{obs: obs, spec: spec} do
      spots = [%{id: "bad", window_start: 0, window_end: 15}]

      assert_raise ArgumentError, ~r/outside the post period/, fn ->
        Pipeline.run(obs, {1, 10}, {11, 20}, spots, spec, num_samples: 2, burn_in: 1, seed: 1)
      end
    end

    test "raises when spot window_start is negative", %{obs: obs, spec: spec} do
      spots = [%{id: "bad", window_start: -1, window_end: 5}]

      assert_raise ArgumentError, ~r/outside the post period/, fn ->
        Pipeline.run(obs, {1, 10}, {11, 20}, spots, spec, num_samples: 2, burn_in: 1, seed: 1)
      end
    end

    test "raises when spot window_start > window_end", %{obs: obs, spec: spec} do
      spots = [%{id: "bad", window_start: 7, window_end: 3}]

      assert_raise ArgumentError, ~r/window_start.*>.*window_end/, fn ->
        Pipeline.run(obs, {1, 10}, {11, 20}, spots, spec, num_samples: 2, burn_in: 1, seed: 1)
      end
    end

    test "raises when spot window is empty", %{obs: obs, spec: spec} do
      spots = [%{id: "bad", window_start: 3, window_end: 3}]

      assert_raise ArgumentError, ~r/empty window/, fn ->
        Pipeline.run(obs, {1, 10}, {11, 20}, spots, spec, num_samples: 2, burn_in: 1, seed: 1)
      end
    end

    test "allows non-contiguous pre and post periods", %{obs: obs, spec: spec} do
      spots = [%{id: "s1", window_start: 0, window_end: 3}]

      result =
        Pipeline.run(obs, {1, 10}, {13, 20}, spots, spec, num_samples: 2, burn_in: 1, seed: 1)

      assert length(result.summary.actual) == 8
      assert length(result.attributions.attributions) == 1
    end
  end

  # ── Composed model (level + seasonal) ─────────────────────────────────

  describe "with composed level + seasonal model" do
    test "pipeline works end-to-end with composed spec" do
      :rand.seed(:exsss, {700, 701, 702})
      pattern = [5.0, 2.0, -3.0, -4.0]
      base = 200.0
      lift = 20.0

      pre =
        Enum.flat_map(1..8, fn _q ->
          Enum.map(pattern, fn s -> base + s + :rand.normal() * 2.0 end)
        end)

      post =
        Enum.flat_map(1..2, fn _q ->
          Enum.map(pattern, fn s -> base + s + lift + :rand.normal() * 2.0 end)
        end)

      obs = pre ++ post
      n_pre = length(pre)
      n_total = length(obs)

      ll =
        Components.local_level_spec(
          process_var: 1.0,
          obs_var: 2.0,
          initial_state: 200.0,
          initial_cov: 50.0
        )

      ss = Components.seasonal_spec(4, process_var: 0.1)
      combined = Components.compose_specs(ll, ss)

      spots = [%{id: "campaign", window_start: 0, window_end: n_total - n_pre}]

      result =
        Pipeline.run(obs, {1, n_pre}, {n_pre + 1, n_total}, spots, combined,
          num_samples: 20,
          burn_in: 10,
          seed: 42
        )

      assert length(result.attributions.attributions) == 1
      [attr] = result.attributions.attributions
      assert attr.spot_id == "campaign"
      # Should detect positive lift
      assert attr.lift > 0
    end
  end

  # ── Deterministic PRNG ────────────────────────────────────────────────

  describe "deterministic results" do
    test "same seed produces identical results" do
      :rand.seed(:exsss, {800, 801, 802})
      pre = Enum.map(1..30, fn _ -> 50.0 + :rand.normal() * 2 end)
      post = Enum.map(1..10, fn _ -> 55.0 + :rand.normal() * 2 end)
      obs = pre ++ post

      spec = Components.local_level_spec(initial_state: 50.0, initial_cov: 10.0)
      spots = [%{id: "s1", window_start: 0, window_end: 5}]
      common_opts = [num_samples: 5, burn_in: 2, seed: 42]

      r1 = Pipeline.run(obs, {1, 30}, {31, 40}, spots, spec, common_opts)
      r2 = Pipeline.run(obs, {1, 30}, {31, 40}, spots, spec, common_opts)

      [a1] = r1.attributions.attributions
      [a2] = r2.attributions.attributions
      assert a1.lift == a2.lift
      assert a1.lift_sd == a2.lift_sd
    end
  end

  # ── Options forwarding ────────────────────────────────────────────────

  describe "options forwarding" do
    test "alpha option affects CI width" do
      {obs, spec, spots} = basic_setup()

      # Need enough posterior draws so that quantile-based CIs differ
      # between alpha=0.01 and alpha=0.20. With burn_in draws discarded,
      # effective draws = num_samples - burn_in.
      r_narrow =
        Pipeline.run(obs, {1, 30}, {31, 40}, spots, spec,
          num_samples: 30,
          burn_in: 5,
          seed: 42,
          alpha: 0.01
        )

      r_wide =
        Pipeline.run(obs, {1, 30}, {31, 40}, spots, spec,
          num_samples: 30,
          burn_in: 5,
          seed: 42,
          alpha: 0.20
        )

      [a_narrow] = r_narrow.attributions.attributions
      [a_wide] = r_wide.attributions.attributions

      # 99% CI should be wider than 80% CI (quantile-based)
      narrow_width = a_narrow.lift_upper - a_narrow.lift_lower
      wide_width = a_wide.lift_upper - a_wide.lift_lower
      assert narrow_width >= wide_width

      narrow_summary_width =
        r_narrow.summary.cumulative_effect.upper - r_narrow.summary.cumulative_effect.lower

      wide_summary_width =
        r_wide.summary.cumulative_effect.upper - r_wide.summary.cumulative_effect.lower

      assert narrow_summary_width >= wide_summary_width
    end
  end

  # ── Helpers ───────────────────────────────────────────────────────────

  defp basic_setup do
    :rand.seed(:exsss, {900, 901, 902})
    pre = Enum.map(1..30, fn _ -> 50.0 + :rand.normal() * 2 end)
    post = Enum.map(1..10, fn _ -> 55.0 + :rand.normal() * 2 end)
    obs = pre ++ post
    spec = Components.local_level_spec(initial_state: 50.0, initial_cov: 10.0)
    spots = [%{id: "s1", window_start: 0, window_end: 10}]
    {obs, spec, spots}
  end
end
