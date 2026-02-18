defmodule BstsNx.CausalImpactStructuredTest do
  use ExUnit.Case, async: true

  alias BstsNx.CausalImpact
  alias BstsNx.Components

  # ── Basic structure and API ────────────────────────────────────────────

  describe "estimate_structured/5 with local_level_spec" do
    test "returns expected result structure" do
      :rand.seed(:exsss, {100, 101, 102})
      pre = Enum.map(1..40, fn _ -> 50.0 + :rand.normal() * 2 end)
      post = Enum.map(1..10, fn _ -> 55.0 + :rand.normal() * 2 end)
      obs = pre ++ post

      spec =
        Components.local_level_spec(
          process_var: 1.0,
          obs_var: 1.0,
          initial_state: 50.0,
          initial_cov: 10.0
        )

      result =
        CausalImpact.estimate_structured(obs, {1, 40}, {41, 50}, spec,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert is_list(result.point_effects)
      assert length(result.point_effects) == 10
      assert Enum.all?(result.point_effects, fn pe -> length(pe) == 10 end)
      assert length(result.cumulative_effects) == 10
      assert length(result.relative_effects) == 10
      assert length(result.actual) == 10
      assert length(result.counterfactual) == 10
      assert result.pre_period == {1, 40}
      assert result.post_period == {41, 50}
    end

    test "summary works on structured results" do
      :rand.seed(:exsss, {200, 201, 202})
      pre = Enum.map(1..40, fn _ -> 50.0 + :rand.normal() * 2 end)
      post = Enum.map(1..10, fn _ -> 55.0 + :rand.normal() * 2 end)
      obs = pre ++ post

      spec = Components.local_level_spec(initial_state: 50.0, initial_cov: 10.0)

      result =
        CausalImpact.estimate_structured(obs, {1, 40}, {41, 50}, spec,
          num_samples: 15,
          burn_in: 5,
          seed: 99
        )

      summary = CausalImpact.summary(result)

      assert is_number(summary.cumulative_effect.mean)
      assert is_number(summary.cumulative_effect.sd)
      assert summary.cumulative_effect.lower <= summary.cumulative_effect.mean
      assert summary.cumulative_effect.mean <= summary.cumulative_effect.upper
      assert length(summary.point_effects) == 10
    end
  end

  # ── Period validation ──────────────────────────────────────────────────

  describe "estimate_structured period validation" do
    setup do
      spec = Components.local_level_spec()
      obs = Enum.map(1..20, fn i -> i * 1.0 end)
      %{spec: spec, obs: obs}
    end

    test "raises for invalid pre_period", %{spec: spec, obs: obs} do
      assert_raise ArgumentError, ~r/pre_period/, fn ->
        CausalImpact.estimate_structured(obs, {0, 10}, {11, 20}, spec, seed: 1)
      end
    end

    test "raises when pre_period exceeds observations", %{spec: spec, obs: obs} do
      assert_raise ArgumentError, ~r/pre_period end/, fn ->
        CausalImpact.estimate_structured(obs, {1, 25}, {26, 30}, spec, seed: 1)
      end
    end

    test "raises when post_period doesn't follow pre_period", %{spec: spec, obs: obs} do
      assert_raise ArgumentError, ~r/post_period must immediately follow/, fn ->
        CausalImpact.estimate_structured(obs, {1, 10}, {12, 20}, spec, seed: 1)
      end
    end

    test "raises when post_period exceeds observations", %{spec: spec, obs: obs} do
      assert_raise ArgumentError, ~r/post_period end/, fn ->
        CausalImpact.estimate_structured(obs, {1, 10}, {11, 25}, spec, seed: 1)
      end
    end
  end

  # ── Seasonal model ─────────────────────────────────────────────────────

  describe "estimate_structured with seasonal model" do
    @tag timeout: 120_000
    test "detects positive effect in seasonal data" do
      # Deterministic seasonal signal with a strong post-period lift to keep
      # this check robust across Elixir/OTP/Nx versions.
      pattern = [3.0, 1.0, -1.0, -2.0, 0.0, 2.0, -3.0]
      base_level = 100.0

      pre =
        Enum.flat_map(1..20, fn _week ->
          Enum.map(pattern, fn s -> base_level + s end)
        end)

      # Post period: same pattern but with clear positive lift
      lift = 30.0

      post =
        Enum.flat_map(1..5, fn _week ->
          Enum.map(pattern, fn s -> base_level + s + lift end)
        end)

      obs = pre ++ post
      n_pre = length(pre)
      n_total = length(obs)

      spec =
        Components.seasonal_spec(7,
          process_var: 0.01,
          obs_var: 0.5,
          initial_cov: 10.0
        )

      result =
        CausalImpact.estimate_structured(obs, {1, n_pre}, {n_pre + 1, n_total}, spec,
          num_samples: 80,
          burn_in: 40,
          seed: 555
        )

      summary = CausalImpact.summary(result)
      # Should detect positive cumulative effect
      assert summary.cumulative_effect.mean > 0
    end
  end

  # ── Composed model: level + seasonal ───────────────────────────────────

  describe "estimate_structured with composed level + seasonal" do
    test "detects effect with level + seasonal model" do
      :rand.seed(:exsss, {400, 401, 402})
      # Quarterly pattern (S=4), 8 quarters pre, 2 quarters post
      pattern = [5.0, 2.0, -3.0, -4.0]
      base_level = 200.0

      pre =
        Enum.flat_map(1..8, fn _q ->
          Enum.map(pattern, fn s -> base_level + s + :rand.normal() * 2.0 end)
        end)

      lift = 20.0

      post =
        Enum.flat_map(1..2, fn _q ->
          Enum.map(pattern, fn s -> base_level + s + lift + :rand.normal() * 2.0 end)
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

      result =
        CausalImpact.estimate_structured(obs, {1, n_pre}, {n_pre + 1, n_total}, combined,
          num_samples: 30,
          burn_in: 15,
          seed: 777
        )

      summary = CausalImpact.summary(result)
      # Positive cumulative effect expected
      assert summary.cumulative_effect.mean > 0
      # Relative effect should be reasonably positive (~10% = 20/200)
      assert summary.relative_effect.mean > 0
    end

    test "no false positive when there is no intervention" do
      :rand.seed(:exsss, {500, 501, 502})
      pattern = [5.0, 2.0, -3.0, -4.0]
      base_level = 200.0

      # No lift in post period
      all_data =
        Enum.flat_map(1..10, fn _q ->
          Enum.map(pattern, fn s -> base_level + s + :rand.normal() * 2.0 end)
        end)

      n_pre = 32
      n_total = 40

      ll =
        Components.local_level_spec(
          process_var: 1.0,
          obs_var: 2.0,
          initial_state: 200.0,
          initial_cov: 50.0
        )

      ss = Components.seasonal_spec(4, process_var: 0.1)
      combined = Components.compose_specs(ll, ss)

      result =
        CausalImpact.estimate_structured(all_data, {1, n_pre}, {n_pre + 1, n_total}, combined,
          num_samples: 30,
          burn_in: 15,
          seed: 888
        )

      summary = CausalImpact.summary(result)
      # Cumulative effect should be near zero (within reasonable tolerance)
      # The CI should include zero for a well-calibrated model
      per_step_mean = summary.cumulative_effect.mean / 8

      assert abs(per_step_mean) < 30.0,
             "Per-step mean #{per_step_mean} seems too large for no intervention"
    end
  end

  # ── Time-varying H (regression) ────────────────────────────────────────

  describe "estimate_structured with time-varying H (regression)" do
    test "handles regression spec with time-varying H" do
      :rand.seed(:exsss, {600, 601, 602})
      n_pre = 30
      n_post = 10
      n_total = n_pre + n_post

      # Single regressor: X_t values
      x_vals = Enum.map(1..n_total, fn t -> :math.sin(t / 5.0) end)
      beta_true = 3.0

      # y_t = beta * X_t + noise (pre) and y_t = beta * X_t + lift + noise (post)
      pre =
        Enum.map(0..(n_pre - 1), fn t ->
          beta_true * Enum.at(x_vals, t) + :rand.normal() * 0.5
        end)

      lift = 5.0

      post =
        Enum.map(n_pre..(n_total - 1), fn t ->
          beta_true * Enum.at(x_vals, t) + lift + :rand.normal() * 0.5
        end)

      obs = pre ++ post

      # Build regression spec with time-varying H
      regressors = Nx.tensor(Enum.map(x_vals, fn v -> [v] end))
      spec = Components.regression_spec(regressors, var_beta: 0.01, obs_var: 1.0)

      # Verify H is time-varying
      assert is_list(spec.h)
      assert length(spec.h) == n_total

      result =
        CausalImpact.estimate_structured(obs, {1, n_pre}, {n_pre + 1, n_total}, spec,
          num_samples: 15,
          burn_in: 5,
          seed: 999
        )

      assert length(result.actual) == n_post
      assert length(result.counterfactual) == 15

      summary = CausalImpact.summary(result)
      # Should detect the positive lift
      assert summary.cumulative_effect.mean > 0
    end

    test "raises when time-varying H does not cover post period" do
      observations = Enum.map(1..12, &(&1 * 1.0))
      pre_regressors = Nx.tensor(Enum.map(1..8, fn i -> [i * 0.1] end))

      spec =
        Components.compose_specs(
          Components.local_linear_trend_spec(initial_level: 1.0),
          Components.regression_spec(pre_regressors)
        )

      assert_raise ArgumentError, ~r/structured model H covers/, fn ->
        CausalImpact.estimate_structured(observations, {1, 8}, {9, 12}, spec,
          num_samples: 6,
          burn_in: 3,
          seed: 42
        )
      end
    end
  end

  # ── Reproducibility ────────────────────────────────────────────────────

  describe "deterministic PRNG" do
    test "same seed produces identical results" do
      :rand.seed(:exsss, {700, 701, 702})
      pre = Enum.map(1..30, fn _ -> 50.0 + :rand.normal() * 2 end)
      post = Enum.map(1..10, fn _ -> 55.0 + :rand.normal() * 2 end)
      obs = pre ++ post

      spec = Components.local_level_spec(initial_state: 50.0, initial_cov: 10.0)
      common_opts = [num_samples: 5, burn_in: 3, seed: 42]

      r1 = CausalImpact.estimate_structured(obs, {1, 30}, {31, 40}, spec, common_opts)
      r2 = CausalImpact.estimate_structured(obs, {1, 30}, {31, 40}, spec, common_opts)

      assert r1.cumulative_effects == r2.cumulative_effects
    end
  end
end
