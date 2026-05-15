defmodule BstsNx.ValidationTest do
  use ExUnit.Case, async: true

  alias BstsNx.Validation

  # ---------------------------------------------------------------
  # Helpers
  # ---------------------------------------------------------------

  # (helpers removed — not needed; synthetic data is built inline)

  describe "shared validation helpers" do
    test "validate_alpha!/2 accepts open-interval alpha values" do
      assert Validation.validate_alpha!(0.05) == :ok
    end

    test "validate_alpha!/2 preserves caller-specific error wording" do
      assert_raise ArgumentError, ~r/alpha must be in \(0, 1\), got: 1.0/, fn ->
        Validation.validate_alpha!(1.0)
      end

      assert_raise ArgumentError, ~r/alpha must be a number in \(0, 1\), got: 0/, fn ->
        Validation.validate_alpha!(0, "alpha must be a number in (0, 1)")
      end
    end

    test "relative effect helpers use the existing zero-baseline convention" do
      assert Validation.relative_effect(12.0, 24.0) == 0.5
      assert Validation.relative_effect(12.0, 0.0) == 0.0

      assert Validation.relative_effects([10.0, 20.0, 30.0], [100.0, 0.0, -60.0]) == [
               0.1,
               0.0,
               -0.5
             ]

      summary = Validation.relative_effect_summary(10.0, 2.0, 20.0, 1.96)
      assert summary == %{mean: 0.5, sd: 0.1, lower: 0.304, upper: 0.696}
    end

    test "fixed variance helpers build diagonal Q matrices with explicit source semantics" do
      q_specs = [
        %{dim_index: 0, initial: 0.2, prior_shape: 3.0, prior_scale: 8.0},
        %{dim_index: 2, initial: 0.4, prior_shape: 1.0, prior_scale: 2.0}
      ]

      initial_q = Validation.fixed_q_matrix(q_specs, 3)
      prior_q = Validation.fixed_q_matrix(q_specs, 3, source: :prior_mean)

      [initial_level, initial_missing, initial_slope] =
        Nx.to_flat_list(Nx.take_diagonal(initial_q))

      assert_in_delta initial_level, 0.2, 1.0e-6
      assert initial_missing == 0.0
      assert_in_delta initial_slope, 0.4, 1.0e-6

      assert Nx.to_flat_list(Nx.take_diagonal(prior_q)) == [4.0, 0.0, 4.0]
      assert Validation.fixed_variance_from_q_spec(hd(q_specs)) == 0.2
      assert Validation.fixed_variance_from_q_spec(hd(q_specs), :prior_mean) == 4.0
      assert Validation.fixed_variance_from_prior(1.0, 2.0) == 4.0
    end

    test "fixed Q matrix rejects duplicate or out-of-range q_specs" do
      assert_raise ArgumentError, ~r/q_specs dim_index values must be unique/, fn ->
        Validation.fixed_q_matrix(
          [
            %{dim_index: 0, initial: 0.2, prior_shape: 3.0, prior_scale: 8.0},
            %{dim_index: 0, initial: 0.4, prior_shape: 3.0, prior_scale: 8.0}
          ],
          2
        )
      end

      assert_raise ArgumentError, ~r/q_specs dim_index must be in \[0, 1\]/, fn ->
        Validation.fixed_q_matrix(
          [%{dim_index: 2, initial: 0.2, prior_shape: 3.0, prior_scale: 8.0}],
          2
        )
      end
    end

    test "validate_spot_windows!/3 preserves post-period and series error wording" do
      valid_spots = [%{id: "ok", window_start: 0, window_end: 5}]
      assert Validation.validate_spot_windows!(valid_spots, 5) == :ok

      assert_raise ArgumentError, ~r/is outside the post period \(0\.\.5\)/, fn ->
        Validation.validate_spot_windows!([%{id: "bad", window_start: 0, window_end: 6}], 5)
      end

      assert_raise ArgumentError, ~r/is out of bounds for series of length 5/, fn ->
        Validation.validate_spot_windows!(
          [%{id: "bad", window_start: -1, window_end: 3}],
          5,
          context: :series
        )
      end

      assert_raise ArgumentError, ~r/window_start.*>.*window_end/, fn ->
        Validation.validate_spot_windows!([%{id: "bad", window_start: 4, window_end: 3}], 5)
      end

      assert_raise ArgumentError,
                   ~r/empty window.*window_start must be less than window_end/,
                   fn ->
                     Validation.validate_spot_windows!(
                       [%{id: "bad", window_start: 3, window_end: 3}],
                       5
                     )
                   end
    end

    test "validate_study_periods!/5 centralizes causal period checks" do
      assert Validation.validate_study_periods!(10, 1, 5, 6, 10) == :ok

      assert_raise ArgumentError,
                   ~r/pre_period end \(11\) extends beyond observations \(10\)/,
                   fn ->
                     Validation.validate_study_periods!(10, 1, 11, 12, 12)
                   end

      assert_raise ArgumentError, ~r/post_period must satisfy pre_end < start <= end/, fn ->
        Validation.validate_study_periods!(10, 1, 5, 5, 6)
      end
    end

    test "attribution_options/1 preserves explicit n_samples over shapley_samples" do
      opts =
        Validation.attribution_options(
          alpha: 0.1,
          shapley_samples: 50,
          n_samples: 5,
          decay: 0.8,
          key: Nx.Random.key(4),
          value_fn_mode: :diminishing,
          ignored: true
        )

      assert opts[:alpha] == 0.1
      assert opts[:n_samples] == 5
      assert opts[:decay] == 0.8
      assert opts[:value_fn_mode] == :diminishing
      refute Keyword.has_key?(opts, :shapley_samples)
      refute Keyword.has_key?(opts, :ignored)
    end
  end

  # ---------------------------------------------------------------
  # 1. prediction_error
  # ---------------------------------------------------------------

  describe "prediction_error/3" do
    test "returns nil for empty indices" do
      a = Nx.tensor([1.0, 2.0, 3.0])
      b = Nx.tensor([1.0, 2.0, 3.0])
      assert Validation.prediction_error(a, b, []) == nil
    end

    test "perfect baseline yields ~0 RMSE and ~0 MAPE" do
      values = Enum.map(1..100, &(&1 * 1.0))
      t = Nx.tensor(values, type: {:f, 32})
      indices = Enum.to_list(0..99)

      result = Validation.prediction_error(t, t, indices)

      assert result.n == 100
      assert_in_delta result.rmse, 0.0, 1.0e-5
      assert_in_delta result.mape, 0.0, 1.0e-5
    end

    test "known constant offset produces correct RMSE and MAPE" do
      n = 50
      actual = Nx.tensor(List.duplicate(110.0, n), type: {:f, 32})
      baseline = Nx.tensor(List.duplicate(100.0, n), type: {:f, 32})
      indices = Enum.to_list(0..(n - 1))

      result = Validation.prediction_error(actual, baseline, indices)

      # Error = 10 everywhere, RMSE = 10
      assert_in_delta result.rmse, 10.0, 1.0e-3
      # MAPE = |10| / max(110, 1) = 10/110 ≈ 0.0909
      assert_in_delta result.mape, 10.0 / 110.0, 1.0e-3
    end

    test "MAPE floors denominator at 1.0 for zero actuals" do
      actual = Nx.tensor([0.0, 0.0, 0.0], type: {:f, 32})
      baseline = Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32})

      result = Validation.prediction_error(actual, baseline, [0, 1, 2])

      # Denominator clamped to 1.0, so MAPE = mean(|1|, |2|, |3|) = 2.0
      assert_in_delta result.mape, 2.0, 1.0e-3
    end
  end

  # ---------------------------------------------------------------
  # 2. coverage
  # ---------------------------------------------------------------

  describe "coverage/5" do
    test "returns nil for empty indices" do
      t = Nx.tensor([1.0])
      assert Validation.coverage(t, t, t, 1.0, []) == nil
    end

    test "perfect coverage with very wide CI" do
      n = 100
      actual = Nx.tensor(Enum.map(1..n, &(&1 * 1.0)), type: {:f, 32})
      baseline = actual
      # Large variance => wide CI => everything covered
      state_vars = Nx.broadcast(Nx.tensor(1.0e6, type: {:f, 32}), {n})
      obs_var = 0.0
      indices = Enum.to_list(0..(n - 1))

      result = Validation.coverage(actual, baseline, state_vars, obs_var, indices)

      assert result.pct == 1.0
      assert result.n_covered == n
      assert result.n_total == n
    end

    test "zero coverage with impossibly narrow CI" do
      n = 100
      # Actuals far from baseline
      actual = Nx.tensor(List.duplicate(100.0, n), type: {:f, 32})
      baseline = Nx.tensor(List.duplicate(0.0, n), type: {:f, 32})
      # Tiny variance => narrow CI => nothing covered
      state_vars = Nx.broadcast(Nx.tensor(1.0e-10, type: {:f, 32}), {n})
      obs_var = 1.0e-10
      indices = Enum.to_list(0..(n - 1))

      result = Validation.coverage(actual, baseline, state_vars, obs_var, indices)

      assert result.pct == 0.0
      assert result.n_covered == 0
    end

    test "calibrated coverage near 95% with Gaussian noise" do
      # Generate actual = baseline + N(0,1) using sin-based approximation
      # With variance = 1.0, 95% CI is ±1.96, so ~95% should be covered
      n = 1000
      baseline_val = 50.0
      baseline = Nx.broadcast(Nx.tensor(baseline_val, type: {:f, 32}), {n})

      # Deterministic "noise" that's small relative to CI width
      noise =
        Enum.map(0..(n - 1), fn i ->
          baseline_val + 0.5 * :math.sin(i * 0.37) + 0.3 * :math.cos(i * 0.71)
        end)

      actual = Nx.tensor(noise, type: {:f, 32})
      # State var + obs var should produce SD ~= 1.0 so CI width = ±1.96
      state_vars = Nx.broadcast(Nx.tensor(0.5, type: {:f, 32}), {n})
      obs_var = 0.5
      indices = Enum.to_list(0..(n - 1))

      result = Validation.coverage(actual, baseline, state_vars, obs_var, indices)

      # With our deterministic noise (max amplitude ~0.8) and CI width ±1.96,
      # we expect very high coverage
      assert result.pct > 0.85
    end
  end

  # ---------------------------------------------------------------
  # 3. durbin_watson
  # ---------------------------------------------------------------

  describe "durbin_watson/2" do
    test "returns nil for fewer than 3 indices" do
      r = Nx.tensor([1.0, 2.0], type: {:f, 32})
      assert Validation.durbin_watson(r, [0, 1]) == nil
    end

    test "returns nil for empty indices" do
      r = Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32})
      assert Validation.durbin_watson(r, []) == nil
    end

    test "constant residuals yield DW = 0" do
      n = 100
      residuals = Nx.broadcast(Nx.tensor(5.0, type: {:f, 32}), {n})
      indices = Enum.to_list(0..(n - 1))

      result = Validation.durbin_watson(residuals, indices)

      # All diffs are 0, numerator = 0
      assert_in_delta result.statistic, 0.0, 1.0e-5
      assert result.n == n
    end

    test "alternating residuals yield DW near 4.0" do
      n = 100
      # +1, -1, +1, -1, ...
      values = Enum.map(0..(n - 1), fn i -> if rem(i, 2) == 0, do: 1.0, else: -1.0 end)
      residuals = Nx.tensor(values, type: {:f, 32})
      indices = Enum.to_list(0..(n - 1))

      result = Validation.durbin_watson(residuals, indices)

      # Each diff = ±2, so numerator = 99 * 4 = 396
      # Each r² = 1, so denominator = 100
      # DW = 396/100 = 3.96
      assert_in_delta result.statistic, 4.0, 0.1
    end

    test "uncorrelated-like residuals yield DW near 2.0" do
      # Use many summed sinusoids at incommensurate frequencies to approximate
      # uncorrelated residuals. DW won't be exactly 2.0 with deterministic
      # data, but should be in the 1.0–3.0 range (well away from 0 or 4).
      n = 500

      values =
        Enum.map(0..(n - 1), fn i ->
          :math.sin(i * 2.31) + :math.cos(i * 3.77) + :math.sin(i * 5.13) +
            :math.cos(i * 7.89) + :math.sin(i * 11.03)
        end)

      residuals = Nx.tensor(values, type: {:f, 32})
      indices = Enum.to_list(0..(n - 1))

      result = Validation.durbin_watson(residuals, indices)

      # Should be roughly in the neighbourhood of 2.0
      assert result.statistic > 1.0
      assert result.statistic < 3.0
    end
  end

  # ---------------------------------------------------------------
  # 4. placebo_test
  # ---------------------------------------------------------------

  describe "placebo_test/3" do
    test "returns nil for empty on_air_indices" do
      result = Validation.placebo_test([1.0, 2.0, 3.0], [], fn _, _ -> %{} end)
      assert result == nil
    end

    test "returns nil for empty sessions" do
      result = Validation.placebo_test([], [0, 1], fn _, _ -> %{} end)
      assert result == nil
    end

    test "returns nil when pre-period is too short" do
      # 10 sessions, 6 on-air => 4 pre-period < 2*6 = 12
      sessions = List.duplicate(10.0, 10)
      on_air = Enum.to_list(0..5)

      result = Validation.placebo_test(sessions, on_air, fn _, _ -> %{} end)

      assert result == nil
    end

    test "constant series should not be significant" do
      n = 200
      sessions = List.duplicate(50.0, n)
      on_air = Enum.to_list(180..199)

      estimate_fn = fn _sessions, _placebo_indices ->
        # A constant series has zero lift
        %{
          lift_sessions: 0.0,
          lift_pct: 0.0,
          lift_ci95: %{lower: -1.0, upper: 1.0, sd: 0.5}
        }
      end

      result = Validation.placebo_test(sessions, on_air, estimate_fn)

      refute result == nil
      assert result.is_significant == false
      assert_in_delta result.lift, 0.0, 1.0e-5
    end

    test "returns expected fields" do
      n = 200
      sessions = List.duplicate(50.0, n)
      on_air = Enum.to_list(180..199)

      estimate_fn = fn _sessions, _indices ->
        %{
          lift_sessions: 10.0,
          lift_pct: 0.02,
          lift_ci95: %{lower: 5.0, upper: 15.0, sd: 2.5}
        }
      end

      result = Validation.placebo_test(sessions, on_air, estimate_fn)

      assert Map.has_key?(result, :lift)
      assert Map.has_key?(result, :lift_pct)
      assert Map.has_key?(result, :ci95)
      assert Map.has_key?(result, :is_significant)
      # CI is [5, 15] which doesn't include 0 => significant
      assert result.is_significant == true
    end

    test "calls estimate_fn with correct placebo indices from middle of pre-period" do
      n = 100
      sessions = Enum.map(1..n, &(&1 * 1.0))
      # On-air at the end: indices 90-99
      on_air = Enum.to_list(90..99)

      called_with = :ets.new(:called_with, [:set, :public])

      estimate_fn = fn _sessions, placebo_indices ->
        :ets.insert(called_with, {:indices, placebo_indices})

        %{
          lift_sessions: 0.0,
          lift_pct: 0.0,
          lift_ci95: %{lower: -1.0, upper: 1.0, sd: 0.5}
        }
      end

      Validation.placebo_test(sessions, on_air, estimate_fn)

      [{:indices, indices}] = :ets.lookup(called_with, :indices)
      :ets.delete(called_with)

      # Placebo should be 10 indices from the middle of 0..89
      assert length(indices) == 10
      # All placebo indices should be in the pre-period (0..89)
      assert Enum.all?(indices, fn i -> i >= 0 and i <= 89 end)
    end
  end

  # ---------------------------------------------------------------
  # 5. effect_stability
  # ---------------------------------------------------------------

  describe "effect_stability/4" do
    test "stable estimator has ~0 max_pct_change" do
      # Estimator always returns the same lift regardless of window
      main_lift = 100.0
      estimate_fn = fn _window -> 100.0 end

      result = Validation.effect_stability(main_lift, estimate_fn, 30)

      assert_in_delta result.max_pct_change, 0.0, 1.0e-10
      assert result.lift_base == 100.0
      assert result.lift_low == 100.0
      assert result.lift_high == 100.0
      assert result.window_low == 25
      assert result.window_high == 35
    end

    test "sensitive estimator has large max_pct_change" do
      main_lift = 100.0
      # Lift scales linearly with window
      estimate_fn = fn window -> window * 5.0 end

      result = Validation.effect_stability(main_lift, estimate_fn, 30)

      # low_window = 25, lift = 125, |125 - 100|/100 = 0.25
      # high_window = 35, lift = 175, |175 - 100|/100 = 0.75
      assert_in_delta result.max_pct_change, 0.75, 1.0e-5
      assert_in_delta result.lift_low, 125.0, 1.0e-5
      assert_in_delta result.lift_high, 175.0, 1.0e-5
    end

    test "near-zero main lift returns 0.0 max_pct_change" do
      main_lift = 0.001
      estimate_fn = fn _window -> 50.0 end

      result = Validation.effect_stability(main_lift, estimate_fn, 30)

      assert result.max_pct_change == 0.0
    end

    test "custom delta" do
      main_lift = 100.0
      estimate_fn = fn _window -> 100.0 end

      result = Validation.effect_stability(main_lift, estimate_fn, 30, 10)

      assert result.window_low == 20
      assert result.window_high == 40
    end

    test "window floor is 5 minutes" do
      main_lift = 100.0
      estimate_fn = fn _window -> 100.0 end

      result = Validation.effect_stability(main_lift, estimate_fn, 8, 5)

      # 8 - 5 = 3, but floored to 5
      assert result.window_low == 5
      assert result.window_high == 13
    end
  end

  # ---------------------------------------------------------------
  # 6. assess
  # ---------------------------------------------------------------

  describe "assess/1" do
    test "all-pass case" do
      results = %{
        prediction_error: %{rmse: 1.0, mape: 0.05, n: 100},
        coverage: %{pct: 0.94, n_covered: 94, n_total: 100},
        durbin_watson: %{statistic: 2.0, n: 100},
        placebo: %{
          lift: 0.5,
          lift_pct: 0.01,
          ci95: %{lower: -1.0, upper: 2.0, sd: 0.5},
          is_significant: false
        },
        effect_stability: %{
          max_pct_change: 0.05,
          lift_base: 100.0,
          lift_low: 95.0,
          lift_high: 105.0,
          window_low: 25,
          window_high: 35
        }
      }

      verdicts = Validation.assess(results)

      assert verdicts.prediction_error == :pass
      assert verdicts.coverage == :pass
      assert verdicts.durbin_watson == :pass
      assert verdicts.placebo == :pass
      assert verdicts.effect_stability == :pass
    end

    test "all nil returns all :skip" do
      verdicts = Validation.assess(%{})

      assert verdicts.prediction_error == :skip
      assert verdicts.coverage == :skip
      assert verdicts.durbin_watson == :skip
      assert verdicts.placebo == :skip
      assert verdicts.effect_stability == :skip
    end

    test "MAPE warn zone (10-20%)" do
      results = %{prediction_error: %{rmse: 5.0, mape: 0.15, n: 100}}
      verdicts = Validation.assess(results)
      assert verdicts.prediction_error == :warn
    end

    test "MAPE fail zone (>20%)" do
      results = %{prediction_error: %{rmse: 10.0, mape: 0.25, n: 100}}
      verdicts = Validation.assess(results)
      assert verdicts.prediction_error == :fail
    end

    test "coverage 80-85% warns" do
      results = %{coverage: %{pct: 0.80, n_covered: 80, n_total: 100}}
      verdicts = Validation.assess(results)
      assert verdicts.coverage == :warn
    end

    test "coverage above 99% warns" do
      results = %{coverage: %{pct: 1.0, n_covered: 100, n_total: 100}}
      verdicts = Validation.assess(results)
      assert verdicts.coverage == :warn
    end

    test "coverage below 80% fails" do
      results = %{coverage: %{pct: 0.70, n_covered: 70, n_total: 100}}
      verdicts = Validation.assess(results)
      assert verdicts.coverage == :fail
    end

    test "DW 1.0-1.5 warns" do
      results = %{durbin_watson: %{statistic: 1.0, n: 100}}
      verdicts = Validation.assess(results)
      assert verdicts.durbin_watson == :warn
    end

    test "DW 2.5-3.0 warns" do
      results = %{durbin_watson: %{statistic: 3.0, n: 100}}
      verdicts = Validation.assess(results)
      assert verdicts.durbin_watson == :warn
    end

    test "DW below 1.0 fails" do
      results = %{durbin_watson: %{statistic: 0.5, n: 100}}
      verdicts = Validation.assess(results)
      assert verdicts.durbin_watson == :fail
    end

    test "DW above 3.0 fails" do
      results = %{durbin_watson: %{statistic: 3.5, n: 100}}
      verdicts = Validation.assess(results)
      assert verdicts.durbin_watson == :fail
    end

    test "significant placebo fails" do
      results = %{
        placebo: %{
          lift: 5.0,
          lift_pct: 0.10,
          ci95: %{lower: 2.0, upper: 8.0, sd: 1.5},
          is_significant: true
        }
      }

      verdicts = Validation.assess(results)
      assert verdicts.placebo == :fail
    end

    test "non-significant placebo with large lift_pct fails" do
      results = %{
        placebo: %{
          lift: 5.0,
          lift_pct: 0.10,
          ci95: %{lower: -1.0, upper: 11.0, sd: 3.0},
          is_significant: false
        }
      }

      verdicts = Validation.assess(results)
      # |0.10| >= 0.05 => fail
      assert verdicts.placebo == :fail
    end

    test "effect stability warn zone (10-25%)" do
      results = %{
        effect_stability: %{
          max_pct_change: 0.20,
          lift_base: 100.0,
          lift_low: 80.0,
          lift_high: 100.0,
          window_low: 25,
          window_high: 35
        }
      }

      verdicts = Validation.assess(results)
      assert verdicts.effect_stability == :warn
    end

    test "effect stability fail zone (>25%)" do
      results = %{
        effect_stability: %{
          max_pct_change: 0.50,
          lift_base: 100.0,
          lift_low: 50.0,
          lift_high: 100.0,
          window_low: 25,
          window_high: 35
        }
      }

      verdicts = Validation.assess(results)
      assert verdicts.effect_stability == :fail
    end

    test "mixed failures" do
      results = %{
        prediction_error: %{rmse: 1.0, mape: 0.05, n: 100},
        coverage: %{pct: 0.70, n_covered: 70, n_total: 100},
        durbin_watson: %{statistic: 2.0, n: 100},
        placebo: %{
          lift: 10.0,
          lift_pct: 0.20,
          ci95: %{lower: 5.0, upper: 15.0, sd: 2.5},
          is_significant: true
        },
        effect_stability: %{
          max_pct_change: 0.05,
          lift_base: 100.0,
          lift_low: 95.0,
          lift_high: 105.0,
          window_low: 25,
          window_high: 35
        }
      }

      verdicts = Validation.assess(results)

      assert verdicts.prediction_error == :pass
      assert verdicts.coverage == :fail
      assert verdicts.durbin_watson == :pass
      assert verdicts.placebo == :fail
      assert verdicts.effect_stability == :pass
    end
  end
end
