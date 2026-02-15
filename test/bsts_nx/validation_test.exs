defmodule BstsNx.Validation.ZScoreAndCoverageTest do
  use ExUnit.Case, async: true

  alias BstsNx.Validation

  # -------------------------------------------------------------------
  # z_score/1 tests
  # -------------------------------------------------------------------

  describe "z_score/1" do
    test "returns ~1.96 for alpha=0.05 (95% CI)" do
      z = Validation.z_score(0.05)
      assert_in_delta z, 1.96, 0.01
    end

    test "returns ~1.28 for alpha=0.20 (80% CI)" do
      z = Validation.z_score(0.20)
      assert_in_delta z, 1.2816, 0.01
    end

    test "returns ~1.645 for alpha=0.10 (90% CI)" do
      z = Validation.z_score(0.10)
      assert_in_delta z, 1.6449, 0.01
    end

    test "returns ~2.576 for alpha=0.01 (99% CI)" do
      z = Validation.z_score(0.01)
      assert_in_delta z, 2.5758, 0.02
    end

    test "returns ~0.6745 for alpha=0.50 (50% CI)" do
      z = Validation.z_score(0.50)
      assert_in_delta z, 0.6745, 0.01
    end

    test "z-scores are monotonically decreasing as alpha increases" do
      alphas = [0.01, 0.05, 0.10, 0.20, 0.50]
      z_scores = Enum.map(alphas, &Validation.z_score/1)

      z_scores
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.each(fn [z1, z2] ->
        assert z1 > z2, "Expected z(#{z1}) > z(#{z2})"
      end)
    end
  end

  # -------------------------------------------------------------------
  # coverage/7 with confidence_level tests
  # -------------------------------------------------------------------

  describe "coverage/7 with confidence_level" do
    setup do
      # Create a simple series where actual = baseline + small noise
      # With known state variances, we can predict coverage behavior
      n = 100
      baseline = Nx.broadcast(100.0, {n})
      state_variances = Nx.broadcast(4.0, {n})
      obs_variance = 1.0
      indices = Enum.to_list(0..(n - 1))

      # Total SD = sqrt(4 + 1) = sqrt(5) ~ 2.236
      # At 95%: z=1.96, half-width = 1.96 * 2.236 = 4.38
      # At 80%: z=1.28, half-width = 1.28 * 2.236 = 2.86

      %{
        n: n,
        baseline: baseline,
        state_variances: state_variances,
        obs_variance: obs_variance,
        indices: indices
      }
    end

    test "all covered at 95% when actual equals baseline", ctx do
      actual = Nx.broadcast(100.0, {ctx.n})

      result =
        Validation.coverage(
          actual,
          ctx.baseline,
          ctx.state_variances,
          ctx.obs_variance,
          ctx.indices,
          1.0,
          confidence_level: 0.95
        )

      assert result.pct == 1.0
      assert result.n_covered == ctx.n
    end

    test "wider CI (99%) covers more than narrower CI (80%)", ctx do
      # Actual is offset by 3.0 from baseline — borderline for 80% CI
      actual = Nx.broadcast(103.0, {ctx.n})

      result_80 =
        Validation.coverage(
          actual,
          ctx.baseline,
          ctx.state_variances,
          ctx.obs_variance,
          ctx.indices,
          1.0,
          confidence_level: 0.80
        )

      result_99 =
        Validation.coverage(
          actual,
          ctx.baseline,
          ctx.state_variances,
          ctx.obs_variance,
          ctx.indices,
          1.0,
          confidence_level: 0.99
        )

      assert result_99.pct >= result_80.pct
    end

    test "50% CI covers none when actual is far from baseline", ctx do
      # Actual is 10.0 away — way outside even 99% CI (half-width ~5.75)
      actual = Nx.broadcast(115.0, {ctx.n})

      result =
        Validation.coverage(
          actual,
          ctx.baseline,
          ctx.state_variances,
          ctx.obs_variance,
          ctx.indices,
          1.0,
          confidence_level: 0.50
        )

      assert result.pct == 0.0
    end

    test "defaults to 95% when no opts given", ctx do
      actual = Nx.broadcast(100.0, {ctx.n})

      result_default =
        Validation.coverage(
          actual,
          ctx.baseline,
          ctx.state_variances,
          ctx.obs_variance,
          ctx.indices
        )

      result_explicit =
        Validation.coverage(
          actual,
          ctx.baseline,
          ctx.state_variances,
          ctx.obs_variance,
          ctx.indices,
          1.0,
          confidence_level: 0.95
        )

      assert result_default.pct == result_explicit.pct
    end

    test "returns nil for empty indices" do
      actual = Nx.broadcast(100.0, {10})
      baseline = Nx.broadcast(100.0, {10})
      state_variances = Nx.broadcast(1.0, {10})

      assert Validation.coverage(actual, baseline, state_variances, 1.0, []) == nil
    end
  end
end
