defmodule BstsNx.DiagnosticsSingleChainTest do
  use ExUnit.Case, async: true

  alias BstsNx.Diagnostics

  describe "ess_single/1" do
    test "returns :nan for very short chains" do
      assert Diagnostics.ess_single([1.0]) == :nan
      assert Diagnostics.ess_single([1.0, 2.0]) == :nan
      assert Diagnostics.ess_single([1.0, 2.0, 3.0]) == :nan
    end

    test "returns high ESS for iid samples" do
      # iid samples should have ESS close to n
      :rand.seed(:exsss, {10, 11, 12})
      iid = Enum.map(1..200, fn _ -> :rand.normal() end)

      ess = Diagnostics.ess_single(iid)
      # ESS should be at least 50% of n for truly independent samples
      assert ess > 100.0, "ESS #{ess} should be > 100 for 200 iid samples"
    end

    test "returns lower ESS for correlated samples" do
      # Create highly correlated samples (random walk)
      :rand.seed(:exsss, {20, 21, 22})

      {_, corr} =
        Enum.reduce(1..200, {0.0, []}, fn _, {prev, acc} ->
          next = prev + :rand.normal() * 0.1
          {next, [next | acc]}
        end)

      corr = Enum.reverse(corr)

      ess = Diagnostics.ess_single(corr)
      # ESS should be much less than n for correlated samples
      assert ess < 200.0, "ESS #{ess} should be < 200 for correlated samples"
      assert ess >= 1.0, "ESS should be at least 1.0"
    end

    test "returns :nan for constant chain" do
      constant = List.duplicate(5.0, 50)
      assert Diagnostics.ess_single(constant) == :nan
    end
  end

  describe "split_r_hat/1" do
    test "returns :nan for very short chains" do
      assert Diagnostics.split_r_hat([1.0]) == :nan
      assert Diagnostics.split_r_hat([1.0, 2.0]) == :nan
      assert Diagnostics.split_r_hat([1.0, 2.0, 3.0]) == :nan
    end

    test "returns near 1.0 for stationary chain" do
      # Generate a stationary chain (iid draws)
      :rand.seed(:exsss, {30, 31, 32})
      chain = Enum.map(1..100, fn _ -> :rand.normal() end)

      r = Diagnostics.split_r_hat(chain)
      assert is_float(r)
      # For a stationary chain, split R-hat should be close to 1.0
      assert_in_delta(r, 1.0, 0.3, "split R-hat #{r} should be near 1.0")
    end

    test "detects non-stationarity" do
      # First half from N(0,1), second half from N(5,1) -> clearly non-stationary
      :rand.seed(:exsss, {40, 41, 42})
      first_half = Enum.map(1..50, fn _ -> :rand.normal() end)
      second_half = Enum.map(1..50, fn _ -> :rand.normal() + 5.0 end)
      chain = first_half ++ second_half

      r = Diagnostics.split_r_hat(chain)
      assert is_float(r)
      # Should detect the mean shift
      assert r > 1.1, "split R-hat #{r} should be > 1.1 for non-stationary chain"
    end

    test "is consistent with r_hat on manually split chain" do
      :rand.seed(:exsss, {50, 51, 52})
      chain = Enum.map(1..100, fn _ -> :rand.normal() end)

      split_result = Diagnostics.split_r_hat(chain)

      # Manually split and compute r_hat
      half = div(length(chain), 2)
      first = Enum.take(chain, half)
      second = Enum.drop(chain, length(chain) - half)
      manual_result = Diagnostics.r_hat([first, second])

      assert_in_delta(split_result, manual_result, 1.0e-10)
    end
  end

  describe "geweke_z/2 and geweke_pass?/2" do
    test "returns :nan for short chains" do
      assert Diagnostics.geweke_z([1.0, 2.0, 3.0]) == :nan
      assert Diagnostics.geweke_pass?([1.0, 2.0, 3.0]) == :nan
    end

    test "returns near-zero z for stationary chain" do
      :rand.seed(:exsss, {60, 61, 62})
      chain = Enum.map(1..400, fn _ -> :rand.normal() end)

      z = Diagnostics.geweke_z(chain)
      assert is_float(z)
      assert abs(z) < 2.0, "Geweke z #{z} should be in a non-significant range"
      assert Diagnostics.geweke_pass?(chain)
    end

    test "detects mean shift non-stationarity" do
      :rand.seed(:exsss, {70, 71, 72})
      first = Enum.map(1..200, fn _ -> :rand.normal() end)
      last = Enum.map(1..200, fn _ -> :rand.normal() + 2.5 end)
      chain = first ++ last

      z = Diagnostics.geweke_z(chain)
      assert is_float(z)
      assert abs(z) > 2.0, "Geweke z #{z} should flag non-stationarity"
      refute Diagnostics.geweke_pass?(chain)
    end
  end

  describe "hpd_interval/2" do
    test "returns :nan for empty samples" do
      assert Diagnostics.hpd_interval([]) == :nan
    end

    test "returns narrow interval around center for Gaussian samples" do
      :rand.seed(:exsss, {80, 81, 82})
      samples = Enum.map(1..2_000, fn _ -> :rand.normal() end)
      {lower, upper} = Diagnostics.hpd_interval(samples, 0.95)
      width = upper - lower

      assert lower < 0.0 and upper > 0.0
      assert width > 3.0 and width < 4.5
    end
  end
end
