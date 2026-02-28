defmodule BstsNxUtilsSafeSolveTest do
  use ExUnit.Case
  alias BstsNx.Utils

  describe "safe_solve/2" do
    test "solves a well-conditioned 2x2 system" do
      a = Nx.tensor([[2.0, 1.0], [1.0, 3.0]])
      b = Nx.tensor([[5.0], [7.0]])
      x = Utils.safe_solve(a, b)
      # A * x should equal b
      result = Nx.multiply(a, Nx.transpose(x)) |> Nx.sum(axes: [1]) |> Nx.reshape({2, 1})
      assert_in_delta Nx.to_number(result[0][0]), 5.0, 1.0e-4
      assert_in_delta Nx.to_number(result[1][0]), 7.0, 1.0e-4
    end

    test "returns finite result for singular matrix" do
      a = Nx.tensor([[1.0, 2.0], [2.0, 4.0]])
      b = Nx.tensor([[3.0], [6.0]])
      x = Utils.safe_solve(a, b)
      assert not Utils.has_non_finite?(x)
    end

    test "returns finite result for near-singular matrix" do
      a = Nx.tensor([[1.0, 2.0], [2.0, 4.0 + 1.0e-12]])
      b = Nx.tensor([[3.0], [6.0]])
      x = Utils.safe_solve(a, b)
      assert not Utils.has_non_finite?(x)
    end

    test "solves 1x1 systems without LinAlg.solve fallback path" do
      a = Nx.tensor([[2.0]])
      b = Nx.tensor([[6.0], [10.0]])
      x = Utils.safe_solve(a, b)

      assert Nx.shape(x) == {2, 1}
      assert_in_delta Nx.to_number(x[0][0]), 3.0, 1.0e-6
      assert_in_delta Nx.to_number(x[1][0]), 5.0, 1.0e-6
    end

    test "handles near-zero scalar denominator with jitter and keeps result finite" do
      a = Nx.tensor(0.0)
      b = Nx.tensor([1.0, -2.0, 3.0])
      x = Utils.safe_solve(a, b)

      assert Nx.shape(x) == {3}
      assert not Utils.has_non_finite?(x)
    end

    test "falls back to finite zeros when matrix solve outputs non-finite values" do
      a = Nx.eye(2)

      b =
        Nx.stack([Nx.Constants.infinity(), Nx.tensor(1.0)])
        |> Nx.reshape({2, 1})

      x = Utils.safe_solve(a, b)

      assert Nx.shape(x) == {2, 1}
      assert Nx.to_flat_list(x) == [0.0, 0.0]
      assert not Utils.has_non_finite?(x)
    end

    test "falls back to scalar zero when scalar division remains non-finite" do
      x = Utils.safe_solve(Nx.tensor(1.0), Nx.Constants.infinity())
      assert Nx.shape(x) == {}
      assert Nx.to_number(x) == 0.0
    end
  end

  describe "has_non_finite?/1" do
    test "returns false for finite tensor" do
      assert not Utils.has_non_finite?(Nx.tensor([1.0, 2.0, 3.0]))
    end

    test "returns true for tensor with NaN" do
      nan_tensor = Nx.stack([Nx.tensor(1.0), Nx.Constants.nan(), Nx.tensor(3.0)])
      assert Utils.has_non_finite?(nan_tensor)
    end

    test "returns true for tensor with Inf" do
      inf_tensor = Nx.stack([Nx.tensor(1.0), Nx.Constants.infinity(), Nx.tensor(3.0)])
      assert Utils.has_non_finite?(inf_tensor)
    end
  end

  describe "missing_observation?/1" do
    test "nil is missing" do
      assert Utils.missing_observation?(nil)
    end

    test ":nan atom is missing" do
      assert Utils.missing_observation?(:nan)
    end

    test "NaN tensor is missing" do
      assert Utils.missing_observation?(Nx.Constants.nan())
    end

    test "NaN float is missing" do
      nan = Nx.to_number(Nx.Constants.nan())
      assert Utils.missing_observation?(nan)
    end

    test "normal number is not missing" do
      refute Utils.missing_observation?(1.0)
    end

    test "normal tensor is not missing" do
      refute Utils.missing_observation?(Nx.tensor(1.0))
    end
  end

  describe "safe_cholesky/1" do
    test "raises after NaN-triggered jitter retries are exhausted" do
      nan = Nx.to_number(Nx.Constants.nan())
      bad = Nx.tensor([[nan, 0.0], [0.0, 1.0]])

      assert_raise ArgumentError, ~r/failed even with jitter/, fn ->
        Utils.safe_cholesky(bad)
      end
    end
  end

  describe "percentile_interval/3 and z_score/1 edge paths" do
    test "percentile_interval handles empty and singleton inputs" do
      assert Utils.percentile_interval([], 0, 0.05) == {0.0, 0.0}
      assert Utils.percentile_interval([3.5], 1, 0.05) == {3.5, 3.5}
    end

    test "z_score exercises erfinv edge and symmetry branches" do
      assert Utils.z_score(1.0) == 0.0
      assert_in_delta Utils.z_score(1.2), -Utils.z_score(0.8), 1.0e-12

      assert_raise ArgumentError, ~r/undefined for x >= 1.0/, fn ->
        Utils.z_score(0.0)
      end
    end
  end
end
