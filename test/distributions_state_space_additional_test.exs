defmodule BstsNx.DistributionsStateSpaceAdditionalTest do
  use ExUnit.Case, async: true

  alias BstsNx.Distributions
  alias BstsNx.StateSpace

  describe "Distributions" do
    test "inv_gamma_sample raises on shape mismatch" do
      assert_raise ArgumentError, ~r/same shape/, fn ->
        Distributions.inv_gamma_sample(Nx.tensor([1.0, 2.0]), Nx.tensor([1.0]))
      end
    end

    test "inv_gamma_sample raises on invalid key shape" do
      bad_key = Nx.tensor([1, 2, 3], type: {:u, 32})

      assert_raise ArgumentError, ~r/Expected Nx.Random key with shape \{2\}/, fn ->
        Distributions.inv_gamma_sample(2.0, 3.0, key: bad_key)
      end
    end

    test "normal_sample returns sample and next key" do
      {sample, next_key} = Distributions.normal_sample(Nx.Random.key(7), mean: 2.0, stddev: 0.5)

      assert is_number(Nx.to_number(sample))
      assert Nx.shape(next_key) == {2}
    end

    test "mv_normal_sample returns vector sample with matching dimension" do
      key = Nx.Random.key(9)
      mean = Nx.tensor([0.0, 1.0, -1.0])
      cov = Nx.eye(3)

      {sample, next_key} = Distributions.mv_normal_sample(key, mean, cov)

      assert Nx.shape(sample) == {3}
      assert Nx.shape(next_key) == {2}
    end
  end

  describe "StateSpace" do
    test "block_diag raises on empty list" do
      assert_raise ArgumentError, ~r/empty list/, fn ->
        StateSpace.block_diag([])
      end
    end

    test "block_diag raises on non-square matrix" do
      assert_raise ArgumentError, ~r/matrix must be square/, fn ->
        StateSpace.block_diag([Nx.tensor([[1.0, 2.0]])])
      end
    end

    test "compose raises when observation row dimensions differ" do
      c1 = %{f: Nx.tensor([[1.0]]), q: Nx.tensor([[1.0]]), h: Nx.tensor([[1.0]])}
      c2 = %{f: Nx.tensor([[1.0]]), q: Nx.tensor([[1.0]]), h: Nx.tensor([[1.0], [2.0]])}

      assert_raise ArgumentError, ~r/same number of rows/, fn ->
        StateSpace.compose(c1, c2)
      end
    end
  end
end
