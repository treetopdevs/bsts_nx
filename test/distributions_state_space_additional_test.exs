defmodule BstsNx.DistributionsStateSpaceAdditionalTest do
  use ExUnit.Case, async: true

  alias BstsNx.Distributions
  alias BstsNx.StateSpace
  alias BstsNx.Utils

  describe "Distributions" do
    test "inv_gamma_sample raises on shape mismatch" do
      assert_raise ArgumentError, ~r/same shape/, fn ->
        Distributions.inv_gamma_sample(Nx.tensor([1.0, 2.0]), Nx.tensor([1.0]))
      end
    end

    test "inv_gamma_sample raises on invalid key shape" do
      bad_key = Nx.tensor([1, 2, 3], type: {:u, 32})

      assert_raise ArgumentError, ~r/shape \{2\}/, fn ->
        Distributions.inv_gamma_sample(2.0, 3.0, key: bad_key)
      end
    end

    test "inv_gamma_sample supports tensor alpha/beta inputs without a key" do
      draws =
        Distributions.inv_gamma_sample(
          Nx.tensor([2.0, 4.0, 6.0]),
          Nx.tensor([1.0, 2.0, 3.0])
        )

      assert Nx.shape(draws) == {3}
      assert Enum.all?(Nx.to_flat_list(draws), &(&1 > 0.0))
    end

    test "inv_gamma_sample supports list alpha/beta inputs with key" do
      key = Nx.Random.key(1234)

      draw1 = Distributions.inv_gamma_sample([2.0, 3.0], [1.0, 1.5], key: key)
      draw2 = Distributions.inv_gamma_sample([2.0, 3.0], [1.0, 1.5], key: key)

      assert Nx.shape(draw1) == {2}
      assert Nx.to_flat_list(draw1) == Nx.to_flat_list(draw2)
      assert Enum.all?(Nx.to_flat_list(draw1), &(&1 > 0.0))
    end

    test "inv_gamma_sample_with_key supports tensor alpha/beta and returns next key" do
      key = Nx.Random.key(99)
      {sample, next_key} = Distributions.inv_gamma_sample_with_key([2.0, 5.0], [1.0, 3.0], key)

      assert Nx.shape(sample) == {2}
      assert Nx.shape(next_key) == {2}
      assert Enum.all?(Nx.to_flat_list(sample), &(&1 > 0.0))
    end

    test "inv_gamma_sample_with_key rejects opts key conflict" do
      assert_raise ArgumentError, ~r/pass the PRNG key as the third argument/, fn ->
        Distributions.inv_gamma_sample_with_key(2.0, 3.0, Nx.Random.key(1), key: Nx.Random.key(2))
      end
    end

    test "inv_gamma_sample rejects invalid max_value option" do
      assert_raise ArgumentError, ~r/max_value must be :infinity or a positive number/, fn ->
        Distributions.inv_gamma_sample(2.0, 3.0, max_value: 0.0)
      end
    end

    test "inv_gamma_sample rejects non-positive alpha" do
      assert_raise ArgumentError, ~r/requires alpha > 0/, fn ->
        Distributions.inv_gamma_sample(0.0, 2.0, key: Nx.Random.key(7))
      end
    end

    test "inv_gamma_sample_with_key rejects invalid key shape" do
      bad_key = Nx.tensor([1, 2, 3], type: {:u, 32})

      assert_raise ArgumentError, ~r/shape \{2\}/, fn ->
        Distributions.inv_gamma_sample_with_key(2.0, 3.0, bad_key)
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

    test "inv_gamma_sample is uncapped by default for heavy-tail settings" do
      key = Nx.Random.key(123)

      draws =
        Enum.map(0..199, fn idx ->
          keys = Nx.Random.split(key, parts: idx + 2)

          subkey =
            Nx.slice_along_axis(keys, idx, 1, axis: 0)
            |> Nx.squeeze(axes: [0])

          Distributions.inv_gamma_sample(1.0, 1.0e12, key: subkey)
          |> Nx.to_number()
        end)

      assert Enum.any?(draws, fn v -> is_number(v) and v > 1.0e10 end)
    end

    test "inv_gamma_sample applies explicit max_value truncation when requested" do
      key = Nx.Random.key(123)

      draws =
        Enum.map(0..49, fn idx ->
          keys = Nx.Random.split(key, parts: idx + 2)

          subkey =
            Nx.slice_along_axis(keys, idx, 1, axis: 0)
            |> Nx.squeeze(axes: [0])

          Distributions.inv_gamma_sample(1.0, 1.0e12, key: subkey, max_value: 1.0e10)
          |> Nx.to_number()
        end)

      assert Enum.all?(draws, fn v -> is_number(v) and v <= 1.0e10 end)
      assert Enum.any?(draws, &(&1 == 1.0e10))
    end

    test "inv_gamma_sample_with_key aligns with split-key behavior and returns next key" do
      base_key = Nx.Random.key(44)
      split = Nx.Random.split(base_key, parts: 2)
      expected_next_key = Utils.split_key_at(split, 1)

      expected =
        Distributions.inv_gamma_sample(3.0, 2.0, key: base_key)
        |> Nx.to_number()

      {actual_t, next_key} = Distributions.inv_gamma_sample_with_key(3.0, 2.0, base_key)
      actual = Nx.to_number(actual_t)

      assert_in_delta actual, expected, 1.0e-12
      assert Nx.to_flat_list(next_key) == Nx.to_flat_list(expected_next_key)
    end

    test "inv_gamma_sample_with_key empirical moments are close to theory" do
      alpha = 4.0
      beta = 3.0
      n = 4_000
      key0 = Nx.Random.key(777)

      {draws, _final_key} =
        Enum.map_reduce(1..n, key0, fn _, key_acc ->
          {draw, key_next} = Distributions.inv_gamma_sample_with_key(alpha, beta, key_acc)
          {Nx.to_number(draw), key_next}
        end)

      mean = Enum.sum(draws) / n
      var = variance(draws, mean)

      # Inv-Gamma(alpha, beta) with alpha > 2:
      # mean = beta / (alpha - 1), var = beta^2 / ((alpha - 1)^2 * (alpha - 2))
      expected_mean = beta / (alpha - 1.0)
      expected_var = beta * beta / ((alpha - 1.0) * (alpha - 1.0) * (alpha - 2.0))

      assert_in_delta mean, expected_mean, 0.08
      assert_in_delta var, expected_var, 0.15
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

  defp variance(samples, mean) do
    samples
    |> Enum.reduce(0.0, fn x, acc ->
      d = x - mean
      acc + d * d
    end)
    |> Kernel./(max(length(samples), 1))
  end
end
