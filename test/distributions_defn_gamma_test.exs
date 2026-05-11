defmodule BstsNx.DistributionsDefnGammaTest do
  use ExUnit.Case, async: true

  alias BstsNx.Distributions

  test "gamma_sample_defn returns f64 positive samples with matching shape" do
    alpha = Nx.tensor([0.25, 0.5, 2.0], type: {:f, 64})
    scale = Nx.tensor([0.75, 1.5, 2.0], type: {:f, 64})
    key = Nx.Random.key(456)

    {draws, next_key} = Distributions.gamma_sample_defn(alpha, scale, key)
    vals = Nx.to_flat_list(draws)

    assert Nx.shape(draws) == {3}
    assert Nx.type(draws) == {:f, 64}
    assert Nx.shape(next_key) == {2}
    assert Enum.all?(vals, &finite_positive?/1)
  end

  test "inv_gamma_sample_defn returns positive finite samples with matching shape" do
    alpha = Nx.tensor([2.5, 4.0, 1.2], type: {:f, 64})
    beta = Nx.tensor([1.5, 3.0, 0.8], type: {:f, 64})
    key = Nx.Random.key(123)

    {draws, next_key} = Distributions.inv_gamma_sample_defn(alpha, beta, key)
    vals = Nx.to_flat_list(draws)

    assert Nx.shape(draws) == {3}
    assert Nx.shape(next_key) == {2}

    assert Enum.all?(vals, fn v ->
             is_number(v) and v > 0.0 and v == v and v != :infinity
           end)
  end

  test "inv_gamma_sample_defn applies max_value truncation" do
    alpha = Nx.tensor([1.0, 1.0, 1.0], type: {:f, 64})
    beta = Nx.tensor([1.0e12, 1.0e12, 1.0e12], type: {:f, 64})
    key = Nx.Random.key(321)

    {draws, _next_key} = Distributions.inv_gamma_sample_defn(alpha, beta, key, max_value: 1.0e10)
    vals = Nx.to_flat_list(draws)

    assert Enum.all?(vals, &(&1 > 0.0 and &1 <= 1.0e10))
  end

  test "inv_gamma_sample_defn empirical mean tracks inverse-gamma mean" do
    alpha = 4.0
    beta = 3.0
    n = 2_000
    key0 = Nx.Random.key(777)

    {draws, _final_key} =
      Enum.map_reduce(1..n, key0, fn _, key_acc ->
        {draw, key_next} = Distributions.inv_gamma_sample_defn(alpha, beta, key_acc)
        {Nx.to_number(draw), key_next}
      end)

    mean = Enum.sum(draws) / n
    expected_mean = beta / (alpha - 1.0)
    assert_in_delta mean, expected_mean, 0.12
  end

  test "defn samplers reject non-positive parameters before entering the kernel" do
    key = Nx.Random.key(987)

    assert_raise ArgumentError, ~r/alpha must be strictly positive/, fn ->
      Distributions.gamma_sample_defn(Nx.tensor([1.0, 0.0]), Nx.tensor([1.0, 1.0]), key)
    end

    assert_raise ArgumentError, ~r/scale must be strictly positive/, fn ->
      Distributions.gamma_sample_defn(Nx.tensor([1.0, 2.0]), Nx.tensor([1.0, -1.0]), key)
    end

    assert_raise ArgumentError, ~r/beta must be strictly positive/, fn ->
      Distributions.inv_gamma_sample_defn(Nx.tensor([1.0, 2.0]), Nx.tensor([1.0, 0.0]), key)
    end
  end

  defp finite_positive?(value) do
    is_number(value) and value > 0.0 and value == value and value != :infinity
  end
end
