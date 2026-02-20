defmodule BstsNxDistributionsInvGammaTest do
  use ExUnit.Case
  alias BstsNx.Distributions

  describe "inv_gamma_sample max_value option" do
    test "max_value caps samples" do
      samples =
        Enum.map(1..100, fn i ->
          Distributions.inv_gamma_sample(1.0, 1.0, max_value: 5.0, key: Nx.Random.key(i))
          |> Nx.to_number()
        end)

      assert Enum.all?(samples, &(&1 > 0))
      assert Enum.all?(samples, &(&1 <= 5.0))
    end

    test "default (no max_value) produces positive finite samples" do
      samples =
        Enum.map(1..100, fn i ->
          Distributions.inv_gamma_sample(2.0, 1.0, key: Nx.Random.key(i))
          |> Nx.to_number()
        end)

      assert Enum.all?(samples, &(&1 > 0))
      assert Enum.all?(samples, &is_float/1)
    end

    test "default allows samples above 1e10 for large beta" do
      # InvGamma(1.0, 1e12) has median ~1.44e12; without capping, samples should go above 1e10
      samples =
        Enum.map(1..200, fn i ->
          Distributions.inv_gamma_sample(1.0, 1.0e12, key: Nx.Random.key(i))
          |> Nx.to_number()
        end)

      assert Enum.any?(samples, &(&1 > 1.0e10)),
             "Expected at least one sample > 1e10 for InvGamma(1, 1e12)"
    end
  end
end
