defmodule BstsNx.ModelSpecTest do
  use ExUnit.Case, async: true

  alias BstsNx.ModelSpec

  test "validate!/1 rejects duplicate q_specs dim_index values" do
    spec = %ModelSpec{
      f: Nx.eye(2),
      h: Nx.tensor([[1.0, 0.0]]),
      x0: Nx.tensor([0.0, 0.0]),
      p0: Nx.eye(2),
      obs_var: 1.0,
      q_specs: [
        %{dim_index: 0, initial: 0.1, prior_shape: 2.0, prior_scale: 1.0},
        %{dim_index: 0, initial: 0.2, prior_shape: 2.0, prior_scale: 1.0}
      ]
    }

    assert_raise ArgumentError, ~r/dim_index values must be unique/, fn ->
      ModelSpec.validate!(spec)
    end
  end

  test "validate!/1 accepts unique q_specs dim_index values" do
    spec = %ModelSpec{
      f: Nx.eye(2),
      h: Nx.tensor([[1.0, 0.0]]),
      x0: Nx.tensor([0.0, 0.0]),
      p0: Nx.eye(2),
      obs_var: 1.0,
      q_specs: [
        %{dim_index: 0, initial: 0.1, prior_shape: 2.0, prior_scale: 1.0},
        %{dim_index: 1, initial: 0.2, prior_shape: 2.0, prior_scale: 1.0}
      ]
    }

    assert ModelSpec.validate!(spec) == spec
  end
end
