defmodule BstsNxDiagnosticsTest do
  use ExUnit.Case
  alias BstsNx.Diagnostics

  @moduletag :external

  test "r_hat and effective_sample_size compute statistics on multiple chains" do
    # Create fake chains of scalar values
    chains = [
      [1.0, 2.0, 3.0, 4.0],
      [0.9, 2.1, 3.2, 3.8],
      [1.1, 2.0, 2.9, 4.2]
    ]

    rhat = Diagnostics.r_hat(chains)
    ess = Diagnostics.effective_sample_size(chains)
    assert is_float(rhat) or rhat == :nan
    assert (is_float(ess) and ess > 0) or ess == :nan
  end

  test "r_hat returns :nan for single chain" do
    rhat = Diagnostics.r_hat([[1.0, 2.0, 3.0]])
    assert rhat == :nan
  end

  test "r_hat returns :nan for constant chains" do
    rhat = Diagnostics.r_hat([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    assert rhat == :nan
  end

  test "effective_sample_size returns :nan for short chains" do
    ess = Diagnostics.effective_sample_size([[1.0]])
    assert ess == :nan
  end
end
