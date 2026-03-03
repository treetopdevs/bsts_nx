defmodule BstsNxDistributionsMvNormalTest do
  use ExUnit.Case, async: true

  alias BstsNx.Distributions

  test "mv_normal_sample_with_chol matches mv_normal_sample for the same key" do
    mean = Nx.tensor([1.5, -0.2], type: {:f, 64})
    cov = Nx.tensor([[0.9, 0.15], [0.15, 0.5]], type: {:f, 64})
    chol = Nx.LinAlg.cholesky(cov)
    key = Nx.Random.key(12345)

    {sample_with_chol, key_a} = Distributions.mv_normal_sample_with_chol(key, mean, cov, chol)
    {sample_direct, key_b} = Distributions.mv_normal_sample(key, mean, cov)

    assert Nx.shape(sample_with_chol) == {2}
    assert Nx.all_close(sample_with_chol, sample_direct, atol: 1.0e-8, rtol: 1.0e-8)
    assert Nx.to_flat_list(key_a) == Nx.to_flat_list(key_b)
  end
end
