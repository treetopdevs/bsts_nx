defmodule BstsNxComponentsReproducibilityTest do
  use ExUnit.Case
  import Nx, only: [to_number: 1]
  alias BstsNx.Components
  alias BstsNx.GibbsSampler

  @moduletag :external

  describe "Components" do
    test "local level component produces expected matrices" do
      comp = Components.local_level(0.5)
      assert Nx.shape(comp.f) == {1, 1}
      assert Nx.shape(comp.q) == {1, 1}
      assert Nx.shape(comp.h) == {1, 1}
      assert to_number(comp.q[0][0]) == 0.5
    end

    test "local linear trend component produces correct F and H" do
      comp = Components.local_linear_trend(0.1, 0.01)
      assert Nx.shape(comp.f) == {2, 2}
      f = comp.f
      # F = [[1,1],[0,1]]
      assert to_number(f[0][1]) == 1.0
      assert to_number(f[1][0]) == 0.0
      # H = [1,0]
      h = comp.h
      assert Nx.shape(h) == {1, 2}
      assert to_number(h[0][0]) == 1.0
      assert to_number(h[0][1]) == 0.0
    end

    test "regression component produces identity F and betas in H" do
      betas = Nx.tensor([0.5, -0.2])
      sigma = Nx.eye(2) |> Nx.multiply(0.01)
      comp = apply(Components, :regression, [betas, sigma])
      assert Nx.shape(comp.f) == {2, 2}
      # F should be identity
      assert to_number(comp.f[0][0]) == 1.0
      assert to_number(comp.f[1][1]) == 1.0
      # H contains betas
      assert_in_delta(to_number(comp.h[0][0]), 0.5, 1.0e-6)
      assert_in_delta(to_number(comp.h[0][1]), -0.2, 1.0e-6)
    end
  end

  describe "GibbsSampler reproducibility" do
    test "samples are reproducible with same seed" do
      :rand.seed(:exsss, {100, 101, 102})
      obs = Enum.map(1..30, fn _ -> :rand.uniform() end)
      samples1 = GibbsSampler.sample(obs, 5, 0.0, 1.0, 1.0, 1.0, seed: 42)
      samples2 = GibbsSampler.sample(obs, 5, 0.0, 1.0, 1.0, 1.0, seed: 42)

      Enum.zip(samples1, samples2)
      |> Enum.each(fn {s1, s2} ->
        assert Nx.to_number(s1.process_var) == Nx.to_number(s2.process_var)
        assert Nx.to_number(s1.obs_var) == Nx.to_number(s2.obs_var)
      end)
    end
  end
end
