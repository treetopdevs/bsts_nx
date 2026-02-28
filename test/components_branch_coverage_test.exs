defmodule BstsNx.ComponentsBranchCoverageTest do
  use ExUnit.Case, async: true

  alias BstsNx.Components
  alias BstsNx.ModelSpec

  describe "predict_with_regressors/2 coercion" do
    test "accepts number and list inputs" do
      assert Nx.to_number(Components.predict_with_regressors(2.0, 3.0)) == 6.0
      assert Nx.to_number(Components.predict_with_regressors([1.0, 2.0], [3.0, 4.0])) == 11.0
    end
  end

  describe "regression/2 and regression_spec/2 guards" do
    test "regression/2 rejects legacy list-like second arg" do
      assert_raise ArgumentError, ~r/Legacy \(betas, sigma_matrix\) is no longer supported/, fn ->
        Components.regression([[1.0], [2.0]], [0.1, 0.2])
      end
    end

    test "regression_spec accepts list regressors and list variance options" do
      spec =
        Components.regression_spec(
          [[1.0, 2.0], [3.0, 4.0]],
          var_beta: [0.2, 0.3],
          initial_betas: Nx.tensor([0.5, -0.5]),
          initial_cov_beta: [5.0, 6.0]
        )

      assert %ModelSpec{} = spec
      assert length(spec.h) == 2
      assert Enum.map(spec.q_specs, & &1.initial) == [0.2, 0.3]
      assert Nx.to_flat_list(spec.x0) == [0.5, -0.5]
      assert Nx.to_flat_list(Nx.take_diagonal(spec.p0)) == [5.0, 6.0]
    end

    test "regression_spec rejects unknown mode" do
      assert_raise ArgumentError, ~r/unknown regression mode/, fn ->
        Components.regression_spec([[1.0], [2.0]], mode: :nope)
      end
    end

    test "spike-and-slab guards prior_inclusion and g" do
      assert_raise ArgumentError, ~r/prior_inclusion must be in \(0, 1\)/, fn ->
        Components.regression_spec([[1.0], [2.0]], mode: :spike_and_slab, prior_inclusion: 1.0)
      end

      assert_raise ArgumentError, ~r/g must be > 0/, fn ->
        Components.regression_spec([[1.0], [2.0]], mode: :spike_and_slab, g: 0.0)
      end
    end

    test "spike-and-slab accepts tensor betas and list initial covariance" do
      spec =
        Components.regression_spec(
          [[1.0, 0.0], [0.0, 1.0]],
          mode: :spike_and_slab,
          initial_betas: Nx.tensor([0.25, -0.25]),
          initial_cov_beta: [7.0, 8.0]
        )

      assert spec.regression.mode == :spike_and_slab
      assert Nx.to_flat_list(spec.x0) == [0.25, -0.25]
      assert Nx.to_flat_list(Nx.take_diagonal(spec.p0)) == [7.0, 8.0]
    end
  end

  describe "compose_specs/2 H and regression branches" do
    test "composes two time-varying H lists" do
      s1 = Components.regression_spec([[1.0], [2.0]])
      s2 = Components.regression_spec([[3.0], [4.0]])

      composed = Components.compose_specs(s1, s2)

      assert is_list(composed.h)
      assert length(composed.h) == 2
      assert Nx.shape(hd(composed.h)) == {1, 2}
    end

    test "composes time-varying H with static H (h1 list, h2 tensor)" do
      s1 = Components.regression_spec([[1.0], [2.0]])
      s2 = Components.local_level_spec()

      composed = Components.compose_specs(s1, s2)

      assert is_list(composed.h)
      assert length(composed.h) == 2
      assert Nx.shape(hd(composed.h)) == {1, 2}
    end

    test "compose_specs preserves spec1-only regression metadata" do
      reg_only =
        Components.regression_spec([[1.0], [2.0]],
          mode: :spike_and_slab,
          prior_inclusion: 0.4,
          g: 2.0
        )

      combined = Components.compose_specs(reg_only, Components.local_level_spec())
      assert combined.regression.mode == :spike_and_slab
      assert combined.regression.start_dim == 0
    end

    test "compose_specs raises when both specs have regression metadata" do
      s1 =
        Components.regression_spec([[1.0], [2.0]],
          mode: :spike_and_slab,
          prior_inclusion: 0.4,
          g: 2.0
        )

      s2 =
        Components.regression_spec([[3.0], [4.0]],
          mode: :spike_and_slab,
          prior_inclusion: 0.3,
          g: 3.0
        )

      assert_raise ArgumentError, ~r/at most one regression metadata block/, fn ->
        Components.compose_specs(s1, s2)
      end
    end

    test "compose_specs reshapes rank-0 and rank-1 H rows" do
      s1 = minimal_spec(Nx.tensor(2.0))
      s2 = minimal_spec(Nx.tensor([3.0]))

      composed = Components.compose_specs(s1, s2)

      assert Nx.shape(composed.h) == {1, 2}
      assert Nx.to_flat_list(composed.h) == [2.0, 3.0]
    end
  end

  defp minimal_spec(h) do
    %ModelSpec{
      f: Nx.tensor([[1.0]]),
      h: h,
      x0: Nx.tensor([0.0]),
      p0: Nx.tensor([[1.0]]),
      obs_var: 1.0,
      q_specs: [%{dim_index: 0, initial: 0.1, prior_shape: 1.0, prior_scale: 1.0}]
    }
  end
end
