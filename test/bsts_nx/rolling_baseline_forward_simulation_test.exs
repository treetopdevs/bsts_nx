defmodule BstsNx.RollingBaselineForwardSimulationTest do
  use ExUnit.Case, async: true

  alias BstsNx.ModelSpec
  alias BstsNx.RollingBaseline

  describe "counterfactual/3 forward simulation" do
    test "static H matches closed-form forward moments" do
      spec =
        model_spec(
          Nx.tensor([[1.0, 1.0], [0.0, 1.0]]),
          Nx.tensor([[1.0, 0.0]])
        )

      fit_result = %{
        posterior_samples: [
          sample(
            [10.0, 2.0],
            [[1.0, 0.2], [0.2, 0.5]],
            [[0.1, 0.0], [0.0, 0.05]]
          )
        ],
        spec: spec,
        n_regression_dims: 0
      }

      cf = RollingBaseline.counterfactual(fit_result, 3)

      assert_close(cf.mean, [12.0, 14.0, 16.0])
      assert_close(cf.variance, [2.0, 4.05, 7.25])
      assert_in_delta cf.obs_variance, 0.25, 1.0e-6
    end

    test "time-varying post-regressor H matches closed-form forward moments" do
      spec =
        model_spec(
          Nx.tensor([[1.0, 0.0], [0.0, 1.0]]),
          [Nx.tensor([[1.0, 0.0]])]
        )

      fit_result = %{
        posterior_samples: [
          sample(
            [10.0, 2.0],
            [[1.0, 0.25], [0.25, 0.5]],
            [[0.1, 0.0], [0.0, 0.2]]
          )
        ],
        spec: spec,
        n_regression_dims: 1
      }

      cf =
        RollingBaseline.counterfactual(fit_result, 3,
          post_regressors: Nx.tensor([[0.0], [1.0], [2.0]])
        )

      assert_close(cf.mean, [10.0, 12.0, 14.0])
      assert_close(cf.variance, [1.1, 2.6, 6.7])
      assert_in_delta cf.obs_variance, 0.25, 1.0e-6
    end

    test "forward simulation stays on the compiled horizon path" do
      source = File.read!("lib/bsts_nx/rolling_baseline.ex")

      assert source =~ "defnp forward_simulate_defn"
      assert source =~ "while {i = Nx.tensor(0, type: {:s, 64})"
      refute source =~ "Enum.reduce(1..horizon"
      refute source =~ "Enum.reduce(h_list"
    end
  end

  defp model_spec(f, h) do
    %ModelSpec{
      f: f,
      h: h,
      x0: Nx.tensor([0.0, 0.0]),
      p0: Nx.eye(2),
      obs_var: 0.25,
      q_specs: []
    }
  end

  defp sample(state, cov, q_matrix) do
    %{
      states: [Nx.tensor(state)],
      state_covs: [Nx.tensor(cov)],
      q_matrix: Nx.tensor(q_matrix),
      obs_var: Nx.tensor(0.25)
    }
  end

  defp assert_close(actual, expected) do
    assert length(actual) == length(expected)

    Enum.zip(actual, expected)
    |> Enum.each(fn {actual_value, expected_value} ->
      assert_in_delta actual_value, expected_value, 1.0e-5
    end)
  end
end
