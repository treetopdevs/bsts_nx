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

    test "baseline and predictive moments retain distinct covariance and noise contracts" do
      spec = model_spec(Nx.tensor([[1.0]]), Nx.tensor([[1.0]]))

      samples = [
        %{
          states: [Nx.tensor([10.0])],
          state_covs: [Nx.tensor([[2.0]])],
          q_matrix: Nx.tensor([[1.0]]),
          obs_var: Nx.tensor(0.5)
        },
        %{
          states: [Nx.tensor([20.0])],
          state_covs: [Nx.tensor([[4.0]])],
          q_matrix: Nx.tensor([[3.0]]),
          obs_var: Nx.tensor(1.5)
        }
      ]

      baseline = RollingBaseline.counterfactual(%{posterior_samples: samples, spec: spec}, 2)

      {means, sds} =
        BstsNx.Forward.structured_moments_from_samples(samples, spec, [
          Nx.tensor([[1.0]]),
          Nx.tensor([[1.0]])
        ])

      assert_close(baseline.mean, [15.0, 15.0])
      assert_close(baseline.variance, [30.0, 32.0])
      assert baseline.obs_variance == 1.0
      assert_close(means, [15.0, 15.0])
      assert_close(sds, [:math.sqrt(53.0), :math.sqrt(55.0)])
    end

    test "missing terminal covariance falls back to zero and full Q is retained" do
      spec = model_spec(Nx.eye(2), Nx.tensor([[1.0, 1.0]]))

      sample = %{
        states: [Nx.tensor([2.0, 3.0])],
        q_matrix: Nx.tensor([[1.0, 0.5], [0.5, 2.0]]),
        obs_var: Nx.tensor(0.25)
      }

      for draw <- [sample, Map.put(sample, :state_covs, [])] do
        baseline = RollingBaseline.counterfactual(%{posterior_samples: [draw], spec: spec}, 2)

        {means, sds} =
          BstsNx.Forward.structured_moments_from_samples([draw], spec, [spec.h, spec.h])

        assert_close(baseline.mean, [5.0, 5.0])
        assert_close(baseline.variance, [4.0, 8.0])
        assert baseline.obs_variance == 0.25
        assert_close(means, [5.0, 5.0])
        assert_close(sds, [:math.sqrt(4.25), :math.sqrt(8.25)])
      end
    end

    test "zero horizon remains empty without requiring retained samples" do
      assert RollingBaseline.counterfactual(%{}, 0) ==
               %{mean: [], variance: [], obs_variance: 0.0}
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
