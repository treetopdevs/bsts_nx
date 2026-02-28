defmodule BstsNx.RollingBaselineBranchCoverageTest do
  use ExUnit.Case, async: true

  alias BstsNx.Components
  alias BstsNx.ModelSpec
  alias BstsNx.RollingBaseline

  describe "fit/3 guard and coercion branches" do
    test "rejects negative burn_in" do
      spec = Components.local_level_spec()

      assert_raise ArgumentError, ~r/burn_in must be a non-negative integer/, fn ->
        RollingBaseline.fit([1.0, 2.0], spec, num_samples: 1, burn_in: -1)
      end
    end

    test "coerces integer and tensor observations in list form" do
      spec = Components.local_level_spec(process_var: 0.1, obs_var: 1.0)

      result =
        RollingBaseline.fit(
          [1, Nx.tensor(2.0), 3],
          spec,
          num_samples: 1,
          burn_in: 0,
          seed: 11
        )

      assert result.num_observations == 3
      assert length(result.posterior_samples) == 1
    end

    test "accepts observations as tensor in non-slow mode" do
      spec = Components.local_level_spec(process_var: 0.1, obs_var: 1.0)
      obs = Nx.tensor([1.0, 2.0, 3.0])
      result = RollingBaseline.fit(obs, spec, num_samples: 1, burn_in: 0, seed: 12)

      assert result.num_observations == 3
      assert length(result.posterior_samples) == 1
    end

    test "validates n_regression_dims against regressor columns" do
      spec = Components.local_level_spec(process_var: 0.1, obs_var: 1.0)

      assert_raise ArgumentError, ~r/must match regressors columns p=2/, fn ->
        RollingBaseline.fit(
          [1.0, 1.5, 2.0],
          spec,
          regressors: [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]],
          n_regression_dims: 1,
          num_samples: 1,
          burn_in: 0,
          seed: 13
        )
      end
    end

    test "rejects invalid n_regression_dims type/value" do
      spec = Components.local_level_spec(process_var: 0.1, obs_var: 1.0)

      assert_raise ArgumentError, ~r/non-negative integer/, fn ->
        RollingBaseline.fit([1.0, 2.0], spec,
          n_regression_dims: -1,
          num_samples: 1,
          burn_in: 0,
          seed: 14
        )
      end
    end

    test "warm_start with empty posterior samples is a no-op" do
      spec = Components.local_level_spec(process_var: 0.1, obs_var: 1.0)

      result =
        RollingBaseline.fit(
          [1.0, 1.0, 1.0],
          spec,
          num_samples: 1,
          burn_in: 0,
          seed: 15,
          warm_start: %{posterior_samples: []}
        )

      assert result.spec.obs_var == spec.obs_var
      assert result.spec.q_specs == spec.q_specs
    end

    test "warm_start falls back when terminal state/covariance candidates are invalid" do
      spec = Components.local_level_spec(process_var: 0.1, obs_var: 1.0)

      warm_start = %{
        posterior_samples: [
          %{
            obs_var: Nx.tensor(0.8),
            q_matrix: Nx.tensor([[0.2]]),
            states: [:invalid_state],
            state_covs: [:invalid_cov]
          }
        ],
        final_states: :not_a_list,
        final_covs: :not_a_list
      }

      result =
        RollingBaseline.fit(
          [1.0, 1.0, 1.0],
          spec,
          num_samples: 1,
          burn_in: 0,
          seed: 16,
          warm_start: warm_start
        )

      assert Nx.shape(result.spec.x0) == Nx.shape(spec.x0)
      assert Nx.shape(result.spec.p0) == Nx.shape(spec.p0)
    end

    test "convergence summaries return :nan for specs with no q_specs" do
      spec = %ModelSpec{
        f: Nx.tensor([[1.0]]),
        h: Nx.tensor([[1.0]]),
        x0: Nx.tensor([0.0]),
        p0: Nx.tensor([[1.0]]),
        obs_var: 1.0,
        q_specs: []
      }

      result = RollingBaseline.fit([1.0, 1.0], spec, num_samples: 1, burn_in: 0, seed: 17)

      assert result.convergence.r_hat_process_var == :nan
      assert result.convergence.ess_process_var == :nan
      refute result.convergence.converged
    end
  end

  describe "regressor normalization guards" do
    test "build_spec accepts list regressors" do
      spec = RollingBaseline.build_spec(2, num_seasons: 2, regressors: [[1.0], [2.0]])
      assert is_list(spec.h)
      assert length(spec.h) == 2
    end

    test "build_spec rejects regressors with wrong rank" do
      assert_raise ArgumentError, ~r/rank-2 tensor/, fn ->
        RollingBaseline.build_spec(2, num_seasons: 2, regressors: Nx.tensor([1.0, 2.0]))
      end
    end

    test "build_spec rejects regressors with wrong number of rows" do
      assert_raise ArgumentError, ~r/must have 3 rows/, fn ->
        RollingBaseline.build_spec(3, num_seasons: 2, regressors: Nx.tensor([[1.0], [2.0]]))
      end
    end

    test "fit_and_predict validates full regressor row count" do
      assert_raise ArgumentError, ~r/full regressors must have 4 rows/, fn ->
        RollingBaseline.fit_and_predict(
          [1.0, 2.0],
          2,
          num_seasons: 2,
          num_samples: 1,
          burn_in: 0,
          regressors: Nx.tensor([[1.0], [2.0], [3.0]])
        )
      end
    end
  end

  describe "counterfactual/3 branches" do
    test "returns empty structure for zero horizon" do
      assert RollingBaseline.counterfactual(%{}, 0) == %{
               mean: [],
               variance: [],
               obs_variance: 0.0
             }
    end

    test "uses zero covariance fallback when sample has no state_covs" do
      spec = Components.local_level_spec(process_var: 0.0, obs_var: 0.0)

      sample = %{
        states: [Nx.tensor([2.0])],
        state_covs: [],
        q_matrix: Nx.tensor([[0.0]]),
        obs_var: Nx.tensor(0.0)
      }

      fit_result = %{posterior_samples: [sample], spec: spec, n_regression_dims: 0}
      cf = RollingBaseline.counterfactual(fit_result, 1)

      assert cf.mean == [2.0]
      assert cf.variance == [0.0]
      assert cf.obs_variance == 0.0
    end

    test "replicates the last time-varying H row for non-regression forecasting" do
      spec = time_varying_h_spec()

      sample = %{
        states: [Nx.tensor([2.0, 5.0])],
        state_covs: [Nx.tensor([[0.0, 0.0], [0.0, 0.0]])],
        q_matrix: Nx.tensor([[0.0, 0.0], [0.0, 0.0]]),
        obs_var: Nx.tensor(0.0)
      }

      fit_result = %{posterior_samples: [sample], spec: spec, n_regression_dims: 0}
      cf = RollingBaseline.counterfactual(fit_result, 2)

      assert cf.mean == [6.0, 6.0]
    end

    test "raises when static H is inconsistent with n_regression_dims" do
      fit_result = static_h_fit_result(1)

      assert_raise ArgumentError, ~r/static H but n_regression_dims=1/, fn ->
        RollingBaseline.counterfactual(fit_result, 1)
      end
    end

    test "raises when static H receives post_regressors" do
      fit_result = static_h_fit_result(1)

      assert_raise ArgumentError, ~r/received :post_regressors but the model has static H/, fn ->
        RollingBaseline.counterfactual(fit_result, 1, post_regressors: Nx.tensor([[1.0]]))
      end
    end

    test "raises when post_regressors columns do not match n_regression_dims" do
      fit_result = %{
        posterior_samples: [time_varying_sample()],
        spec: time_varying_h_spec(),
        n_regression_dims: 1
      }

      assert_raise ArgumentError,
                   ~r/post_regressors has p=2 columns, but fit_result.n_regression_dims=1/,
                   fn ->
                     RollingBaseline.counterfactual(
                       fit_result,
                       2,
                       post_regressors: Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
                     )
                   end
    end

    test "raises when inferred regression dims exceed model state dimension" do
      fit_result = %{
        posterior_samples: [time_varying_sample()],
        spec: time_varying_h_spec(),
        n_regression_dims: 0
      }

      assert_raise ArgumentError, ~r/model state dimension is 2/, fn ->
        RollingBaseline.counterfactual(
          fit_result,
          1,
          post_regressors: Nx.tensor([[1.0, 2.0, 3.0]])
        )
      end
    end
  end

  defp static_h_fit_result(n_regression_dims) do
    spec = Components.local_level_spec(process_var: 0.0, obs_var: 0.0)

    sample = %{
      states: [Nx.tensor([1.0])],
      state_covs: [Nx.tensor([[0.0]])],
      q_matrix: Nx.tensor([[0.0]]),
      obs_var: Nx.tensor(0.0)
    }

    %{posterior_samples: [sample], spec: spec, n_regression_dims: n_regression_dims}
  end

  defp time_varying_h_spec do
    %ModelSpec{
      f: Nx.eye(2),
      h: [Nx.tensor([[1.0, 0.0]]), Nx.tensor([[3.0, 0.0]])],
      x0: Nx.tensor([0.0, 0.0]),
      p0: Nx.eye(2),
      obs_var: 0.0,
      q_specs: []
    }
  end

  defp time_varying_sample do
    %{
      states: [Nx.tensor([2.0, 5.0])],
      state_covs: [Nx.tensor([[0.0, 0.0], [0.0, 0.0]])],
      q_matrix: Nx.tensor([[0.0, 0.0], [0.0, 0.0]]),
      obs_var: Nx.tensor(0.0)
    }
  end
end
