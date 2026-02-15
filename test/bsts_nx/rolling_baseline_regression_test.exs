defmodule BstsNx.RollingBaselineRegressionTest do
  use ExUnit.Case, async: true

  alias BstsNx.RollingBaseline

  # All tests in this module exercise the MCMC sampler
  @moduletag :external

  setup do
    :rand.seed(:exsss, {202, 203, 204})
    :ok
  end

  # -- Helpers --

  defp synthetic_regression_data(n, beta) do
    # y_t = beta * x_t + 50 + 0.1*t + small noise
    # x_t is a simple sinusoid so it's not collinear with trend
    Enum.map(0..(n - 1), fn t ->
      x = :math.sin(2 * :math.pi() * t / 12) * 10 + 20
      y = beta * x + 50.0 + 0.1 * t + :rand.normal() * 0.5
      {y, x}
    end)
  end

  # ---------------------------------------------------------------
  # 1. build_spec with regressors produces time-varying H
  # ---------------------------------------------------------------

  describe "build_spec/2 with regressors" do
    test "produces time-varying H (list) of correct length" do
      n_pre = 20
      regressors = Nx.broadcast(1.0, {n_pre, 2})

      spec = RollingBaseline.build_spec(n_pre, num_seasons: 4, regressors: regressors)

      assert is_list(spec.h)
      assert length(spec.h) == n_pre

      # State dim = 2 (trend) + 3 (seasonal, S=4) + 2 (regressors) = 7
      expected_state_dim = 2 + 3 + 2
      assert Nx.axis_size(spec.f, 0) == expected_state_dim

      # Each H entry should be {1, 7}
      h_entry = hd(spec.h)
      assert Nx.shape(h_entry) == {1, expected_state_dim}
    end

    test "q_specs includes regression coefficient entries" do
      regressors = Nx.broadcast(1.0, {10, 1})
      spec = RollingBaseline.build_spec(10, num_seasons: 4, regressors: regressors)

      # 2 trend + 1 seasonal + 1 regression = 4 q_specs
      assert length(spec.q_specs) == 4

      # Regression dim_index should be 5 (2 trend + 3 seasonal states)
      reg_q = List.last(spec.q_specs)
      assert reg_q.dim_index == 5
    end
  end

  # ---------------------------------------------------------------
  # 2. fit with regression spec succeeds
  # ---------------------------------------------------------------

  describe "fit/3 with regression" do
    test "returns fit_result with correct n_regression_dims" do
      n_pre = 20
      pairs = synthetic_regression_data(n_pre, 2.0)
      {ys, xs} = Enum.unzip(pairs)
      regressors = xs |> Nx.tensor() |> Nx.reshape({n_pre, 1})

      spec = RollingBaseline.build_spec(n_pre, num_seasons: 4, regressors: regressors)

      fit_result =
        RollingBaseline.fit(ys, spec,
          num_samples: 10,
          burn_in: 5,
          seed: 42,
          n_regression_dims: 1
        )

      assert fit_result.n_regression_dims == 1
      assert fit_result.num_observations == n_pre
      assert length(fit_result.posterior_samples) == 10

      # State dimension = 2 + 3 + 1 = 6
      sample = hd(fit_result.posterior_samples)
      first_state = hd(sample.states)
      assert Nx.axis_size(first_state, 0) == 6
    end
  end

  # ---------------------------------------------------------------
  # 3. counterfactual with post_regressors returns correct format
  # ---------------------------------------------------------------

  describe "counterfactual/3 with post_regressors" do
    test "returns correct format with time-varying H" do
      n_pre = 20
      horizon = 5
      pairs = synthetic_regression_data(n_pre, 2.0)
      {ys, xs} = Enum.unzip(pairs)
      regressors = xs |> Nx.tensor() |> Nx.reshape({n_pre, 1})

      spec = RollingBaseline.build_spec(n_pre, num_seasons: 4, regressors: regressors)

      fit_result =
        RollingBaseline.fit(ys, spec,
          num_samples: 10,
          burn_in: 5,
          seed: 42,
          n_regression_dims: 1
        )

      # Create post-period regressors
      post_regressors = Nx.broadcast(25.0, {horizon, 1})

      cf = RollingBaseline.counterfactual(fit_result, horizon, post_regressors: post_regressors)

      assert is_map(cf)
      assert length(cf.mean) == horizon
      assert length(cf.variance) == horizon
      assert cf.obs_variance > 0.0

      # All variances non-negative
      Enum.each(cf.variance, fn v -> assert v >= 0.0 end)
    end

    test "infers regression dims when fit_result.n_regression_dims is 0" do
      n_pre = 20
      horizon = 5
      pairs = synthetic_regression_data(n_pre, 2.0)
      {ys, xs} = Enum.unzip(pairs)
      regressors = xs |> Nx.tensor() |> Nx.reshape({n_pre, 1})

      spec = RollingBaseline.build_spec(n_pre, num_seasons: 4, regressors: regressors)

      fit_result =
        RollingBaseline.fit(ys, spec,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert fit_result.n_regression_dims == 0

      post_regressors = Nx.broadcast(25.0, {horizon, 1})
      cf = RollingBaseline.counterfactual(fit_result, horizon, post_regressors: post_regressors)

      assert length(cf.mean) == horizon
      assert length(cf.variance) == horizon
      assert cf.obs_variance > 0.0
    end
  end

  # ---------------------------------------------------------------
  # 4. fit_and_predict with full regressors end-to-end
  # ---------------------------------------------------------------

  describe "fit_and_predict/3 with regressors" do
    test "end-to-end with synthetic regression data" do
      n_pre = 20
      horizon = 5
      total = n_pre + horizon
      pairs = synthetic_regression_data(total, 2.0)
      {ys, xs} = Enum.unzip(pairs)

      # Only pre-period observations go to fit
      pre_ys = Enum.take(ys, n_pre)

      # Full regressor matrix (pre + post)
      full_regressors = xs |> Nx.tensor() |> Nx.reshape({total, 1})

      {cf, fit_result} =
        RollingBaseline.fit_and_predict(pre_ys, horizon,
          num_seasons: 4,
          num_samples: 10,
          burn_in: 5,
          seed: 42,
          regressors: full_regressors
        )

      assert length(cf.mean) == horizon
      assert length(cf.variance) == horizon
      assert cf.obs_variance > 0.0
      assert fit_result.n_regression_dims == 1
      assert fit_result.num_observations == n_pre
    end
  end

  # ---------------------------------------------------------------
  # 5. backward compat: fit_and_predict without regressors
  # ---------------------------------------------------------------

  describe "backward compatibility" do
    test "fit_and_predict without regressors works as before" do
      data =
        Enum.map(0..19, fn t ->
          100.0 + 0.1 * t + 10.0 * :math.sin(2 * :math.pi() * t / 4)
        end)

      {cf, fit_result} =
        RollingBaseline.fit_and_predict(data, 5,
          num_seasons: 4,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert length(cf.mean) == 5
      assert fit_result.n_regression_dims == 0
    end
  end

  # ---------------------------------------------------------------
  # 6. counterfactual raises when regression fit but no post_regressors
  # ---------------------------------------------------------------

  describe "counterfactual/3 error cases" do
    test "raises when regression model but no post_regressors" do
      n_pre = 20
      pairs = synthetic_regression_data(n_pre, 2.0)
      {ys, xs} = Enum.unzip(pairs)
      regressors = xs |> Nx.tensor() |> Nx.reshape({n_pre, 1})

      spec = RollingBaseline.build_spec(n_pre, num_seasons: 4, regressors: regressors)

      fit_result =
        RollingBaseline.fit(ys, spec,
          num_samples: 10,
          burn_in: 5,
          seed: 42,
          n_regression_dims: 1
        )

      assert_raise ArgumentError, ~r/requires :post_regressors/, fn ->
        RollingBaseline.counterfactual(fit_result, 5)
      end
    end
  end
end
