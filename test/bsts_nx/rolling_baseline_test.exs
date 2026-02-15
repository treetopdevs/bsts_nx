defmodule BstsNx.RollingBaselineTest do
  use ExUnit.Case, async: true

  alias BstsNx.RollingBaseline
  alias BstsNx.ModelSpec
  alias BstsNx.Components

  # -- Helpers --

  defp synthetic_data(n, opts) do
    num_seasons = Keyword.get(opts, :num_seasons, 12)
    base = Keyword.get(opts, :base, 100.0)
    trend = Keyword.get(opts, :trend, 0.1)
    amplitude = Keyword.get(opts, :amplitude, 10.0)

    Enum.map(0..(n - 1), fn t ->
      base + trend * t + amplitude * :math.sin(2 * :math.pi() * t / num_seasons)
    end)
  end

  # ---------------------------------------------------------------
  # 1. build_spec/2
  # ---------------------------------------------------------------

  describe "build_spec/2" do
    test "returns a ModelSpec with correct state dimension for default seasons" do
      spec = RollingBaseline.build_spec(100)

      assert %ModelSpec{} = spec
      # State dim = 2 (trend) + (96 - 1) (seasonal) = 97
      assert Nx.axis_size(spec.f, 0) == 97
      assert Nx.axis_size(spec.f, 1) == 97
    end

    test "respects custom num_seasons" do
      spec = RollingBaseline.build_spec(50, num_seasons: 12)

      # State dim = 2 (trend) + (12 - 1) (seasonal) = 13
      expected_dim = 2 + (12 - 1)
      assert Nx.axis_size(spec.f, 0) == expected_dim
      assert Nx.axis_size(spec.f, 1) == expected_dim
    end

    test "H observation matrix has correct dimensions" do
      spec = RollingBaseline.build_spec(50, num_seasons: 7)

      # State dim = 2 + 6 = 8
      h = spec.h
      assert Nx.shape(h) == {1, 8}
    end

    test "initial state vector has correct length" do
      spec = RollingBaseline.build_spec(50, num_seasons: 4)

      # State dim = 2 + 3 = 5
      assert Nx.axis_size(spec.x0, 0) == 5
    end

    test "q_specs covers trend and seasonal components" do
      spec = RollingBaseline.build_spec(50, num_seasons: 7)

      # 2 trend q_specs (level, slope) + 1 seasonal q_spec = 3
      assert length(spec.q_specs) == 3

      dim_indices = Enum.map(spec.q_specs, & &1.dim_index)
      # Trend: dims 0, 1; Seasonal: dim 2 (shifted by trend state size)
      assert 0 in dim_indices
      assert 1 in dim_indices
      assert 2 in dim_indices
    end

    test "custom variance options are applied" do
      spec =
        RollingBaseline.build_spec(50,
          num_seasons: 4,
          obs_var: 2.0,
          trend_var_level: 0.05,
          trend_var_slope: 0.005,
          seasonal_var: 0.01
        )

      assert spec.obs_var == 2.0

      level_spec = Enum.find(spec.q_specs, &(&1.dim_index == 0))
      assert level_spec.initial == 0.05

      slope_spec = Enum.find(spec.q_specs, &(&1.dim_index == 1))
      assert slope_spec.initial == 0.005

      seasonal_spec = Enum.find(spec.q_specs, &(&1.dim_index == 2))
      assert seasonal_spec.initial == 0.01
    end
  end

  # ---------------------------------------------------------------
  # 2. fit/3
  # ---------------------------------------------------------------

  describe "fit/3" do
    @tag :slow
    test "returns fit_result with expected keys and shapes" do
      num_seasons = 4
      data = synthetic_data(20, num_seasons: num_seasons)
      spec = RollingBaseline.build_spec(length(data), num_seasons: num_seasons)

      result = RollingBaseline.fit(data, spec, num_samples: 10, burn_in: 5, seed: 42)

      assert is_map(result)
      assert Map.has_key?(result, :posterior_samples)
      assert Map.has_key?(result, :spec)
      assert Map.has_key?(result, :num_observations)
      assert Map.has_key?(result, :final_states)
      assert Map.has_key?(result, :final_covs)

      assert result.num_observations == 20
      assert length(result.posterior_samples) == 10

      # Each sample should have states, state_covs, q_matrix, obs_var
      sample = hd(result.posterior_samples)
      assert Map.has_key?(sample, :states)
      assert Map.has_key?(sample, :state_covs)
      assert Map.has_key?(sample, :q_matrix)
      assert Map.has_key?(sample, :obs_var)

      # States should have one entry per observation
      assert length(sample.states) == 20

      # State dimension = 2 + (4 - 1) = 5
      state_dim = 2 + (num_seasons - 1)
      first_state = hd(sample.states)
      assert Nx.axis_size(first_state, 0) == state_dim
    end

    @tag :slow
    test "accepts Nx tensor observations" do
      data = Nx.tensor(synthetic_data(15, num_seasons: 4))
      spec = RollingBaseline.build_spec(15, num_seasons: 4)

      result = RollingBaseline.fit(data, spec, num_samples: 5, burn_in: 3, seed: 123)

      assert result.num_observations == 15
      assert length(result.posterior_samples) == 5
    end

    test "rejects non-positive num_samples" do
      data = synthetic_data(15, num_seasons: 4)
      spec = RollingBaseline.build_spec(15, num_seasons: 4)

      assert_raise ArgumentError, ~r/num_samples/, fn ->
        RollingBaseline.fit(data, spec, num_samples: 0)
      end
    end
  end

  # ---------------------------------------------------------------
  # 3. counterfactual/2
  # ---------------------------------------------------------------

  describe "counterfactual/2" do
    @tag :slow
    test "returns correct format for SpotAttributor" do
      num_seasons = 4
      data = synthetic_data(20, num_seasons: num_seasons)
      spec = RollingBaseline.build_spec(length(data), num_seasons: num_seasons)
      fit_result = RollingBaseline.fit(data, spec, num_samples: 10, burn_in: 5, seed: 42)

      horizon = 5
      cf = RollingBaseline.counterfactual(fit_result, horizon)

      assert is_map(cf)
      assert Map.has_key?(cf, :mean)
      assert Map.has_key?(cf, :variance)
      assert Map.has_key?(cf, :obs_variance)

      assert is_list(cf.mean)
      assert is_list(cf.variance)
      assert is_float(cf.obs_variance)

      assert length(cf.mean) == horizon
      assert length(cf.variance) == horizon
      assert cf.obs_variance > 0.0
    end

    @tag :slow
    test "all variances are non-negative" do
      num_seasons = 4
      data = synthetic_data(20, num_seasons: num_seasons)
      spec = RollingBaseline.build_spec(length(data), num_seasons: num_seasons)
      fit_result = RollingBaseline.fit(data, spec, num_samples: 10, burn_in: 5, seed: 42)

      cf = RollingBaseline.counterfactual(fit_result, 10)

      Enum.each(cf.variance, fn v ->
        assert v >= 0.0
      end)
    end

    @tag :slow
    test "variance grows with horizon" do
      num_seasons = 4
      data = synthetic_data(30, num_seasons: num_seasons)
      spec = RollingBaseline.build_spec(length(data), num_seasons: num_seasons)
      # Use more samples for stable variance estimates
      fit_result = RollingBaseline.fit(data, spec, num_samples: 50, burn_in: 25, seed: 42)

      cf = RollingBaseline.counterfactual(fit_result, 20)

      # Compare first vs last quarter for a stronger signal than halves.
      # Last-step variance should exceed first-step variance (monotone property
      # of forward uncertainty propagation through a state-space model).
      first_var = hd(cf.variance)
      last_var = List.last(cf.variance)

      assert last_var >= first_var
    end

    test "uses posterior state covariance as initial predictive uncertainty" do
      spec = Components.local_level_spec(initial_state: 0.0, initial_cov: 1.0, process_var: 0.0)

      sample = %{
        states: [Nx.tensor([0.0])],
        state_covs: [Nx.tensor([[4.0]])],
        q_matrix: Nx.tensor([[0.0]]),
        obs_var: Nx.tensor(0.0)
      }

      fit_result = %{
        posterior_samples: [sample],
        spec: spec,
        num_observations: 1,
        final_states: sample.states,
        final_covs: sample.state_covs,
        convergence: %{}
      }

      cf = RollingBaseline.counterfactual(fit_result, 1)

      # With F=1, H=1, Q=0 and P_T=4, first predictive variance should be 4.
      assert_in_delta hd(cf.variance), 4.0, 1.0e-8
    end

    test "rejects empty posterior sample list" do
      fit_result = %{
        posterior_samples: [],
        spec: Components.local_level_spec(),
        n_regression_dims: 0
      }

      assert_raise ArgumentError, ~r/posterior_samples/, fn ->
        RollingBaseline.counterfactual(fit_result, 3)
      end
    end

    test "uses last pre-period H row when stitching structural and post regressors" do
      spec = %ModelSpec{
        f: Nx.tensor([[1.0, 0.0], [0.0, 1.0]]),
        h: [Nx.tensor([[1.0, 0.5]]), Nx.tensor([[2.0, 0.7]])],
        x0: Nx.tensor([0.0, 0.0]),
        p0: Nx.tensor([[1.0, 0.0], [0.0, 1.0]]),
        obs_var: 0.0,
        q_specs: []
      }

      sample = %{
        states: [Nx.tensor([10.0, 3.0])],
        state_covs: [Nx.tensor([[0.0, 0.0], [0.0, 0.0]])],
        q_matrix: Nx.tensor([[0.0, 0.0], [0.0, 0.0]]),
        obs_var: Nx.tensor(0.0)
      }

      fit_result = %{
        posterior_samples: [sample],
        spec: spec,
        num_observations: 2,
        final_states: sample.states,
        final_covs: sample.state_covs,
        convergence: %{},
        n_regression_dims: 1
      }

      cf = RollingBaseline.counterfactual(fit_result, 1, post_regressors: Nx.tensor([[1.0]]))

      # last structural H coefficient (2.0) should be used, not the first row (1.0)
      assert_in_delta hd(cf.mean), 23.0, 1.0e-8
    end
  end

  # ---------------------------------------------------------------
  # 4. fit_and_predict/3
  # ---------------------------------------------------------------

  describe "fit_and_predict/3" do
    @tag :slow
    test "end-to-end with synthetic sinusoidal data" do
      num_seasons = 4
      data = synthetic_data(20, num_seasons: num_seasons)

      {cf, fit_result} =
        RollingBaseline.fit_and_predict(data, 5,
          num_seasons: num_seasons,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      # Verify counterfactual shape
      assert length(cf.mean) == 5
      assert length(cf.variance) == 5
      assert cf.obs_variance > 0.0

      # Verify fit_result
      assert fit_result.num_observations == 20
      assert length(fit_result.posterior_samples) == 10
    end

    @tag :slow
    test "returns same result as manual build_spec + fit + counterfactual" do
      num_seasons = 4
      data = synthetic_data(20, num_seasons: num_seasons)
      opts = [num_seasons: num_seasons, num_samples: 10, burn_in: 5, seed: 42]

      # Manual path
      spec = RollingBaseline.build_spec(length(data), opts)
      fit_result = RollingBaseline.fit(data, spec, opts)
      cf_manual = RollingBaseline.counterfactual(fit_result, 5)

      # Convenience path
      {cf_auto, _fit_result} = RollingBaseline.fit_and_predict(data, 5, opts)

      # Should produce identical results with same seed
      Enum.zip(cf_manual.mean, cf_auto.mean)
      |> Enum.each(fn {m, a} ->
        assert_in_delta m, a, 1.0e-10
      end)

      assert_in_delta cf_manual.obs_variance, cf_auto.obs_variance, 1.0e-10
    end
  end

  # ---------------------------------------------------------------
  # 5. SpotAttributor compatibility
  # ---------------------------------------------------------------

  describe "SpotAttributor compatibility" do
    @tag :slow
    test "counterfactual output plugs into SpotAttributor.attribute/4" do
      num_seasons = 4
      data = synthetic_data(20, num_seasons: num_seasons)
      spec = RollingBaseline.build_spec(length(data), num_seasons: num_seasons)
      fit_result = RollingBaseline.fit(data, spec, num_samples: 10, burn_in: 5, seed: 42)

      horizon = 10
      cf = RollingBaseline.counterfactual(fit_result, horizon)

      # Create synthetic post-intervention observations (baseline + lift)
      post_obs = Enum.map(cf.mean, fn m -> m + 5.0 end)

      spots = [%{id: "spot_1", window_start: 2, window_end: 7}]

      # Should not raise
      result = BstsNx.SpotAttributor.attribute(post_obs, spots, cf)

      assert length(result.attributions) == 1
      [attr] = result.attributions
      assert attr.spot_id == "spot_1"
      assert is_float(attr.lift)
      assert is_float(attr.lift_sd)
      assert attr.lift_sd >= 0.0
      assert attr.p_positive >= 0.0 and attr.p_positive <= 1.0

      # With a constant +5 lift, total lift should be approximately 5 * 5 = 25
      assert_in_delta attr.lift, 25.0, 1.0e-6
    end
  end
end
