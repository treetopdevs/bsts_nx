defmodule BstsNx.ModelBuilderTest do
  use ExUnit.Case, async: true

  alias BstsNx.Components
  alias BstsNx.ModelBuilder

  describe "coerce_obs/1" do
    test "preserves missing observations as NaN sentinels" do
      nan = Nx.Constants.nan() |> Nx.to_number()

      result =
        ModelBuilder.coerce_obs([
          nil,
          :nan,
          nan,
          Nx.Constants.nan(),
          Nx.tensor(7),
          8
        ])

      assert Enum.take(result, 4) |> Enum.all?(&BstsNx.Utils.missing_observation?/1)
      assert Enum.at(result, 4) == 7.0
      assert Enum.at(result, 5) == 8.0
    end

    test "first_obs/1 skips missing observations before falling back" do
      assert ModelBuilder.first_obs([nil, :nan, Nx.Constants.nan(), 12.5]) == 12.5
      assert ModelBuilder.first_obs([nil, :nan, Nx.Constants.nan()]) == 0.0
    end

    test "default structured specs use the first observed value for initial level" do
      {spec, :structured} = ModelBuilder.build_spec([nil, :nan, 10.0, 11.0], seasonality: 3)

      assert spec.x0 |> Nx.to_flat_list() |> hd() == 10.0
    end
  end

  describe "build_spec/2" do
    test "returns scalar when no options given" do
      assert {nil, :scalar} = ModelBuilder.build_spec([1.0, 2.0, 3.0])
    end

    test "returns structured with seasonality" do
      {spec, :structured} = ModelBuilder.build_spec([1.0, 2.0], seasonality: 7)
      assert %BstsNx.ModelSpec{} = spec
    end

    test "returns structured with model_spec" do
      spec = Components.local_level_spec(initial_state: 1.0, initial_cov: 1.0)
      assert {^spec, :structured} = ModelBuilder.build_spec([1.0], model_spec: spec)
    end

    test "returns structured with regressors" do
      regressors = Nx.tensor([[1.0], [2.0], [3.0]])
      {spec, :structured} = ModelBuilder.build_spec([1.0, 2.0, 3.0], regressors: regressors)

      assert %BstsNx.ModelSpec{} = spec
      # Trend (2 states) + 1 regressor = 3 state dimensions
      assert Nx.axis_size(spec.f, 0) == 3
      # H should be time-varying (list of length 3)
      assert is_list(spec.h)
      assert length(spec.h) == 3
    end

    test "returns structured with regressors and seasonality" do
      regressors = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

      {spec, :structured} =
        ModelBuilder.build_spec([1.0, 2.0, 3.0], regressors: regressors, seasonality: 3)

      assert %BstsNx.ModelSpec{} = spec
      # Trend (2) + seasonal (2 for period 3) + 2 regressors = 6 state dims
      assert Nx.axis_size(spec.f, 0) == 6
      assert is_list(spec.h)
      assert length(spec.h) == 3
    end

    test "model_spec takes precedence over regressors" do
      spec = Components.local_level_spec(initial_state: 1.0, initial_cov: 1.0)
      regressors = Nx.tensor([[1.0], [2.0]])

      {result_spec, :structured} =
        ModelBuilder.build_spec([1.0, 2.0], model_spec: spec, regressors: regressors)

      # Should use the explicit model_spec, ignoring regressors
      assert Nx.all_close(result_spec.f, spec.f)
    end

    test "multi-column regressors produce correct state dimension" do
      regressors = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      {spec, :structured} = ModelBuilder.build_spec([1.0, 2.0], regressors: regressors)

      # Trend (2) + 3 regressors = 5 state dims
      assert Nx.axis_size(spec.f, 0) == 5
    end
  end

  describe "build_opts_with_controls/3" do
    test "returns opts without controls" do
      result = ModelBuilder.build_opts_with_controls([1.0, 2.0], nil, alpha: 0.05, seed: 42)
      assert result[:alpha] == 0.05
      assert result[:seed] == 42
      refute Keyword.has_key?(result, :model_spec)
    end

    test "preserves method without controls" do
      result =
        ModelBuilder.build_opts_with_controls([1.0, 2.0], nil,
          method: :filter,
          alpha: 0.05,
          seed: 42
        )

      assert result[:method] == :filter
    end

    test "treats empty controls as no controls" do
      result = ModelBuilder.build_opts_with_controls([1.0, 2.0], [], alpha: 0.05, seed: 42)
      assert result[:alpha] == 0.05
      refute Keyword.has_key?(result, :model_spec)
    end

    test "produces structured spec with controls" do
      obs = [1.0, 2.0, 3.0]
      controls = [[10.0, 20.0, 30.0]]
      result = ModelBuilder.build_opts_with_controls(obs, controls, seed: 42)

      assert %BstsNx.ModelSpec{} = result[:model_spec]
      spec = result[:model_spec]
      # Trend (2) + 1 regressor = 3 state dims
      assert Nx.axis_size(spec.f, 0) == 3
    end

    test "preserves method with controls" do
      obs = [1.0, 2.0, 3.0]
      controls = [[10.0, 20.0, 30.0]]
      result = ModelBuilder.build_opts_with_controls(obs, controls, method: :filter)

      assert result[:method] == :filter
      assert %BstsNx.ModelSpec{} = result[:model_spec]
    end

    test "preserves execution and sampling passthrough options with controls" do
      obs = [1.0, 2.0, 3.0]
      controls = [[10.0, 20.0, 30.0]]
      key = Nx.Random.key(123)

      result =
        ModelBuilder.build_opts_with_controls(obs, controls,
          mode: :operational,
          fallback: :mcmc,
          allow_mcmc_fallback: true,
          return: :tensors,
          key: key,
          thin: 3,
          n_samples: 17,
          seed: 42
        )

      assert result[:mode] == :operational
      assert result[:fallback] == :mcmc
      assert result[:allow_mcmc_fallback] == true
      assert result[:return] == :tensors
      assert result[:key] == key
      assert result[:thin] == 3
      assert result[:n_samples] == 17
      assert result[:seed] == 42
      assert %BstsNx.ModelSpec{} = result[:model_spec]
    end

    test "produces structured spec with controls and seasonality" do
      obs = [1.0, 2.0, 3.0]
      controls = [[10.0, 20.0, 30.0]]
      result = ModelBuilder.build_opts_with_controls(obs, controls, seed: 42, seasonality: 3)

      spec = result[:model_spec]
      # Trend (2) + seasonal (2 for period 3) + 1 regressor = 5
      assert Nx.axis_size(spec.f, 0) == 5
    end

    test "composes controls into provided model_spec" do
      obs = [1.0, 2.0, 3.0]
      controls = [[10.0, 20.0, 30.0]]
      base_spec = Components.local_level_spec(initial_state: 1.0, initial_cov: 1.0)

      result =
        ModelBuilder.build_opts_with_controls(obs, controls,
          model_spec: base_spec,
          seasonality: 7
        )

      spec = result[:model_spec]

      # local_level (1 state) + 1 regressor = 2 states.
      # seasonality is ignored because explicit model_spec takes precedence.
      assert Nx.axis_size(spec.f, 0) == 2
    end

    test "rejects mismatched control series length" do
      assert_raise ArgumentError, ~r/control series length/, fn ->
        ModelBuilder.build_opts_with_controls([1.0, 2.0, 3.0], [[1.0, 2.0]], seed: 42)
      end
    end

    test "control_selection can choose a subset of controls" do
      obs = [1.0, 2.0, 3.0, 4.0, 20.0, 21.0]
      control_signal = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
      control_constant = [10.0, 10.0, 10.0, 10.0, 0.0, 1.0]

      result =
        ModelBuilder.build_opts_with_controls(obs, [control_signal, control_constant],
          control_selection: [threshold: 0.8, max_controls: 1],
          control_selection_pre_period: {1, 4}
        )

      spec = result[:model_spec]
      assert %BstsNx.ModelSpec{} = spec
      # Trend (2) + 1 selected regressor = 3 state dims
      assert Nx.axis_size(spec.f, 0) == 3
    end

    test "control_selection can drop controls and keep seasonal structure" do
      obs = [1.0, 2.0, 3.0, 4.0]
      controls = [[5.0, 5.0, 5.0, 5.0]]

      result =
        ModelBuilder.build_opts_with_controls(obs, controls,
          seasonality: 3,
          control_selection: [threshold: 0.9]
        )

      spec = result[:model_spec]
      assert %BstsNx.ModelSpec{} = spec
      # Trend (2) + seasonal (2 for period 3) with no retained controls.
      assert Nx.axis_size(spec.f, 0) == 4
    end

    test "control_regression_mode uses in-loop spike-and-slab metadata" do
      obs = [1.0, 2.0, 3.0, 4.0]
      controls = [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]

      result =
        ModelBuilder.build_opts_with_controls(obs, controls,
          control_regression_mode: :spike_and_slab,
          control_regression_opts: [prior_inclusion: 0.2, g: 4.0]
        )

      spec = result[:model_spec]
      assert %BstsNx.ModelSpec{} = spec
      assert spec.regression.mode == :spike_and_slab
      assert spec.regression.prior_inclusion == 0.2
      assert spec.regression.g == 4.0
      # Trend q_specs only (level + slope); regression variances are not in q_specs.
      assert length(spec.q_specs) == 2
    end
  end

  describe "build_intervention_spec/3" do
    test "uses pre-period values for default initial level" do
      observations = [100.0, 101.0, 102.0, 10.0, 11.0, 12.0]

      {spec, :structured} =
        ModelBuilder.build_intervention_spec(observations, {4, 6}, seasonality: 3)

      assert Nx.to_flat_list(spec.x0) |> hd() == 10.0
    end

    test "preserves full-series regressor rows" do
      observations = [1.0, 2.0, 3.0, 4.0]
      regressors = Nx.tensor([[10.0], [20.0], [30.0], [40.0]])

      {spec, :structured} =
        ModelBuilder.build_intervention_spec(observations, {1, 2}, regressors: regressors)

      assert is_list(spec.h)
      assert length(spec.h) == 4
    end

    test "rejects regressors that do not cover the full series" do
      assert_raise ArgumentError, ~r/regressors row count/, fn ->
        ModelBuilder.build_intervention_spec([1.0, 2.0, 3.0], {1, 2},
          regressors: Nx.tensor([[1.0], [2.0]])
        )
      end
    end
  end

  describe "build_future_h/3" do
    test "produces per-step observation matrices" do
      regressors = Nx.tensor([[1.0], [2.0], [3.0]])
      {spec, :structured} = ModelBuilder.build_spec([1.0, 2.0, 3.0], regressors: regressors)

      future_reg = Nx.tensor([[4.0], [5.0]])
      future_h = ModelBuilder.build_future_h(spec, future_reg, 1)

      assert length(future_h) == 2

      Enum.each(future_h, fn h ->
        assert Nx.rank(h) == 2
        # Should span the full state dimension (trend 2 + regressor 1 = 3)
        assert Nx.axis_size(h, 1) == 3
      end)
    end

    test "preserves f64 precision in future regressor rows" do
      sentinel = 16_777_217.0
      regressors = Nx.tensor([[1.0], [2.0], [3.0]], type: {:f, 64})
      {spec, :structured} = ModelBuilder.build_spec([1.0, 2.0, 3.0], regressors: regressors)

      future_reg = Nx.tensor([[sentinel], [sentinel + 2.0]], type: {:f, 64})
      [first_h | _] = ModelBuilder.build_future_h(spec, future_reg, 1)

      assert Nx.type(first_h) == {:f, 64}
      assert List.last(Nx.to_flat_list(first_h)) == sentinel
    end

    test "rejects mismatched future regressor columns" do
      regressors = Nx.tensor([[1.0], [2.0], [3.0]])
      {spec, :structured} = ModelBuilder.build_spec([1.0, 2.0, 3.0], regressors: regressors)

      wrong_cols = Nx.tensor([[1.0, 2.0]])

      assert_raise ArgumentError, ~r/future_regressors columns/, fn ->
        ModelBuilder.build_future_h(spec, wrong_cols, 1)
      end
    end
  end
end
