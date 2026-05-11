defmodule BstsNx.CausalImpactFilterTest do
  use ExUnit.Case, async: true

  alias BstsNx.CausalImpact
  alias BstsNx.Components
  alias BstsNx.TestHelpers

  @emlx_backend? match?(EMLX.Backend, Nx.default_backend()) or
                   match?({EMLX.Backend, _}, Nx.default_backend())

  test "estimate_from_filter detects a constant uplift" do
    # 100 time steps: baseline of 10.0, then a +5.0 uplift on indices 50-79
    on_air = Enum.to_list(50..79)
    obs = TestHelpers.uplift_series(100, on_air, 10.0, 5.0)

    result =
      CausalImpact.estimate_from_filter(obs, on_air,
        q: 1.0e-3,
        r: 1.0e-3,
        x0: 10.0,
        p0: 1.0
      )

    assert is_map(result)
    assert is_list(result.actual)
    assert is_list(result.baseline)
    assert length(result.actual) == 30
    assert length(result.baseline) == 30
    assert length(result.baseline_variance) == 30
    assert result.obs_variance > 0.0

    # All actual values should be 15.0
    Enum.each(result.actual, fn a -> assert_in_delta(a, 15.0, 0.1) end)

    # Cumulative lift should be approximately 30 * 5 = 150
    assert_in_delta(result.cumulative_effect.mean, 150.0, 10.0)
    assert result.cumulative_effect.mean > 0.0

    # CI should bracket the true effect
    assert result.cumulative_effect.lower < 150.0
    assert result.cumulative_effect.upper > 150.0

    # Relative effect: 150/300 = 0.5
    assert_in_delta(result.relative_effect.mean, 0.5, 0.1)
  end

  test "estimate_from_filter with no effect returns near-zero lift" do
    # Flat series with no treatment effect
    obs = List.duplicate(10.0, 100)
    on_air = Enum.to_list(50..79)

    result =
      CausalImpact.estimate_from_filter(obs, on_air,
        q: 1.0e-3,
        r: 1.0e-3,
        x0: 10.0,
        p0: 1.0
      )

    # No effect: cumulative should be near zero
    assert_in_delta(result.cumulative_effect.mean, 0.0, 1.0)
  end

  test "estimate_from_filter accepts Nx tensor input" do
    obs = Nx.tensor(Enum.map(0..49, fn _ -> 10.0 end), type: {:f, 32})
    on_air = Enum.to_list(20..29)

    result = CausalImpact.estimate_from_filter(obs, on_air, q: 0.1, r: 0.5, x0: 10.0, p0: 1.0)

    assert is_map(result)
    assert length(result.actual) == 10
  end

  test "estimate_from_filter point effects have correct structure" do
    obs = Enum.map(0..49, fn idx -> if idx >= 30, do: 12.0, else: 10.0 end)
    on_air = Enum.to_list(30..49)

    result =
      CausalImpact.estimate_from_filter(obs, on_air,
        q: 1.0e-3,
        r: 1.0e-3,
        x0: 10.0,
        p0: 1.0
      )

    assert is_list(result.point_effects.mean)
    assert is_list(result.point_effects.lower)
    assert is_list(result.point_effects.upper)
    assert length(result.point_effects.mean) == 20

    # Each lower should be < mean < upper (for non-zero effects)
    Enum.zip([result.point_effects.lower, result.point_effects.mean, result.point_effects.upper])
    |> Enum.each(fn {l, m, u} ->
      assert l <= m, "lower (#{l}) should be <= mean (#{m})"
      assert m <= u, "mean (#{m}) should be <= upper (#{u})"
    end)
  end

  test "estimate_from_filter ignores out-of-range intervention indices" do
    obs = Enum.map(0..9, fn _ -> 10.0 end)
    on_air = [-1, 2, 3, 99]

    result =
      CausalImpact.estimate_from_filter(obs, on_air,
        q: 1.0e-3,
        r: 1.0e-3,
        x0: 10.0,
        p0: 1.0
      )

    # Only valid indices (2 and 3) should be used.
    assert length(result.actual) == 2
    assert length(result.baseline) == 2
    assert length(result.point_effects.mean) == 2
  end

  test "estimate_from_filter cumulative variance includes observation noise" do
    obs = List.duplicate(10.0, 20)
    on_air = Enum.to_list(10..14)

    result =
      CausalImpact.estimate_from_filter(obs, on_air,
        q: 0.0,
        r: 1.0e-3,
        x0: 10.0,
        p0: 0.0
      )

    assert_in_delta result.cumulative_effect.mean, 0.0, 1.0e-10
    assert_in_delta result.cumulative_effect.sd, :math.sqrt(5.0e-3), 1.0e-10
  end

  if @emlx_backend? do
    @tag skip:
           "EMLX backend does not implement Nx.Backend.lu/3 required by structured filter smoothing"
  end

  test "estimate_structured_from_filter returns structured summary fields" do
    observations = TestHelpers.uplift_series(60, Enum.to_list(40..49), 20.0, 4.0)
    intervention_indices = Enum.to_list(40..49)
    spec = Components.local_level_spec(initial_state: 20.0, initial_cov: 1.0, process_var: 0.1)

    result =
      CausalImpact.estimate_structured_from_filter(observations, intervention_indices, spec,
        alpha: 0.05
      )

    assert is_map(result.point_effects)
    assert length(result.actual) == 10
    assert length(result.baseline) == 10
    assert result.cumulative_effect.cross_cov_included == false
    assert length(result.baseline_variance) == 10
    assert is_float(result.obs_variance)
  end

  if @emlx_backend? do
    @tag skip:
           "EMLX backend does not implement Nx.Backend.lu/3 required by structured filter smoothing"
  end

  test "estimate_structured_from_filter uses ModelSpec initial state and covariance" do
    observations = List.duplicate(50.0, 12)
    intervention_indices = Enum.to_list(8..11)

    anchored_spec =
      Components.local_level_spec(
        initial_state: 50.0,
        initial_cov: 1.0e-6,
        process_var: 1.0e-9,
        obs_var: 1.0e-6
      )

    diffuse_wrong_spec =
      Components.local_level_spec(
        initial_state: 0.0,
        initial_cov: 1.0e-6,
        process_var: 1.0e-9,
        obs_var: 1.0e-6
      )

    anchored =
      CausalImpact.estimate_structured_from_filter(
        observations,
        intervention_indices,
        anchored_spec
      )

    diffuse_wrong =
      CausalImpact.estimate_structured_from_filter(
        observations,
        intervention_indices,
        diffuse_wrong_spec
      )

    assert abs(Enum.at(anchored.baseline, 0) - 50.0) < 1.0
    assert abs(Enum.at(diffuse_wrong.baseline, 0) - 50.0) > 5.0
  end

  if @emlx_backend? do
    @tag skip:
           "EMLX backend does not implement Nx.Backend.lu/3 required by structured filter smoothing"
  end

  test "estimate_structured_from_filter accepts rank-1 static H tensors" do
    observations = TestHelpers.uplift_series(50, Enum.to_list(35..44), 15.0, 3.0)
    intervention_indices = Enum.to_list(35..44)

    spec =
      Components.local_linear_trend_spec(
        initial_level: 15.0,
        initial_slope: 0.0,
        initial_cov_level: 1.0,
        initial_cov_slope: 1.0,
        var_level: 0.1,
        var_slope: 0.01
      )
      |> Map.put(:h, Nx.tensor([1.0, 0.0]))

    result =
      CausalImpact.estimate_structured_from_filter(observations, intervention_indices, spec)

    assert length(result.actual) == 10
    assert length(result.baseline) == 10
    assert length(result.point_effects.mean) == 10
  end

  if @emlx_backend? do
    @tag skip:
           "EMLX backend does not implement Nx.Backend.lu/3 required by structured filter smoothing"
  end

  test "estimate_structured_from_filter accepts rank-2 time-varying H tensors" do
    observations = Enum.map(1..8, &(&1 * 1.0))
    intervention_indices = [5, 6]

    spec =
      Components.local_linear_trend_spec(
        initial_level: 1.0,
        initial_slope: 1.0,
        initial_cov_level: 1.0,
        initial_cov_slope: 1.0,
        var_level: 0.1,
        var_slope: 0.01
      )
      |> Map.put(:h, Nx.tensor(Enum.map(1..8, fn i -> [1.0, i / 10] end), type: {:f, 64}))

    result =
      CausalImpact.estimate_structured_from_filter(observations, intervention_indices, spec)

    assert length(result.actual) == 2
    assert length(result.baseline) == 2
    assert length(result.point_effects.mean) == 2
  end
end
