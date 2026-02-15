defmodule BstsNx.CausalImpactFilterTest do
  use ExUnit.Case, async: true

  alias BstsNx.CausalImpact

  test "estimate_from_filter detects a constant uplift" do
    # 100 time steps: baseline of 10.0, then a +5.0 uplift on indices 50-79
    on_air = Enum.to_list(50..79)

    obs =
      Enum.map(0..99, fn idx ->
        if idx in on_air, do: 15.0, else: 10.0
      end)

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
end
