defmodule BstsNx.ValidationWorkflowTest do
  use ExUnit.Case, async: true
  alias BstsNx.Validation

  test "assembles evidence with caller indices, coverage options, and ordered refits" do
    actual = Nx.tensor([10.0, 12.0, 14.0, 16.0])
    baseline = Nx.tensor([10.0, 11.0, 12.0, 13.0])
    variances = Nx.tensor([1.0, 1.0, 1.0, 1.0])
    h = Nx.tensor([1.0, 0.0, 2.0, 1.0])
    indices = [0, 2, 3]
    sessions = Enum.to_list(1..8)

    result =
      Validation.evaluate(%{
        actual: actual,
        baseline: baseline,
        indices: indices,
        coverage: %{state_variances: variances, obs_variance: 0.0, h: h, confidence_level: 0.8},
        placebo: %{
          sessions: sessions,
          on_air_indices: [6, 7],
          estimate_fn: fn obs, idx ->
            send(self(), {:refit, :placebo, obs, idx})
            %{lift_sessions: 0.0, lift_pct: 0.0, lift_ci95: %{lower: -1.0, upper: 1.0, sd: 1.0}}
          end
        },
        effect_stability: %{
          main_lift: 10.0,
          window: 10,
          delta: 2,
          estimate_at_window_fn: fn window ->
            send(self(), {:refit, :stability, window})
            window * 1.0
          end
        }
      })

    calls =
      for _ <- 1..3 do
        assert_receive call when elem(call, 0) == :refit
        call
      end

    assert calls == [
             {:refit, :placebo, sessions, [2, 3]},
             {:refit, :stability, 8},
             {:refit, :stability, 12}
           ]

    refute_received {:refit, _, _}
    refute_received {:refit, _, _, _}
    assert Nx.to_flat_list(result.residuals) == [0.0, 1.0, 2.0, 3.0]

    assert result.details.prediction_error ==
             Validation.prediction_error(actual, baseline, indices)

    assert result.details.coverage ==
             Validation.coverage(actual, baseline, variances, 0.0, indices, h,
               confidence_level: 0.8
             )

    assert result.details.durbin_watson == Validation.durbin_watson(result.residuals, indices)
    assert result.details.placebo.is_significant == false
    assert result.details.effect_stability.window_low == 8
    assert result.details.effect_stability.window_high == 12

    assert result.verdicts == %{
             prediction_error: :warn,
             coverage: :fail,
             durbin_watson: :fail,
             placebo: :pass,
             effect_stability: :warn
           }

    assert result.verdicts == Validation.assess(result.details)
  end

  test "omitted checks and empty evaluation indices remain skipped" do
    result =
      Validation.evaluate(%{
        actual: Nx.tensor([1.0]),
        baseline: Nx.tensor([1.0]),
        indices: [],
        coverage: nil,
        placebo: nil
      })

    assert Enum.all?(result.details, fn {_name, detail} -> is_nil(detail) end)
    assert Enum.all?(result.verdicts, fn {_name, verdict} -> verdict == :skip end)
  end

  test "a short placebo period skips its callback and default stability windows are retained" do
    result =
      Validation.evaluate(%{
        actual: Nx.tensor([1.0]),
        baseline: Nx.tensor([1.0]),
        indices: [0],
        placebo: %{
          sessions: [1.0, 2.0],
          on_air_indices: [1],
          estimate_fn: fn _, _ -> flunk("unexpected refit") end
        },
        effect_stability: %{main_lift: 1.0, window: 6, estimate_at_window_fn: fn _ -> 1.0 end}
      })

    assert result.details.placebo == nil
    assert result.verdicts.placebo == :skip
    assert result.details.effect_stability.window_low == 5
    assert result.details.effect_stability.window_high == 11
    assert result.verdicts.effect_stability == :pass
  end

  test "rejects incompatible vectors and indices before refitting" do
    request = %{actual: Nx.tensor([1.0, 2.0]), baseline: Nx.tensor([1.0, 2.0]), indices: [0]}

    for change <- [
          %{baseline: Nx.tensor([1.0])},
          %{indices: [-1]},
          %{indices: [2]},
          %{indices: [0.5]},
          %{actual: Nx.tensor([[1.0, 2.0]])}
        ] do
      assert_raise ArgumentError, fn -> Validation.evaluate(Map.merge(request, change)) end
    end
  end

  test "invalid coverage configuration is rejected even with empty indices" do
    request = %{actual: Nx.tensor([1.0]), baseline: Nx.tensor([1.0]), indices: []}
    valid = %{state_variances: Nx.tensor([1.0]), obs_variance: 1.0}

    for change <- [
          %{state_variances: :invalid},
          %{state_variances: Nx.tensor([1.0, 2.0])},
          %{obs_variance: :invalid},
          %{h: Nx.tensor([1.0, 2.0])},
          %{confidence_level: 2.0}
        ] do
      assert_raise ArgumentError, fn ->
        Validation.evaluate(Map.put(request, :coverage, Map.merge(valid, change)))
      end
    end

    assert Validation.evaluate(Map.put(request, :coverage, valid)).verdicts.coverage == :skip
  end

  test "malformed optional checks fail and callback exceptions propagate" do
    request = %{actual: Nx.tensor([1.0]), baseline: Nx.tensor([1.0]), indices: [0]}

    for check <- [:coverage, :placebo, :effect_stability] do
      assert_raise KeyError, fn -> Validation.evaluate(Map.put(request, check, %{})) end
    end

    assert_raise RuntimeError, "refit failed", fn ->
      Validation.evaluate(
        Map.put(request, :effect_stability, %{
          main_lift: 1.0,
          window: 10,
          estimate_at_window_fn: fn _ -> raise "refit failed" end
        })
      )
    end
  end
end
