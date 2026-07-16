defmodule BstsNx.InterventionAnalysisTest do
  use ExUnit.Case, async: true

  alias BstsNx.Components
  alias BstsNx.InterventionAnalysis

  describe "analyze/3 with simple local level" do
    test "detects positive intervention effect" do
      :rand.seed(:exsss, {100, 101, 102})
      pre = Enum.map(1..30, fn _ -> 50.0 + :rand.normal() * 2 end)
      post = Enum.map(1..10, fn _ -> 55.0 + :rand.normal() * 2 end)
      obs = pre ++ post

      result =
        InterventionAnalysis.analyze(
          obs,
          %{pre_period: {1, 30}, post_period: {31, 40}},
          num_samples: 20,
          burn_in: 10,
          seed: 42
        )

      assert is_map(result.summary)
      assert is_boolean(result.significant?)
      assert result.alpha == 0.05
      assert result.impact != nil
      assert result.execution.mode == :bayesian
      assert result.execution.method_used == :elixir_mcmc

      # The cumulative effect should be positive (around 5 * 10 = 50)
      assert result.summary.cumulative_effect.mean > 0
    end

    test "no effect when pre and post are the same" do
      :rand.seed(:exsss, {200, 201, 202})
      pre = Enum.map(1..30, fn _ -> 50.0 + :rand.normal() * 2 end)
      post = Enum.map(1..10, fn _ -> 50.0 + :rand.normal() * 2 end)
      obs = pre ++ post

      result =
        InterventionAnalysis.analyze(
          obs,
          %{pre_period: {1, 30}, post_period: {31, 40}},
          num_samples: 20,
          burn_in: 10,
          seed: 42
        )

      # Effect should be near zero (within noise)
      assert abs(result.summary.cumulative_effect.mean) < 50
    end

    test "mcmc summaries honor non-default alpha" do
      :rand.seed(:exsss, {210, 211, 212})
      pre = Enum.map(1..30, fn _ -> 50.0 + :rand.normal() * 2 end)
      post = Enum.map(1..10, fn _ -> 55.0 + :rand.normal() * 2 end)
      obs = pre ++ post

      wide =
        InterventionAnalysis.analyze(
          obs,
          %{pre_period: {1, 30}, post_period: {31, 40}},
          num_samples: 30,
          burn_in: 5,
          seed: 42,
          alpha: 0.01
        )

      narrow =
        InterventionAnalysis.analyze(
          obs,
          %{pre_period: {1, 30}, post_period: {31, 40}},
          num_samples: 30,
          burn_in: 5,
          seed: 42,
          alpha: 0.20
        )

      wide_width = wide.summary.cumulative_effect.upper - wide.summary.cumulative_effect.lower

      narrow_width =
        narrow.summary.cumulative_effect.upper - narrow.summary.cumulative_effect.lower

      assert wide.alpha == 0.01
      assert narrow.alpha == 0.20
      assert wide_width >= narrow_width
    end
  end

  describe "analyze/3 with seasonality" do
    test "handles seasonal model" do
      :rand.seed(:exsss, {300, 301, 302})

      # Weekly seasonality
      pre =
        Enum.map(1..28, fn i ->
          seasonal = :math.sin(2 * :math.pi() * i / 7) * 5
          50.0 + seasonal + :rand.normal()
        end)

      post =
        Enum.map(29..35, fn i ->
          seasonal = :math.sin(2 * :math.pi() * i / 7) * 5
          60.0 + seasonal + :rand.normal()
        end)

      obs = pre ++ post

      result =
        InterventionAnalysis.analyze(
          obs,
          %{pre_period: {1, 28}, post_period: {29, 35}},
          seasonality: 7,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert is_map(result.summary)
      assert result.model_spec != nil
    end
  end

  describe "analyze_at/2" do
    test "auto-detects periods from intervention start" do
      :rand.seed(:exsss, {400, 401, 402})
      pre = Enum.map(1..20, fn _ -> 50.0 + :rand.normal() end)
      post = Enum.map(1..10, fn _ -> 55.0 + :rand.normal() end)
      obs = pre ++ post

      result =
        InterventionAnalysis.analyze_at(obs,
          intervention_start: 21,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert result.summary.cumulative_effect.mean > 0
    end
  end

  describe "analyze/3 with filter method" do
    test "filter-based estimation works" do
      :rand.seed(:exsss, {500, 501, 502})
      pre = Enum.map(1..30, fn _ -> 50.0 + :rand.normal() end)
      post = Enum.map(1..10, fn _ -> 55.0 + :rand.normal() end)
      obs = pre ++ post

      result =
        InterventionAnalysis.analyze(
          obs,
          %{pre_period: {1, 30}, post_period: {31, 40}},
          method: :filter
        )

      assert result.impact == nil
      assert is_map(result.summary)
      assert result.summary.cumulative_effect.mean > 0
      assert result.execution.mode == :operational
      assert result.execution.method_used == :scalar_forecast_filter
    end

    test "default initial state only uses the configured pre-period" do
      config = %{pre_period: {3, 5}, post_period: {6, 7}}
      opts = [method: :filter, q: 0.1, r: 0.1]
      nan = Nx.to_number(Nx.Constants.nan())

      low_prefix =
        InterventionAnalysis.analyze(
          [-1_000.0, -1_000.0, nan, 10.0, 10.0, 12.0, 12.0],
          config,
          opts
        )

      high_prefix =
        InterventionAnalysis.analyze(
          [1_000.0, 1_000.0, nan, 10.0, 10.0, 12.0, 12.0],
          config,
          opts
        )

      assert_in_delta low_prefix.summary.cumulative_effect.mean,
                      high_prefix.summary.cumulative_effect.mean,
                      1.0e-10
    end

    test "method mcmc compatibility alias runs bayesian lane" do
      :rand.seed(:exsss, {510, 511, 512})
      pre = Enum.map(1..20, fn _ -> 50.0 + :rand.normal() end)
      post = Enum.map(1..5, fn _ -> 55.0 + :rand.normal() end)
      obs = pre ++ post

      result =
        InterventionAnalysis.analyze(
          obs,
          %{pre_period: {1, 20}, post_period: {21, 25}},
          method: :mcmc,
          num_samples: 5,
          burn_in: 2,
          seed: 42
        )

      assert result.impact != nil
      assert result.execution.mode == :bayesian
      assert result.execution.method_used == :elixir_mcmc
    end
  end

  describe "report/1" do
    test "generates readable report" do
      :rand.seed(:exsss, {600, 601, 602})
      pre = Enum.map(1..20, fn _ -> 50.0 + :rand.normal() end)
      post = Enum.map(1..5, fn _ -> 55.0 + :rand.normal() end)
      obs = pre ++ post

      result =
        InterventionAnalysis.analyze(
          obs,
          %{pre_period: {1, 20}, post_period: {21, 25}},
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      report = InterventionAnalysis.report(result)
      assert is_binary(report)
      assert String.contains?(report, "Intervention Analysis Report")
      assert String.contains?(report, "Cumulative effect")
    end
  end

  describe "analyze/3 with :control_series" do
    test "produces structured result with control series" do
      :rand.seed(:exsss, {700, 701, 702})
      # Control series correlated with response
      control = Enum.map(1..35, fn _ -> 50.0 + :rand.normal() * 2 end)
      pre = Enum.map(Enum.take(control, 28), fn c -> c + :rand.normal() end)
      post = Enum.map(Enum.drop(control, 28), fn c -> c + 10.0 + :rand.normal() end)
      obs = pre ++ post

      result =
        InterventionAnalysis.analyze(
          obs,
          %{pre_period: {1, 28}, post_period: {29, 35}},
          control_series: [control],
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert result.model_spec != nil
      assert is_map(result.summary)
      # Model should have regression state dimensions
      assert Nx.axis_size(result.model_spec.f, 0) >= 3
    end

    test "control_series with seasonality composes all components" do
      :rand.seed(:exsss, {710, 711, 712})
      control = Enum.map(1..35, fn _ -> 50.0 + :rand.normal() * 2 end)
      pre = Enum.map(1..28, fn i -> Enum.at(control, i - 1) + :math.sin(i / 3.5) * 2 end)
      post = Enum.map(29..35, fn i -> Enum.at(control, i - 1) + 10.0 end)
      obs = pre ++ post

      result =
        InterventionAnalysis.analyze(
          obs,
          %{pre_period: {1, 28}, post_period: {29, 35}},
          control_series: [control],
          seasonality: 7,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert result.model_spec != nil
      # Trend (2) + seasonal (6 for period 7) + 1 regressor = 9
      assert Nx.axis_size(result.model_spec.f, 0) == 9
    end

    test "control_selection screens controls before composing the model" do
      control_signal = Enum.map(1..35, &(&1 * 1.0))
      control_constant = List.duplicate(10.0, 35)
      pre = Enum.map(1..28, fn i -> Enum.at(control_signal, i - 1) + 0.05 * :math.sin(i) end)
      post = Enum.map(29..35, fn i -> Enum.at(control_signal, i - 1) + 10.0 end)
      obs = pre ++ post

      result =
        InterventionAnalysis.analyze(
          obs,
          %{pre_period: {1, 28}, post_period: {29, 35}},
          control_series: [control_signal, control_constant],
          control_selection: [threshold: 0.7, max_controls: 1],
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert result.model_spec != nil
      # Trend (2) + 1 selected regressor = 3
      assert Nx.axis_size(result.model_spec.f, 0) == 3
    end

    test "control_regression_mode enables in-loop spike-and-slab regression" do
      :rand.seed(:exsss, {720, 721, 722})
      n = 40
      control1 = Enum.map(1..n, fn _ -> :rand.normal() end)
      control2 = Enum.map(1..n, fn _ -> :rand.normal() end)
      pre = Enum.map(1..30, fn i -> 3.0 * Enum.at(control1, i - 1) + :rand.normal() * 0.2 end)

      post =
        Enum.map(31..40, fn i -> 3.0 * Enum.at(control1, i - 1) + 4.0 + :rand.normal() * 0.2 end)

      obs = pre ++ post

      result =
        InterventionAnalysis.analyze(
          obs,
          %{pre_period: {1, 30}, post_period: {31, 40}},
          control_series: [control1, control2],
          control_regression_mode: :spike_and_slab,
          control_regression_opts: [prior_inclusion: 0.2, g: 30.0],
          num_samples: 20,
          burn_in: 20,
          seed: 4242
        )

      assert result.model_spec != nil
      assert result.model_spec.regression.mode == :spike_and_slab
      assert is_map(result.summary)
    end
  end

  describe "validation" do
    test "rejects empty observations" do
      assert_raise ArgumentError, ~r/non-empty/, fn ->
        InterventionAnalysis.analyze([], %{pre_period: {1, 5}, post_period: {6, 10}})
      end
    end

    test "rejects invalid alpha" do
      assert_raise ArgumentError, ~r/alpha/, fn ->
        InterventionAnalysis.analyze(
          [1.0, 2.0, 3.0],
          %{pre_period: {1, 2}, post_period: {3, 3}},
          alpha: 0.0
        )
      end
    end

    test "rejects invalid period" do
      assert_raise ArgumentError, ~r/pre_period/, fn ->
        InterventionAnalysis.analyze(
          [1.0, 2.0, 3.0],
          %{pre_period: {5, 2}, post_period: {6, 8}}
        )
      end
    end

    test "allows custom model_spec even when seasonality option is invalid" do
      spec = Components.local_level_spec(initial_state: 1.0, initial_cov: 1.0)

      result =
        InterventionAnalysis.analyze(
          [1.0, 1.1, 1.2, 1.4, 1.5],
          %{pre_period: {1, 3}, post_period: {4, 5}},
          model_spec: spec,
          seasonality: 1,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert result.model_spec != nil
      assert Nx.all_close(result.model_spec.f, spec.f)
    end

    test "rejects unsupported method" do
      assert_raise ArgumentError, ~r/method/, fn ->
        InterventionAnalysis.analyze(
          [1.0, 2.0, 3.0],
          %{pre_period: {1, 2}, post_period: {3, 3}},
          method: :unsupported
        )
      end
    end

    test "accepts control_series with filter method (fast structured path)" do
      result =
        InterventionAnalysis.analyze(
          [1.0, 2.0, 3.0, 4.0, 5.0],
          %{pre_period: {1, 3}, post_period: {4, 5}},
          control_series: [[10.0, 20.0, 30.0, 40.0, 50.0]],
          method: :filter
        )

      assert result.model_spec != nil
      assert is_map(result.summary)
      assert result.execution.fallback? == false
    end

    test "rejects unknown analysis options before they can be silently dropped" do
      assert_raise ArgumentError, ~r/unknown option.*:surprise/, fn ->
        InterventionAnalysis.analyze(
          [1.0, 2.0, 3.0, 4.0],
          %{pre_period: {1, 2}, post_period: {3, 4}},
          num_samples: 1,
          burn_in: 0,
          seed: 42,
          surprise: true
        )
      end
    end
  end
end
