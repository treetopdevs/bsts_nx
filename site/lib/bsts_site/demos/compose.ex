defmodule BstsSite.Demos.Compose do
  @moduledoc """
  The /engine/compose demo: build a `%BstsNx.ModelSpec{}` from component
  bricks and watch the composed transition matrix take shape.

  No sampling runs here — `BstsNx.Components.compose_specs/2` is pure
  matrix bookkeeping, so every chip click rebuilds the spec live in
  well under a millisecond (warm).
  """

  alias BstsNx.Components

  @trend_choices [:local_level, :local_linear_trend]
  @seasonal_choices [:none, 4, 7]
  @regression_choices [:none, :two_controls]

  # A small fixed {12, 2} regressor matrix: twelve weeks of two control
  # series (a paid-search index and a slow market swing). Only its shape
  # matters for composition — each row becomes the regression part of the
  # observation matrix H at that timestep.
  @regressors (for week <- 0..11 do
                 [
                   Float.round(1.0 + 0.1 * week, 2),
                   Float.round(1.0 + 0.5 * :math.sin(week / 2.0), 2)
                 ]
               end)

  def trend_choices, do: @trend_choices
  def seasonal_choices, do: @seasonal_choices
  def regression_choices, do: @regression_choices
  def regressors, do: @regressors

  @doc "Parses a chip value coming from the client back into a known choice."
  def parse_choice("trend", value) do
    parse_from(value, @trend_choices, :local_level)
  end

  def parse_choice("seasonal", value) do
    parse_from(value, @seasonal_choices, :none)
  end

  def parse_choice("regression", value) do
    parse_from(value, @regression_choices, :none)
  end

  defp parse_from(value, choices, default) do
    Enum.find(choices, default, fn choice -> to_string(choice) == value end)
  end

  @doc """
  Builds the composed spec for the current chip configuration and returns
  everything the page renders: the F matrix as lists, block outlines,
  state dimension, the variances the sampler would learn, and timing.
  """
  def build(trend, seasonal, regression) do
    t0 = System.monotonic_time(:microsecond)

    parts = component_parts(trend, seasonal, regression)

    spec =
      parts
      |> Enum.map(& &1.spec)
      |> Enum.reduce(fn next, acc -> Components.compose_specs(acc, next) end)

    elapsed_us = System.monotonic_time(:microsecond) - t0

    state_dim = Nx.axis_size(spec.f, 0)

    {blocks, _offset} =
      Enum.map_reduce(parts, 0, fn part, offset ->
        {%{from: offset, size: part.dim, label: part.label}, offset + part.dim}
      end)

    %{
      trend: trend,
      seasonal: seasonal,
      regression: regression,
      f_matrix: Nx.to_list(spec.f),
      state_dim: state_dim,
      blocks: blocks,
      part_labels: Enum.map(parts, & &1.label),
      variance_labels: Enum.flat_map(parts, & &1.variances),
      q_count: length(spec.q_specs),
      time_varying_h?: is_list(spec.h),
      h_steps: if(is_list(spec.h), do: length(spec.h)),
      execution: %{
        method_used: :compose_specs,
        elapsed_ms: max(Float.round(elapsed_us / 1000, 1), 0.1)
      }
    }
  end

  # One entry per active component: the spec, its state dimension, a short
  # label for the block outline, and the names of the variances it asks the
  # sampler to learn (one per q_spec entry, in order).
  defp component_parts(trend, seasonal, regression) do
    trend_part =
      case trend do
        :local_level ->
          %{
            spec: Components.local_level_spec(process_var: 0.5),
            dim: 1,
            label: "local level",
            variances: ["level noise"]
          }

        :local_linear_trend ->
          %{
            spec: Components.local_linear_trend_spec(var_level: 0.5, var_slope: 0.05),
            dim: 2,
            label: "local linear trend",
            variances: ["level noise", "slope noise"]
          }
      end

    seasonal_part =
      case seasonal do
        :none ->
          []

        s when s in [4, 7] ->
          [
            %{
              spec: Components.seasonal_spec(s, process_var: 0.1),
              dim: s - 1,
              label: "seasonal-#{s}",
              variances: ["seasonal innovation"]
            }
          ]
      end

    regression_part =
      case regression do
        :none ->
          []

        :two_controls ->
          [
            %{
              spec: Components.regression_spec(@regressors, var_beta: 0.01),
              dim: 2,
              label: "regression (2 controls)",
              variances: ["β₁ drift", "β₂ drift"]
            }
          ]
      end

    [trend_part] ++ seasonal_part ++ regression_part
  end

  @doc "The code the current configuration runs, shown in the under-the-hood panel."
  def code_snippet(demo) do
    trend_line =
      case demo.trend do
        :local_level ->
          "trend = BstsNx.Components.local_level_spec(process_var: 0.5)"

        :local_linear_trend ->
          "trend = BstsNx.Components.local_linear_trend_spec(var_level: 0.5, var_slope: 0.05)"
      end

    seasonal_line =
      case demo.seasonal do
        :none -> nil
        s -> "seasonal = BstsNx.Components.seasonal_spec(#{s}, process_var: 0.1)"
      end

    regression_line =
      case demo.regression do
        :none ->
          nil

        :two_controls ->
          "# regressors is a fixed {12, 2} matrix of two control series\n" <>
            "regression = BstsNx.Components.regression_spec(regressors, var_beta: 0.01)"
      end

    compose_lines =
      case Enum.reject([seasonal: seasonal_line, regression: regression_line], fn {_, l} ->
             is_nil(l)
           end) do
        [] ->
          "spec = trend"

        parts ->
          pipes =
            Enum.map_join(parts, "\n", fn {name, _} ->
              "  |> BstsNx.Components.compose_specs(#{name})"
            end)

          "spec =\n  trend\n" <> pipes
      end

    [trend_line, seasonal_line, regression_line]
    |> Enum.reject(&is_nil/1)
    |> Enum.join("\n")
    |> Kernel.<>("\n\n")
    |> Kernel.<>(compose_lines)
    |> Kernel.<>("""


    Nx.shape(spec.f)      #=> {#{demo.state_dim}, #{demo.state_dim}}
    length(spec.q_specs)  #=> #{demo.q_count}  (+ 1 observation variance)
    """)
  end
end
