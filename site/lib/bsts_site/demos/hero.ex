defmodule BstsSite.Demos.Hero do
  @moduledoc """
  The hub hero demo: the Operational fast lane answering the site's opening
  question on every slider drag.

  `prepare/0` precomputes model artifacts once per LiveView mount;
  `run/3` re-runs the counterfactual in single-digit milliseconds.
  """

  alias BstsNx.{Components, Operational}
  alias BstsSite.Demos.Scenarios

  # Slider bounds are the server-side contract: values outside are clamped.
  @lift_range 0..25
  @noise_range 1..12

  def lift_range, do: @lift_range
  def noise_range, do: @noise_range

  @doc """
  Prepared artifacts keyed by the noise slider: the model's observation
  variance tracks the noise the visitor dials in, so cranking up noise
  genuinely widens the interval instead of just roughing up the line.
  Preparation is ~7 ms, so building all twelve up front is trivial.
  """
  def prepare do
    Map.new(@noise_range, fn noise_sd ->
      spec =
        Components.local_level_spec(
          initial_state: 84.0,
          initial_cov: 10.0,
          process_var: 0.8,
          obs_var: noise_sd * noise_sd * 1.0
        )

      {noise_sd, Operational.prepare(spec, {1, 96}, {97, 144})}
    end)
  end

  def run(prepared_by_noise, lift_param, noise_param) do
    lift = clamp(lift_param, @lift_range)
    noise_sd = clamp(noise_param, @noise_range)
    prepared = Map.fetch!(prepared_by_noise, noise_sd)
    scenario = Scenarios.hero(lift, noise_sd)

    result = Operational.run(prepared, scenario.observations, [], return: :lists)

    band_z = 1.96

    band =
      Enum.zip_with(
        [result.summary.baseline, result.counterfactual.variance],
        fn [mean, var] ->
          sd = :math.sqrt(max(var, 0.0))
          {mean - band_z * sd, mean + band_z * sd}
        end
      )

    %{
      scenario: scenario,
      lift: lift,
      noise_sd: noise_sd,
      observations: scenario.observations,
      counterfactual_mean: result.summary.baseline,
      band_lower: Enum.map(band, &elem(&1, 0)),
      band_upper: Enum.map(band, &elem(&1, 1)),
      cumulative: result.summary.cumulative_effect,
      relative: result.summary.relative_effect,
      execution: result.execution,
      significant?: result.summary.cumulative_effect.lower > 0
    }
  end

  defp clamp(value, range) when is_integer(value) do
    value |> max(range.first) |> min(range.last)
  end

  defp clamp(value, range) when is_binary(value) do
    case Integer.parse(value) do
      {n, _} -> clamp(n, range)
      :error -> range.first
    end
  end

  defp clamp(_value, range), do: range.first

  @doc "The code this demo runs, shown verbatim in the under-the-hood panel."
  def code_snippet do
    ~S"""
    # Prepare once (7 ms) — reused across every slider drag.
    spec =
      BstsNx.Components.local_level_spec(
        initial_state: 84.0,
        initial_cov: 10.0,
        process_var: 0.8,
        obs_var: 16.0  # tracks the noise slider: sd²
      )

    prepared = BstsNx.Operational.prepare(spec, {1, 96}, {97, 144})

    # Re-run per interaction (~6 ms warm). The baseline is forecast-first:
    # the model never sees post-period outcomes when projecting them.
    result = BstsNx.Operational.run(prepared, observations, [], return: :lists)

    result.summary.cumulative_effect
    #=> %{mean: ..., lower: ..., upper: ..., sd: ...}
    result.execution.method_used
    #=> :scalar_forecast_filter
    """
  end
end
