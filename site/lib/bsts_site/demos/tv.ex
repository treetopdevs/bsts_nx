defmodule BstsSite.Demos.Tv do
  @moduledoc """
  The TV attribution demo: three spot windows inside a 48-hour post period,
  each with a planted adstock-style pulse, credited by the Operational
  fast lane with Shapley splitting wherever windows overlap.

  `prepare/0` builds the model artifacts once per LiveView mount;
  `run/5` regenerates the scenario and re-attributes on every slider drag.
  """

  alias BstsNx.{Components, Operational}
  alias BstsSite.Demos.Scenarios

  @total 144
  @pre_end 96
  @post_hours 48
  @window_hours 8
  @decay_tau 2.5
  @level 90.0
  @noise_sd 4.0
  @seed 25

  # {id, short label, share of the strength slider}
  @spot_defs [{"Spot A", "A", 1.0}, {"Spot B", "B", 0.7}, {"Spot C", "C", 0.4}]

  # Slider bounds are the server-side contract: values outside are clamped.
  # A window must fit inside the post period: start + 8 <= 48.
  @start_range 0..(@post_hours - @window_hours)
  @strength_range 8..30

  def start_range, do: @start_range
  def strength_range, do: @strength_range
  def window_hours, do: @window_hours
  def post_hours, do: @post_hours
  def pre_end, do: @pre_end
  def noise_sd, do: @noise_sd

  def defaults, do: %{start_a: 6, start_b: 18, start_c: 24, strength: 20}

  @doc """
  Prepared Operational artifacts. The noise level is fixed for this page,
  so a single prepared model serves every interaction.
  """
  def prepare do
    spec =
      Components.local_level_spec(
        initial_state: @level,
        initial_cov: 10.0,
        process_var: 0.5,
        obs_var: @noise_sd * @noise_sd
      )

    Operational.prepare(spec, {1, @pre_end}, {@pre_end + 1, @total})
  end

  @doc """
  Rebuilds the scenario from the slider values, runs the prepared
  counterfactual, and attributes lift to the three spot windows.

  Windows are 0-based half-open within the post period and may overlap;
  the library's SpotAttributor Shapley-splits any shared hours.
  """
  def run(prepared, start_a, start_b, start_c, strength) do
    starts = Enum.map([start_a, start_b, start_c], &clamp(&1, @start_range))
    strength = clamp(strength, @strength_range)

    spots =
      @spot_defs
      |> Enum.zip(starts)
      |> Enum.map(fn {{id, short, share}, start} ->
        %{
          id: id,
          short: short,
          window_start: start,
          window_end: start + @window_hours,
          peak: strength * share
        }
      end)

    # Planted truth: each spot adds a decaying pulse (peak * exp(-k/tau))
    # over its own 8-hour window. Overlapping pulses add up in the data.
    post_effects =
      for t <- 0..(@post_hours - 1) do
        Enum.reduce(spots, 0.0, fn s, acc -> acc + pulse(t, s.window_start, s.peak) end)
      end

    observations =
      (List.duplicate(0.0, @pre_end) ++ post_effects)
      |> Enum.zip_with(Scenarios.noise(@total, @noise_sd, @seed), fn effect, e ->
        @level + effect + e
      end)

    lib_spots = Enum.map(spots, &Map.take(&1, [:id, :window_start, :window_end]))
    result = Operational.run(prepared, observations, lib_spots, return: :lists)

    attr_by_id = Map.new(result.attributions.attributions, &{&1.spot_id, &1})

    spots =
      Enum.map(spots, fn spot ->
        attr = Map.fetch!(attr_by_id, spot.id)

        Map.merge(spot, %{
          lift: attr.lift,
          lift_sd: attr.lift_sd,
          lift_lower: attr.lift_lower,
          lift_upper: attr.lift_upper,
          p_positive: attr.p_positive,
          truth: spot.peak * pulse_sum(),
          earned?: attr.lift_lower > 0
        })
      end)

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
      strength: strength,
      spots: spots,
      observations: observations,
      counterfactual_mean: result.summary.baseline,
      band_lower: Enum.map(band, &elem(&1, 0)),
      band_upper: Enum.map(band, &elem(&1, 1)),
      truth_path: Enum.map(post_effects, &(@level + &1)),
      attributed_sum: spots |> Enum.map(& &1.lift) |> Enum.sum(),
      attributed_total_sd: result.attributions.total_lift_sd,
      union_lift: union_lift(result.summary.point_effects.mean, spots),
      overlap_groups: result.attributions.overlap_groups,
      execution: result.execution
    }
  end

  # The lift actually measured across the union of the spot windows,
  # computed independently of the attributor so the "attributions sum to
  # total lift" reconciliation in the caption is a real check, not an echo.
  defp union_lift(point_effect_means, spots) do
    effects = List.to_tuple(point_effect_means)

    spots
    |> Enum.map(&{&1.window_start, &1.window_end})
    |> Enum.sort()
    |> Enum.reduce([], fn {ws, we}, acc ->
      case acc do
        [{prev_s, prev_e} | rest] when ws <= prev_e -> [{prev_s, max(prev_e, we)} | rest]
        _ -> [{ws, we} | acc]
      end
    end)
    |> Enum.reduce(0.0, fn {ws, we}, acc ->
      Enum.reduce(ws..(we - 1), acc, fn t, inner -> inner + elem(effects, t) end)
    end)
  end

  defp pulse(t, start, peak) when t >= start and t < start + @window_hours do
    peak * :math.exp(-(t - start) / @decay_tau)
  end

  defp pulse(_t, _start, _peak), do: 0.0

  defp pulse_sum do
    Enum.reduce(0..(@window_hours - 1), 0.0, fn k, acc ->
      acc + :math.exp(-k / @decay_tau)
    end)
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
    # Prepare once per visit: model, periods, filter artifacts (~7 ms).
    spec =
      BstsNx.Components.local_level_spec(
        initial_state: 90.0,
        initial_cov: 10.0,
        process_var: 0.5,
        obs_var: 16.0  # noise sd 4, squared
      )

    prepared = BstsNx.Operational.prepare(spec, {1, 96}, {97, 144})

    # Re-run on every slider drag. Spot windows are 0-based, half-open,
    # indexed within the 48-hour post period — and they may overlap.
    spots = [
      %{id: "Spot A", window_start: 6, window_end: 14},
      %{id: "Spot B", window_start: 18, window_end: 26},
      %{id: "Spot C", window_start: 24, window_end: 32}
    ]

    result = BstsNx.Operational.run(prepared, observations, spots, return: :lists)

    # Per-spot credit — Shapley-split wherever windows overlap.
    result.attributions.attributions
    #=> [%{spot_id: "Spot A", lift: ..., lift_lower: ..., lift_upper: ...,
    #      p_positive: ...}, ...]
    result.attributions.total_lift  # the per-spot lifts sum exactly to this
    """
  end
end
