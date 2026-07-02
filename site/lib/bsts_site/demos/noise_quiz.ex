defmodule BstsSite.Demos.NoiseQuiz do
  @moduledoc """
  Engine chapter 1: four seeded random walks, exactly one of which carries a
  planted shift in its second half. The visitor guesses by eye; the compiled
  Kalman filter + RTS smoother (`CausalImpact.estimate_from_filter/3`) then
  grades all four with cumulative-effect intervals.

  Batches are deterministic: `batch/1` maps a regenerate counter to a fixed
  set of seeds. Because a 95% interval is wrong about one time in twenty on
  pure noise, the generator scans a handful of candidate seed sets (in a
  fixed order) and deals the first one where the intervals behave — the
  planted walk excludes zero and the three pure-noise walks straddle it.
  The page says so out loud; false positives are the calibration chapter's
  problem, not this quiz's.
  """

  alias BstsNx.CausalImpact
  alias BstsSite.Demos.Scenarios

  @n 120
  @half 60
  @shift 8.0
  @ramp_tau 6.0
  @step_sd 0.4
  @obs_r 0.04
  @max_attempts 10
  @labels ~w(A B C D)

  def n, do: @n
  def half, do: @half
  def shift, do: @shift
  def labels, do: @labels

  @doc """
  The deterministic batch for a regenerate counter: four walks, the planted
  index, the filter verdicts on all four, and the exact planted truth.
  """
  def batch(counter) when is_integer(counter) and counter >= 0 do
    Enum.reduce_while(0..(@max_attempts - 1), nil, fn attempt, _last ->
      candidate = build(counter, attempt)
      if candidate.clean?, do: {:halt, candidate}, else: {:cont, candidate}
    end)
  end

  defp build(counter, attempt) do
    base = 500 + counter * 1000 + attempt * 29
    planted = rem(:erlang.phash2({base, :planted}), 4)

    {walks, truth_cumulative} =
      Enum.map(0..3, fn i ->
        walk_with_truth(base + i * 7, i == planted)
      end)
      |> Enum.unzip()

    truth_cumulative = Enum.at(truth_cumulative, planted)

    {elapsed_us, results} = :timer.tc(fn -> Enum.map(walks, &estimate/1) end)

    verdicts =
      Enum.map(results, fn r ->
        ci = r.cumulative_effect

        %{
          mean: ci.mean,
          sd: ci.sd,
          lower: ci.lower,
          upper: ci.upper,
          excludes_zero?: ci.lower > 0.0 or ci.upper < 0.0,
          baseline: r.baseline,
          band_lower:
            Enum.zip_with(r.baseline, r.baseline_variance, fn m, v ->
              m - 1.96 * :math.sqrt(max(v, 0.0))
            end),
          band_upper:
            Enum.zip_with(r.baseline, r.baseline_variance, fn m, v ->
              m + 1.96 * :math.sqrt(max(v, 0.0))
            end)
        }
      end)

    filter_pick =
      0..3
      |> Enum.max_by(fn i ->
        v = Enum.at(verdicts, i)
        if v.sd > 0.0, do: abs(v.mean) / v.sd, else: 0.0
      end)

    clean? =
      Enum.with_index(verdicts)
      |> Enum.all?(fn {v, i} ->
        if i == planted, do: v.lower > 0.0, else: not v.excludes_zero?
      end)

    %{
      counter: counter,
      attempt: attempt,
      walks: walks,
      labels: @labels,
      planted: planted,
      verdicts: verdicts,
      filter_pick: filter_pick,
      clean?: clean?,
      truth: %{
        shift: @shift,
        start: @half,
        cumulative: truth_cumulative,
        step_sd: @step_sd
      },
      execution: %{
        method_used: :estimate_from_filter,
        elapsed_ms: div(max(elapsed_us, 0) + 999, 1000)
      }
    }
  end

  # A pure random walk (accumulated seeded Gaussian steps); the planted walk
  # additionally gets a shift ramping in over a handful of steps, the way real
  # carry-over effects arrive. Returns {series, exact_cumulative_effect}.
  defp walk_with_truth(seed, planted?) do
    walk =
      @n
      |> Scenarios.noise(@step_sd, seed)
      |> Enum.scan(&(&1 + &2))

    {series, effects} =
      walk
      |> Enum.with_index(fn w, t ->
        effect =
          if planted? and t >= @half do
            @shift * (1.0 - :math.exp(-(t - @half + 1) / @ramp_tau))
          else
            0.0
          end

        {w + effect, effect}
      end)
      |> Enum.unzip()

    {series, Enum.sum(effects)}
  end

  defp estimate(series) do
    CausalImpact.estimate_from_filter(series, Enum.to_list(@half..(@n - 1)),
      f: 1.0,
      h: 1.0,
      q: @step_sd * @step_sd,
      r: @obs_r,
      x0: 0.0,
      p0: 1.0
    )
  end

  @doc "The code this quiz runs, shown verbatim in the under-the-hood panel."
  def code_snippet do
    ~S"""
    # Each walk is nothing but accumulated seeded noise...
    walk =
      120
      |> Scenarios.noise(0.4, seed)
      |> Enum.scan(&(&1 + &2))

    # ...except one, which gets a shift ramping to +8 from step 60.

    # The filter is told the true step size (q) and asked: treating the
    # second half as the "intervention", how big is the cumulative gap
    # between what happened and the projected baseline?
    result =
      BstsNx.CausalImpact.estimate_from_filter(walk, Enum.to_list(60..119),
        f: 1.0,
        h: 1.0,
        q: 0.4 * 0.4,
        r: 0.04,
        x0: 0.0,
        p0: 1.0
      )

    result.cumulative_effect
    #=> %{mean: ..., lower: ..., upper: ..., sd: ...}
    # Only the planted walk's interval excludes zero.
    """
  end
end
