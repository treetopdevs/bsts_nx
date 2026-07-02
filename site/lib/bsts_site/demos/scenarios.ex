defmodule BstsSite.Demos.Scenarios do
  @moduledoc """
  Seeded synthetic scenario builders.

  Every demo on the site runs on data generated here, with a fixed seed and a
  known planted truth. That is the point: the site never asks for trust — it
  plants an answer, hides it, and lets the model try to recover it.

  All generators are deterministic: the same parameters always produce the
  same series, so what a visitor sees is reproducible.
  """

  @doc """
  Deterministic Gaussian noise stream (Box-Muller over a seeded exsss state).
  Independent of the process dictionary, safe under concurrent LiveViews.
  """
  def noise(n, sd, seed) when is_integer(n) and n >= 0 do
    state = :rand.seed_s(:exsss, {seed, seed + 7919, seed + 104_729})

    {values, _state} =
      Enum.map_reduce(1..n//1, state, fn _, st ->
        {u1, st} = :rand.uniform_s(st)
        {u2, st} = :rand.uniform_s(st)
        z = :math.sqrt(-2.0 * :math.log(max(u1, 1.0e-12))) * :math.cos(2.0 * :math.pi() * u2)
        {z * sd, st}
      end)

    values
  end

  @doc """
  The hub hero: 144 hours of website traffic around a steady baseline. A
  campaign goes live after 96 untreated hours (1-based hour 97, 0-based
  index 96) and its effect ramps in over the first hours
  (like real carry-over effects do), peaking at `lift` visits/hour. The noise
  seed is fixed, so moving the lift slider changes only the planted effect —
  the noise realization stays put.

  The baseline is deliberately within the local-level model class, so the
  hero demo's recovery is honest: no hidden model mismatch flattering or
  damning the estimate. The engine chapters are where mismatch gets taught.

  Returns the observations plus the exact planted truth.
  """
  def hero(lift, noise_sd, opts \\ []) do
    seed = Keyword.get(opts, :seed, 42)
    total = 144
    pre_end = 96
    noise = noise(total, noise_sd, seed)

    {observations, truth_effect} =
      0..(total - 1)
      |> Enum.map(fn t ->
        baseline = 84.0

        effect =
          if t >= pre_end do
            lift * (1.0 - :math.exp(-(t - pre_end + 1) / 6.0))
          else
            0.0
          end

        {baseline + effect + Enum.at(noise, t), effect}
      end)
      |> Enum.unzip()

    %{
      observations: observations,
      pre_period: {1, pre_end},
      post_period: {pre_end + 1, total},
      pre_end: pre_end,
      truth: %{
        lift_per_hour: lift * 1.0,
        cumulative_lift: Enum.sum(truth_effect),
        post_hours: total - pre_end
      }
    }
  end
end
