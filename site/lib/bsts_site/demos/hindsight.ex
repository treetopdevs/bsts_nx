defmodule BstsSite.Demos.Hindsight do
  @moduledoc """
  Engine chapter 3: the same wandering-level series as the Kalman chapter,
  now smoothed backwards with `BstsNx.Smoother.rts/3` and sampled with the
  Carter-Kohn simulation smoother (`Smoother.simulate_with_key/5`).

  We generated the latent level, so both the filter and the smoother can be
  graded against it — RMSE vs truth is the planted-truth device here. Each
  press of "draw another possible history" pulls one full trajectory from
  the state posterior with an incrementing seed, teaching that the smoothed
  line is the mean of a distribution over histories, not the history.
  """

  alias BstsNx.{KalmanFilter, Smoother}
  alias BstsSite.Demos.KalmanTuner

  @f 1.0
  @draw_seed_base 7000
  @max_draws 12

  def max_draws, do: @max_draws

  @doc """
  Filters and RTS-smooths the fixed scenario with the true Q and R, and
  grades both passes against the latent level. Keeps the raw
  {state, covariance} pairs around so `draw/2` can sample trajectories
  without re-running the forward pass.
  """
  def prepare do
    %{observations: obs, latent: latent, q_true: q, r_true: r} = KalmanTuner.scenario()
    x0 = hd(obs)
    p0 = q + r

    {elapsed_us, {filtered, predicted, smoothed}} =
      :timer.tc(fn ->
        {filtered, predicted} = KalmanFilter.filter_with_pred(obs, @f, 1.0, q, r, x0, p0)
        smoothed = Smoother.rts(filtered, predicted, @f)
        {filtered, predicted, smoothed}
      end)

    filt_means = Enum.map(filtered, fn {x, _p} -> Nx.to_number(x) end)
    filt_sds = Enum.map(filtered, fn {_x, p} -> :math.sqrt(max(Nx.to_number(p), 0.0)) end)
    smooth_means = Enum.map(smoothed, fn {x, _p} -> Nx.to_number(x) end)
    smooth_sds = Enum.map(smoothed, fn {_x, p} -> :math.sqrt(max(Nx.to_number(p), 0.0)) end)

    %{
      observations: obs,
      latent: latent,
      q: q,
      r: r,
      filtered: filtered,
      predicted: predicted,
      smoothed: smoothed,
      filt_means: filt_means,
      smooth_means: smooth_means,
      avg_sd_filt: mean(filt_sds),
      avg_sd_smooth: mean(smooth_sds),
      rmse_filt: rmse(filt_means, latent),
      rmse_smooth: rmse(smooth_means, latent),
      execution: %{
        method_used: :rts_smoother,
        elapsed_ms: div(max(elapsed_us, 0) + 999, 1000)
      }
    }
  end

  @doc """
  Draws the `i`-th posterior trajectory via the Carter-Kohn simulation
  smoother, seeded deterministically from the draw counter. Returns the
  trajectory as floats plus its own measured execution metadata.
  """
  def draw(prepared, i) when is_integer(i) do
    key = Nx.Random.key(@draw_seed_base + i)

    {elapsed_us, {states, _key_out}} =
      :timer.tc(fn ->
        Smoother.simulate_with_key(
          prepared.smoothed,
          prepared.filtered,
          prepared.predicted,
          @f,
          key: key
        )
      end)

    %{
      points: Enum.map(states, &Nx.to_number/1),
      execution: %{
        method_used: :carter_kohn_draw,
        elapsed_ms: div(max(elapsed_us, 0) + 999, 1000)
      }
    }
  end

  defp mean(xs), do: Enum.sum(xs) / length(xs)

  defp rmse(estimates, truth) do
    estimates
    |> Enum.zip_with(truth, fn e, t -> (e - t) * (e - t) end)
    |> mean()
    |> :math.sqrt()
  end

  @doc "The code this demo runs, shown verbatim in the under-the-hood panel."
  def code_snippet do
    ~S"""
    # Forward pass — same series and true Q/R as the Kalman chapter.
    {filtered, predicted} =
      BstsNx.KalmanFilter.filter_with_pred(observations, 1.0, 1.0, 0.3, 3.0, x0, p0)

    # Backward pass: the RTS smoother revisits every estimate with the
    # benefit of everything that came after it.
    smoothed = BstsNx.Smoother.rts(filtered, predicted, 1.0)

    # "Draw another possible history": one full trajectory from the
    # posterior over latent paths (Carter-Kohn simulation smoother).
    {states, _key} =
      BstsNx.Smoother.simulate_with_key(smoothed, filtered, predicted, 1.0,
        key: Nx.Random.key(7000 + press_count)
      )
    """
  end
end
