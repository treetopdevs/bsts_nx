defmodule BstsSite.Demos.KalmanTuner do
  @moduledoc """
  Engine chapter 2: one fixed seeded series (a meandering level under
  observation noise), two knobs (process variance Q, observation variance R),
  and `BstsNx.KalmanFilter.filter_with_pred/7` re-run on every drag.

  The series is generated with Q = 0.3 and R = 3.0 — exactly a local-level
  state-space model, so there is no hidden model mismatch. The sliders start
  on the truth; the chapter is about what happens when you walk away from it.

  The eager filter returns no execution metadata, so `run/2` measures the
  call itself with `:timer.tc` and reports honest wall-clock milliseconds.
  """

  alias BstsNx.KalmanFilter
  alias BstsSite.Demos.Scenarios

  @t 96
  @seed 314
  @level_start 20.0
  @q_true 0.3
  @r_true 3.0

  # Log-ish grid shared by both sliders; slider values are indices into it.
  @grid [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

  def grid, do: @grid
  def max_idx, do: length(@grid) - 1
  def q_true, do: @q_true
  def r_true, do: @r_true
  def default_q_idx, do: Enum.find_index(@grid, &(&1 == @q_true))
  def default_r_idx, do: Enum.find_index(@grid, &(&1 == @r_true))
  def t, do: @t

  @doc """
  The fixed scenario: a level that wanders as a random walk with step
  variance Q = 0.3, observed through noise with variance R = 3.0.
  Returns both the observations and the latent level (the smoother
  chapter grades against it; this chapter never peeks).
  """
  def scenario do
    level_steps = Scenarios.noise(@t, :math.sqrt(@q_true), @seed)
    obs_noise = Scenarios.noise(@t, :math.sqrt(@r_true), @seed + 1)

    latent =
      level_steps
      |> Enum.scan(&(&1 + &2))
      |> Enum.map(&(&1 + @level_start))

    observations = Enum.zip_with(latent, obs_noise, &(&1 + &2))

    %{observations: observations, latent: latent, q_true: @q_true, r_true: @r_true}
  end

  @doc """
  Runs the eager Kalman filter at the slider settings (indices into the grid,
  clamped server-side) and returns everything the page draws: the one-step
  predictions, the filtered estimates with a 95% band, and both MAEs.
  """
  def run(q_idx, r_idx) do
    q_idx = clamp_idx(q_idx)
    r_idx = clamp_idx(r_idx)
    q = Enum.at(@grid, q_idx)
    r = Enum.at(@grid, r_idx)

    %{observations: obs} = scenario()
    x0 = hd(obs)
    p0 = @q_true + @r_true

    {elapsed_us, {filtered, predicted}} =
      :timer.tc(fn -> KalmanFilter.filter_with_pred(obs, 1.0, 1.0, q, r, x0, p0) end)

    filt_means = Enum.map(filtered, fn {x, _p} -> Nx.to_number(x) end)
    filt_sds = Enum.map(filtered, fn {_x, p} -> :math.sqrt(max(Nx.to_number(p), 0.0)) end)
    pred_means = Enum.map(predicted, fn {x, _p} -> Nx.to_number(x) end)

    %{
      q: q,
      r: r,
      q_idx: q_idx,
      r_idx: r_idx,
      observations: obs,
      pred_means: pred_means,
      filt_means: filt_means,
      band_lower: Enum.zip_with(filt_means, filt_sds, fn m, s -> m - 1.96 * s end),
      band_upper: Enum.zip_with(filt_means, filt_sds, fn m, s -> m + 1.96 * s end),
      mae_pred: mae(pred_means, obs),
      mae_filt: mae(filt_means, obs),
      regime: regime(q / r),
      execution: %{
        method_used: :kalman_filter,
        elapsed_ms: div(max(elapsed_us, 0) + 999, 1000)
      }
    }
  end

  defp mae(estimates, obs) do
    estimates
    |> Enum.zip_with(obs, fn e, o -> abs(e - o) end)
    |> Enum.sum()
    |> Kernel./(length(obs))
  end

  # The teaching regimes. The truth sits at Q/R = 0.1.
  defp regime(ratio) when ratio >= 1.0, do: :trusts_data
  defp regime(ratio) when ratio <= 0.01, do: :trusts_model
  defp regime(_ratio), do: :balanced

  defp clamp_idx(v) when is_integer(v), do: v |> max(0) |> min(length(@grid) - 1)

  defp clamp_idx(v) when is_binary(v) do
    case Integer.parse(v) do
      {n, _} -> clamp_idx(n)
      :error -> 0
    end
  end

  defp clamp_idx(_), do: 0

  @doc "The code this demo runs, shown verbatim in the under-the-hood panel."
  def code_snippet do
    ~S"""
    # 96 observations of a wandering level, generated with Q = 0.3, R = 3.0.
    # Every slider drag re-runs the eager filter with YOUR Q and R:
    {filtered, predicted} =
      BstsNx.KalmanFilter.filter_with_pred(
        observations,
        1.0,   # F — the level carries over
        1.0,   # H — we observe the level directly
        q,     # process variance: how much the level may move per step
        r,     # observation variance: how noisy each reading is
        hd(observations),  # x0
        3.3    # p0
      )

    # `predicted` holds the one-step-ahead estimates (made BEFORE seeing
    # each observation); `filtered` the updated estimates (made after).
    pred_means = Enum.map(predicted, fn {x, _p} -> Nx.to_number(x) end)
    filt_means = Enum.map(filtered, fn {x, _p} -> Nx.to_number(x) end)
    """
  end
end
