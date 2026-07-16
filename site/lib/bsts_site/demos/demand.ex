defmodule BstsSite.Demos.Demand do
  @moduledoc """
  The stocking demo: forecast three weeks of demand from 100 days of history,
  then turn the posterior into an order quantity at a chosen service level.

  `scenario/0` builds the seeded world — the history the model sees plus the
  21 held-back days it is graded against. `fit_and_forecast/1` is the
  short-tier MCMC call (run under `Limiter` via `start_async`). It keeps the
  raw posterior trajectories, so `decision/2` can re-price the order for any
  service level instantly, without refitting.
  """

  alias BstsNx.Applications.DemandForecaster
  alias BstsNx.{Forecaster, Forward}
  alias BstsSite.Demos.Scenarios

  @history_days 100
  @horizon 21
  @base_level 140.0
  @level_sd 1.2
  @obs_sd 8.0
  @scenario_seed 2027
  @fit_seed 4242
  @forecast_seed 777
  @num_samples 80
  @burn_in 60
  @service_range 80..99

  def history_days, do: @history_days
  def horizon, do: @horizon
  def num_samples, do: @num_samples
  def service_range, do: @service_range

  @doc """
  The seeded world: #{@history_days} days of daily demand around a gently
  meandering level (a random walk — deliberately inside the local-level model
  class), plus #{@horizon} more days generated the same way and held back.
  Deterministic: every visitor sees the same series and the same sealed truth.
  """
  def scenario do
    total = @history_days + @horizon
    level_steps = Scenarios.noise(total, @level_sd, @scenario_seed)
    obs_noise = Scenarios.noise(total, @obs_sd, @scenario_seed + 1)

    levels = Enum.scan(level_steps, @base_level, fn step, level -> level + step end)

    observations =
      Enum.zip_with(levels, obs_noise, fn level, eps ->
        max(Float.round(level + eps, 0), 0.0)
      end)

    {history, future} = Enum.split(observations, @history_days)
    future_total = Enum.sum(future)

    %{
      history: history,
      future: future,
      history_mean: Enum.sum(history) / @history_days,
      truth: %{
        future_total: future_total,
        future_daily_mean: future_total / @horizon
      }
    }
  end

  @doc """
  The short-tier call: scalar Gibbs sampler (#{@num_samples} retained draws,
  #{@burn_in} burn-in) fitted to the history, then #{@horizon}-day forecasts.

  `Forecaster.predict/2` returns aggregated bands; the same PRNG key handed to
  `Forward.posterior_scalar_trajectories/3` reproduces the exact trajectories
  behind those bands, and we keep them for the instant service-level slider.
  """
  def fit_and_forecast(scenario) do
    {elapsed_us, {fit, fan95, fan50, trajectories}} =
      :timer.tc(fn ->
        fit =
          Forecaster.fit(scenario.history,
            num_samples: @num_samples,
            burn_in: @burn_in,
            seed: @fit_seed
          )

        key = Nx.Random.key(@forecast_seed)
        fan95 = Forecaster.predict(fit, horizon: @horizon, alpha: 0.05, key: key)
        fan50 = Forecaster.predict(fit, horizon: @horizon, alpha: 0.50, key: key)

        trajectories =
          Forward.posterior_scalar_trajectories(fit.posterior_samples, @horizon, key)

        {fit, fan95, fan50, trajectories}
      end)

    totals = trajectories |> Enum.map(&Enum.sum/1) |> Enum.sort()

    %{
      fan95: fan95,
      band50: %{lower: fan50.lower, upper: fan50.upper},
      mean: fan95.mean,
      totals: totals,
      expected_total: Enum.sum(totals) / length(totals),
      n_draws: length(totals),
      execution: %{
        method_used: :"#{fit.method}_mcmc_forecast",
        elapsed_ms: System.convert_time_unit(elapsed_us, :microsecond, :millisecond)
      }
    }
  end

  @doc """
  The instant interaction: given the stored posterior, price an order for a
  target service level. The order quantity is the empirical quantile of
  total 21-day demand across trajectories — trajectory sums keep the
  day-to-day correlation that per-day standard deviations throw away.

  Also computes the textbook safety stock (`DemandForecaster.safety_stock/2`,
  which assumes independent daily errors) for an honest comparison.
  """
  def decision(forecast, service_level) do
    sl = clamp(service_level, @service_range)
    q = sl / 100
    totals = forecast.totals
    n = forecast.n_draws

    order_qty = Enum.at(totals, min(ceil(q * (n - 1)), n - 1))
    order_units = ceil(order_qty)
    expected = forecast.expected_total

    naive =
      DemandForecaster.safety_stock(forecast.fan95,
        service_level: q,
        lead_time: @horizon
      )

    %{
      service_level: sl,
      order_units: order_units,
      expected_total: expected,
      safety_stock: order_units - expected,
      exceed_draws: Enum.count(totals, &(&1 > order_qty)),
      n_draws: n,
      naive_safety_stock: naive.total
    }
  end

  @doc "How many of the held-back days fell inside the 95% band."
  def coverage(forecast, future) do
    [future, forecast.fan95.lower, forecast.fan95.upper]
    |> Enum.zip()
    |> Enum.count(fn {y, lo, hi} -> y >= lo and y <= hi end)
  end

  def clamp(value, range), do: BstsSite.Demos.Params.clamp(value, range)

  @doc "The code this demo runs, shown verbatim in the under-the-hood panel."
  def code_snippet do
    ~S"""
    # Fit once (~2 s here): scalar Gibbs sampler on 100 days of history.
    fit = BstsNx.Forecaster.fit(history, num_samples: 80, burn_in: 60, seed: 4242)
    fit.method  #=> :scalar

    # Forecast 21 days ahead. Same key twice -> the 95% and 50% bands
    # come from the same posterior trajectories, so they nest exactly.
    key = Nx.Random.key(777)
    fan95 = BstsNx.Forecaster.predict(fit, horizon: 21, alpha: 0.05, key: key)
    fan50 = BstsNx.Forecaster.predict(fit, horizon: 21, alpha: 0.50, key: key)

    # Keep the raw trajectories: one simulated future per posterior draw.
    trajectories =
      BstsNx.Forward.posterior_scalar_trajectories(fit.posterior_samples, 21, key)

    # The service-level slider is just a quantile of trajectory totals —
    # no refit. Correlation between days comes along for free.
    totals = trajectories |> Enum.map(&Enum.sum/1) |> Enum.sort()
    order_qty = Enum.at(totals, ceil(0.95 * (length(totals) - 1)))
    """
  end
end
