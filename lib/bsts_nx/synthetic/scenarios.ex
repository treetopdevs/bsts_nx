defmodule BstsNx.Synthetic.Scenarios do
  @moduledoc """
  Pre-configured scenarios for BSTS validation testing.

  Each scenario targets a specific aspect of the estimator:

  | ID | Name | Tests |
  |----|------|-------|
  | `:scenario_a` | Single Campaign | Basic lift recovery with clear on/off |
  | `:scenario_b` | Overlapping Campaigns | Shapley allocation for temporal overlap |
  | `:scenario_c` | Varying Effect Sizes | Sensitivity across strong/medium/weak effects |
  | `:scenario_d` | Confounded | Seasonal correlation with TV flighting |
  | `:scenario_e` | Diminishing Returns | High intensity with steep Hill saturation |
  """

  @doc "Returns all scenarios as `[{id, name, config}]`."
  @spec all() :: [{atom(), String.t(), map()}]
  def all do
    [
      {:scenario_a, "Single Campaign", scenario_a()},
      {:scenario_b, "Overlapping Campaigns", scenario_b()},
      {:scenario_c, "Varying Effect Sizes", scenario_c()},
      {:scenario_d, "Confounded", scenario_d()},
      {:scenario_e, "Diminishing Returns", scenario_e()}
    ]
  end

  @doc """
  Scenario A: Single Campaign — clear on/off signal.

  14-day window (20,160 minutes). Spots in days 5-9 (end exclusive).
  Clear separation between pre-period, treatment, and post-period.
  The estimator should recover lift accurately with tight CI.
  """
  @spec scenario_a() :: map()
  def scenario_a do
    total_minutes = 14 * 1440
    # Spots in days 5-10 (4 spots per day at hours 8, 12, 18, 22)
    spots = generate_regular_spots("a", 5, 10, [8, 12, 18, 22], 30)

    %{
      name: "Single Campaign",
      total_minutes: total_minutes,
      baseline: %{
        intercept: 50.0,
        trend: 0.5,
        daily_fourier: [{10.0, 5.0}, {3.0, 2.0}],
        weekly_fourier: [{4.0, 3.0}]
      },
      spots: spots,
      noise_sd: 3.0,
      effect: %{
        lambda: 0.5,
        ec: 1.0,
        slope: 2.0,
        coefficient: 15.0
      },
      controls: %{count: 3, correlation: 0.3},
      seed: 42
    }
  end

  @doc """
  Scenario B: Overlapping Campaigns — tests Shapley allocation.

  21-day window. Two campaign groups with temporal overlap in days 8-11.
  Campaign 1: days 5-11 (end excl.). Campaign 2: days 8-15 (end excl.).
  The overlapping period should require fair allocation via Shapley values.
  """
  @spec scenario_b() :: map()
  def scenario_b do
    total_minutes = 21 * 1440

    # Both campaigns share hours 12 and 20 to create true temporal overlap
    spots_1 = generate_regular_spots("b1", 5, 12, [9, 12, 20], 30)
    spots_2 = generate_regular_spots("b2", 8, 16, [12, 16, 20], 30)

    %{
      name: "Overlapping Campaigns",
      total_minutes: total_minutes,
      baseline: %{
        intercept: 60.0,
        trend: 0.3,
        daily_fourier: [{12.0, 6.0}, {4.0, 2.0}],
        weekly_fourier: [{5.0, 3.0}]
      },
      spots: spots_1 ++ spots_2,
      noise_sd: 4.0,
      effect: %{
        lambda: 0.6,
        ec: 1.0,
        slope: 2.0,
        coefficient: 12.0
      },
      controls: %{count: 3, correlation: 0.4},
      seed: 123
    }
  end

  @doc """
  Scenario C: Varying Effect Sizes — sensitivity test.

  21-day window. Three campaign groups with different spot densities
  producing different total impacts (same per-spot coefficient):
  - Strong (5 spots/day, days 3-6 excl.) — highest total lift
  - Medium (3 spots/day, days 10-13 excl.) — moderate total lift
  - Weak (1 spot/day, days 17-20 excl.) — lowest total lift
  Tests whether the estimator distinguishes signal strengths.
  """
  @spec scenario_c() :: map()
  def scenario_c do
    total_minutes = 21 * 1440

    # Vary spot density to produce different total impacts per group:
    # Strong: 5 hours/day (high density), Medium: 3 hours/day, Weak: 1 hour/day
    spots_strong = generate_regular_spots("c_strong", 3, 7, [8, 10, 13, 16, 20], 30)
    spots_medium = generate_regular_spots("c_medium", 10, 14, [9, 13, 18], 30)
    spots_weak = generate_regular_spots("c_weak", 17, 21, [13], 30)

    %{
      name: "Varying Effect Sizes",
      total_minutes: total_minutes,
      baseline: %{
        intercept: 45.0,
        trend: 0.2,
        daily_fourier: [{8.0, 4.0}, {2.0, 1.0}],
        weekly_fourier: [{3.0, 2.0}]
      },
      spots: spots_strong ++ spots_medium ++ spots_weak,
      noise_sd: 2.5,
      effect: %{
        lambda: 0.4,
        ec: 1.0,
        slope: 2.0,
        coefficient: 10.0
      },
      controls: %{count: 4, correlation: 0.2},
      seed: 456
    }
  end

  @doc """
  Scenario D: Confounded — TV flighting correlated with seasonal peaks.

  28-day window. Spots are concentrated during natural traffic peaks
  (weekday primetime). This tests whether the model can separate
  genuine TV lift from coincidental timing with seasonal highs.
  """
  @spec scenario_d() :: map()
  def scenario_d do
    total_minutes = 28 * 1440

    # Spots only on weekdays at peak hours (primetime 19-22)
    spots =
      for day <- 0..27,
          rem(day, 7) not in [5, 6],
          hour <- [19, 20, 21],
          do: %{
            id: "d_day#{day}_h#{hour}",
            window_start: day * 1440 + hour * 60,
            window_end: day * 1440 + hour * 60 + 30
          }

    %{
      name: "Confounded",
      total_minutes: total_minutes,
      baseline: %{
        intercept: 55.0,
        trend: 0.1,
        # Strong daily pattern peaking at primetime
        daily_fourier: [{15.0, 8.0}, {5.0, 3.0}, {2.0, 1.0}],
        # Strong weekly pattern peaking on weekdays
        weekly_fourier: [{8.0, 5.0}, {3.0, 2.0}]
      },
      spots: spots,
      noise_sd: 5.0,
      effect: %{
        lambda: 0.5,
        ec: 1.0,
        slope: 2.0,
        coefficient: 8.0
      },
      controls: %{count: 5, correlation: 0.5},
      seed: 789
    }
  end

  @doc """
  Scenario E: Diminishing Returns — high intensity with steep saturation.

  14-day window. Many spots with high frequency, combined with a steep
  Hill saturation curve (low ec, high slope). Tests whether the estimator
  correctly handles non-linear response where additional spots yield
  progressively smaller incremental effects.
  """
  @spec scenario_e() :: map()
  def scenario_e do
    total_minutes = 14 * 1440

    # High-frequency spots: every 2 hours during days 3-11
    spots =
      for day <- 3..11,
          hour <- 0..23//2,
          do: %{
            id: "e_day#{day}_h#{hour}",
            window_start: day * 1440 + hour * 60,
            window_end: day * 1440 + hour * 60 + 30
          }

    %{
      name: "Diminishing Returns",
      total_minutes: total_minutes,
      baseline: %{
        intercept: 40.0,
        trend: 0.3,
        daily_fourier: [{8.0, 4.0}],
        weekly_fourier: [{3.0, 2.0}]
      },
      spots: spots,
      noise_sd: 3.5,
      effect: %{
        lambda: 0.7,
        ec: 0.3,
        slope: 4.0,
        coefficient: 20.0
      },
      controls: %{count: 2, correlation: 0.3},
      seed: 999
    }
  end

  # -------------------------------------------------------------------
  # Helpers
  # -------------------------------------------------------------------

  defp generate_regular_spots(prefix, day_start, day_end, hours, window_minutes) do
    for day <- day_start..(day_end - 1),
        hour <- hours do
      start_minute = day * 1440 + hour * 60

      %{
        id: "#{prefix}_day#{day}_h#{hour}",
        window_start: start_minute,
        window_end: start_minute + window_minutes
      }
    end
  end
end
