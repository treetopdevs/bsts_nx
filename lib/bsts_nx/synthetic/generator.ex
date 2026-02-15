defmodule BstsNx.Synthetic.Generator do
  @moduledoc """
  Synthetic data generator for BSTS validation.

  Produces time series with exact known ground truth, enabling quantitative
  measurement of how well the BSTS estimator recovers true effects.

  The data generating process (DGP) is:

      y(t) = baseline(t) + tv_contribution(t) + noise(t)

  where:
  - `baseline` = intercept + trend + daily Fourier + weekly Fourier
  - `tv_contribution` = sum of per-spot effects after adstock + Hill saturation
  - `noise` ~ N(0, noise_sd)

  Every component is stored in the ground truth so validation can compare
  estimated vs true values at any granularity.
  """

  alias BstsNx.Synthetic.Adstock
  alias BstsNx.ShapleyAllocator

  @type scenario_config :: %{
          name: String.t(),
          total_minutes: pos_integer(),
          baseline: %{
            intercept: float(),
            trend: float(),
            daily_fourier: [{float(), float()}],
            weekly_fourier: [{float(), float()}]
          },
          spots: [spot_config()],
          noise_sd: float(),
          effect: %{
            lambda: float(),
            ec: float(),
            slope: float(),
            coefficient: float()
          },
          controls: %{
            count: non_neg_integer(),
            correlation: float()
          },
          seed: integer()
        }

  @type spot_config :: %{
          id: String.t(),
          window_start: non_neg_integer(),
          window_end: non_neg_integer()
        }

  @type generation_result :: %{
          observations: [float()],
          ground_truth: %{
            baseline: [float()],
            tv_contribution: [float()],
            spot_attributions: %{String.t() => float()},
            coalition_values: %{[String.t()] => float()},
            total_lift: float()
          },
          spots: [spot_config()],
          controls: [[float()]],
          config: scenario_config()
        }

  @doc """
  Generates a synthetic time series with known ground truth.

  Returns a `generation_result()` containing the observable time series,
  exact ground truth decomposition, spot definitions, control series,
  and the config used.

  ## Example

      config = BstsNx.Synthetic.Scenarios.scenario_a()
      result = BstsNx.Synthetic.Generator.generate(config)
      # result.observations is the full time series
      # result.ground_truth.total_lift is the exact TV lift
  """
  @spec generate(scenario_config()) :: generation_result()
  def generate(config) do
    rng = :rand.seed_s(:exsss, config.seed)

    total = config.total_minutes
    spots = config.spots

    # 1. Generate baseline
    baseline = generate_baseline(total, config.baseline)

    # 2. Build impulse signal from spots and apply effect chain
    impulse = build_impulse_signal(total, spots)
    tv_contribution = apply_effect(impulse, config.effect)

    # 3. Compute per-spot attributions (true coalition values)
    {spot_attributions, coalition_values} =
      compute_spot_attributions(total, spots, config.effect)

    # 4. Generate noise
    {noise, rng} = generate_noise(total, config.noise_sd, rng)

    # 5. Combine: y(t) = baseline(t) + tv_contribution(t) + noise(t)
    observations =
      Enum.zip_with([baseline, tv_contribution, noise], fn [b, tv, n] -> b + tv + n end)

    # 6. Compute total lift (full impulse response, including carryover)
    total_lift = Enum.sum(tv_contribution)

    # 7. Generate control series
    {controls, _rng} = generate_controls(total, tv_contribution, config.controls, rng)

    %{
      observations: observations,
      ground_truth: %{
        baseline: baseline,
        tv_contribution: tv_contribution,
        spot_attributions: spot_attributions,
        coalition_values: coalition_values,
        total_lift: total_lift
      },
      spots: spots,
      controls: controls,
      config: config
    }
  end

  # -------------------------------------------------------------------
  # Baseline generation
  # -------------------------------------------------------------------

  @doc false
  def generate_baseline(total, baseline_config) do
    %{
      intercept: intercept,
      trend: trend,
      daily_fourier: daily_fourier,
      weekly_fourier: weekly_fourier
    } = baseline_config

    two_pi = 2.0 * :math.pi()

    Enum.map(0..(total - 1), fn t ->
      # Linear trend
      base = intercept + trend * (t / 1440.0)

      # Daily Fourier terms (period = 1440 minutes)
      daily =
        daily_fourier
        |> Enum.with_index(1)
        |> Enum.reduce(0.0, fn {{a, b}, k}, acc ->
          angle = two_pi * k * t / 1440.0
          acc + a * :math.sin(angle) + b * :math.cos(angle)
        end)

      # Weekly Fourier terms (period = 10080 minutes)
      weekly =
        weekly_fourier
        |> Enum.with_index(1)
        |> Enum.reduce(0.0, fn {{a, b}, k}, acc ->
          angle = two_pi * k * t / 10080.0
          acc + a * :math.sin(angle) + b * :math.cos(angle)
        end)

      base + daily + weekly
    end)
  end

  # -------------------------------------------------------------------
  # Spot impulse and effect chain
  # -------------------------------------------------------------------

  defp build_impulse_signal(total, spots) do
    # Build a binary impulse signal: 1.0 during any spot window, 0.0 otherwise
    on_air = build_on_air_set(spots)

    Enum.map(0..(total - 1), fn t ->
      if MapSet.member?(on_air, t), do: 1.0, else: 0.0
    end)
  end

  @doc false
  def apply_effect(impulse, effect_config) do
    %{lambda: lambda, ec: ec, slope: slope, coefficient: coefficient} = effect_config

    impulse
    |> Adstock.geometric_adstock(lambda)
    |> Adstock.hill_saturation(ec, slope)
    |> Enum.map(&(&1 * coefficient))
  end

  # -------------------------------------------------------------------
  # Per-spot attributions and coalition values
  # -------------------------------------------------------------------

  defp compute_spot_attributions(total, spots, effect_config) do
    # Solo spot contributions (single-spot counterfactual value)
    solo_lifts =
      Map.new(spots, fn spot ->
        impulse = build_impulse_signal(total, [spot])
        effect = apply_effect(impulse, effect_config)

        # Sum effect within the spot's own window
        lift =
          effect
          |> Enum.with_index()
          |> Enum.reduce(0.0, fn {val, idx}, acc ->
            if idx >= spot.window_start and idx < spot.window_end do
              acc + val
            else
              acc
            end
          end)

        {spot.id, lift}
      end)

    groups = ShapleyAllocator.detect_overlaps(spots)
    spot_map = Map.new(spots, &{&1.id, &1})

    coalition_values =
      Enum.reduce(groups, %{}, fn group, acc ->
        ids = Enum.map(group, & &1.id)

        group_values =
          if length(group) <= 8 do
            ids
            |> all_nonempty_subsets()
            |> Map.new(fn subset ->
              coalition_spots = Enum.map(subset, &Map.fetch!(spot_map, &1))
              {Enum.sort(subset), coalition_lift(total, coalition_spots, effect_config)}
            end)
          else
            # For very large overlap groups, keep singleton coalition values only.
            Map.new(ids, fn id -> {[id], Map.fetch!(solo_lifts, id)} end)
          end

        Map.merge(acc, group_values)
      end)

    # Ground-truth per-spot values should follow the same overlap logic as
    # SpotAttributor: Shapley allocations inside each overlap-connected group.
    spot_attributions =
      Enum.reduce(groups, %{}, fn group, acc ->
        ids = Enum.map(group, & &1.id)

        allocations =
          case group do
            [single] ->
              %{single.id => Map.fetch!(solo_lifts, single.id)}

            _ ->
              coalition_key = Enum.sort(ids)

              coalition_total =
                Map.get(coalition_values, coalition_key) ||
                  coalition_lift(total, group, effect_config)

              value_fn =
                if length(group) <= 8 do
                  fn subset_ids ->
                    Map.get(coalition_values, Enum.sort(subset_ids), 0.0)
                  end
                else
                  group_solo = Map.take(solo_lifts, ids)
                  ShapleyAllocator.default_value_function(group_solo)
                end

              ShapleyAllocator.allocate(group, coalition_total, value_fn)
          end

        Map.merge(acc, allocations)
      end)

    {spot_attributions, coalition_values}
  end

  defp coalition_lift(total, coalition_spots, effect_config) do
    impulse = build_impulse_signal(total, coalition_spots)
    effect = apply_effect(impulse, effect_config)
    Enum.sum(effect)
  end

  # -------------------------------------------------------------------
  # Noise generation
  # -------------------------------------------------------------------

  defp generate_noise(total, noise_sd, rng) do
    if not is_number(noise_sd) or noise_sd < 0.0 do
      raise ArgumentError, "noise_sd must be a non-negative number, got: #{inspect(noise_sd)}"
    end

    Enum.map_reduce(1..total, rng, fn _i, state ->
      {z, state} = normal_sample(state)
      {z * noise_sd, state}
    end)
  end

  # Box-Muller transform for a single standard normal draw.
  # Clamps u1 away from zero to prevent log(0) producing -infinity.
  defp normal_sample(state) do
    {u1_raw, state} = :rand.uniform_s(state)
    {u2, state} = :rand.uniform_s(state)
    u1 = max(u1_raw, 1.0e-300)
    z = :math.sqrt(-2.0 * :math.log(u1)) * :math.cos(2.0 * :math.pi() * u2)
    {z, state}
  end

  # -------------------------------------------------------------------
  # Control series generation
  # -------------------------------------------------------------------

  @doc false
  def generate_controls(total, tv_signal, controls_config, rng) do
    %{count: count, correlation: correlation} = controls_config

    if not is_integer(count) or count < 0 do
      raise ArgumentError,
            "controls.count must be a non-negative integer, got: #{inspect(count)}"
    end

    if not is_number(correlation) or correlation < -1.0 or correlation > 1.0 do
      raise ArgumentError,
            "controls.correlation must be in [-1.0, 1.0], got: #{inspect(correlation)}"
    end

    if count == 0 do
      {[], rng}
    else
      {tv_mean, tv_sd} = mean_sd(tv_signal)

      if tv_sd <= 1.0e-10 and abs(correlation) > 1.0e-10 do
        raise ArgumentError,
              "controls.correlation=#{inspect(correlation)} cannot be satisfied when tv_signal has near-zero variance"
      end

      tv_norm = standardize(tv_signal, tv_mean, tv_sd)
      tv_scale = if tv_sd > 1.0e-10, do: tv_sd, else: 1.0

      Enum.map_reduce(1..count, rng, fn _i, state ->
        {noise_raw, state} =
          Enum.map_reduce(1..total, state, fn _j, s ->
            normal_sample(s)
          end)

        {noise_mean, noise_sd} = mean_sd(noise_raw)
        noise_norm = standardize(noise_raw, noise_mean, noise_sd)
        coeff = :math.sqrt(max(1.0 - correlation * correlation, 0.0))

        control =
          Enum.zip_with(tv_norm, noise_norm, fn tv_z, n_z ->
            (correlation * tv_z + coeff * n_z) * tv_scale + tv_mean
          end)

        {control, state}
      end)
    end
  end

  defp mean_sd([]), do: {0.0, 0.0}

  defp mean_sd(series) do
    n = length(series)
    mean = Enum.sum(series) / n

    var =
      series
      |> Enum.reduce(0.0, fn x, acc -> acc + (x - mean) * (x - mean) end)
      |> Kernel./(n)

    {mean, :math.sqrt(max(var, 0.0))}
  end

  defp standardize(series, _mean, sd) when sd <= 1.0e-10 do
    Enum.map(series, fn _ -> 0.0 end)
  end

  defp standardize(series, mean, sd) do
    Enum.map(series, fn x -> (x - mean) / sd end)
  end

  # -------------------------------------------------------------------
  # Helpers
  # -------------------------------------------------------------------

  defp build_on_air_set(spots) do
    Enum.reduce(spots, MapSet.new(), fn spot, acc ->
      if spot.window_start < spot.window_end do
        Enum.reduce(spot.window_start..(spot.window_end - 1), acc, &MapSet.put(&2, &1))
      else
        acc
      end
    end)
  end

  defp all_nonempty_subsets([]), do: []

  defp all_nonempty_subsets(list) do
    n = length(list)
    arr = :array.from_list(list)

    for mask <- 1..(Bitwise.bsl(1, n) - 1) do
      for i <- 0..(n - 1), Bitwise.band(mask, Bitwise.bsl(1, i)) != 0 do
        :array.get(i, arr)
      end
    end
  end
end
