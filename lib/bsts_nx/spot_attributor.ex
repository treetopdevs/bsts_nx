defmodule BstsNx.SpotAttributor do
  @moduledoc """
  Per-spot TV lift attribution with Shapley allocation for overlaps.

  Given pre-computed counterfactual baselines (mean + variance from the
  Kalman filter/smoother) and spot window definitions, computes per-spot
  lift with uncertainty bounds and P(lift > 0).

  This module is **self-contained** — it does NOT run the Kalman filter
  itself.  The caller provides `%{mean, variance, obs_variance}` from
  whatever estimation pipeline they use.

  For overlapping spots, `BstsNx.ShapleyAllocator` is used to fairly
  distribute the coalition lift among contributors.

  **Variance allocation note**: For overlap groups, the coalition variance is
  distributed proportionally to each spot's absolute Shapley allocation.
  This is a practical heuristic — Shapley values decompose expected values
  but not variances.  The resulting per-spot SDs and CIs should be treated
  as approximate.

  ## Usage

      observations = [100.0, 102.0, 150.0, 148.0, 101.0]
      counterfactual = %{
        mean: [100.0, 101.0, 102.0, 103.0, 104.0],
        variance: [1.0, 1.0, 1.0, 1.0, 1.0],
        obs_variance: 0.5
      }
      spots = [%{id: "spot_1", window_start: 2, window_end: 4}]

      result = BstsNx.SpotAttributor.attribute(observations, spots, counterfactual)
      # => %{attributions: [...], total_lift: ..., total_lift_sd: ..., overlap_groups: []}
  """

  alias BstsNx.ShapleyAllocator

  @typedoc "A spot with an ID and half-open time window [window_start, window_end)."
  @type spot :: %{id: String.t(), window_start: non_neg_integer(), window_end: non_neg_integer()}

  @typedoc "Pre-computed counterfactual baseline from Kalman filter/smoother."
  @type counterfactual :: %{
          mean: [float()],
          variance: [float()],
          obs_variance: float()
        }

  @typedoc "Attribution result for a single spot."
  @type spot_attribution :: %{
          spot_id: String.t(),
          lift: float(),
          lift_sd: float(),
          lift_lower: float(),
          lift_upper: float(),
          p_positive: float(),
          window_start: non_neg_integer(),
          window_end: non_neg_integer()
        }

  @typedoc "Overall attribution result."
  @type attribution_result :: %{
          attributions: [spot_attribution()],
          total_lift: float(),
          total_lift_sd: float(),
          overlap_groups: [[String.t()]]
        }

  # -------------------------------------------------------------------
  # 1. attribute
  # -------------------------------------------------------------------

  @doc """
  Main entry point: computes per-spot lift attributions.

  Returns an `attribution_result()` with per-spot lift, uncertainty
  bounds, and overlap group information.

  ## Options

  - `:alpha` — significance level for CI (default: 0.05 → 95% CI)
  - `:n_samples` — MC samples for Shapley when groups > 12 (default: 10_000)
  - `:key` — `Nx.Random.key` for deterministic MC
  - `:decay` — diminishing returns decay for default value function (default: 0.7)

  ## Examples

      iex> obs = [100.0, 102.0, 150.0, 148.0, 101.0]
      iex> cf = %{mean: [100.0, 101.0, 102.0, 103.0, 104.0], variance: [1.0, 1.0, 1.0, 1.0, 1.0], obs_variance: 0.5}
      iex> spots = [%{id: "s1", window_start: 2, window_end: 4}]
      iex> result = BstsNx.SpotAttributor.attribute(obs, spots, cf)
      iex> length(result.attributions)
      1
  """
  @spec attribute([float()], [spot()], counterfactual(), keyword()) :: attribution_result()
  def attribute(observations, spots, counterfactual, opts \\ [])

  def attribute(_observations, [], _counterfactual, _opts) do
    %{attributions: [], total_lift: 0.0, total_lift_sd: 0.0, overlap_groups: []}
  end

  def attribute(observations, spots, counterfactual, opts) do
    t = length(observations)
    validate_inputs!(observations, spots, counterfactual, t)

    alpha = Keyword.get(opts, :alpha, 0.05)
    validate_alpha!(alpha)
    z = BstsNx.Utils.z_score(alpha)

    # Convert to :array for O(1) access
    obs_arr = :array.from_list(observations)
    mean_arr = :array.from_list(counterfactual.mean)
    var_arr = :array.from_list(counterfactual.variance)
    obs_var = counterfactual.obs_variance

    # Build a map of spot_id → spot for lookups
    spot_map = Map.new(spots, fn s -> {s.id, s} end)

    # Detect overlap groups
    groups = ShapleyAllocator.detect_overlaps(spots)

    evaluate_attribution(groups, spot_map, obs_arr, mean_arr, var_arr, obs_var, z, opts)
  end

  # -------------------------------------------------------------------
  # 1b. attribute_posterior
  # -------------------------------------------------------------------

  @doc """
  Posterior-propagated attribution: runs `attribute/4` on each posterior
  counterfactual draw, then aggregates per-spot lift distributions.

  This preserves the correlation structure across time steps that would
  be lost by collapsing draws into mean/variance before attribution.

  ## Parameters

    * `observations` — actual post-period values (list of floats)
    * `spots` — spot definitions with `%{id, window_start, window_end}`
    * `counterfactual_draws` — list of lists, each a posterior counterfactual trajectory
    * `obs_variance` — mean observation variance (0.0 if already included in draws)
    * `opts` — keyword options (`:alpha`, `:n_samples`, `:key`, `:decay`)

  ## Examples

      iex> obs = [110.0, 112.0, 108.0]
      iex> spots = [%{id: "s1", window_start: 0, window_end: 3}]
      iex> draws = [[100.0, 100.0, 100.0], [101.0, 101.0, 101.0], [99.0, 99.0, 99.0]]
      iex> result = BstsNx.SpotAttributor.attribute_posterior(obs, spots, draws, 0.0)
      iex> length(result.attributions)
      1
  """
  @spec attribute_posterior([float()], [spot()], [[float()]], float(), keyword()) ::
          attribution_result()
  def attribute_posterior(observations, spots, counterfactual_draws, obs_variance, opts \\ [])

  def attribute_posterior(_observations, [], _draws, _obs_variance, _opts) do
    %{attributions: [], total_lift: 0.0, total_lift_sd: 0.0, overlap_groups: []}
  end

  def attribute_posterior(_observations, _spots, [], _obs_variance, _opts) do
    %{attributions: [], total_lift: 0.0, total_lift_sd: 0.0, overlap_groups: []}
  end

  def attribute_posterior(observations, spots, counterfactual_draws, obs_variance, opts) do
    alpha = Keyword.get(opts, :alpha, 0.05)
    validate_alpha!(alpha)
    t = length(observations)
    n_draws = length(counterfactual_draws)
    zeros_arr = :array.new(t, default: 0.0)

    base_counterfactual = %{
      mean: hd(counterfactual_draws),
      variance: List.duplicate(0.0, t),
      obs_variance: obs_variance
    }

    validate_inputs!(observations, spots, base_counterfactual, t)

    Enum.each(counterfactual_draws, fn draw ->
      if length(draw) != t do
        raise ArgumentError,
              "each counterfactual draw length (#{length(draw)}) must match observations length (#{t})"
      end
    end)

    obs_arr = :array.from_list(observations)
    spot_map = Map.new(spots, fn s -> {s.id, s} end)
    groups = ShapleyAllocator.detect_overlaps(spots)

    overlap_groups =
      groups |> Enum.filter(&(length(&1) > 1)) |> Enum.map(&Enum.map(&1, fn s -> s.id end))

    z = BstsNx.Utils.z_score(alpha)

    # Collect per-spot lifts across draws
    spot_ids = Enum.map(spots, & &1.id)
    init_lifts = Map.new(spot_ids, &{&1, []})

    {per_spot_lifts, per_draw_totals} =
      Enum.reduce(counterfactual_draws, {init_lifts, []}, fn draw, {lift_acc, total_acc} ->
        mean_arr = :array.from_list(draw)

        result =
          evaluate_attribution(
            groups,
            spot_map,
            obs_arr,
            mean_arr,
            zeros_arr,
            obs_variance,
            z,
            opts
          )

        attr_map = Map.new(result.attributions, fn a -> {a.spot_id, a.lift} end)

        updated_lifts =
          Map.new(lift_acc, fn {sid, lifts} ->
            {sid, [Map.get(attr_map, sid, 0.0) | lifts]}
          end)

        {updated_lifts, [result.total_lift | total_acc]}
      end)
      |> then(fn {lift_map, totals} ->
        {
          Map.new(lift_map, fn {sid, lifts} -> {sid, Enum.reverse(lifts)} end),
          Enum.reverse(totals)
        }
      end)

    # Aggregate per-spot statistics
    attributions =
      Enum.map(spots, fn spot ->
        lifts = Map.fetch!(per_spot_lifts, spot.id)
        mean_lift = Enum.sum(lifts) / n_draws

        lift_sd =
          if n_draws < 2 do
            0.0
          else
            ss =
              Enum.reduce(lifts, 0.0, fn l, acc -> acc + (l - mean_lift) * (l - mean_lift) end)

            safe_sqrt(ss / (n_draws - 1))
          end

        sorted = Enum.sort(lifts)
        {lower, upper} = quantiles(sorted, n_draws, alpha)
        p_pos = Enum.count(lifts, fn l -> l > 0.0 end) / n_draws

        %{
          spot_id: spot.id,
          lift: mean_lift,
          lift_sd: lift_sd,
          lift_lower: lower,
          lift_upper: upper,
          p_positive: p_pos,
          window_start: spot.window_start,
          window_end: spot.window_end
        }
      end)

    total_lift = Enum.reduce(attributions, 0.0, fn a, acc -> acc + a.lift end)

    total_lift_sd =
      if n_draws < 2 do
        0.0
      else
        mean_total = Enum.sum(per_draw_totals) / n_draws

        ss =
          Enum.reduce(per_draw_totals, 0.0, fn dt, acc ->
            acc + (dt - mean_total) * (dt - mean_total)
          end)

        safe_sqrt(ss / (n_draws - 1))
      end

    %{
      attributions: attributions,
      total_lift: total_lift,
      total_lift_sd: total_lift_sd,
      overlap_groups: overlap_groups
    }
  end

  defp quantiles(sorted, n, alpha) do
    BstsNx.Utils.percentile_interval(sorted, n, alpha)
  end

  # Shared attribution core used by `attribute/4` and `attribute_posterior/5`.
  defp evaluate_attribution(groups, spot_map, obs_arr, mean_arr, var_arr, obs_var, z, opts) do
    overlap_groups =
      groups
      |> Enum.filter(fn g -> length(g) > 1 end)
      |> Enum.map(fn g -> Enum.map(g, & &1.id) end)

    # Process each group
    attributions =
      Enum.flat_map(groups, fn group ->
        if length(group) == 1 do
          # Singleton — direct computation
          spot = hd(group)

          %{lift: lift, variance: var} =
            compute_window_lift_arr(
              obs_arr,
              mean_arr,
              var_arr,
              obs_var,
              spot.window_start,
              spot.window_end
            )

          sd = safe_sqrt(var)
          [build_attribution(spot, lift, sd, z)]
        else
          # Multi-spot overlap group — use Shapley allocation
          allocate_group(group, obs_arr, mean_arr, var_arr, obs_var, spot_map, z, opts)
        end
      end)

    # Compute totals.
    # Variance summation assumes independence across overlap groups, which holds
    # because detect_overlaps partitions spots into non-overlapping groups.
    # Within each group, allocated variances already account for shared time steps.
    # Note: smoothed state uncertainties are weakly correlated across groups through
    # the state-space model, but correcting for this would require the full smoother
    # cross-covariance matrix and is not attempted here.
    total_lift = Enum.reduce(attributions, 0.0, fn a, acc -> acc + a.lift end)
    total_var = Enum.reduce(attributions, 0.0, fn a, acc -> acc + a.lift_sd * a.lift_sd end)
    total_lift_sd = safe_sqrt(total_var)

    %{
      attributions: attributions,
      total_lift: total_lift,
      total_lift_sd: total_lift_sd,
      overlap_groups: overlap_groups
    }
  end

  # -------------------------------------------------------------------
  # 2. compute_window_lift
  # -------------------------------------------------------------------

  @doc """
  Computes lift for a single half-open window `[window_start, window_end)`.

  Returns `%{lift: float(), variance: float()}`.

  `lift` is the sum of `(obs[t] - baseline_mean[t])` over the window.
  `variance` is the sum of `(baseline_variance[t] + obs_variance)` over
  the window.

  ## Examples

      iex> obs = [100.0, 102.0, 150.0, 148.0, 101.0]
      iex> cf = %{mean: [100.0, 101.0, 102.0, 103.0, 104.0], variance: [1.0, 1.0, 1.0, 1.0, 1.0], obs_variance: 0.5}
      iex> %{lift: lift, variance: var} = BstsNx.SpotAttributor.compute_window_lift(obs, cf, 2, 4)
      iex> lift == (150.0 - 102.0) + (148.0 - 103.0)
      true
      iex> var == (1.0 + 0.5) + (1.0 + 0.5)
      true
  """
  @spec compute_window_lift([float()], counterfactual(), non_neg_integer(), non_neg_integer()) ::
          %{lift: float(), variance: float()}
  def compute_window_lift(observations, counterfactual, window_start, window_end) do
    obs_arr = :array.from_list(observations)
    mean_arr = :array.from_list(counterfactual.mean)
    var_arr = :array.from_list(counterfactual.variance)

    compute_window_lift_arr(
      obs_arr,
      mean_arr,
      var_arr,
      counterfactual.obs_variance,
      window_start,
      window_end
    )
  end

  # -------------------------------------------------------------------
  # 3. compute_coalition_lift
  # -------------------------------------------------------------------

  @doc """
  Computes lift for the union of windows from a group of spots.

  Deduplicates overlapping indices so each time step is counted once.
  Returns `%{lift: float(), variance: float()}`.

  ## Examples

      iex> obs = [100.0, 102.0, 150.0, 148.0, 101.0]
      iex> cf = %{mean: [100.0, 101.0, 102.0, 103.0, 104.0], variance: [1.0, 1.0, 1.0, 1.0, 1.0], obs_variance: 0.5}
      iex> spots = [%{id: "a", window_start: 1, window_end: 3}, %{id: "b", window_start: 2, window_end: 4}]
      iex> %{lift: lift} = BstsNx.SpotAttributor.compute_coalition_lift(obs, cf, spots)
      iex> lift == (102.0 - 101.0) + (150.0 - 102.0) + (148.0 - 103.0)
      true
  """
  @spec compute_coalition_lift([float()], counterfactual(), [spot()]) ::
          %{lift: float(), variance: float()}
  def compute_coalition_lift(observations, counterfactual, spots) do
    obs_arr = :array.from_list(observations)
    mean_arr = :array.from_list(counterfactual.mean)
    var_arr = :array.from_list(counterfactual.variance)
    obs_var = counterfactual.obs_variance

    compute_coalition_lift_arr(obs_arr, mean_arr, var_arr, obs_var, spots)
  end

  # -------------------------------------------------------------------
  # 4. p_positive
  # -------------------------------------------------------------------

  @doc """
  Computes P(X > 0) where X ~ N(mean, variance).

  Uses the Gaussian CDF: P(X > 0) = Φ(mean / √variance).

  ## Examples

      iex> BstsNx.SpotAttributor.p_positive(10.0, 1.0)
      1.0

      iex> BstsNx.SpotAttributor.p_positive(-10.0, 1.0)
      0.0

      iex> BstsNx.SpotAttributor.p_positive(0.0, 1.0)
      0.5
  """
  @spec p_positive(float(), float()) :: float()
  def p_positive(mean, variance) when variance <= 1.0e-15 do
    cond do
      mean > 0.0 -> 1.0
      mean < 0.0 -> 0.0
      true -> 0.5
    end
  end

  def p_positive(mean, variance) do
    z = mean / :math.sqrt(2.0 * variance)
    0.5 * (1.0 + :math.erf(z))
  end

  # -------------------------------------------------------------------
  # 5. default_spot_value_function
  # -------------------------------------------------------------------

  @doc """
  Factory returning a value function for `ShapleyAllocator.allocate/4`.

  The returned callback maps a sorted list of spot IDs to the coalition
  lift (sum of lifts over the union of their windows). It captures
  `observations` and `counterfactual` in its closure.

  For small groups (n <= 6), computes actual counterfactual-based lift
  per subset. For larger groups, falls back to
  `ShapleyAllocator.default_value_function` with individual lifts.

  ## Options

  - `:value_fn_mode` — `:exact_lift` or `:diminishing` (auto-selected by group size)
  - `:decay` — decay for diminishing returns mode (default: 0.7)

  ## Examples

      iex> obs = [100.0, 102.0, 150.0, 148.0, 101.0]
      iex> cf = %{mean: [100.0, 101.0, 102.0, 103.0, 104.0], variance: [1.0, 1.0, 1.0, 1.0, 1.0], obs_variance: 0.5}
      iex> spots = [%{id: "a", window_start: 2, window_end: 4}]
      iex> vf = BstsNx.SpotAttributor.default_spot_value_function(obs, cf, spots)
      iex> vf.([])
      0.0
  """
  @spec default_spot_value_function([float()], counterfactual(), [spot()], keyword()) ::
          ShapleyAllocator.value_function()
  def default_spot_value_function(observations, counterfactual, spots, opts \\ []) do
    spot_map = Map.new(spots, fn s -> {s.id, s} end)
    n = length(spots)
    mode = Keyword.get(opts, :value_fn_mode, if(n <= 6, do: :exact_lift, else: :diminishing))

    case mode do
      :exact_lift ->
        obs_arr = :array.from_list(observations)
        mean_arr = :array.from_list(counterfactual.mean)
        var_arr = :array.from_list(counterfactual.variance)
        obs_var = counterfactual.obs_variance

        fn sorted_ids ->
          if sorted_ids == [] do
            0.0
          else
            coalition_spots = Enum.map(sorted_ids, fn id -> Map.fetch!(spot_map, id) end)

            %{lift: lift} =
              compute_coalition_lift_arr(obs_arr, mean_arr, var_arr, obs_var, coalition_spots)

            lift
          end
        end

      :diminishing ->
        obs_arr = :array.from_list(observations)
        mean_arr = :array.from_list(counterfactual.mean)
        var_arr = :array.from_list(counterfactual.variance)
        obs_var = counterfactual.obs_variance

        individual_lifts =
          Map.new(spots, fn spot ->
            %{lift: lift} =
              compute_window_lift_arr(
                obs_arr,
                mean_arr,
                var_arr,
                obs_var,
                spot.window_start,
                spot.window_end
              )

            {spot.id, lift}
          end)

        decay = Keyword.get(opts, :decay, 0.7)
        ShapleyAllocator.default_value_function(individual_lifts, decay: decay)
    end
  end

  # -------------------------------------------------------------------
  # Private helpers
  # -------------------------------------------------------------------

  defp validate_alpha!(alpha) do
    if not is_number(alpha) or alpha <= 0 or alpha >= 1 do
      raise ArgumentError,
            "alpha must be a number in (0, 1), got: #{inspect(alpha)}"
    end
  end

  defp validate_inputs!(observations, spots, counterfactual, t) do
    if length(counterfactual.mean) != t do
      raise ArgumentError,
            "counterfactual.mean length (#{length(counterfactual.mean)}) must match observations length (#{t})"
    end

    if length(counterfactual.variance) != t do
      raise ArgumentError,
            "counterfactual.variance length (#{length(counterfactual.variance)}) must match observations length (#{t})"
    end

    if counterfactual.obs_variance < 0.0 do
      raise ArgumentError, "counterfactual.obs_variance must be non-negative"
    end

    spot_ids = Enum.map(spots, & &1.id)

    if length(spot_ids) != length(Enum.uniq(spot_ids)) do
      dupes = spot_ids -- Enum.uniq(spot_ids)

      raise ArgumentError,
            "duplicate spot IDs: #{inspect(Enum.uniq(dupes))}"
    end

    Enum.each(spots, fn spot ->
      if spot.window_start < 0 or spot.window_end > t do
        raise ArgumentError,
              "spot #{spot.id} window [#{spot.window_start}, #{spot.window_end}) is out of bounds for series of length #{t}"
      end

      if spot.window_start > spot.window_end do
        raise ArgumentError,
              "spot #{spot.id} has window_start (#{spot.window_start}) > window_end (#{spot.window_end})"
      end
    end)

    observations
  end

  defp compute_window_lift_arr(obs_arr, mean_arr, var_arr, obs_var, window_start, window_end) do
    if window_start >= window_end do
      %{lift: 0.0, variance: 0.0}
    else
      Enum.reduce(window_start..(window_end - 1), %{lift: 0.0, variance: 0.0}, fn t, acc ->
        obs_t = :array.get(t, obs_arr)
        mean_t = :array.get(t, mean_arr)
        var_t = :array.get(t, var_arr)

        %{
          lift: acc.lift + (obs_t - mean_t),
          variance: acc.variance + (var_t + obs_var)
        }
      end)
    end
  end

  defp compute_coalition_lift_arr(obs_arr, mean_arr, var_arr, obs_var, spots) do
    # Merge overlapping [window_start, window_end) intervals instead of
    # materializing all individual indices (avoids O(sum of window sizes) memory).
    merged = merge_intervals(spots)

    Enum.reduce(merged, %{lift: 0.0, variance: 0.0}, fn {ws, we}, outer_acc ->
      if ws >= we do
        outer_acc
      else
        Enum.reduce(ws..(we - 1), outer_acc, fn t, acc ->
          obs_t = :array.get(t, obs_arr)
          mean_t = :array.get(t, mean_arr)
          var_t = :array.get(t, var_arr)

          %{
            lift: acc.lift + (obs_t - mean_t),
            variance: acc.variance + (var_t + obs_var)
          }
        end)
      end
    end)
  end

  # Merges a list of spots' [window_start, window_end) intervals into
  # non-overlapping sorted intervals. O(n log n) sort + O(n) merge.
  defp merge_intervals(spots) do
    spots
    |> Enum.map(fn s -> {s.window_start, s.window_end} end)
    |> Enum.filter(fn {ws, we} -> ws < we end)
    |> Enum.sort()
    |> Enum.reduce([], fn {ws, we}, acc ->
      case acc do
        [{prev_s, prev_e} | rest] when ws <= prev_e ->
          [{prev_s, max(prev_e, we)} | rest]

        _ ->
          [{ws, we} | acc]
      end
    end)
    |> Enum.reverse()
  end

  defp allocate_group(group, obs_arr, mean_arr, var_arr, obs_var, spot_map, z, opts) do
    # Compute coalition lift for the full group
    %{lift: coalition_lift, variance: coalition_var} =
      compute_coalition_lift_arr(obs_arr, mean_arr, var_arr, obs_var, group)

    # Compute individual lifts for building the value function
    n = length(group)
    mode = Keyword.get(opts, :value_fn_mode, if(n <= 6, do: :exact_lift, else: :diminishing))

    value_fn =
      case mode do
        :exact_lift ->
          fn sorted_ids ->
            if sorted_ids == [] do
              0.0
            else
              coalition_spots = Enum.map(sorted_ids, fn id -> Map.fetch!(spot_map, id) end)

              %{lift: lift} =
                compute_coalition_lift_arr(obs_arr, mean_arr, var_arr, obs_var, coalition_spots)

              lift
            end
          end

        :diminishing ->
          individual_lifts =
            Map.new(group, fn spot ->
              %{lift: lift} =
                compute_window_lift_arr(
                  obs_arr,
                  mean_arr,
                  var_arr,
                  obs_var,
                  spot.window_start,
                  spot.window_end
                )

              {spot.id, lift}
            end)

          decay = Keyword.get(opts, :decay, 0.7)
          ShapleyAllocator.default_value_function(individual_lifts, decay: decay)
      end

    # Allocate via Shapley
    shapley_opts = Keyword.take(opts, [:n_samples, :key])
    allocations = ShapleyAllocator.allocate(group, coalition_lift, value_fn, shapley_opts)

    # Distribute variance proportionally to absolute Shapley allocation shares.
    #
    # NOTE: This is a practical heuristic, not a decomposition derived from
    # Shapley axioms.  The assumption is that spots with larger absolute lift
    # contributions carry proportionally more uncertainty.  This can
    # underestimate variance for small-lift spots that overlap temporally
    # with large-lift spots.  When all allocations are near zero, variance
    # is split equally.
    abs_total = allocations |> Map.values() |> Enum.map(&abs/1) |> Enum.sum()

    Enum.map(group, fn spot ->
      allocated_lift = Map.get(allocations, spot.id, 0.0)

      allocated_var =
        if abs_total > 1.0e-15 do
          coalition_var * (abs(allocated_lift) / abs_total)
        else
          coalition_var / n
        end

      sd = safe_sqrt(allocated_var)
      build_attribution(spot, allocated_lift, sd, z)
    end)
  end

  defp build_attribution(spot, lift, sd, z) do
    %{
      spot_id: spot.id,
      lift: lift,
      lift_sd: sd,
      lift_lower: lift - z * sd,
      lift_upper: lift + z * sd,
      p_positive: p_positive(lift, sd * sd),
      window_start: spot.window_start,
      window_end: spot.window_end
    }
  end

  defp safe_sqrt(x) when x <= 0.0, do: 0.0
  defp safe_sqrt(x), do: :math.sqrt(x)
end
