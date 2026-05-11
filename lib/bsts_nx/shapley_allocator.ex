defmodule BstsNx.ShapleyAllocator do
  @moduledoc """
  Cooperative game theory utilities for Shapley value allocation.

  Provides functions for detecting overlapping time windows and fairly
  allocating a total value (e.g., lift from overlapping ad spots) among
  contributors using Shapley values.

  The four Shapley axioms guarantee fair allocation:

  1. **Efficiency** — allocations sum to the coalition's total value
  2. **Symmetry** — identical contributors receive equal allocations
  3. **Dummy** — zero-contribution players receive nothing
  4. **Linearity** — Shapley values are additive across games

  ## Usage

      spots = [
        %{id: "spot_a", window_start: 0, window_end: 10},
        %{id: "spot_b", window_start: 5, window_end: 15}
      ]

      value_fn = fn ids -> length(ids) * 50.0 end
      allocations = BstsNx.ShapleyAllocator.allocate(spots, 100.0, value_fn)
      # => %{"spot_a" => 50.0, "spot_b" => 50.0}
  """

  import Bitwise
  alias BstsNx.Utils

  @exact_threshold 12

  @typedoc "A spot with an ID and time window."
  @type spot :: %{id: String.t(), window_start: number(), window_end: number()}

  @typedoc "A value function mapping sorted player ID lists to a numeric value."
  @type value_function :: ([String.t()] -> float())

  @typedoc "A map from spot IDs to their allocated values."
  @type allocation :: %{String.t() => float()}

  # -------------------------------------------------------------------
  # 1. detect_overlaps
  # -------------------------------------------------------------------

  @doc """
  Groups spots with overlapping time windows into connected components.

  Uses a sweep-line algorithm: sorts spots by `window_start`, then walks
  the list maintaining the running maximum `window_end`. If a spot's
  `window_start` is strictly less than the group's end, it joins the
  current group; otherwise a new group begins.

  Returns a list of groups (each a list of spots). Non-overlapping spots
  appear as singleton groups. Spots that merely touch (one ends exactly
  where the next begins) are placed in separate groups.

  ## Examples

      iex> BstsNx.ShapleyAllocator.detect_overlaps([])
      []

      iex> spots = [%{id: "a", window_start: 0, window_end: 10}, %{id: "b", window_start: 5, window_end: 15}]
      iex> BstsNx.ShapleyAllocator.detect_overlaps(spots) |> length()
      1
  """
  @spec detect_overlaps([spot()]) :: [[spot()]]
  def detect_overlaps([]), do: []

  def detect_overlaps(spots) do
    sorted = Enum.sort_by(spots, & &1.window_start)

    {groups, _group_end} =
      Enum.reduce(sorted, {[], nil}, fn spot, {groups, group_end} ->
        case groups do
          [] ->
            {[[spot]], spot.window_end}

          [current | rest] ->
            if spot.window_start < group_end do
              new_end = max(group_end, spot.window_end)
              {[[spot | current] | rest], new_end}
            else
              {[[spot] | groups], spot.window_end}
            end
        end
      end)

    groups
    |> Enum.map(&Enum.reverse/1)
    |> Enum.reverse()
  end

  # -------------------------------------------------------------------
  # 2. exact_shapley
  # -------------------------------------------------------------------

  @doc """
  Computes exact Shapley values via O(n·2^n) subset enumeration.

  Raises `ArgumentError` if `length(spots)` exceeds #{@exact_threshold}.
  For larger groups, use `monte_carlo_shapley/4` or `allocate/4`.

  For each player i, enumerates all 2^(n−1) subsets S of N\\{i} and
  sums the weighted marginal contributions:

      φ_i = Σ_S  [|S|!(n−|S|−1)!/n!] × [v(S ∪ {i}) − v(S)]

  `value_fn` receives sorted ID lists and returns a float. It will be
  called with the empty list (which should return 0.0 by convention).

  ## Examples

      iex> spots = [%{id: "a", window_start: 0, window_end: 10}]
      iex> value_fn = fn ["a"] -> 100.0; [] -> 0.0 end
      iex> BstsNx.ShapleyAllocator.exact_shapley(spots, value_fn)
      %{"a" => 100.0}
  """
  @spec exact_shapley([spot()], value_function()) :: allocation()
  def exact_shapley([], _value_fn), do: %{}

  def exact_shapley(spots, value_fn) do
    ids = spots |> Enum.map(& &1.id) |> Enum.sort()
    n = length(ids)

    if n > @exact_threshold do
      raise ArgumentError,
            "exact_shapley is O(n*2^n) and is limited to #{@exact_threshold} players, " <>
              "got: #{n}; use monte_carlo_shapley/4 or allocate/4 for larger groups"
    end

    factorials = precompute_factorials(n)
    n_fact = Map.fetch!(factorials, n)
    num_coalitions = bsl(1, n)

    coalition_cache =
      Map.new(0..(num_coalitions - 1), fn mask ->
        {mask, value_fn.(subset_from_mask(ids, mask))}
      end)

    ids
    |> Enum.with_index()
    |> Enum.map(fn {player_id, player_idx} ->
      player_bit = bsl(1, player_idx)

      shapley_value =
        Enum.reduce(0..(num_coalitions - 1), 0.0, fn mask, acc ->
          if band(mask, player_bit) == 0 do
            s_size = popcount(mask)

            weight =
              Map.fetch!(factorials, s_size) *
                Map.fetch!(factorials, n - s_size - 1) / n_fact

            marginal =
              Map.fetch!(coalition_cache, bor(mask, player_bit)) -
                Map.fetch!(coalition_cache, mask)

            acc + weight * marginal
          else
            acc
          end
        end)

      {player_id, shapley_value}
    end)
    |> Map.new()
  end

  # -------------------------------------------------------------------
  # 3. monte_carlo_shapley
  # -------------------------------------------------------------------

  @doc """
  Approximates Shapley values via random permutation sampling.

  For each of `n_samples` random permutations, walks left-to-right
  accumulating marginal contributions for each player. Averages across
  all samples. Uses an `Nx.Random.key` for deterministic PRNG.

  ## Examples

      iex> spots = [
      ...>   %{id: "a", window_start: 0, window_end: 10},
      ...>   %{id: "b", window_start: 5, window_end: 15}
      ...> ]
      iex> value_fn = fn ids -> length(ids) * 50.0 end
      iex> key = Nx.Random.key(42)
      iex> result = BstsNx.ShapleyAllocator.monte_carlo_shapley(spots, value_fn, 1000, key)
      iex> abs(result["a"] - 50.0) < 5.0 and abs(result["b"] - 50.0) < 5.0
      true
  """
  @spec monte_carlo_shapley([spot()], value_function(), pos_integer(), Nx.Tensor.t()) ::
          allocation()
  def monte_carlo_shapley([], _value_fn, _n_samples, _key), do: %{}

  def monte_carlo_shapley(_spots, _value_fn, n_samples, _key) when n_samples <= 0 do
    raise ArgumentError,
          "n_samples must be positive, got: #{n_samples}"
  end

  def monte_carlo_shapley(spots, value_fn, n_samples, key) when n_samples > 0 do
    ids = Enum.map(spots, & &1.id)

    # Nx.Random doesn't support efficient permutation shuffling, so we derive
    # a deterministic :rand seed from the Nx key for tag-and-sort shuffling.
    {a58, b58, c58} = key |> Nx.to_flat_list() |> Utils.derive_exsss_seed()
    initial_state = :rand.seed_s(:exsss, {a58, b58, c58})

    {contributions, _final_state} =
      Enum.reduce(1..n_samples, {%{}, initial_state}, fn _sample, {acc, rand_state} ->
        {perm, new_rand_state} = shuffle_list(ids, rand_state)

        {marginals, _coalition, _v_last} =
          Enum.reduce(perm, {%{}, [], value_fn.([])}, fn player_id,
                                                         {marg, coalition, v_without} ->
            coalition_with = sorted_insert(coalition, player_id)
            v_with = value_fn.(coalition_with)
            marginal = v_with - v_without

            {Map.put(marg, player_id, marginal), coalition_with, v_with}
          end)

        new_acc =
          Enum.reduce(marginals, acc, fn {id, val}, a ->
            Map.update(a, id, val, &(&1 + val))
          end)

        {new_acc, new_rand_state}
      end)

    contributions
    |> Enum.map(fn {id, total} -> {id, total / n_samples} end)
    |> Map.new()
  end

  # -------------------------------------------------------------------
  # 4. default_value_function
  # -------------------------------------------------------------------

  @doc """
  Factory returning a diminishing-returns value function callback.

  `spot_lifts` is a map from spot IDs to their individual lift estimates.
  The returned callback, given a sorted list of IDs, computes:

      Σ lift_i × decay^rank

  where rank is the 0-indexed position in the sorted coalition.
  An empty coalition returns 0.0.

  ## Options

  - `:decay` — per-rank decay factor (default: 0.7)

  ## Examples

      iex> lifts = %{"a" => 100.0, "b" => 80.0}
      iex> vf = BstsNx.ShapleyAllocator.default_value_function(lifts)
      iex> vf.([])
      0.0

      iex> lifts = %{"a" => 100.0}
      iex> vf = BstsNx.ShapleyAllocator.default_value_function(lifts)
      iex> vf.(["a"])
      100.0
  """
  @spec default_value_function(%{String.t() => float()}, keyword()) :: value_function()
  def default_value_function(spot_lifts, opts \\ []) do
    decay = Keyword.get(opts, :decay, 0.7)
    max_rank = max(map_size(spot_lifts) - 1, 0)

    decay_powers =
      0..max_rank
      |> Enum.map(&:math.pow(decay, &1))
      |> List.to_tuple()

    fn sorted_ids ->
      if sorted_ids == [] do
        0.0
      else
        # Rank by absolute lift magnitude. Players tied on magnitude share the
        # same average rank weight, preserving symmetry for equal-lift players.
        sorted_pairs =
          sorted_ids
          |> Enum.map(fn id -> {id, Map.get(spot_lifts, id, 0.0)} end)
          |> Enum.sort_by(fn {_id, lift} -> -abs(lift) end)

        {groups, current_group} =
          Enum.reduce(sorted_pairs, {[], []}, fn pair, {groups_acc, group_acc} ->
            case group_acc do
              [] ->
                {groups_acc, [pair]}

              [{_id0, lift0} | _] ->
                {_id, lift} = pair

                if abs(abs(lift) - abs(lift0)) <= 1.0e-12 do
                  {groups_acc, [pair | group_acc]}
                else
                  {[Enum.reverse(group_acc) | groups_acc], [pair]}
                end
            end
          end)

        tie_groups = Enum.reverse([Enum.reverse(current_group) | groups])

        {value, _rank_after} =
          Enum.reduce(tie_groups, {0.0, 0}, fn group, {acc, start_rank} ->
            group_size = length(group)

            avg_weight =
              start_rank..(start_rank + group_size - 1)
              |> Enum.map(fn rank ->
                if rank <= max_rank do
                  elem(decay_powers, rank)
                else
                  :math.pow(decay, rank)
                end
              end)
              |> Enum.sum()
              |> Kernel./(group_size)

            group_value =
              Enum.reduce(group, 0.0, fn {_id, lift}, group_acc ->
                group_acc + lift * avg_weight
              end)

            {acc + group_value, start_rank + group_size}
          end)

        value
      end
    end
  end

  # -------------------------------------------------------------------
  # 5. time_weighted_value_function
  # -------------------------------------------------------------------

  @doc """
  Factory returning a time-weighted value function callback.

  `spot_lifts` is a map from spot IDs to their individual lift estimates.
  `spot_times` is a map from spot IDs to their representative time (e.g., window midpoint).

  The returned callback, given a sorted list of IDs, computes:

      Σ lift_i × exp(-λ × (t_max - t_i))

  where λ = ln(2) / half_life and t_max is the most recent time in the coalition.

  ## Options

    - `:half_life` — time units for half decay (default: 7.0)

  ## Examples

      iex> lifts = %{"a" => 100.0, "b" => 80.0}
      iex> times = %{"a" => 0, "b" => 7}
      iex> vf = BstsNx.ShapleyAllocator.time_weighted_value_function(lifts, times, half_life: 7.0)
      iex> vf.([])
      0.0

      iex> lifts = %{"a" => 100.0}
      iex> times = %{"a" => 5}
      iex> vf = BstsNx.ShapleyAllocator.time_weighted_value_function(lifts, times)
      iex> vf.(["a"])
      100.0
  """
  @spec time_weighted_value_function(
          %{String.t() => float()},
          %{String.t() => number()},
          keyword()
        ) ::
          value_function()
  def time_weighted_value_function(spot_lifts, spot_times, opts \\ []) do
    half_life = Keyword.get(opts, :half_life, 7.0)

    if half_life <= 0 do
      raise ArgumentError, "half_life must be positive, got: #{half_life}"
    end

    lambda = :math.log(2) / half_life

    global_t_max =
      if map_size(spot_times) == 0 do
        0
      else
        spot_times |> Map.values() |> Enum.max()
      end

    fn sorted_ids ->
      if sorted_ids == [] do
        0.0
      else
        Enum.reduce(sorted_ids, 0.0, fn id, acc ->
          lift = Map.get(spot_lifts, id, 0.0)
          t_i = Map.get(spot_times, id, 0)
          weight = :math.exp(-lambda * (global_t_max - t_i))
          acc + lift * weight
        end)
      end
    end
  end

  # -------------------------------------------------------------------
  # 6. normalize_to_total
  # -------------------------------------------------------------------

  @doc """
  Rescales allocation values proportionally so they sum to `target_total`.

  If the raw sum is approximately zero (< 1.0e-10), distributes
  `target_total` equally among all entries.

  ## Examples

      iex> BstsNx.ShapleyAllocator.normalize_to_total(%{"a" => 30.0, "b" => 70.0}, 200.0)
      %{"a" => 60.0, "b" => 140.0}

      iex> BstsNx.ShapleyAllocator.normalize_to_total(%{}, 100.0)
      %{}
  """
  @spec normalize_to_total(allocation(), float()) :: allocation()
  def normalize_to_total(allocations, _target_total) when map_size(allocations) == 0, do: %{}

  def normalize_to_total(allocations, target_total) do
    raw_sum = allocations |> Map.values() |> Enum.sum()

    if abs(raw_sum) < 1.0e-10 do
      n = map_size(allocations)
      equal_share = target_total / n
      allocations |> Map.keys() |> Enum.map(&{&1, equal_share}) |> Map.new()
    else
      scale = target_total / raw_sum
      allocations |> Enum.map(fn {k, v} -> {k, v * scale} end) |> Map.new()
    end
  end

  # -------------------------------------------------------------------
  # 7. allocate
  # -------------------------------------------------------------------

  @doc """
  Main entry point: computes fair per-spot Shapley allocations.

  Short-circuits for n=0 (empty map) and n=1 (full lift to single spot).
  For n ≤ #{@exact_threshold}, uses exact subset enumeration; otherwise
  falls back to Monte Carlo permutation sampling. Results are normalized
  to sum to `coalition_lift`.

  ## Options

  - `:n_samples` — number of MC permutations (default: 10_000)
  - `:key` — `Nx.Random.key` for deterministic MC (default: key(42))

  ## Examples

      iex> spots = [%{id: "x", window_start: 0, window_end: 10}]
      iex> BstsNx.ShapleyAllocator.allocate(spots, 75.0, fn _ -> 1.0 end)
      %{"x" => 75.0}

      iex> BstsNx.ShapleyAllocator.allocate([], 100.0, fn _ -> 0.0 end)
      %{}
  """
  @spec allocate([spot()], float(), value_function(), keyword()) :: allocation()
  def allocate(spots, coalition_lift, value_fn, opts \\ [])

  def allocate([], _coalition_lift, _value_fn, _opts), do: %{}

  def allocate([single], coalition_lift, _value_fn, _opts) do
    %{single.id => coalition_lift}
  end

  def allocate(spots, coalition_lift, value_fn, opts) do
    n = length(spots)

    raw =
      if n <= @exact_threshold do
        exact_shapley(spots, value_fn)
      else
        n_samples = Keyword.get(opts, :n_samples, 10_000)
        key = Keyword.get(opts, :key, Nx.Random.key(42))
        monte_carlo_shapley(spots, value_fn, n_samples, key)
      end

    normalize_to_total(raw, coalition_lift)
  end

  # -------------------------------------------------------------------
  # Private helpers
  # -------------------------------------------------------------------

  defp precompute_factorials(n) do
    Enum.reduce(1..max(n, 1), %{0 => 1.0}, fn i, acc ->
      Map.put(acc, i, Map.fetch!(acc, i - 1) * i)
    end)
  end

  defp subset_from_mask(list, mask) do
    list
    |> Enum.with_index()
    |> Enum.filter(fn {_elem, idx} -> band(mask, bsl(1, idx)) != 0 end)
    |> Enum.map(fn {elem, _idx} -> elem end)
  end

  defp shuffle_list(list, rand_state) do
    {tagged, final_state} =
      Enum.reduce(list, {[], rand_state}, fn elem, {acc, state} ->
        {val, new_state} = :rand.uniform_s(state)
        {[{val, elem} | acc], new_state}
      end)

    shuffled = tagged |> Enum.sort_by(&elem(&1, 0)) |> Enum.map(&elem(&1, 1))
    {shuffled, final_state}
  end

  defp popcount(mask), do: popcount(mask, 0)
  defp popcount(0, acc), do: acc
  defp popcount(mask, acc), do: popcount(band(mask, mask - 1), acc + 1)

  defp sorted_insert([], value), do: [value]

  defp sorted_insert([h | t] = list, value) do
    if value <= h, do: [value | list], else: [h | sorted_insert(t, value)]
  end
end
