defmodule BstsNx.CausalImpact do
  @moduledoc """
  Causal impact analysis for Bayesian Structural Time Series models.

  This module provides functions to estimate the causal impact of an
  intervention by comparing observed data to a counterfactual baseline
  predicted from a state-space model.  The approach is inspired by the
  CausalImpact framework of Brodersen et al. and uses posterior
  predictive samples from a Gibbs sampler to quantify uncertainty.

  Two MCMC-based estimators are provided:

    * `estimate/4` — fits a simple local level model via
      `BstsNx.GibbsSampler.sample/7`.

    * `estimate_structured/5` — fits any `%BstsNx.ModelSpec{}` (local
      level, local linear trend, seasonal, regression, composed models)
      via `BstsNx.GibbsSampler.sample_structured/4`.

  A non-MCMC alternative is also available:

    * `estimate_from_filter/3` — uses the compiled Kalman filter and
      RTS smoother for fast estimation on long time series.
  """

  alias BstsNx.GibbsSampler

  @type impact_result :: %{
          point_effects: list(list(float())),
          cumulative_effects: list(float()),
          relative_effects: list(float()),
          actual: list(float()),
          counterfactual: list(list(float())),
          pre_period: {non_neg_integer(), non_neg_integer()},
          post_period: {non_neg_integer(), non_neg_integer()}
        }

  @doc """
  Estimates causal impact of an intervention by comparing observed values
  to a counterfactual baseline predicted from the pre-intervention
  period.  Returns a structure containing posterior samples of
  pointwise effects, cumulative effects and relative effects, along
  with the underlying actual and counterfactual series.

  * `observations` – list of numeric observations for the entire study period.
  * `pre_period` – tuple `{start_index, end_index}` (1-based) indicating the
    inclusive range of observations used for model fitting.  Must satisfy
    `1 <= start_index <= end_index <= length(observations)`.
  * `post_period` – tuple `{start_index, end_index}` (1-based) indicating
    the inclusive range of observations over which impact is assessed.
    Must follow immediately after `pre_period`.
  * `opts` – keyword list forwarded to `GibbsSampler.sample/7`.  Options
    include `:num_samples`, `:burn_in`, `:thin`, `:seed`, etc.

  The function fits a local level model to the pre-period data using
  Gibbs sampling, then projects the latent state forward to generate
  counterfactual predictions for the post-period.  The difference
  between the observed post-period and the counterfactual distribution
  yields the causal impact.
  """
  @spec estimate(
          [number | Nx.t()],
          {pos_integer(), pos_integer()},
          {pos_integer(), pos_integer()},
          keyword()
        ) :: impact_result()
  def estimate(observations, {pre_start, pre_end}, {post_start, post_end}, opts \\ []) do
    # Coerce any Nx scalar elements to plain numbers so downstream
    # Enum arithmetic (:math.pow, Enum.sum, etc.) doesn't crash.
    observations = Enum.map(observations, &to_number/1)
    obs_count = length(observations)

    # Validate indices
    if pre_start < 1 or pre_end < pre_start do
      raise ArgumentError, "invalid pre_period: must satisfy 1 <= start <= end"
    end

    if pre_end > obs_count do
      raise ArgumentError,
            "pre_period end (#{pre_end}) extends beyond observations (#{obs_count})"
    end

    if post_start != pre_end + 1 or post_end < post_start do
      raise ArgumentError, "post_period must immediately follow pre_period and have end >= start"
    end

    if post_end > obs_count do
      raise ArgumentError,
            "post_period end (#{post_end}) extends beyond observations (#{obs_count})"
    end

    # Extract pre and post subsets
    {pre_obs, post_obs} = Enum.split(observations, pre_end)
    pre_data = pre_obs |> Enum.drop(pre_start - 1)
    post_data = post_obs |> Enum.take(post_end - pre_end)
    # Determine number of samples
    num_samples = Keyword.get(opts, :num_samples, 200)
    burn_in = Keyword.get(opts, :burn_in, div(num_samples, 2))
    thin = Keyword.get(opts, :thin, 1)
    # Fit Gibbs sampler on the pre-period data only to sample latent states and variances.
    # Use provided burn_in and thin options to control chain length.
    # Default initial state to first observation for faster convergence.
    init_state = Keyword.get(opts, :initial_state, List.first(pre_data) || 0.0)
    init_cov = Keyword.get(opts, :initial_cov, 1.0)
    process_var = Keyword.get(opts, :process_var, 1.0)
    obs_var_init = Keyword.get(opts, :obs_var, 1.0)

    gibbs_opts =
      opts
      |> Keyword.delete(:num_samples)
      |> Keyword.put(:burn_in, burn_in)
      |> Keyword.put(:thin, thin)

    pre_samples =
      GibbsSampler.sample(
        pre_data,
        num_samples,
        init_state,
        init_cov,
        process_var,
        obs_var_init,
        gibbs_opts
      )

    # Determine number of post steps
    n_post = post_end - pre_end
    # Create Nx.Random key for counterfactual simulation (consistent PRNG approach)
    seed_base = Keyword.get(opts, :seed, System.os_time())
    cf_base_key = Keyword.get(opts, :key, Nx.Random.key(seed_base))
    # Split into one key per sample for independent counterfactual draws
    cf_key_rows = Nx.Random.split(cf_base_key, parts: length(pre_samples)) |> Nx.to_list()

    # Collect point effects and counterfactual predictions per sample by forward simulation
    effects =
      Enum.zip(pre_samples, cf_key_rows)
      |> Enum.map(fn {sample, cf_key_row} ->
        # final state of pre-period
        final_state = List.last(sample.states)
        init_val = Nx.to_number(final_state)
        q = Nx.to_number(sample.process_var)
        r = Nx.to_number(sample.obs_var)
        # generate counterfactual by forward simulation (random walk with observation noise)
        cf_key = Nx.tensor(cf_key_row, type: Nx.type(cf_base_key))
        cf = generate_counterfactual(init_val, q, r, n_post, cf_key)
        # compute point effects as difference between actual post data and counterfactual
        point_effects =
          Enum.zip(post_data, cf)
          |> Enum.map(fn {a, b} -> a - b end)

        {cf, point_effects}
      end)

    # separate baselines and effects
    {baselines, point_effects_list} = Enum.unzip(effects)
    # compute cumulative and relative effects per sample
    cumulative_effects = Enum.map(point_effects_list, &Enum.sum/1)
    baseline_sums = Enum.map(baselines, &Enum.sum/1)

    relative_effects =
      Enum.zip(cumulative_effects, baseline_sums)
      |> Enum.map(fn {cum, base} -> if abs(base) > 1.0e-10, do: cum / base, else: 0.0 end)

    # build result struct
    %{
      point_effects: point_effects_list,
      cumulative_effects: cumulative_effects,
      relative_effects: relative_effects,
      actual: post_data,
      counterfactual: baselines,
      pre_period: {pre_start, pre_end},
      post_period: {post_start, post_end}
    }
  end

  @doc """
  MCMC-based causal impact estimation using a structured model spec.

  Unlike `estimate/4` which is limited to a scalar local-level model,
  this function accepts any `%BstsNx.ModelSpec{}` — local-linear-trend,
  regression, composed models, etc.  The model is fit on the pre-period
  via `GibbsSampler.sample_structured/4`, then counterfactual predictions
  are generated by forward-simulating the state-space model.

  The period arguments and return format match `estimate/4`.

  ## Options

    * `:num_samples` — posterior samples (default: 200)
    * `:burn_in` — samples to discard (default: num_samples / 2)
    * `:thin` — thinning interval (default: 1)
    * `:seed` — integer PRNG seed
    * `:key` — Nx.Random PRNG key (overrides `:seed`)
  """
  @spec estimate_structured(
          [number | Nx.t()],
          {pos_integer(), pos_integer()},
          {pos_integer(), pos_integer()},
          BstsNx.ModelSpec.t(),
          keyword()
        ) :: impact_result()
  def estimate_structured(
        observations,
        {pre_start, pre_end},
        {post_start, post_end},
        %BstsNx.ModelSpec{} = spec,
        opts \\ []
      ) do
    observations = Enum.map(observations, &to_number/1)
    obs_count = length(observations)

    if pre_start < 1 or pre_end < pre_start do
      raise ArgumentError, "invalid pre_period: must satisfy 1 <= start <= end"
    end

    if pre_end > obs_count do
      raise ArgumentError,
            "pre_period end (#{pre_end}) extends beyond observations (#{obs_count})"
    end

    if post_start != pre_end + 1 or post_end < post_start do
      raise ArgumentError, "post_period must immediately follow pre_period and have end >= start"
    end

    if post_end > obs_count do
      raise ArgumentError,
            "post_period end (#{post_end}) extends beyond observations (#{obs_count})"
    end

    {pre_obs, post_obs} = Enum.split(observations, pre_end)
    pre_data = pre_obs |> Enum.drop(pre_start - 1)
    post_data = post_obs |> Enum.take(post_end - pre_end)

    num_samples = Keyword.get(opts, :num_samples, 200)
    burn_in = Keyword.get(opts, :burn_in, div(num_samples, 2))
    thin = Keyword.get(opts, :thin, 1)

    gibbs_opts =
      opts
      |> Keyword.delete(:num_samples)
      |> Keyword.put(:burn_in, burn_in)
      |> Keyword.put(:thin, thin)

    # Slice H for the pre-period so sample_structured gets matching lengths
    pre_spec = slice_spec_h(spec, pre_start - 1, length(pre_data))
    pre_samples = GibbsSampler.sample_structured(pre_data, pre_spec, num_samples, gibbs_opts)

    n_post = post_end - pre_end
    seed_base = Keyword.get(opts, :seed, System.os_time())
    cf_base_key = Keyword.get(opts, :key, Nx.Random.key(seed_base))
    cf_key_rows = Nx.Random.split(cf_base_key, parts: length(pre_samples)) |> Nx.to_list()

    # Get post-period H entries for counterfactual forward simulation
    post_h = post_period_h(spec.h, pre_end, n_post)

    effects =
      Enum.zip(pre_samples, cf_key_rows)
      |> Enum.map(fn {sample, cf_key_row} ->
        final_state = List.last(sample.states)
        q_mat = sample.q_matrix
        r = Nx.to_number(sample.obs_var)
        cf_key = Nx.tensor(cf_key_row, type: Nx.type(cf_base_key))

        # Forward-simulate the state-space model for counterfactual
        cf =
          generate_structured_counterfactual(
            final_state,
            spec.f,
            post_h,
            q_mat,
            r,
            n_post,
            cf_key
          )

        point_effects =
          Enum.zip(post_data, cf)
          |> Enum.map(fn {a, b} -> a - b end)

        {cf, point_effects}
      end)

    {baselines, point_effects_list} = Enum.unzip(effects)
    cumulative_effects = Enum.map(point_effects_list, &Enum.sum/1)
    baseline_sums = Enum.map(baselines, &Enum.sum/1)

    relative_effects =
      Enum.zip(cumulative_effects, baseline_sums)
      |> Enum.map(fn {cum, base} -> if abs(base) > 1.0e-10, do: cum / base, else: 0.0 end)

    %{
      point_effects: point_effects_list,
      cumulative_effects: cumulative_effects,
      relative_effects: relative_effects,
      actual: post_data,
      counterfactual: baselines,
      pre_period: {pre_start, pre_end},
      post_period: {post_start, post_end}
    }
  end

  @doc """
  Summarises the posterior causal impact results.

  Given an `impact_result` returned by `estimate/4` or
  `estimate_structured/5`, this function computes summary statistics
  including the posterior mean, standard deviation and 95% credible
  interval for the point effects at each post‑period time step, as
  well as for the cumulative and relative
  effects.  Returns a map with keys `:point_effects`, `:cumulative_effect`,
  and `:relative_effect`.  Each entry contains the mean, sd and
  interval bounds.  If fewer than 2 posterior samples are provided,
  the standard deviation and interval bounds are returned as `:nan`.

  ## Examples

      iex> result = %{
      ...>   point_effects: [[1.0, 2.0], [2.0, 4.0]],
      ...>   cumulative_effects: [3.0, 6.0],
      ...>   relative_effects: [0.5, 1.0],
      ...>   actual: [1.0, 2.0],
      ...>   counterfactual: [[0.0, 0.0], [0.0, 0.0]],
      ...>   pre_period: {1, 2},
      ...>   post_period: {3, 4}
      ...> }
      iex> summary = BstsNx.CausalImpact.summary(result)
      iex> summary.cumulative_effect.mean
      4.5
  """
  @spec summary(impact_result()) :: map()
  def summary(result) do
    n_post = length(result.actual)
    m = length(result.point_effects)
    lower_idx = trunc(Float.floor(0.025 * max(m - 1, 0)))
    upper_idx = trunc(Float.ceil(0.975 * max(m - 1, 0)))

    point_summaries =
      cond do
        m == 0 ->
          List.duplicate(%{mean: :nan, sd: :nan, lower: :nan, upper: :nan}, n_post)

        true ->
          effects_t = Nx.tensor(result.point_effects, type: {:f, 64})
          means = Nx.mean(effects_t, axes: [0]) |> Nx.to_flat_list()

          sds =
            if m < 2 do
              List.duplicate(:nan, n_post)
            else
              Nx.standard_deviation(effects_t, axes: [0], ddof: 1) |> Nx.to_flat_list()
            end

          {lowers, uppers} =
            if m < 2 do
              {List.duplicate(:nan, n_post), List.duplicate(:nan, n_post)}
            else
              sorted = Nx.sort(effects_t, axis: 0)

              lower =
                sorted
                |> Nx.slice([lower_idx, 0], [1, n_post])
                |> Nx.squeeze(axes: [0])
                |> Nx.to_flat_list()

              upper =
                sorted
                |> Nx.slice([upper_idx, 0], [1, n_post])
                |> Nx.squeeze(axes: [0])
                |> Nx.to_flat_list()

              {lower, upper}
            end

          Enum.map(0..(n_post - 1), fn idx ->
            %{mean: Enum.at(means, idx), sd: Enum.at(sds, idx), lower: Enum.at(lowers, idx), upper: Enum.at(uppers, idx)}
          end)
      end

    sum_stats = fn vals ->
      cond do
        m == 0 ->
          %{mean: :nan, sd: :nan, lower: :nan, upper: :nan}

        m == 1 ->
          v = hd(vals)
          %{mean: v, sd: :nan, lower: :nan, upper: :nan}

        true ->
          v_t = Nx.tensor(vals, type: {:f, 64})
          sorted = Nx.sort(v_t)

          lower = sorted |> Nx.slice([lower_idx], [1]) |> Nx.squeeze() |> Nx.to_number()
          upper = sorted |> Nx.slice([upper_idx], [1]) |> Nx.squeeze() |> Nx.to_number()

          %{
            mean: v_t |> Nx.mean() |> Nx.to_number(),
            sd: v_t |> Nx.standard_deviation(ddof: 1) |> Nx.to_number(),
            lower: lower,
            upper: upper
          }
      end
    end

    cum_stats = sum_stats.(result.cumulative_effects)
    rel_stats = sum_stats.(result.relative_effects)

    %{
      point_effects: point_summaries,
      cumulative_effect: cum_stats,
      relative_effect: rel_stats
    }
  end

  @doc """
  Non-MCMC causal impact estimation using the compiled Kalman filter and
  RTS smoother.

  This function provides a lightweight alternative to `estimate/4` that
  avoids the cost of Gibbs sampling.  It is suitable for long time series
  (e.g. minute-level data) where MCMC would be too slow.

  The algorithm:
  1. Masks the intervention period observations as NaN.
  2. Runs `KalmanFilter.filter_defn/7` on the masked series.
  3. Runs `Smoother.rts_defn/4` for backward smoothing.
  4. Computes lift = actual − smoothed baseline on the intervention period.
  5. Derives credible intervals from the smoothed variances.

  **Causal note**: RTS smoothing uses all non-masked observations, including
  those after the intervention period. The baseline is therefore an
  interpolation rather than a pure counterfactual forecast. This is
  acceptable when post-intervention observations outside the masked window
  are unaffected by treatment, but can introduce bias otherwise.

  **Cumulative variance**: Cross-covariance between time steps is
  approximated via smoother gain propagation rather than exact RTS
  lag-one covariance. This is accurate at steady state but may
  under/overestimate uncertainty during transients.

  * `observations` – Nx tensor of shape `{t}` or a list of numbers.
  * `intervention_indices` – list of 0-based integer indices marking
    the intervention (on-air) period.
  * `opts` – keyword list with `:f`, `:h`, `:q`, `:r`, `:x0`, `:p0`,
    `:alpha` (significance level for CI, default: 0.05).

  Returns a map with the same shape as `summary/1` for consistency:
    * `:point_effects` – `%{mean: [...], lower: [...], upper: [...]}`
    * `:cumulative_effect` – `%{mean, sd, lower, upper}`
    * `:relative_effect` – `%{mean, sd, lower, upper}`
    * `:actual` – list of actual values during intervention
    * `:baseline` – list of smoothed baseline values during intervention
  """
  @spec estimate_from_filter(Nx.t() | [number()], [non_neg_integer()], keyword()) :: map()
  def estimate_from_filter(observations, intervention_indices, opts \\ []) do
    obs_tensor =
      case observations do
        %Nx.Tensor{} -> observations
        list when is_list(list) -> Nx.tensor(list, type: {:f, 32})
      end

    t = Nx.axis_size(obs_tensor, 0)
    f = Keyword.get(opts, :f, 1.0)
    h = Keyword.get(opts, :h, 1.0)
    q = Keyword.get(opts, :q, 1.0)
    r = Keyword.get(opts, :r, 1.0)
    x0 = Keyword.get(opts, :x0, 0.0)
    p0 = Keyword.get(opts, :p0, 1.0)
    alpha = Keyword.get(opts, :alpha, 0.05)

    if not is_number(alpha) or alpha <= 0 or alpha >= 1 do
      raise ArgumentError, "alpha must be in (0, 1), got: #{inspect(alpha)}"
    end

    z = BstsNx.Utils.z_score(alpha)

    valid_indices =
      intervention_indices
      |> Enum.filter(fn idx -> idx >= 0 and idx < t end)
      |> Enum.uniq()
      |> Enum.sort()

    if valid_indices == [] do
      # No valid intervention indices — return zero-effect result
      %{
        point_effects: %{mean: [], lower: [], upper: []},
        cumulative_effect: %{mean: 0.0, sd: 0.0, lower: 0.0, upper: 0.0},
        relative_effect: %{mean: 0.0, sd: 0.0, lower: 0.0, upper: 0.0},
        actual: [],
        baseline: []
      }
    else
      # Build NaN mask for intervention period
      mask_flags = :array.new(t, default: 0)

      mask_flags =
        Enum.reduce(valid_indices, mask_flags, fn idx, arr ->
          :array.set(idx, 1, arr)
        end)

      mask_list = Enum.map(0..(t - 1), fn i -> :array.get(i, mask_flags) end)
      mask = Nx.equal(Nx.tensor(mask_list, type: {:u, 8}), 1)
      nan_vec = Nx.broadcast(Nx.Constants.nan(), {t})
      masked_obs = Nx.select(mask, nan_vec, obs_tensor)

      # Filter
      {xs, ps} = BstsNx.KalmanFilter.filter_defn(masked_obs, f, h, q, r, x0, p0)

      # Smooth
      {sxs, sps} = BstsNx.Smoother.rts_defn(xs, ps, f, q)

      # Extract intervention period values
      idx_tensor = Nx.tensor(valid_indices, type: {:s, 64})
      actual_vals = Nx.take(obs_tensor, idx_tensor) |> Nx.to_flat_list()
      raw_state_vals = Nx.take(sxs, idx_tensor) |> Nx.to_flat_list()
      state_var_vals = Nx.take(sps, idx_tensor) |> Nx.to_flat_list()
      # Project smoothed states through h to get baseline in observation space:
      # y_t = h_t * x_t, so baseline = h_t * smoothed_x_t
      h_vals = h_intervention_values(h, valid_indices)
      baseline_vals = Enum.zip(h_vals, raw_state_vals) |> Enum.map(fn {hi, xi} -> hi * xi end)

      # Point effects
      point_effects_mean =
        Enum.zip(actual_vals, baseline_vals) |> Enum.map(fn {a, b} -> a - b end)

      # Observation-level variance: h_t^2 * p_t + r
      r_num = r_to_number(r)

      point_sds =
        Enum.zip(h_vals, state_var_vals)
        |> Enum.map(fn {hi, v} -> :math.sqrt(max(hi * hi * v + r_num, 0.0)) end)

      point_lower =
        Enum.zip(point_effects_mean, point_sds)
        |> Enum.map(fn {m, s} -> m - z * s end)

      point_upper =
        Enum.zip(point_effects_mean, point_sds)
        |> Enum.map(fn {m, s} -> m + z * s end)

      # Cumulative effect with cross-covariance correction.
      # Smoothed state errors at adjacent time steps are correlated through
      # the smoother gain G_k = P^filt_k * f / (f * P^filt_k * f + q).
      # The accumulator A_t propagates through ALL time steps between the
      # first and last intervention index so that gaps (non-consecutive
      # indices) correctly transmit cross-covariance via intermediate G_k.
      #
      # Recurrence: A_{t+1} = G_t * (A_t + delta_t)
      # where delta_t = h_t * P^s_t if t is an intervention index, else 0.
      # The cross-covariance sum accumulates h_j * A_j only at intervention
      # indices j.
      cum_mean = Enum.sum(point_effects_mean)

      f_num = if is_number(f), do: f * 1.0, else: Nx.to_number(Nx.tensor(f))
      q_num = if is_number(q), do: q * 1.0, else: Nx.to_number(Nx.tensor(q))

      n_intervention = length(valid_indices)

      # Diagonal variance: sum(h_i^2 * P^s_i)
      diag_var =
        Enum.zip(h_vals, state_var_vals)
        |> Enum.map(fn {hi, v} -> hi * hi * v end)
        |> Enum.sum()

      # Cross-covariance correction propagated through all time steps
      cross_var_sum =
        if n_intervention >= 2 do
          min_idx = Enum.min(valid_indices)
          max_idx = Enum.max(valid_indices)

          # Extract filtered variances and smoothed variances for the full
          # range [min_idx, max_idx] so we can propagate through gaps.
          # Use :array for O(1) random access (lists would be O(n) per Enum.at call).
          all_ps = ps |> Nx.to_flat_list() |> :array.from_list()
          all_sps = sps |> Nx.to_flat_list() |> :array.from_list()

          # Build intervention set for O(1) membership checks
          intervention_set = MapSet.new(valid_indices)

          # Compute h value at any time step
          h_at = h_at_step_fn(h, t)

          # Smoother gain at time step k
          gain_at = fn k ->
            pf = :array.get(k, all_ps)
            denom = f_num * pf * f_num + q_num
            if abs(denom) < 1.0e-15, do: 0.0, else: pf * f_num / denom
          end

          # Propagate accumulator A through every time step from
          # min_idx+1 to max_idx, accumulating cross terms at intervention steps.
          {_a_final, cv_sum} =
            Enum.reduce((min_idx + 1)..max_idx//1, {0.0, 0.0}, fn step, {a_prev, cs} ->
              prev = step - 1
              g_prev = gain_at.(prev)

              delta =
                if MapSet.member?(intervention_set, prev) do
                  h_at.(prev) * :array.get(prev, all_sps)
                else
                  0.0
                end

              a_step = g_prev * (a_prev + delta)

              cs_new =
                if MapSet.member?(intervention_set, step) do
                  cs + h_at.(step) * a_step
                else
                  cs
                end

              {a_step, cs_new}
            end)

          cv_sum
        else
          0.0
        end

      cum_var = diag_var + 2.0 * cross_var_sum + n_intervention * r_num
      cum_sd = :math.sqrt(max(cum_var, 0.0))

      # Relative
      baseline_sum = Enum.sum(baseline_vals)

      rel_mean =
        if abs(baseline_sum) < 1.0e-10, do: 0.0, else: cum_mean / baseline_sum

      rel_sd =
        if abs(baseline_sum) < 1.0e-10, do: 0.0, else: cum_sd / abs(baseline_sum)

      %{
        point_effects: %{
          mean: point_effects_mean,
          lower: point_lower,
          upper: point_upper
        },
        cumulative_effect: %{
          mean: cum_mean,
          sd: cum_sd,
          lower: cum_mean - z * cum_sd,
          upper: cum_mean + z * cum_sd
        },
        relative_effect: %{
          mean: rel_mean,
          sd: rel_sd,
          lower: rel_mean - z * rel_sd,
          upper: rel_mean + z * rel_sd
        },
        actual: actual_vals,
        baseline: baseline_vals
      }
    end
  end

  defp r_to_number(r) when is_number(r), do: r * 1.0
  defp r_to_number(%Nx.Tensor{} = r), do: Nx.to_number(r)

  # Returns a function that gives h at any time step index.
  defp h_at_step_fn(h, _t) when is_number(h), do: fn _step -> h * 1.0 end

  defp h_at_step_fn(%Nx.Tensor{} = h, t) do
    if Nx.rank(h) == 0 do
      val = Nx.to_number(h)
      fn _step -> val end
    else
      h_arr = :array.from_list(Nx.to_flat_list(h) |> Enum.take(t))
      fn step -> :array.get(step, h_arr) end
    end
  end

  # Returns h values at the given intervention indices, handling scalar and tensor h.
  defp h_intervention_values(h, indices) when is_number(h) do
    List.duplicate(h * 1.0, length(indices))
  end

  defp h_intervention_values(%Nx.Tensor{} = h, indices) do
    if Nx.rank(h) == 0 do
      List.duplicate(Nx.to_number(h), length(indices))
    else
      idx_tensor = Nx.tensor(indices, type: {:s, 64})
      Nx.take(h, idx_tensor) |> Nx.to_flat_list()
    end
  end

  # Generates a counterfactual trajectory by forward simulation of a random walk
  # with observation noise, using Nx.Random for PRNG consistency with the rest
  # of the library.
  #
  # Given an `initial_state` (number), process variance `process_var`,
  # observation variance `obs_var`, and the number of steps `n_steps`, this
  # helper produces a list of counterfactual predicted observations by
  # iteratively adding Gaussian process noise and then observation noise.
  # An Nx.Random `key` controls reproducibility.  The returned list has
  # length `n_steps`.
  defp generate_counterfactual(initial_state, process_var, obs_var, n_steps, key) do
    sd_process = :math.sqrt(max(process_var, 0.0))
    sd_obs = :math.sqrt(max(obs_var, 0.0))

    # Split into 3 independent sub-keys: one for process noise, one for
    # observation noise, and one unused.  Using 3 parts ensures process
    # and observation streams share no common ancestor from a single split,
    # consistent with the per-step splitting in generate_structured_counterfactual.
    [key_process_row, key_obs_row] = Nx.Random.split(key, parts: 2) |> Nx.to_list()
    key_process = Nx.tensor(key_process_row, type: Nx.type(key))
    key_obs = Nx.tensor(key_obs_row, type: Nx.type(key))
    {process_noise, _} = Nx.Random.normal(key_process, 0.0, 1.0, shape: {n_steps})
    {obs_noise, _} = Nx.Random.normal(key_obs, 0.0, 1.0, shape: {n_steps})

    process_noise_list = Nx.to_flat_list(process_noise)
    obs_noise_list = Nx.to_flat_list(obs_noise)

    {_, values} =
      Enum.zip(process_noise_list, obs_noise_list)
      |> Enum.reduce({initial_state, []}, fn {pn, on}, {prev, acc} ->
        next_state = prev + pn * sd_process
        cf_obs = next_state + on * sd_obs
        {next_state, [cf_obs | acc]}
      end)

    Enum.reverse(values)
  end

  # Forward-simulates a multi-dimensional state-space model for counterfactual
  # generation: x_{t+1} = F * x_t + w_t,  y_t = H_t * x_t + v_t.
  #
  # Uses diagonal sampling for process noise (Q is always diagonal in the
  # structured sampler) to correctly handle zero-variance dimensions like
  # the lagged seasonal states.
  #
  # `post_h` is a list of {1,n} tensors, one per post-period time step.
  defp generate_structured_counterfactual(final_state, f, post_h, q_matrix, obs_var, n_steps, key) do
    state_dim = Nx.axis_size(f, 0)

    # Diagonal Q sampling: sqrt of each diagonal entry
    q_diag = Nx.take_diagonal(q_matrix)
    q_sds = Nx.sqrt(Nx.max(q_diag, Nx.tensor(0.0)))
    obs_sd = :math.sqrt(max(obs_var, 0.0))

    # Pre-extract all state/observation subkeys once to avoid per-step tensor slicing.
    subkey_rows = Nx.Random.split(key, parts: n_steps * 2) |> Nx.to_list()
    state_key_rows = Enum.take_every(subkey_rows, 2)
    obs_key_rows = subkey_rows |> Enum.drop(1) |> Enum.take_every(2)

    # Iterate directly over post_h list to avoid O(n²) Enum.at access
    {_, values} =
      Enum.zip(post_h, Enum.zip(state_key_rows, obs_key_rows))
      |> Enum.reduce({Nx.flatten(final_state), []}, fn {h_t, {state_key_row, obs_key_row}},
                                                       {prev_state, acc} ->
        key_state = Nx.tensor(state_key_row, type: Nx.type(key))
        key_obs = Nx.tensor(obs_key_row, type: Nx.type(key))

        # Process noise: sample N(0, I) then scale by sqrt(Q_diag)
        # Zero-variance dimensions get exactly zero noise
        {z_state, _} = Nx.Random.normal(key_state, 0.0, 1.0, shape: {state_dim})
        process_noise = Nx.multiply(z_state, q_sds)

        # State transition: x_{t+1} = F * x_t + w_t
        next_state = Nx.add(compat_dot(f, prev_state), process_noise)

        # Observation: y_t = H_t * x_t + v_t (using per-step H)
        h_row = if Nx.rank(h_t) == 2, do: Nx.squeeze(h_t, axes: [0]), else: Nx.flatten(h_t)
        predicted_y = Nx.to_number(compat_dot(h_row, next_state))

        {z_obs, _} = Nx.Random.normal(key_obs, 0.0, 1.0)
        cf_obs = predicted_y + Nx.to_number(z_obs) * obs_sd

        {next_state, [cf_obs | acc]}
      end)

    Enum.reverse(values)
  end

  # Returns a copy of spec with H sliced for the pre-period.
  # Handles: list of per-step tensors, static tensor (rank <= 2), and
  # time-varying tensor (rank 3, shape {t, ...}).
  defp slice_spec_h(%BstsNx.ModelSpec{} = spec, start_idx, count) do
    case spec.h do
      list when is_list(list) ->
        %{spec | h: Enum.slice(list, start_idx, count)}

      %Nx.Tensor{} = h ->
        if Nx.rank(h) >= 3 do
          %{spec | h: Nx.slice_along_axis(h, start_idx, count, axis: 0)}
        else
          spec
        end
    end
  end

  # Returns a list of H tensors for the post-period forward simulation.
  # Handles: list of per-step tensors, static tensor (rank <= 2), and
  # time-varying tensor (rank 3, shape {t, ...}).
  defp post_period_h(h, post_start_0based, n_post) do
    case h do
      list when is_list(list) ->
        require_h_coverage!(length(list), post_start_0based, n_post)
        Enum.slice(list, post_start_0based, n_post)

      %Nx.Tensor{} = t ->
        if Nx.rank(t) >= 3 do
          require_h_coverage!(Nx.axis_size(t, 0), post_start_0based, n_post)

          # Slice the time axis then split into a list of per-step tensors
          sliced = Nx.slice_along_axis(t, post_start_0based, n_post, axis: 0)
          Enum.map(0..(n_post - 1), fn i -> take_time_slice_at(sliced, i) end)
        else
          List.duplicate(t, n_post)
        end
    end
  end

  # Nx.dot handles all rank combinations used here; keep scalar/scalar explicit.
  defp compat_dot(a, b) do
    case {Nx.rank(a), Nx.rank(b)} do
      {0, 0} ->
        Nx.multiply(a, b)

      _ ->
        Nx.dot(a, b)
    end
  end

  defp require_h_coverage!(h_len, post_start_0based, n_post) do
    required_end = post_start_0based + n_post

    if required_end > h_len do
      raise ArgumentError,
            "structured model H covers #{h_len} steps, but post_period requires index range " <>
              "[#{post_start_0based}, #{required_end - 1}]"
    end
  end

  defp to_number(%Nx.Tensor{} = t), do: Nx.to_number(t)
  defp to_number(x) when is_number(x), do: x + 0.0

  defp take_time_slice_at(tensor, idx) do
    Nx.slice_along_axis(tensor, idx, 1, axis: 0)
    |> Nx.squeeze(axes: [0])
  end
end
