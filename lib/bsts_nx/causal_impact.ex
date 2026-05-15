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
  alias BstsNx.Validation
  import BstsNx.Utils, only: [split_key_at: 2]
  import Nx.Defn

  @default_alpha 0.05
  @near_zero_denominator 1.0e-15

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
    Must satisfy `pre_end < start_index <= end_index`.
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
  def estimate(observations, pre_period, post_period, opts \\ []) do
    period = period_context(observations, pre_period, post_period)

    # Determine number of samples
    num_samples = Keyword.get(opts, :num_samples, 200)
    burn_in = Keyword.get(opts, :burn_in, div(num_samples, 2))
    thin = Keyword.get(opts, :thin, 1)
    # Fit Gibbs sampler on the pre-period data only to sample latent states and variances.
    # Use provided burn_in and thin options to control chain length.
    # Default initial state to first observation for faster convergence.
    init_state = Keyword.get(opts, :initial_state, List.first(period.pre_data) || 0.0)
    init_cov = Keyword.get(opts, :initial_cov, 1.0)
    process_var = Keyword.get(opts, :process_var, 1.0)
    obs_var_init = Keyword.get(opts, :obs_var, 1.0)

    {sampler_key, cf_base_key} = derive_estimate_keys(opts)
    gibbs_opts = mcmc_opts(opts, sampler_key, burn_in, thin)

    pre_samples =
      GibbsSampler.sample(
        period.pre_data,
        num_samples,
        init_state,
        init_cov,
        process_var,
        obs_var_init,
        gibbs_opts
      )

    # Collect point effects and counterfactual predictions per sample by forward simulation
    {baselines, point_effects_list} =
      simulate_effects(pre_samples, period.post_data, cf_base_key, fn sample, key ->
        # final state of pre-period
        final_state = List.last(sample.states)
        init_val = Nx.to_number(final_state)
        q = Nx.to_number(sample.process_var)
        r = Nx.to_number(sample.obs_var)
        # generate counterfactual by forward simulation (random walk with observation noise)
        generate_counterfactual(init_val, q, r, period.n_post, key, period.gap_count)
      end)

    build_impact_result(
      period.post_data,
      baselines,
      point_effects_list,
      period.pre_period,
      period.post_period
    )
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
        pre_period,
        post_period,
        %BstsNx.ModelSpec{} = spec,
        opts \\ []
      ) do
    period = period_context(observations, pre_period, post_period)

    num_samples = Keyword.get(opts, :num_samples, 200)
    burn_in = Keyword.get(opts, :burn_in, div(num_samples, 2))
    thin = Keyword.get(opts, :thin, 1)

    {sampler_key, cf_base_key} = derive_estimate_keys(opts)
    gibbs_opts = mcmc_opts(opts, sampler_key, burn_in, thin)

    # Slice H for the pre-period so sample_structured gets matching lengths
    state_dim = Nx.axis_size(spec.f, 0)
    pre_spec = slice_spec_h(spec, period.pre_start - 1, length(period.pre_data), state_dim)

    pre_samples =
      GibbsSampler.sample_structured(period.pre_data, pre_spec, num_samples, gibbs_opts)

    # Get post-period H entries for counterfactual forward simulation
    post_h = post_period_h(spec.h, period.post_start - 1, period.n_post, state_dim)

    {baselines, point_effects_list} =
      simulate_effects(pre_samples, period.post_data, cf_base_key, fn sample, key ->
        final_state = List.last(sample.states)
        q_mat = sample.q_matrix
        r = Nx.to_number(sample.obs_var)

        # Forward-simulate the state-space model for counterfactual
        generate_structured_counterfactual(
          final_state,
          spec.f,
          post_h,
          q_mat,
          r,
          key,
          period.gap_count
        )
      end)

    build_impact_result(
      period.post_data,
      baselines,
      point_effects_list,
      period.pre_period,
      period.post_period
    )
  end

  defp period_context(
         observations,
         {pre_start, pre_end} = pre_period,
         {post_start, post_end} = post_period
       ) do
    # Coerce any Nx scalar elements to plain numbers so downstream
    # Enum arithmetic (:math.pow, Enum.sum, etc.) doesn't crash.
    observations = Enum.map(observations, &to_number/1)
    obs_count = length(observations)

    Validation.validate_study_periods!(obs_count, pre_start, pre_end, post_start, post_end)

    pre_data = Enum.slice(observations, pre_start - 1, pre_end - pre_start + 1)
    post_data = Enum.slice(observations, post_start - 1, post_end - post_start + 1)

    %{
      observations: observations,
      pre_data: pre_data,
      post_data: post_data,
      pre_period: pre_period,
      post_period: post_period,
      pre_start: pre_start,
      pre_end: pre_end,
      post_start: post_start,
      post_end: post_end,
      n_post: post_end - post_start + 1,
      gap_count: post_start - pre_end - 1
    }
  end

  defp mcmc_opts(opts, sampler_key, burn_in, thin) do
    opts
    |> Keyword.delete(:num_samples)
    |> Keyword.delete(:seed)
    |> Keyword.put(:key, sampler_key)
    |> Keyword.put(:burn_in, burn_in)
    |> Keyword.put(:thin, thin)
  end

  defp simulate_effects(samples, post_data, cf_base_key, counterfactual_fun) do
    cf_keys = Nx.Random.split(cf_base_key, parts: length(samples))

    samples
    |> Enum.with_index()
    |> Enum.map(fn {sample, idx} ->
      cf = counterfactual_fun.(sample, split_key_at(cf_keys, idx))
      point_effects = point_effects(post_data, cf)

      {cf, point_effects}
    end)
    |> Enum.unzip()
  end

  defp point_effects(actual_vals, baseline_vals) do
    Enum.zip(actual_vals, baseline_vals)
    |> Enum.map(fn {actual, baseline} -> actual - baseline end)
  end

  defp build_impact_result(post_data, baselines, point_effects_list, pre_period, post_period) do
    cumulative_effects = Enum.map(point_effects_list, &Enum.sum/1)
    baseline_sums = Enum.map(baselines, &Enum.sum/1)
    relative_effects = Validation.relative_effects(cumulative_effects, baseline_sums)

    %{
      point_effects: point_effects_list,
      cumulative_effects: cumulative_effects,
      relative_effects: relative_effects,
      actual: post_data,
      counterfactual: baselines,
      pre_period: pre_period,
      post_period: post_period
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
    lower_q = 0.025
    upper_q = 0.975

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
              effect_columns =
                result.point_effects
                |> Enum.zip()
                |> Enum.map(&Tuple.to_list/1)

              {lower, upper} =
                effect_columns
                |> Enum.map(&percentile_bounds_linear(&1, lower_q, upper_q))
                |> Enum.unzip()

              {lower, upper}
            end

          Enum.map(0..(n_post - 1), fn idx ->
            %{
              mean: Enum.at(means, idx),
              sd: Enum.at(sds, idx),
              lower: Enum.at(lowers, idx),
              upper: Enum.at(uppers, idx)
            }
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
          {lower, upper} = percentile_bounds_linear(vals, lower_q, upper_q)

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
  def estimate_from_filter(observations, intervention_indices, opts \\ [])

  def estimate_from_filter(observations, intervention_indices, opts) when is_list(observations) do
    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      observations
      |> Nx.tensor(type: {:f, 64})
      |> estimate_from_filter(intervention_indices, opts)
    end)
  end

  def estimate_from_filter(%Nx.Tensor{} = obs_tensor, intervention_indices, opts) do
    t = Nx.axis_size(obs_tensor, 0)
    f = Keyword.get(opts, :f, 1.0)
    h = Keyword.get(opts, :h, 1.0)
    q = Keyword.get(opts, :q, 1.0)
    r = Keyword.get(opts, :r, 1.0)
    x0 = Keyword.get(opts, :x0, 0.0)
    p0 = Keyword.get(opts, :p0, 1.0)
    alpha = Keyword.get(opts, :alpha, @default_alpha)

    Validation.validate_alpha!(alpha)

    valid_indices = valid_filter_indices(intervention_indices, t)

    if valid_indices == [] do
      # No valid intervention indices — return zero-effect result
      empty_filter_summary(r_to_number(r), false)
    else
      masked_obs = mask_observations_at_indices(obs_tensor, t, valid_indices)

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

      baseline_variance_vals =
        Enum.zip(h_vals, state_var_vals)
        |> Enum.map(fn {hi, v} -> max(hi * hi * v, 0.0) end)

      # Point effects
      point_effects_mean = point_effects(actual_vals, baseline_vals)

      # Observation-level variance: h_t^2 * p_t + r
      r_num = r_to_number(r)

      point_sds =
        Enum.zip(h_vals, state_var_vals)
        |> Enum.map(fn {hi, v} -> :math.sqrt(max(hi * hi * v + r_num, 0.0)) end)

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
      cum_sd =
        scalar_cumulative_sd_with_cross_covariance(
          valid_indices,
          h,
          h_vals,
          state_var_vals,
          ps,
          sps,
          f,
          q,
          r_num,
          t
        )

      build_filter_summary(%{
        actual: actual_vals,
        baseline: baseline_vals,
        baseline_variance: baseline_variance_vals,
        point_effects: point_effects_mean,
        point_sds: point_sds,
        cumulative_sd: cum_sd,
        obs_variance: r_num,
        alpha: alpha,
        cross_cov_included: true
      })
    end
  end

  @doc """
  Non-MCMC causal impact estimation for structured models using the
  compiled Kalman filter and RTS matrix smoother.

  This provides a fast path for `estimate_structured/5` without Gibbs sampling.
  It computes the block-diagonal F, Q, and observation variance R from the
  priors (or initialized values) of the `.model_spec` and executes a single
  pass of the state-space smoother.

  * `observations` – list or tensor of observations.
  * `intervention_indices` – list of 0-based index marking the on-air period.
  * `spec` – A `%BstsNx.ModelSpec{}` defining the state space architecture.
  * `opts` – keyword list of options.

  Returns the standard impact summary map.
  """
  @spec estimate_structured_from_filter(
          Nx.t() | [number()],
          [non_neg_integer()],
          BstsNx.ModelSpec.t(),
          keyword()
        ) :: map()
  def estimate_structured_from_filter(observations, intervention_indices, spec, opts \\ [])

  def estimate_structured_from_filter(observations, intervention_indices, spec, opts)
      when is_list(observations) do
    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      observations
      |> Nx.tensor(type: {:f, 64})
      |> estimate_structured_from_filter(intervention_indices, spec, opts)
    end)
  end

  def estimate_structured_from_filter(%Nx.Tensor{} = obs_tensor, intervention_indices, spec, opts) do
    t = Nx.axis_size(obs_tensor, 0)
    alpha = Keyword.get(opts, :alpha, @default_alpha)

    Validation.validate_alpha!(alpha)

    valid_indices = valid_filter_indices(intervention_indices, t)

    if valid_indices == [] do
      empty_filter_summary(0.0, false)
    else
      # Extract fixed structural matrices from the configured ModelSpec values.
      # This filter-only path does not resample variances; callers that want
      # prior/posterior adaptation should use the MCMC path.
      state_dim = Nx.axis_size(spec.f, 0)
      q_matrix = Validation.fixed_q_matrix(spec.q_specs, state_dim)

      r_val = to_number(spec.obs_var)
      r_tensor = Nx.tensor(r_val, type: {:f, 64})

      x0 = spec.x0 |> BstsNx.Utils.to_tensor() |> Nx.as_type({:f, 64}) |> Nx.flatten()
      p0 = spec.p0 |> BstsNx.Utils.to_tensor() |> Nx.as_type({:f, 64})

      masked_obs = mask_observations_at_indices(obs_tensor, t, valid_indices)

      h_tensor = normalize_structured_h_for_filter(spec.h, t, state_dim)

      # Filter
      {filt_xs, filt_ps} =
        BstsNx.KalmanFilter.filter_defn_multi(
          masked_obs,
          spec.f,
          h_tensor,
          q_matrix,
          r_tensor,
          x0,
          p0
        )

      # Smooth
      {sxs, sps} = BstsNx.Smoother.rts_defn_matrix(filt_xs, filt_ps, spec.f, q_matrix)

      # Extract intervention period values
      idx_tensor = Nx.tensor(valid_indices, type: {:s, 64})
      actual_vals = Nx.take(obs_tensor, idx_tensor) |> Nx.to_flat_list()

      # Project smoothed states through H at each valid interval
      h_interventions = Nx.take(h_tensor, idx_tensor)
      sxs_interventions = Nx.take(sxs, idx_tensor)
      sps_interventions = Nx.take(sps, idx_tensor)

      # Baseline: H_t * x_t  →  one scalar per time step.
      # h_interventions may be {n, 1, state_dim} (3D) or {n, state_dim} (2D)
      # depending on how H was broadcast.  Flatten all but the leading batch axis.
      h_flat = Nx.reshape(h_interventions, {Nx.axis_size(h_interventions, 0), :auto})

      baseline_vals =
        Nx.sum(Nx.multiply(h_flat, sxs_interventions), axes: [1]) |> Nx.to_flat_list()

      point_effects_mean = point_effects(actual_vals, baseline_vals)

      # Observation-level variance: H_t * P_t * H_t^T + R.
      # Compute this with explicit batched multiplies to avoid backend-specific
      # broadcasting differences in Nx.dot for rank-3 tensors.
      h_rows =
        case Nx.rank(h_interventions) do
          2 ->
            h_interventions

          3 ->
            cond do
              Nx.axis_size(h_interventions, 1) == 1 ->
                Nx.squeeze(h_interventions, axes: [1])

              Nx.axis_size(h_interventions, 2) == 1 ->
                Nx.squeeze(h_interventions, axes: [2])

              true ->
                raise ArgumentError, "expected singleton observation axis in structured H rows"
            end
        end

      # (P * h^T) for each time step -> {n, dim}
      ph = Nx.sum(Nx.multiply(sps_interventions, Nx.new_axis(h_rows, 1)), axes: [2])

      baseline_projected_vars = Nx.sum(Nx.multiply(ph, h_rows), axes: [1])

      # Ensure no negative variances due to precision
      baseline_projected_vars =
        Nx.select(
          Nx.less(baseline_projected_vars, 0.0),
          Nx.tensor(0.0),
          baseline_projected_vars
        )

      p_projected_vars = Nx.add(baseline_projected_vars, r_tensor)

      point_sds = Nx.sqrt(p_projected_vars) |> Nx.to_flat_list()
      baseline_variance_vals = baseline_projected_vars |> Nx.to_flat_list()

      diag_var = Nx.sum(p_projected_vars) |> Nx.to_number()

      cross_var_sum =
        structured_smoother_cross_covariance_sum(
          valid_indices,
          h_tensor,
          filt_ps,
          sps,
          spec.f,
          q_matrix
        )

      cum_sd = :math.sqrt(max(diag_var + 2.0 * cross_var_sum, 0.0))

      build_filter_summary(%{
        actual: actual_vals,
        baseline: baseline_vals,
        baseline_variance: baseline_variance_vals,
        point_effects: point_effects_mean,
        point_sds: point_sds,
        cumulative_sd: cum_sd,
        obs_variance: r_val,
        alpha: alpha,
        cross_cov_included: true
      })
    end
  end

  defp valid_filter_indices(intervention_indices, t) do
    intervention_indices
    |> Enum.filter(fn idx -> idx >= 0 and idx < t end)
    |> Enum.uniq()
    |> Enum.sort()
  end

  defp empty_filter_summary(obs_variance, cross_cov_included) do
    %{
      point_effects: %{mean: [], lower: [], upper: []},
      cumulative_effect: %{
        mean: 0.0,
        sd: 0.0,
        lower: 0.0,
        upper: 0.0,
        cross_cov_included: cross_cov_included
      },
      relative_effect: %{mean: 0.0, sd: 0.0, lower: 0.0, upper: 0.0},
      actual: [],
      baseline: [],
      baseline_variance: [],
      obs_variance: obs_variance
    }
  end

  defp build_filter_summary(%{
         actual: actual_vals,
         baseline: baseline_vals,
         baseline_variance: baseline_variance_vals,
         point_effects: point_effects_mean,
         point_sds: point_sds,
         cumulative_sd: cum_sd,
         obs_variance: obs_variance,
         alpha: alpha,
         cross_cov_included: cross_cov_included
       }) do
    z = BstsNx.Utils.z_score(alpha)
    cum_mean = Enum.sum(point_effects_mean)
    baseline_sum = Enum.sum(baseline_vals)
    relative_effect = Validation.relative_effect_summary(cum_mean, cum_sd, baseline_sum, z)

    point_lower =
      Enum.zip(point_effects_mean, point_sds)
      |> Enum.map(fn {m, s} -> m - z * s end)

    point_upper =
      Enum.zip(point_effects_mean, point_sds)
      |> Enum.map(fn {m, s} -> m + z * s end)

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
        upper: cum_mean + z * cum_sd,
        cross_cov_included: cross_cov_included
      },
      relative_effect: relative_effect,
      actual: actual_vals,
      baseline: baseline_vals,
      baseline_variance: baseline_variance_vals,
      obs_variance: obs_variance
    }
  end

  defp scalar_cumulative_sd_with_cross_covariance(
         valid_indices,
         h,
         h_vals,
         state_var_vals,
         ps,
         sps,
         f,
         q,
         r_num,
         t
       ) do
    n_intervention = length(valid_indices)

    # Diagonal variance: sum(h_i^2 * P^s_i)
    diag_var =
      Enum.zip(h_vals, state_var_vals)
      |> Enum.map(fn {hi, v} -> hi * hi * v end)
      |> Enum.sum()

    cross_var_sum =
      if n_intervention >= 2 do
        min_idx = Enum.min(valid_indices)
        max_idx = Enum.max(valid_indices)
        f_num = if is_number(f), do: f * 1.0, else: Nx.to_number(Nx.tensor(f))
        q_num = if is_number(q), do: q * 1.0, else: Nx.to_number(Nx.tensor(q))

        all_ps = ps |> Nx.to_flat_list() |> :array.from_list()
        all_sps = sps |> Nx.to_flat_list() |> :array.from_list()
        intervention_set = MapSet.new(valid_indices)
        h_at = h_at_step_fn(h, t)

        gain_at = fn k ->
          pf = :array.get(k, all_ps)
          denom = f_num * pf * f_num + q_num
          if abs(denom) < @near_zero_denominator, do: 0.0, else: pf * f_num / denom
        end

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

    cum_var = diag_var + n_intervention * r_num + 2.0 * cross_var_sum
    :math.sqrt(max(cum_var, 0.0))
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

  defp normalize_structured_h_for_filter(h, t, state_dim) when is_list(h) do
    h_rows = h |> Enum.map(&BstsNx.Utils.h_to_row_tensor/1) |> Nx.stack()

    cond do
      Nx.axis_size(h_rows, 0) == t ->
        h_rows

      Nx.axis_size(h_rows, 0) == 1 ->
        Nx.broadcast(Nx.squeeze(h_rows, axes: [0]), {t, state_dim})

      true ->
        raise ArgumentError,
              "structured model H covers #{Nx.axis_size(h_rows, 0)} steps, but observations require #{t}"
    end
  end

  defp normalize_structured_h_for_filter(%Nx.Tensor{} = h, t, state_dim) do
    cond do
      Nx.rank(h) == 0 and state_dim == 1 ->
        Nx.broadcast(h, {t, 1})

      Nx.rank(h) == 1 and Nx.axis_size(h, 0) == state_dim ->
        Nx.broadcast(h, {t, state_dim})

      Nx.rank(h) == 2 and Nx.shape(h) == {1, state_dim} ->
        Nx.broadcast(Nx.squeeze(h, axes: [0]), {t, state_dim})

      Nx.rank(h) == 2 and Nx.shape(h) == {t, state_dim} ->
        h

      Nx.rank(h) == 3 and Nx.shape(h) == {t, 1, state_dim} ->
        Nx.squeeze(h, axes: [1])

      true ->
        raise ArgumentError,
              "h must be shape {#{state_dim}}, {1, #{state_dim}}, or {#{t}, #{state_dim}} for structured filter, got: #{inspect(Nx.shape(h))}"
    end
  end

  defp structured_smoother_cross_covariance_sum(valid_indices, h_tensor, filt_ps, sps, f, q) do
    case valid_indices do
      [] ->
        0.0

      [_single] ->
        0.0

      indices ->
        f = Nx.as_type(f, {:f, 64})
        q = Nx.as_type(q, {:f, 64})

        gains =
          indices
          |> Enum.min_max()
          |> then(fn {min_idx, max_idx} ->
            if min_idx >= max_idx do
              %{}
            else
              Map.new(min_idx..(max_idx - 1), fn idx ->
                {idx, smoother_gain_at(filt_ps, f, q, idx)}
              end)
            end
          end)

        for i <- indices, j <- indices, i < j, reduce: 0.0 do
          acc ->
            cov_ij = smooth_state_cross_covariance(i, j, sps, gains)
            h_i = BstsNx.Utils.take_time_slice_at(h_tensor, i)
            h_j = BstsNx.Utils.take_time_slice_at(h_tensor, j)

            projected =
              h_i
              |> Nx.dot(cov_ij)
              |> Nx.dot(h_j)
              |> Nx.to_number()

            acc + projected
        end
    end
  end

  defp smoother_gain_at(filt_ps, f, q, idx) do
    p_filt = BstsNx.Utils.take_time_slice_at(filt_ps, idx) |> Nx.as_type({:f, 64})
    p_pred_next = Nx.add(Nx.dot(Nx.dot(f, p_filt), Nx.transpose(f)), q)
    rhs = Nx.dot(f, p_filt)
    BstsNx.Utils.safe_solve(p_pred_next, rhs) |> Nx.transpose()
  end

  defp smooth_state_cross_covariance(i, j, sps, gains) when i < j do
    Enum.reduce((j - 1)..i//-1, BstsNx.Utils.take_time_slice_at(sps, j), fn idx, cov ->
      gains |> Map.fetch!(idx) |> Nx.dot(cov)
    end)
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
  defp generate_counterfactual(initial_state, process_var, obs_var, n_steps, key, gap_steps) do
    sd_process = :math.sqrt(max(process_var, 0.0))
    sd_obs = :math.sqrt(max(obs_var, 0.0))
    total_steps = gap_steps + n_steps

    # Split into 2 independent sub-keys: one for process noise, one for
    # observation noise. Sharing no common ancestor from a single split
    # ensures mathematical safety against PRNG collision.
    keys = Nx.Random.split(key, parts: 2)
    key_process = split_key_at(keys, 0)
    key_obs = split_key_at(keys, 1)

    {z_process, _} = Nx.Random.normal(key_process, 0.0, 1.0, shape: {total_steps})
    {z_obs, _} = Nx.Random.normal(key_obs, 0.0, 1.0, shape: {n_steps})

    # Vectorized state simulation: x_t = x_0 + sum_{i=1}^t w_i
    process_noise = Nx.multiply(z_process, sd_process)
    states = Nx.add(initial_state, Nx.cumulative_sum(process_noise))
    post_states = Nx.slice_along_axis(states, gap_steps, n_steps, axis: 0)

    # Vectorized observation simulation: y_t = x_t + v_t
    obs_noise = Nx.multiply(z_obs, sd_obs)
    cf_obs = Nx.add(post_states, obs_noise)

    Nx.to_flat_list(cf_obs)
  end

  defp generate_structured_counterfactual(
         final_state,
         f,
         post_h,
         q_matrix,
         obs_var,
         key,
         gap_steps
       ) do
    # Convert list of post-period H tensors into a single stacked tensor {n_steps, state_dim}
    # This enables running the entire simulation gracefully inside a compiled Nx.Defn.while loop
    post_h_tensor =
      post_h
      |> Enum.map(fn h_t ->
        if Nx.rank(h_t) == 2, do: Nx.squeeze(h_t, axes: [0]), else: Nx.flatten(h_t)
      end)
      |> Nx.stack()

    state_dim = Nx.axis_size(f, 0)
    total_steps = gap_steps + Nx.axis_size(post_h_tensor, 0)
    keys = Nx.Random.split(key, parts: 2)
    key_state = split_key_at(keys, 0)
    key_obs = split_key_at(keys, 1)

    {z_state, _} = Nx.Random.normal(key_state, 0.0, 1.0, shape: {total_steps, state_dim})
    {z_obs, _} = Nx.Random.normal(key_obs, 0.0, 1.0, shape: {Nx.axis_size(post_h_tensor, 0)})

    cf_obs_t =
      generate_structured_counterfactual_defn(
        Nx.flatten(final_state),
        f,
        post_h_tensor,
        q_matrix,
        obs_var,
        z_state,
        z_obs,
        gap_steps
      )

    Nx.to_flat_list(cf_obs_t)
  end

  defnp take_time_slice_defn(tensor, idx) do
    Nx.slice_along_axis(tensor, idx, 1, axis: 0) |> Nx.squeeze(axes: [0])
  end

  defn generate_structured_counterfactual_defn(
         final_state,
         f_mat,
         post_h_tensor,
         q_matrix,
         obs_var,
         z_state_raw,
         z_obs_raw,
         gap_steps
       ) do
    type = Nx.type(final_state)
    f_t = Nx.as_type(f_mat, type)
    h_t = Nx.as_type(post_h_tensor, type)
    q_t = Nx.as_type(q_matrix, type)
    obs_var_t = Nx.as_type(obs_var, type)
    {n_steps, _} = Nx.shape(h_t)
    z_state = Nx.as_type(z_state_raw, type)
    z_obs = Nx.as_type(z_obs_raw, type)

    q_diag = Nx.take_diagonal(q_t)
    q_sds = Nx.sqrt(Nx.max(q_diag, Nx.tensor(0.0, type: type)))
    obs_sd = Nx.sqrt(Nx.max(obs_var_t, Nx.tensor(0.0, type: type)))

    {_, gap_state, _, _, _, _} =
      while {i = 0, prev_state = final_state, z_s = z_state, f = f_t, q_s = q_sds,
             gap = gap_steps},
            i < gap do
        z_s_i = take_time_slice_defn(z_s, i)
        process_noise = Nx.multiply(z_s_i, q_s)
        next_state = Nx.add(Nx.dot(f, prev_state), process_noise)
        {i + 1, next_state, z_s, f, q_s, gap}
      end

    cf_obs = Nx.broadcast(Nx.tensor(0.0, type: type), {n_steps})

    {_, _, final_obs, _, _, _, _, _, _, _} =
      while {i = 0, prev_state = gap_state, obs_acc = cf_obs, z_s = z_state, z_o = z_obs,
             h_tensor = h_t, f = f_t, q_s = q_sds, o_s = obs_sd, gap = gap_steps},
            i < n_steps do
        z_s_i = take_time_slice_defn(z_s, i + gap)
        process_noise = Nx.multiply(z_s_i, q_s)

        # State transition: x_{t+1} = F * x_t + w_t
        next_state = Nx.add(Nx.dot(f, prev_state), process_noise)

        # Observation: y_t = H_t * x_t + v_t
        h_row = take_time_slice_defn(h_tensor, i)
        predicted_y = Nx.dot(h_row, next_state)

        z_o_i = take_time_slice_defn(z_o, i)
        cf_obs_i = Nx.add(predicted_y, Nx.multiply(z_o_i, o_s))

        new_obs_acc = Nx.put_slice(obs_acc, [i], Nx.new_axis(Nx.squeeze(cf_obs_i), 0))

        {i + 1, next_state, new_obs_acc, z_s, z_o, h_tensor, f, q_s, o_s, gap}
      end

    final_obs
  end

  # Returns a copy of spec with H sliced for the pre-period.
  # Handles: list of per-step tensors, static tensor (rank <= 2), and
  # time-varying tensor (rank 3, shape {t, ...}).
  defp slice_spec_h(%BstsNx.ModelSpec{} = spec, start_idx, count, state_dim) do
    case spec.h do
      list when is_list(list) ->
        %{spec | h: Enum.slice(list, start_idx, count)}

      %Nx.Tensor{} = h ->
        if time_varying_h_tensor?(h, state_dim) do
          %{spec | h: Nx.slice_along_axis(h, start_idx, count, axis: 0)}
        else
          spec
        end
    end
  end

  # Returns a list of H tensors for the post-period forward simulation.
  # Handles: list of per-step tensors, static tensor (rank <= 2), and
  # time-varying tensor (rank 3, shape {t, ...}).
  defp post_period_h(h, post_start_0based, n_post, state_dim) do
    case h do
      list when is_list(list) ->
        require_h_coverage!(length(list), post_start_0based, n_post)
        Enum.slice(list, post_start_0based, n_post)

      %Nx.Tensor{} = t ->
        if time_varying_h_tensor?(t, state_dim) do
          require_h_coverage!(Nx.axis_size(t, 0), post_start_0based, n_post)

          # Slice the time axis then split into a list of per-step tensors
          sliced = Nx.slice_along_axis(t, post_start_0based, n_post, axis: 0)
          Enum.map(0..(n_post - 1), fn i -> BstsNx.Utils.take_time_slice_at(sliced, i) end)
        else
          List.duplicate(t, n_post)
        end
    end
  end

  defp time_varying_h_tensor?(%Nx.Tensor{} = h, state_dim) do
    case Nx.rank(h) do
      2 -> Nx.axis_size(h, 0) > 1 and Nx.axis_size(h, 1) == state_dim
      rank when rank >= 3 -> true
      _ -> false
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

  defp mask_observations_at_indices(obs_tensor, t, indices) do
    mask_flags = :array.new(t, default: 0)

    mask_flags =
      Enum.reduce(indices, mask_flags, fn idx, arr ->
        :array.set(idx, 1, arr)
      end)

    mask_list = Enum.map(0..(t - 1), fn i -> :array.get(i, mask_flags) end)
    mask = Nx.equal(Nx.tensor(mask_list, type: {:u, 8}), 1)
    nan_vec = Nx.broadcast(Nx.Constants.nan(), {t})
    Nx.select(mask, nan_vec, obs_tensor)
  end

  defp to_number(%Nx.Tensor{} = t), do: Nx.to_number(t)
  defp to_number(x) when is_number(x), do: x + 0.0

  defp percentile_bounds_linear(values, lower_quantile, upper_quantile) when is_list(values) do
    sorted =
      values
      |> Enum.map(&to_number/1)
      |> Enum.sort()

    case sorted do
      [] ->
        {:nan, :nan}

      [v] ->
        {v, v}

      _ ->
        last = length(sorted) - 1
        sorted_t = List.to_tuple(sorted)

        {
          linear_percentile(sorted_t, last, lower_quantile),
          linear_percentile(sorted_t, last, upper_quantile)
        }
    end
  end

  defp linear_percentile(sorted_t, last, quantile) do
    position = quantile * last
    lo = trunc(Float.floor(position))
    hi = trunc(Float.ceil(position))
    weight = position - lo

    lo_v = elem(sorted_t, lo)
    hi_v = elem(sorted_t, hi)
    lo_v + weight * (hi_v - lo_v)
  end

  defp derive_estimate_keys(opts) do
    root_key =
      case Keyword.fetch(opts, :key) do
        {:ok, %Nx.Tensor{} = key} ->
          validate_prng_key!(key)

        {:ok, other} ->
          raise ArgumentError,
                "key must be an Nx.Random key tensor of shape {2}, got: #{inspect(other)}"

        :error ->
          Nx.Random.key(Keyword.get(opts, :seed, System.os_time()))
      end

    split_keys = Nx.Random.split(root_key, parts: 2)
    {split_key_at(split_keys, 0), split_key_at(split_keys, 1)}
  end

  defp validate_prng_key!(%Nx.Tensor{} = key) do
    if Nx.shape(key) == {2} do
      key
    else
      raise ArgumentError,
            "key must be an Nx.Random key tensor of shape {2}, got shape #{inspect(Nx.shape(key))}"
    end
  end
end
