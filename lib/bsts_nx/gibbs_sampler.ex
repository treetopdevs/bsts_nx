defmodule BstsNx.GibbsSampler do
  @moduledoc """
  Gibbs sampler for linear Gaussian state-space models.

  This module implements Gibbs samplers for Bayesian structural time series,
  sampling both the latent state trajectory and variance parameters from
  their inverse-gamma posteriors.

  Two sampler families are provided:

    * **Scalar samplers** (`sample/6`, `sample/7`, `sample_general/5`,
      `sample_chains/8`) — for local level and scalar regression models
      with a single process variance Q and observation variance R.

    * **Structured samplers** (`sample_structured/4`,
      `sample_structured_chains/5`) — for multi-dimensional state-space
      models (local-linear-trend, regression, composed models) where each
      diagonal entry of Q is resampled independently from its own
      inverse-gamma posterior.

  Both use the Kalman filter, RTS smoother and Carter-Kohn simulation
  smoother for state trajectory sampling.  Burn-in, thinning and
  deterministic PRNG control are supported throughout.
  """

  alias BstsNx.Distributions
  alias BstsNx.KalmanFilter
  alias BstsNx.ModelSpec

  import BstsNx.Utils,
    only: [to_tensor: 1, missing_observation?: 1, compat_dot: 2, split_key_at: 2]

  require Logger

  @rank_update_pivot_floor 1.0e-12

  @type sample_result :: %{
          states: [Nx.t()],
          state_covs: [Nx.t()],
          process_var: Nx.t(),
          obs_var: Nx.t()
        }

  @doc """
  Runs a Gibbs sampler for a simple local level state-space model
  `x_t = x_{t-1} + w_t` and `y_t = x_t + v_t` for a given
  observation sequence.  Returns a list of samples where each
  sample includes the latent state sequence and the sampled
  process and observation variances.

  * `observations` – list of scalar observations `y_t`. Missing values may
    be expressed as `nil` or NaN (number/tensor). These are supported but
    are excluded from the observation variance update, and a warning is logged.
  * `num_samples` – number of Gibbs iterations to run.
  * `initial_state` – prior mean for the latent state at time 0.
  * `initial_cov` – prior variance for the latent state at time 0.
  * `process_var` – initial guess for the process noise variance.
  * `obs_var` – initial guess for the observation noise variance.

  Returns a list of maps; each map has keys `:states`, `:state_covs`,
  `:process_var` and `:obs_var`.
  """
  @spec sample(
          [number | list | Nx.t()],
          pos_integer,
          number | list | Nx.t(),
          number | list | Nx.t(),
          number | list | Nx.t(),
          number | list | Nx.t()
        ) :: [sample_result]
  def sample(observations, num_samples, initial_state, initial_cov, process_var, obs_var) do
    # backward compatible call without options delegates to extended version
    sample(observations, num_samples, initial_state, initial_cov, process_var, obs_var, [])
  end

  @doc """
  Runs a Gibbs sampler for the local level state‑space model with optional burn‑in,
  thinning and reproducible randomness.

  This function extends the six‑argument `sample/6` by accepting an options
  keyword list.  Recognised options include:

    * `:burn_in` – number of initial iterations to discard (default: 0).
    * `:thin` – interval between retained samples.  For example, `thin: 2`
      keeps every second sample after burn‑in (default: 1).
    * `:seed` – integer seed used to initialise the PRNG key.  If not
      provided, `System.os_time/0` is used.
    * `:key` – an `Nx.Random` PRNG key.  If supplied, this key takes
      precedence over `:seed` for controlling randomness.
    * `:prior_shape` – shape parameter for the inverse‑gamma prior on both
      process and observation variances (default: 1.0).
    * `:prior_scale` – scale parameter for the inverse‑gamma prior on both
      process and observation variances (default: 1.0).

  The sampler returns a list of `num_samples` draws, after applying burn‑in
  and thinning.  Each draw is a map containing the latent state trajectory,
  filtered state covariances, and sampled process and observation variances.
  """
  @spec sample(
          [number | list | Nx.t()],
          pos_integer,
          number | list | Nx.t(),
          number | list | Nx.t(),
          number | list | Nx.t(),
          number | list | Nx.t(),
          keyword()
        ) :: [sample_result]
  def sample(observations, num_samples, initial_state, initial_cov, process_var, obs_var, opts) do
    merged_opts =
      Keyword.merge(opts,
        initial_state: initial_state,
        initial_cov: initial_cov,
        process_var: process_var,
        obs_var: obs_var
      )

    sample_general(observations, 1.0, 1.0, num_samples, merged_opts)
  end

  @doc """
  Runs multiple independent Gibbs sampler chains.

  * `observations` – list of observations.
  * `num_chains` – number of independent chains to run.
  * `num_samples` – number of samples to collect per chain (after burn‑in and thinning).
  * `initial_state`, `initial_cov`, `process_var`, `obs_var` – as in `sample/7`.
  * `opts` – options forwarded to `sample/7`.  Each chain uses a distinct
    seed derived from `:seed` and the chain index.  You can override this
    behaviour by supplying a list of seeds via `:seeds` with length equal
    to `num_chains`.  When a `:key` is provided instead of `:seed`, it is
    split into `num_chains` independent subkeys.
  * `:timeout` – per-chain timeout in milliseconds (default: 300_000 / 5 min).

  Returns a list of chains, where each chain is the result of a call to
  `sample/7`.
  """
  @spec sample_chains(
          [number | list | Nx.t()],
          pos_integer,
          pos_integer,
          number | list | Nx.t(),
          number | list | Nx.t(),
          number | list | Nx.t(),
          number | list | Nx.t(),
          keyword()
        ) :: [[sample_result]]
  def sample_chains(
        observations,
        num_chains,
        num_samples,
        initial_state,
        initial_cov,
        process_var,
        obs_var,
        opts \\ []
      ) do
    validate_positive!(:num_chains, num_chains)

    run_chains(
      num_chains,
      opts,
      "Gibbs sampler",
      "Gibbs sampler",
      fn chain_opts ->
        sample(
          observations,
          num_samples,
          initial_state,
          initial_cov,
          process_var,
          obs_var,
          chain_opts
        )
      end
    )
  end

  @doc """
  Generalized Gibbs sampler that accepts a time-varying observation vector `h`.

  Unlike `sample/7` which hardcodes a scalar local level model (`F=1, H=1`),
  this function accepts explicit `f` (transition matrix) and `h` (observation
  vector) arguments.  When `h` is a 1-D tensor of length `t`, element `h[i]`
  is the observation coefficient at time step `i`.  This enables seasonal
  regression models where Fourier predictions serve as the time-varying
  regressor.

  * `observations` – list of scalar observations. Missing values may be `nil`
    or NaN (number/tensor).
  * `f` – scalar state transition parameter.
  * `h` – observation coefficient. Either a scalar (broadcast to all steps) or
    a 1-D tensor of length `t`.
  * `num_samples` – number of post-burn-in samples to return.
  * `opts` – keyword list with `:burn_in`, `:thin`, `:seed`, `:key`,
    `:initial_state`, `:initial_cov`, `:process_var`, `:obs_var`,
    `:prior_shape`, `:prior_scale`.

  Returns a list of sample maps identical in shape to `sample/7`.
  """
  @spec sample_general(
          [number | nil | Nx.t()],
          number | Nx.t(),
          number | Nx.t(),
          pos_integer(),
          keyword()
        ) :: [sample_result]
  def sample_general(observations, f, h, num_samples, opts \\ []) do
    validate_positive!(:num_samples, num_samples)

    if Enum.any?(observations, &missing_observation?/1) do
      Logger.warning(
        "GibbsSampler received missing observations (nil/NaN); missing values are skipped in the observation variance update"
      )
    end

    f_t = to_tensor(f)
    x0 = to_tensor(Keyword.get(opts, :initial_state, 0.0))
    p0 = to_tensor(Keyword.get(opts, :initial_cov, 1.0))
    process_var_raw = Keyword.get(opts, :process_var, 1.0)
    obs_var_raw = Keyword.get(opts, :obs_var, 1.0)
    validate_non_negative_var!(:process_var, process_var_raw)
    validate_non_negative_var!(:obs_var, obs_var_raw)
    process_var_t = to_tensor(process_var_raw)
    obs_var_t = to_tensor(obs_var_raw)

    prior_shape = Keyword.get(opts, :prior_shape, 1.0)
    prior_scale = Keyword.get(opts, :prior_scale, 1.0)
    validate_prior_params!(prior_shape, prior_scale)
    burn_in = Keyword.get(opts, :burn_in, 0)
    validate_non_negative_integer!(:burn_in, burn_in)
    thin = Keyword.get(opts, :thin, 1)
    validate_positive!(:thin, thin)
    seed = Keyword.get(opts, :seed, System.os_time())
    key = Keyword.get(opts, :key, Nx.Random.key(seed))
    total_iters = burn_in + num_samples * thin

    max_iters = 1_000_000

    if total_iters > max_iters do
      raise ArgumentError,
            "total iterations (burn_in + num_samples * thin = #{total_iters}) " <>
              "exceeds safety cap of #{max_iters}; reduce burn_in, num_samples or thin"
    end

    t = length(observations)
    h_vals = h_to_number_list(h, t)

    if length(h_vals) != t do
      raise ArgumentError,
            "h vector length (#{length(h_vals)}) must match observations length (#{t})"
    end

    obs_tensor = observations_to_filter_tensor(observations)
    h_tensor = Nx.tensor(h_vals, type: {:f, 32})
    obs_present_mask = Nx.equal(obs_tensor, obs_tensor)
    t_steps = obs_present_mask |> Nx.sum() |> Nx.to_number() |> round()
    num_diffs = latent_transition_count(observations)

    {_, _, samples_acc, _key_acc} =
      Enum.reduce(1..total_iters, {process_var_t, obs_var_t, [], key}, fn iter,
                                                                          {q_prev, r_prev, acc, k} ->
        {xs, ps} = KalmanFilter.filter_defn(obs_tensor, f_t, h_tensor, q_prev, r_prev, x0, p0)

        {sampled_xs, new_key_smooth} =
          BstsNx.Smoother.simulate_from_filtered_defn(xs, ps, f_t, q_prev, k)

        process_ss = process_sum_of_squares(sampled_xs, f_t)
        obs_ss = obs_sum_of_squares(obs_tensor, sampled_xs, h_tensor, obs_present_mask)

        {q_sample, r_sample, next_key} =
          resample_variances(
            new_key_smooth,
            prior_shape,
            prior_scale,
            {process_ss, num_diffs},
            {obs_ss, t_steps}
          )

        acc2 =
          if iter > burn_in and rem(iter - burn_in, thin) == 0 do
            sample_map = %{
              states: scalar_tensor_list(sampled_xs),
              state_covs: scalar_tensor_list(ps),
              process_var: q_sample,
              obs_var: r_sample
            }

            [sample_map | acc]
          else
            acc
          end

        {q_sample, r_sample, acc2, next_key}
      end)

    Enum.reverse(samples_acc)
  end

  # ---------------------------------------------------------------------------
  # Structured sampler for multi-dimensional state-space models
  # ---------------------------------------------------------------------------

  @type structured_result :: %{
          states: [Nx.t()],
          state_covs: [Nx.t()],
          q_matrix: Nx.t(),
          obs_var: Nx.t(),
          regression_beta: Nx.t() | nil,
          regression_gamma: [0 | 1] | nil
        }

  @doc """
  Gibbs sampler for structured multi-dimensional state-space models.

  Unlike `sample_general/5` which works with scalar state and a single
  process variance, this function accepts a `%ModelSpec{}` that fully
  describes a multi-dimensional model.  Each diagonal entry of the
  process covariance matrix Q is resampled independently from its own
  inverse-gamma posterior.

  ## Arguments

    * `observations` — list of scalar observations. Missing values as `nil`
      or NaN (number/tensor).
    * `spec` — a `%BstsNx.ModelSpec{}` defining the model.
    * `num_samples` — number of post-burn-in, post-thinning samples.

  ## Options

    * `:burn_in` — iterations to discard (default: 0)
    * `:thin` — keep every nth sample after burn-in (default: 1)
    * `:seed` — integer PRNG seed (default: `System.os_time()`)
    * `:key` — Nx.Random PRNG key (overrides `:seed`)

  ## Returns

  A list of maps, each with:

    * `:states` — list of state vectors (one per timestep)
    * `:state_covs` — list of filtered covariance matrices
    * `:q_matrix` — the resampled diagonal Q matrix
    * `:obs_var` — the resampled observation variance (scalar tensor)
  """
  @spec sample_structured([number | nil | Nx.t()], ModelSpec.t(), pos_integer(), keyword()) ::
          [structured_result()]
  def sample_structured(observations, %ModelSpec{} = spec, num_samples, opts \\ []) do
    validate_positive!(:num_samples, num_samples)
    spec = ModelSpec.validate!(spec)

    if Enum.any?(observations, &missing_observation?/1) do
      Logger.warning(
        "GibbsSampler.sample_structured received missing observations (nil/NaN); " <>
          "missing values are skipped in the observation variance update"
      )
    end

    burn_in = Keyword.get(opts, :burn_in, 0)
    validate_non_negative_integer!(:burn_in, burn_in)
    thin = Keyword.get(opts, :thin, 1)
    validate_positive!(:thin, thin)
    seed = Keyword.get(opts, :seed, System.os_time())
    key = Keyword.get(opts, :key, Nx.Random.key(seed))
    total_iters = burn_in + num_samples * thin

    max_iters = 1_000_000

    if total_iters > max_iters do
      raise ArgumentError,
            "total iterations (burn_in + num_samples * thin = #{total_iters}) " <>
              "exceeds safety cap of #{max_iters}; reduce burn_in, num_samples or thin"
    end

    t = length(observations)

    state_dim = Nx.axis_size(spec.f, 0)

    # Build initial Q matrix from q_specs
    q_matrix = build_initial_q(spec.q_specs, state_dim)
    r_var = to_tensor(spec.obs_var)

    # Normalize H to a per-timestep list for residual computation
    h_list = normalize_h_for_structured(spec.h, t)

    case spec.regression do
      %{mode: :spike_and_slab} = regression ->
        sample_structured_spike_slab(
          observations,
          spec,
          regression,
          q_matrix,
          r_var,
          h_list,
          num_samples,
          burn_in,
          thin,
          key,
          total_iters
        )

      _ ->
        sample_structured_standard(
          observations,
          spec,
          q_matrix,
          r_var,
          h_list,
          burn_in,
          thin,
          key,
          total_iters
        )
    end
  end

  @doc """
  Runs multiple independent structured Gibbs sampler chains in parallel.

  Each chain runs `sample_structured/4` with an independent PRNG seed.

  ## Arguments

    * `observations` — list of scalar observations
    * `spec` — `%BstsNx.ModelSpec{}`
    * `num_chains` — number of parallel chains
    * `num_samples` — samples per chain (after burn-in/thinning)

  ## Options

    * `:seed` — base seed; chain `i` uses `seed + i` (default: `System.os_time()`)
    * `:seeds` — explicit list of seeds, one per chain (overrides `:seed`)
    * `:key` — PRNG key; split into `num_chains` subkeys (overrides `:seed`)
    * `:timeout` — per-chain timeout in ms (default: 300_000)
    * All other options forwarded to `sample_structured/4`

  ## Returns

  A list of chains, where each chain is a list of `structured_result` maps.
  """
  @spec sample_structured_chains(
          [number | nil | Nx.t()],
          ModelSpec.t(),
          pos_integer(),
          pos_integer(),
          keyword()
        ) ::
          [[structured_result()]]
  def sample_structured_chains(
        observations,
        %ModelSpec{} = spec,
        num_chains,
        num_samples,
        opts \\ []
      ) do
    validate_positive!(:num_chains, num_chains)

    run_chains(
      num_chains,
      opts,
      "Structured Gibbs sampler",
      "structured Gibbs sampler",
      fn chain_opts ->
        sample_structured(observations, spec, num_samples, chain_opts)
      end
    )
  end

  defp run_chains(num_chains, opts, warning_label, error_label, fun) do
    timeout = Keyword.get(opts, :timeout, 300_000)

    results =
      num_chains
      |> chain_opts_list(opts)
      |> Task.async_stream(fun,
        max_concurrency: System.schedulers_online(),
        timeout: timeout,
        on_timeout: :kill_task
      )
      |> Enum.with_index()
      |> Enum.reduce([], fn
        {{:ok, res}, _idx}, acc ->
          [res | acc]

        {{:exit, reason}, idx}, acc ->
          Logger.warning("#{warning_label} chain #{idx} failed: #{inspect(reason)}")
          acc
      end)
      |> Enum.reverse()

    if results == [] do
      raise RuntimeError, "all #{num_chains} #{error_label} chains failed"
    end

    results
  end

  defp chain_opts_list(num_chains, opts) do
    base_seed = Keyword.get(opts, :seed, System.os_time())
    seeds = Keyword.get(opts, :seeds)
    key_opt = Keyword.get(opts, :key)

    cond do
      is_list(seeds) and length(seeds) == num_chains ->
        opts_clean = opts |> Keyword.delete(:key) |> Keyword.delete(:seeds)
        seeds_t = List.to_tuple(seeds)

        Enum.map(0..(num_chains - 1), fn idx ->
          Keyword.put(opts_clean, :seed, elem(seeds_t, idx))
        end)

      key_opt != nil ->
        keys = Nx.Random.split(key_opt, parts: num_chains)

        opts_clean =
          opts |> Keyword.delete(:key) |> Keyword.delete(:seed) |> Keyword.delete(:seeds)

        Enum.map(0..(num_chains - 1), fn idx ->
          Keyword.put(opts_clean, :key, split_key_at(keys, idx))
        end)

      true ->
        opts_clean = opts |> Keyword.delete(:seeds)

        Enum.map(0..(num_chains - 1), fn idx ->
          Keyword.put(opts_clean, :seed, base_seed + idx)
        end)
    end
  end

  # -- Structured sampler helpers ----------------------------------------------

  defp sample_structured_standard(
         observations,
         spec,
         q_matrix,
         r_var,
         h_list,
         burn_in,
         thin,
         key,
         total_iters
       ) do
    # Convert inputs bounds into Nx.Tensors once
    obs_tensor = observations_to_filter_tensor(observations)

    # H list needs to be converted into a tensor {t, n} for filter_defn_multi
    h_tensor = Nx.stack(h_list)

    h_rows_tensor =
      h_list |> Enum.map(&structured_h_row/1) |> Enum.map(&Nx.flatten/1) |> Nx.stack()

    num_diffs = latent_transition_count(observations)

    {_, _, samples_acc, _key_acc} =
      Enum.reduce(1..total_iters, {q_matrix, r_var, [], key}, fn iter, {q_prev, r_prev, acc, k} ->
        # 1. Kalman filter (multi-dimensional) via compiled Defn
        {filtered_xs, filtered_ps} =
          KalmanFilter.filter_defn_multi(
            obs_tensor,
            spec.f,
            h_tensor,
            q_prev,
            r_prev,
            spec.x0,
            spec.p0
          )

        # 2. Carter-Kohn simulation smoother via compiled Defn
        {sampled_states_tensor, new_key} =
          BstsNx.Smoother.simulate_from_filtered_defn_matrix(
            filtered_xs,
            filtered_ps,
            spec.f,
            q_prev,
            k
          )

        # 3. Process residuals: e_t = x_t - F * x_{t-1}
        per_dim_ss = process_residuals_structured(sampled_states_tensor, spec.f, spec.q_specs)

        # 4. Observation residuals: SS_obs = Σ (y_t - H_t · x_t)²
        {obs_ss, t_obs} =
          obs_residuals_structured(observations, sampled_states_tensor, h_rows_tensor)

        # 5. Resample each Q diagonal entry independently
        {q_new, key_after_q} =
          resample_q_components(spec.q_specs, per_dim_ss, num_diffs, new_key)

        # 6. Rebuild Q matrix
        q_matrix_new = rebuild_q(q_prev, q_new)

        # 8. Resample observation variance
        shape_r = safe_to_number(spec.obs_prior_shape) + safe_to_number(t_obs) / 2
        scale_r = safe_to_number(spec.obs_prior_scale) + safe_to_number(obs_ss) / 2

        {r_sample, next_key} =
          Distributions.inv_gamma_sample_defn(shape_r, scale_r, key_after_q)

        acc2 =
          if iter > burn_in and rem(iter - burn_in, thin) == 0 do
            covs =
              tensor_time_slices(filtered_ps)

            sample_map = %{
              states: tensor_time_slices(sampled_states_tensor),
              state_covs: covs,
              q_matrix: q_matrix_new,
              obs_var: r_sample,
              regression_beta: nil,
              regression_gamma: nil
            }

            [sample_map | acc]
          else
            acc
          end

        {q_matrix_new, r_sample, acc2, next_key}
      end)

    Enum.reverse(samples_acc)
  end

  defp sample_structured_spike_slab(
         observations,
         spec,
         regression,
         q_matrix,
         r_var,
         h_list,
         _num_samples,
         burn_in,
         thin,
         key,
         total_iters
       ) do
    t = length(observations)
    num_diffs = latent_transition_count(observations)

    ctx = prepare_spike_slab_context(spec, regression, h_list)

    q_struct0 =
      if ctx.struct_dim == 0 do
        nil
      else
        submatrix(q_matrix, ctx.struct_full_indices, ctx.struct_full_indices)
      end

    beta0 = slice_vector(spec.x0, ctx.reg_full_indices)
    {y_obs_init, x_obs_init} = observed_regression_pairs(observations, ctx.x_rows)
    gamma0 = init_gamma_from_data(y_obs_init, x_obs_init, ctx.prior_inclusion, ctx.reg_dim)
    gamma_xtx_stats = if x_obs_init == [], do: nil, else: spike_slab_xtx_stats(x_obs_init)

    {_, _, _, _, samples_acc, _key_acc} =
      Enum.reduce(
        1..total_iters,
        {q_struct0, r_var, beta0, gamma0, [], key},
        fn iter, {q_struct_prev, r_prev, beta_prev, gamma_prev, acc, key_prev} ->
          y_adjusted = adjust_observations_for_regression(observations, ctx.x_rows, beta_prev)

          {sampled_struct_states, struct_covs, q_struct_new, key_after_struct} =
            if ctx.struct_dim == 0 do
              {List.duplicate([], t), List.duplicate(nil, t), q_struct_prev, key_prev}
            else
              y_adjusted_tensor = observations_to_filter_tensor(y_adjusted)

              {filtered_xs, filtered_ps} =
                KalmanFilter.filter_defn_multi(
                  y_adjusted_tensor,
                  ctx.f_struct,
                  ctx.h_struct_tensor,
                  q_struct_prev,
                  r_prev,
                  ctx.x0_struct,
                  ctx.p0_struct
                )

              {sampled_states_tensor, key_after_smooth} =
                BstsNx.Smoother.simulate_from_filtered_defn_matrix(
                  filtered_xs,
                  filtered_ps,
                  ctx.f_struct,
                  q_struct_prev,
                  key_prev
                )

              sampled_states =
                tensor_time_slices(sampled_states_tensor)

              covs =
                tensor_time_slices(filtered_ps)

              per_dim_ss =
                process_residuals_structured(
                  sampled_states_tensor,
                  ctx.f_struct,
                  ctx.q_specs_struct
                )

              {q_new_vals, key_after_q} =
                resample_q_components(
                  ctx.q_specs_struct,
                  per_dim_ss,
                  num_diffs,
                  key_after_smooth
                )

              {sampled_states, covs, rebuild_q(q_struct_prev, q_new_vals), key_after_q}
            end

          {y_reg_obs, x_reg_obs} =
            regression_residual_pairs(
              observations,
              sampled_struct_states,
              ctx.h_struct_rows,
              ctx.x_rows
            )

          sigma2 = max(Nx.to_number(r_prev), 1.0e-12)

          {gamma_new, key_after_gamma} =
            resample_gamma_g_prior(
              gamma_prev,
              y_reg_obs,
              x_reg_obs,
              sigma2,
              ctx.prior_inclusion,
              ctx.g_prior,
              key_after_struct,
              gamma_xtx_stats
            )

          {beta_new, beta_cov, key_after_beta} =
            sample_beta_g_prior(
              gamma_new,
              y_reg_obs,
              x_reg_obs,
              sigma2,
              ctx.g_prior,
              key_after_gamma
            )

          {obs_ss, t_obs} =
            obs_residuals_spike_slab(
              observations,
              sampled_struct_states,
              ctx.h_struct_rows,
              ctx.x_rows,
              beta_new
            )

          shape_r = safe_to_number(spec.obs_prior_shape) + safe_to_number(t_obs) / 2
          scale_r = safe_to_number(spec.obs_prior_scale) + safe_to_number(obs_ss) / 2

          {r_sample, key_next} =
            Distributions.inv_gamma_sample_defn(shape_r, scale_r, key_after_beta)

          acc2 =
            if iter > burn_in and rem(iter - burn_in, thin) == 0 do
              q_matrix_full =
                build_full_q_matrix(
                  ctx.state_dim,
                  ctx.struct_full_indices,
                  q_struct_new
                )

              full_states =
                build_full_state_trajectory(
                  sampled_struct_states,
                  beta_new,
                  ctx.state_dim,
                  ctx.struct_full_indices,
                  ctx.reg_full_indices
                )

              full_covs =
                build_full_covariances(
                  struct_covs,
                  beta_cov,
                  ctx.state_dim,
                  ctx.struct_full_indices,
                  ctx.reg_full_indices
                )

              sample_map = %{
                states: full_states,
                state_covs: full_covs,
                q_matrix: q_matrix_full,
                obs_var: r_sample,
                regression_beta: Nx.tensor(beta_new),
                regression_gamma: gamma_new
              }

              [sample_map | acc]
            else
              acc
            end

          {q_struct_new, r_sample, beta_new, gamma_new, acc2, key_next}
        end
      )

    Enum.reverse(samples_acc)
  end

  defp prepare_spike_slab_context(spec, regression, h_list) do
    state_dim = Nx.axis_size(spec.f, 0)
    reg_start = Map.get(regression, :start_dim, 0)
    reg_dim = Map.get(regression, :num_dims, 0)

    if not is_integer(reg_start) or not is_integer(reg_dim) or reg_dim <= 0 do
      raise ArgumentError,
            "invalid regression metadata for spike-and-slab: #{inspect(regression)}"
    end

    reg_end = reg_start + reg_dim - 1

    if reg_start < 0 or reg_end >= state_dim do
      raise ArgumentError,
            "regression block [#{reg_start}, #{reg_end}] is out of bounds for state dim #{state_dim}"
    end

    reg_full_indices = Enum.to_list(reg_start..reg_end)
    reg_set = MapSet.new(reg_full_indices)
    struct_full_indices = Enum.reject(0..(state_dim - 1), &MapSet.member?(reg_set, &1))
    struct_dim = length(struct_full_indices)

    index_map =
      struct_full_indices
      |> Enum.with_index()
      |> Map.new(fn {full_idx, local_idx} -> {full_idx, local_idx} end)

    q_specs_struct =
      spec.q_specs
      |> Enum.reject(fn qs -> MapSet.member?(reg_set, qs.dim_index) end)
      |> Enum.map(fn qs ->
        local_dim = Map.fetch!(index_map, qs.dim_index)
        %{qs | dim_index: local_dim}
      end)

    h_struct_rows =
      if struct_dim == 0 do
        List.duplicate([], length(h_list))
      else
        Enum.map(h_list, fn h_t -> extract_row_values(h_t, struct_full_indices) end)
      end

    {h_struct_list, h_struct_tensor} =
      if struct_dim == 0 do
        {[], nil}
      else
        h_struct_list =
          Enum.map(h_struct_rows, fn row ->
            Nx.tensor([row])
          end)

        {h_struct_list, Nx.stack(h_struct_list)}
      end

    x_rows = Enum.map(h_list, &extract_row_values(&1, reg_full_indices))

    %{
      state_dim: state_dim,
      reg_dim: reg_dim,
      reg_full_indices: reg_full_indices,
      struct_dim: struct_dim,
      struct_full_indices: struct_full_indices,
      q_specs_struct: q_specs_struct,
      h_struct_rows: h_struct_rows,
      h_struct_list: h_struct_list,
      h_struct_tensor: h_struct_tensor,
      x_rows: x_rows,
      f_struct:
        if(struct_dim == 0,
          do: nil,
          else: submatrix(spec.f, struct_full_indices, struct_full_indices)
        ),
      x0_struct: if(struct_dim == 0, do: [], else: slice_vector(spec.x0, struct_full_indices)),
      p0_struct:
        if(struct_dim == 0,
          do: nil,
          else: submatrix(spec.p0, struct_full_indices, struct_full_indices)
        ),
      prior_inclusion: Map.get(regression, :prior_inclusion, min(0.5, 3.0 / reg_dim)),
      g_prior: Map.get(regression, :g, max(length(h_list), 1))
    }
    |> then(fn ctx ->
      if not is_number(ctx.prior_inclusion) or ctx.prior_inclusion <= 0.0 or
           ctx.prior_inclusion >= 1.0 do
        raise ArgumentError,
              "spike-and-slab prior_inclusion must be in (0, 1), got: #{inspect(ctx.prior_inclusion)}"
      end

      if not is_number(ctx.g_prior) or ctx.g_prior <= 0.0 do
        raise ArgumentError, "spike-and-slab g must be > 0, got: #{inspect(ctx.g_prior)}"
      end

      ctx
    end)
  end

  defp regression_residual_pairs(observations, sampled_struct_states, h_struct_rows, x_rows) do
    observations
    |> Enum.zip(sampled_struct_states)
    |> Enum.zip(h_struct_rows)
    |> Enum.zip(x_rows)
    |> Enum.reduce({[], []}, fn {{{y, state}, h_row}, x_row}, {acc_y, acc_x} ->
      if missing_observation?(y) do
        {acc_y, acc_x}
      else
        y_val = observation_to_number(y)
        struct_pred = dot_list(h_row, state_values(state))
        {[y_val - struct_pred | acc_y], [x_row | acc_x]}
      end
    end)
    |> then(fn {ys, xs} -> {Enum.reverse(ys), Enum.reverse(xs)} end)
  end

  defp observed_regression_pairs(observations, x_rows) do
    observations
    |> Enum.zip(x_rows)
    |> Enum.reduce({[], []}, fn {y, x_row}, {acc_y, acc_x} ->
      if missing_observation?(y) do
        {acc_y, acc_x}
      else
        {[observation_to_number(y) | acc_y], [x_row | acc_x]}
      end
    end)
    |> then(fn {ys, xs} -> {Enum.reverse(ys), Enum.reverse(xs)} end)
  end

  defp adjust_observations_for_regression(observations, x_rows, beta) do
    observations
    |> Enum.zip(x_rows)
    |> Enum.map(fn {y, x_row} ->
      if missing_observation?(y) do
        y
      else
        observation_to_number(y) - dot_list(x_row, beta)
      end
    end)
  end

  defp obs_residuals_spike_slab(observations, sampled_struct_states, h_struct_rows, x_rows, beta) do
    observations
    |> Enum.zip(sampled_struct_states)
    |> Enum.zip(h_struct_rows)
    |> Enum.zip(x_rows)
    |> Enum.reduce({0.0, 0}, fn {{{y, state}, h_row}, x_row}, {acc_ss, acc_count} ->
      if missing_observation?(y) do
        {acc_ss, acc_count}
      else
        y_val = observation_to_number(y)
        struct_pred = dot_list(h_row, state_values(state))
        reg_pred = dot_list(x_row, beta)
        diff = y_val - struct_pred - reg_pred
        {acc_ss + diff * diff, acc_count + 1}
      end
    end)
  end

  defp resample_gamma_g_prior(gamma, _y_obs, _x_obs, _sigma2, _pi, _g, key, _xtx_stats)
       when gamma == [] do
    {gamma, key}
  end

  defp resample_gamma_g_prior(
         gamma,
         y_obs,
         x_obs_rows,
         sigma2,
         prior_inclusion,
         g_prior,
         key,
         xtx_stats
       ) do
    if y_obs == [] do
      {gamma, key}
    else
      p = length(gamma)
      log_prior_odds = :math.log(prior_inclusion / (1.0 - prior_inclusion))
      {u_vec, key_next} = Nx.Random.uniform(key, 0.0, 1.0, shape: {p})
      uniforms = u_vec |> Nx.to_flat_list() |> List.to_tuple()
      stats = spike_slab_sufficient_stats(y_obs, x_obs_rows, xtx_stats)
      model0 = g_prior_model_from_gamma(stats, gamma)

      {gamma_new, _model_new} =
        Enum.reduce(0..(p - 1), {gamma, model0}, fn j, {gamma_curr, model_curr} ->
          {model_off, model_on} =
            if Enum.at(gamma_curr, j) == 1 do
              {remove_g_prior_column(stats, model_curr, j), model_curr}
            else
              {model_curr, add_g_prior_column(stats, model_curr, j)}
            end

          log_ml_off = log_marginal_g_prior(model_off, sigma2, g_prior)
          log_ml_on = log_marginal_g_prior(model_on, sigma2, g_prior)
          prob_on = logistic(log_prior_odds + log_ml_on - log_ml_off)
          gamma_j = if elem(uniforms, j) < prob_on, do: 1, else: 0

          {
            List.replace_at(gamma_curr, j, gamma_j),
            if(gamma_j == 1, do: model_on, else: model_off)
          }
        end)

      {gamma_new, key_next}
    end
  end

  defp spike_slab_xtx_stats(x_obs_rows) do
    p = x_obs_rows |> hd() |> length()
    x_t = Nx.tensor(x_obs_rows)
    xtx = compat_dot(Nx.transpose(x_t), x_t)

    %{
      p: p,
      x_cols: BstsNx.Utils.transpose_rows(x_obs_rows),
      xtx:
        xtx
        |> Nx.to_flat_list()
        |> Enum.chunk_every(p)
        |> to_tuple_matrix()
    }
  end

  defp spike_slab_sufficient_stats(y_obs, x_obs_rows, nil) do
    spike_slab_sufficient_stats(y_obs, x_obs_rows, spike_slab_xtx_stats(x_obs_rows))
  end

  defp spike_slab_sufficient_stats(y_obs, _x_obs_rows, xtx_stats) do
    Map.put(xtx_stats, :xty, spike_slab_xty(xtx_stats.x_cols, y_obs))
  end

  defp spike_slab_xty(x_cols, y_obs) do
    x_cols
    |> Enum.map(&dot_list(&1, y_obs))
    |> List.to_tuple()
  end

  defp g_prior_model_from_gamma(stats, gamma) do
    gamma
    |> active_indices()
    |> g_prior_model_from_active(stats)
  end

  defp g_prior_model_from_active([], _stats), do: empty_g_prior_model()

  defp g_prior_model_from_active(active, stats) do
    k = length(active)

    xtx =
      Enum.map(active, fn row_idx ->
        Enum.map(active, &xtx_value(stats, row_idx, &1))
      end)
      |> Nx.tensor()

    inv_xtx =
      BstsNx.Utils.safe_solve(xtx, Nx.eye(k))
      |> Nx.to_flat_list()
      |> Enum.chunk_every(k)

    if finite_matrix?(inv_xtx) do
      build_g_prior_model(stats, active, inv_xtx)
    else
      empty_g_prior_model()
    end
  end

  defp empty_g_prior_model do
    %{active: [], inv_xtx: [], score: 0.0}
  end

  defp build_g_prior_model(stats, active, inv_xtx) do
    score = score_from_inverse(stats, active, inv_xtx)

    if finite_number?(score) do
      %{active: active, inv_xtx: inv_xtx, score: score}
    else
      empty_g_prior_model()
    end
  end

  defp add_g_prior_column(stats, model, j) do
    if j in model.active do
      model
    else
      do_add_g_prior_column(stats, model, j)
    end
  end

  defp do_add_g_prior_column(stats, %{active: []}, j) do
    pivot = xtx_value(stats, j, j)

    if usable_rank_pivot?(pivot) do
      build_g_prior_model(stats, [j], [[1.0 / pivot]])
    else
      g_prior_model_from_active([j], stats)
    end
  end

  defp do_add_g_prior_column(stats, model, j) do
    b = Enum.map(model.active, &xtx_value(stats, &1, j))
    inv_b = matrix_vector_product(model.inv_xtx, b)
    pivot = xtx_value(stats, j, j) - dot_list(b, inv_b)

    if usable_rank_pivot?(pivot) do
      top_left =
        Enum.zip(model.inv_xtx, inv_b)
        |> Enum.map(fn {row, row_scale} ->
          Enum.zip(row, inv_b)
          |> Enum.map(fn {value, col_scale} ->
            value + row_scale * col_scale / pivot
          end)
        end)

      edge = Enum.map(inv_b, &(-&1 / pivot))

      inv_xtx =
        top_left
        |> Enum.zip(edge)
        |> Enum.map(fn {row, edge_value} -> row ++ [edge_value] end)
        |> Kernel.++([edge ++ [1.0 / pivot]])

      build_g_prior_model(stats, model.active ++ [j], inv_xtx)
    else
      g_prior_model_from_active(model.active ++ [j], stats)
    end
  end

  defp remove_g_prior_column(stats, model, j) do
    case Enum.find_index(model.active, &(&1 == j)) do
      nil ->
        model

      pos ->
        do_remove_g_prior_column(stats, model, pos)
    end
  end

  defp do_remove_g_prior_column(_stats, %{active: [_only]}, _pos), do: empty_g_prior_model()

  defp do_remove_g_prior_column(stats, model, pos) do
    pivot =
      model.inv_xtx
      |> Enum.at(pos)
      |> Enum.at(pos)

    active = List.delete_at(model.active, pos)

    if usable_rank_pivot?(pivot) do
      edge =
        model.inv_xtx
        |> Enum.with_index()
        |> Enum.reject(fn {_row, row_idx} -> row_idx == pos end)
        |> Enum.map(fn {row, _row_idx} -> Enum.at(row, pos) end)

      core =
        model.inv_xtx
        |> List.delete_at(pos)
        |> Enum.map(&List.delete_at(&1, pos))

      inv_xtx =
        core
        |> Enum.zip(edge)
        |> Enum.map(fn {row, row_scale} ->
          Enum.zip(row, edge)
          |> Enum.map(fn {value, col_scale} ->
            value - row_scale * col_scale / pivot
          end)
        end)

      build_g_prior_model(stats, active, inv_xtx)
    else
      g_prior_model_from_active(active, stats)
    end
  end

  defp log_marginal_g_prior(%{active: []}, _sigma2, _g_prior), do: 0.0

  defp log_marginal_g_prior(model, sigma2, g_prior) do
    k = length(model.active)
    denom = 2.0 * max(sigma2, 1.0e-12) * (1.0 + g_prior)
    -0.5 * k * :math.log(1.0 + g_prior) + g_prior * model.score / denom
  end

  defp score_from_inverse(stats, active, inv_xtx) do
    xty = Enum.map(active, &xty_value(stats, &1))
    dot_list(xty, matrix_vector_product(inv_xtx, xty))
  end

  defp matrix_vector_product(matrix, vector) do
    Enum.map(matrix, &dot_list(&1, vector))
  end

  defp xtx_value(stats, i, j), do: stats.xtx |> elem(i) |> elem(j)
  defp xty_value(stats, i), do: elem(stats.xty, i)

  defp usable_rank_pivot?(value) do
    finite_number?(value) and value > @rank_update_pivot_floor
  end

  defp finite_matrix?(matrix) do
    Enum.all?(matrix, fn row -> Enum.all?(row, &finite_number?/1) end)
  end

  defp finite_number?(value) do
    is_number(value) and value == value and abs(value) < 1.0e300
  end

  defp sample_beta_g_prior(gamma, y_obs, x_obs_rows, sigma2, g_prior, key) do
    p = length(gamma)
    active = active_indices(gamma)

    if active == [] or y_obs == [] do
      {List.duplicate(0.0, p), Nx.broadcast(0.0, {p, p}), key}
    else
      x_active = select_active_columns(x_obs_rows, active)
      x_t = Nx.tensor(x_active)
      y_t = Nx.tensor(y_obs)

      xtx = compat_dot(Nx.transpose(x_t), x_t)
      xty = compat_dot(Nx.transpose(x_t), y_t)
      inv_xtx = BstsNx.Utils.safe_solve(xtx, Nx.eye(length(active)))
      beta_ols = BstsNx.Utils.safe_solve(xtx, xty)

      shrink = g_prior / (1.0 + g_prior)
      mean_active = Nx.multiply(beta_ols, shrink)

      cov_active =
        inv_xtx
        |> Nx.multiply(shrink * max(sigma2, 1.0e-12))
        |> symmetrize()

      jitter = Nx.eye(length(active)) |> Nx.multiply(1.0e-9)
      chol = BstsNx.Utils.safe_cholesky(Nx.add(cov_active, jitter))
      {z, key_next} = Nx.Random.normal(key, 0.0, 1.0, shape: {length(active)})

      draw_active =
        Nx.add(mean_active, compat_dot(chol, z))
        |> then(fn draw ->
          if BstsNx.Utils.has_non_finite?(draw), do: mean_active, else: draw
        end)
        |> Nx.to_flat_list()

      cov_rows = Nx.to_flat_list(cov_active) |> Enum.chunk_every(length(active))

      beta_full =
        List.duplicate(0.0, p)
        |> put_active_values(active, draw_active)

      beta_cov =
        List.duplicate(List.duplicate(0.0, p), p)
        |> put_active_covariance(active, cov_rows)
        |> Nx.tensor()

      {beta_full, beta_cov, key_next}
    end
  end

  defp init_gamma_from_data(y_obs, x_obs_rows, prior_inclusion, p) do
    if y_obs == [] do
      List.duplicate(0, p)
    else
      expected = max(round(prior_inclusion * p), 1)
      x_cols = BstsNx.Utils.transpose_rows(x_obs_rows)

      active =
        x_cols
        |> Enum.with_index()
        |> Enum.map(fn {col, idx} -> {idx, abs(dot_list(col, y_obs))} end)
        |> Enum.sort_by(fn {_idx, score} -> score end, :desc)
        |> Enum.take(min(expected, p))
        |> Enum.map(&elem(&1, 0))
        |> MapSet.new()

      Enum.map(0..(p - 1), fn idx -> if MapSet.member?(active, idx), do: 1, else: 0 end)
    end
  end

  defp build_full_q_matrix(state_dim, struct_full_indices, q_struct) do
    n_struct = length(struct_full_indices)

    if n_struct == 0 do
      Nx.broadcast(0.0, {state_dim, state_dim})
    else
      base = Nx.broadcast(0.0, {state_dim, state_dim})

      indices =
        for full_i <- struct_full_indices, full_j <- struct_full_indices do
          [full_i, full_j]
        end

      updates = Nx.reshape(q_struct, {n_struct * n_struct})
      Nx.indexed_put(base, Nx.tensor(indices), updates)
    end
  end

  defp build_full_state_trajectory(
         sampled_struct_states,
         beta,
         state_dim,
         struct_full_indices,
         reg_full_indices
       ) do
    Enum.map(sampled_struct_states, fn struct_state ->
      struct_vals = state_values(struct_state)

      List.duplicate(0.0, state_dim)
      |> put_active_values(struct_full_indices, struct_vals)
      |> put_active_values(reg_full_indices, beta)
      |> Nx.tensor()
    end)
  end

  defp build_full_covariances(
         struct_covs,
         beta_cov,
         state_dim,
         struct_full_indices,
         reg_full_indices
       ) do
    if struct_full_indices == [] do
      List.duplicate(beta_cov, length(struct_covs))
    else
      do_build_full_covariances(
        struct_covs,
        beta_cov,
        state_dim,
        struct_full_indices,
        reg_full_indices
      )
    end
  end

  defp do_build_full_covariances(
         struct_covs,
         beta_cov,
         state_dim,
         struct_full_indices,
         reg_full_indices
       ) do
    struct_map =
      struct_full_indices
      |> Enum.with_index()
      |> Map.new(fn {full_idx, local_idx} -> {full_idx, local_idx} end)

    reg_map =
      reg_full_indices
      |> Enum.with_index()
      |> Map.new(fn {full_idx, local_idx} -> {full_idx, local_idx} end)

    beta_rows =
      beta_cov
      |> Nx.to_flat_list()
      |> Enum.chunk_every(max(length(reg_full_indices), 1))
      |> to_tuple_matrix()

    Enum.map(struct_covs, fn struct_cov ->
      struct_rows =
        struct_cov
        |> Nx.to_flat_list()
        |> Enum.chunk_every(max(length(struct_full_indices), 1))
        |> to_tuple_matrix()

      rows =
        Enum.map(0..(state_dim - 1), fn i ->
          Enum.map(0..(state_dim - 1), fn j ->
            cond do
              Map.get(struct_map, i) != nil and Map.get(struct_map, j) != nil ->
                i_local = Map.get(struct_map, i)
                j_local = Map.get(struct_map, j)
                struct_rows |> elem(i_local) |> elem(j_local)

              Map.get(reg_map, i) != nil and Map.get(reg_map, j) != nil ->
                i_local = Map.get(reg_map, i)
                j_local = Map.get(reg_map, j)
                beta_rows |> elem(i_local) |> elem(j_local)

              true ->
                0.0
            end
          end)
        end)

      Nx.tensor(rows)
    end)
  end

  defp extract_row_values(h_t, indices) do
    row = structured_h_row(h_t) |> Nx.to_flat_list() |> List.to_tuple()
    Enum.map(indices, &elem(row, &1))
  end

  defp state_values(%Nx.Tensor{} = state), do: Nx.to_flat_list(state)
  defp state_values(values) when is_list(values), do: values

  defp active_indices(gamma) do
    gamma
    |> Enum.with_index()
    |> Enum.filter(fn {g, _idx} -> g == 1 end)
    |> Enum.map(&elem(&1, 1))
  end

  defp select_active_columns(rows, active) do
    rows_t = Enum.map(rows, &List.to_tuple/1)
    Enum.map(rows_t, fn row -> Enum.map(active, &elem(row, &1)) end)
  end

  defp put_active_values(base, indices, values) do
    Enum.zip(indices, values)
    |> Enum.reduce(base, fn {idx, val}, acc ->
      List.replace_at(acc, idx, val)
    end)
  end

  defp put_active_covariance(base_rows, active_indices, cov_rows) do
    cov_rows_t = to_tuple_matrix(cov_rows)

    active_pos =
      active_indices
      |> Enum.with_index()
      |> Map.new(fn {idx, pos} -> {idx, pos} end)

    Enum.with_index(base_rows)
    |> Enum.map(fn {row, i} ->
      Enum.with_index(row)
      |> Enum.map(fn {_v, j} ->
        i_pos = Map.get(active_pos, i)
        j_pos = Map.get(active_pos, j)

        if i_pos != nil and j_pos != nil do
          cov_rows_t |> elem(i_pos) |> elem(j_pos)
        else
          0.0
        end
      end)
    end)
  end

  defp dot_list(xs, ys) do
    Enum.zip(xs, ys)
    |> Enum.reduce(0.0, fn {x, y}, acc -> acc + x * y end)
  end

  defp slice_vector(_tensor, []), do: []

  defp slice_vector(tensor, indices) do
    vals = tensor |> Nx.flatten() |> Nx.to_flat_list() |> List.to_tuple()
    Enum.map(indices, &elem(vals, &1))
  end

  defp submatrix(_tensor, [], []), do: Nx.broadcast(0.0, {0, 0})

  defp submatrix(tensor, row_indices, col_indices) do
    {_rows, cols} = Nx.shape(tensor)

    row_data =
      tensor
      |> Nx.to_flat_list()
      |> Enum.chunk_every(cols)
      |> Enum.map(&List.to_tuple/1)
      |> List.to_tuple()

    row_indices
    |> Enum.map(fn r ->
      row = elem(row_data, r)
      Enum.map(col_indices, &elem(row, &1))
    end)
    |> Nx.tensor()
  end

  defp symmetrize(matrix), do: Nx.multiply(Nx.add(matrix, Nx.transpose(matrix)), 0.5)

  defp logistic(logit) when logit >= 35.0, do: 1.0
  defp logistic(logit) when logit <= -35.0, do: 0.0
  defp logistic(logit), do: 1.0 / (1.0 + :math.exp(-logit))

  # Builds the initial diagonal Q matrix from q_specs.
  defp build_initial_q(q_specs, n) do
    BstsNx.Validation.fixed_q_matrix(q_specs, n)
  end

  # Computes per-dimension sum of squares for process residuals.
  # Returns a map %{dim_index => sum_of_squares}.
  defp process_residuals_structured(%Nx.Tensor{} = sampled_states, f_t, q_specs) do
    t = Nx.axis_size(sampled_states, 0)
    state_dim = Nx.axis_size(f_t, 0)

    if t < 2 do
      Map.new(q_specs, fn qs -> {qs.dim_index, 0.0} end)
    else
      prev_states = Nx.slice(sampled_states, [0, 0], [t - 1, state_dim])
      next_states = Nx.slice(sampled_states, [1, 0], [t - 1, state_dim])
      predicted = Nx.dot(prev_states, Nx.transpose(f_t))
      residuals = Nx.subtract(next_states, predicted)

      ss_vec =
        residuals
        |> Nx.multiply(residuals)
        |> Nx.sum(axes: [0])
        |> Nx.to_flat_list()
        |> List.to_tuple()

      Map.new(q_specs, fn qs ->
        ss_val =
          if qs.dim_index < tuple_size(ss_vec), do: elem(ss_vec, qs.dim_index), else: 0.0

        {qs.dim_index, ss_val |> safe_to_number()}
      end)
    end
  end

  defp process_residuals_structured(sampled_states, f_t, q_specs) when is_list(sampled_states) do
    sampled_states
    |> Nx.stack()
    |> process_residuals_structured(f_t, q_specs)
  end

  # Computes observation sum of squares with multi-dimensional state support.
  # h_rows is tensor {t, n}.
  defp obs_residuals_structured(
         observations,
         %Nx.Tensor{} = sampled_states,
         %Nx.Tensor{} = h_rows
       ) do
    t = Nx.axis_size(sampled_states, 0)

    if t == 0 do
      {0.0, 0}
    else
      preds = Nx.sum(Nx.multiply(h_rows, sampled_states), axes: [1])

      obs_tensor = observations_to_filter_tensor(observations)
      mask = Nx.equal(obs_tensor, obs_tensor)
      diffs = Nx.subtract(obs_tensor, preds)

      ss =
        diffs
        |> Nx.multiply(diffs)
        |> then(&Nx.select(mask, &1, 0.0))
        |> Nx.sum()
        |> Nx.to_number()

      count = mask |> Nx.sum() |> Nx.to_number() |> round()
      {ss, count}
    end
  end

  defp obs_residuals_structured(observations, sampled_states, h_list)
       when is_list(sampled_states) and is_list(h_list) do
    h_rows = h_list |> Enum.map(&structured_h_row/1) |> Enum.map(&Nx.flatten/1) |> Nx.stack()

    observations
    |> obs_residuals_structured(Nx.stack(sampled_states), h_rows)
  end

  # Resamples each Q diagonal entry from its inverse-gamma posterior.
  # Returns {list_of_{dim_index, new_value}, next_key}.
  defp resample_q_components(q_specs, per_dim_ss, num_diffs, key) do
    case q_specs do
      [] ->
        {[], key}

      _ ->
        dim_indices = Enum.map(q_specs, & &1.dim_index)

        shapes =
          Enum.map(q_specs, fn qs ->
            safe_to_number(qs.prior_shape) + num_diffs / 2
          end)

        scales =
          Enum.map(q_specs, fn qs ->
            ss = Map.get(per_dim_ss, qs.dim_index, 0.0) |> safe_to_number()
            safe_to_number(qs.prior_scale) + ss / 2
          end)

        {samples, next_key} =
          Distributions.inv_gamma_sample_defn(
            Nx.tensor(shapes, type: {:f, 64}),
            Nx.tensor(scales, type: {:f, 64}),
            key
          )

        {Enum.zip(dim_indices, Nx.to_flat_list(samples)), next_key}
    end
  end

  defp safe_to_number(%Nx.Tensor{} = t), do: Nx.to_number(t) |> safe_to_number()

  defp safe_to_number(val) when is_number(val) do
    if val == val and abs(val) < 1.0e300 do
      val * 1.0
    else
      raise ArgumentError, "expected a finite numeric posterior parameter, got: #{inspect(val)}"
    end
  end

  defp safe_to_number(other) do
    raise ArgumentError, "expected a finite numeric posterior parameter, got: #{inspect(other)}"
  end

  # Updates the diagonal entries of Q matrix after resampling.
  defp rebuild_q(q_prev, new_vals) do
    diag = Nx.take_diagonal(q_prev)

    case new_vals do
      [] ->
        q_prev

      _ ->
        {dim_indices, values} = Enum.unzip(new_vals)
        index_tensor = dim_indices |> Enum.map(&[&1]) |> Nx.tensor(type: {:s, 64})
        value_tensor = Nx.tensor(values, type: Nx.type(diag))
        Nx.indexed_put(diag, index_tensor, value_tensor) |> Nx.make_diagonal()
    end
  end

  # Normalizes spec.h into a per-timestep list of tensors for residual computation.
  defp normalize_h_for_structured(h, t) when is_list(h) do
    if length(h) != t do
      raise ArgumentError,
            "time-varying H list length (#{length(h)}) must match observations length (#{t})"
    end

    h
  end

  defp normalize_h_for_structured(%Nx.Tensor{} = h, t) do
    rank = Nx.rank(h)

    unless rank >= 1 and rank <= 2 do
      raise ArgumentError,
            "observation matrix H must be rank 1 or 2, got rank #{rank}"
    end

    cond do
      rank == 1 ->
        # Static observation vector.
        List.duplicate(h, t)

      rank == 2 and Nx.axis_size(h, 0) == t ->
        # Time-varying scalar-observation rows encoded as {T, n}.
        Enum.map(0..(t - 1), fn i ->
          Nx.slice(h, [i, 0], [1, Nx.axis_size(h, 1)])
        end)

      rank == 2 ->
        # Static scalar-observation matrix (must have at least one singleton axis).
        if Nx.axis_size(h, 0) != 1 and Nx.axis_size(h, 1) != 1 do
          raise ArgumentError,
                "structured sampler expects static H with a singleton axis, got shape #{inspect(Nx.shape(h))}"
        end

        List.duplicate(h, t)
    end
  end

  # Converts a per-step H tensor into a row vector for scalar-observation residuals.
  defp structured_h_row(%Nx.Tensor{} = h_t) do
    case Nx.rank(h_t) do
      0 ->
        Nx.reshape(h_t, {1})

      1 ->
        h_t

      2 ->
        cond do
          Nx.axis_size(h_t, 0) == 1 ->
            Nx.squeeze(h_t, axes: [0])

          Nx.axis_size(h_t, 1) == 1 ->
            Nx.squeeze(h_t, axes: [1])

          true ->
            raise ArgumentError,
                  "structured scalar-observation residual expects rank-2 H with a singleton axis, got shape #{inspect(Nx.shape(h_t))}"
        end

      _ ->
        raise ArgumentError,
              "structured scalar-observation residual expects rank <= 2 H, got rank #{Nx.rank(h_t)}"
    end
  end

  defp observations_to_filter_tensor(observations) do
    nan = Nx.Constants.nan() |> Nx.to_number()

    observations
    |> Enum.map(fn y ->
      if missing_observation?(y), do: nan, else: observation_to_number(y)
    end)
    |> Nx.tensor(type: {:f, 64})
  end

  defp scalar_tensor_list(tensor) do
    tensor
    |> Nx.to_flat_list()
    |> Enum.map(&Nx.tensor/1)
  end

  defp tensor_time_slices(tensor) do
    tensor
    |> Nx.to_batched(1)
    |> Enum.map(&Nx.squeeze(&1, axes: [0]))
  end

  defp latent_transition_count(observations) do
    # Structured samplers impute the full latent trajectory via simulation smoother.
    # Therefore process residuals use all adjacent transitions, not only observed pairs.
    max(length(observations) - 1, 0)
  end

  defp to_tuple_matrix(rows) do
    rows
    |> Enum.map(&List.to_tuple/1)
    |> List.to_tuple()
  end

  defp observation_to_number(%Nx.Tensor{} = v), do: Nx.to_number(v)
  defp observation_to_number(v) when is_number(v), do: v * 1.0

  # -- Input validation helpers ------------------------------------------------

  defp validate_positive!(_name, value) when is_integer(value) and value > 0, do: :ok

  defp validate_positive!(name, value) do
    raise ArgumentError, "#{name} must be a positive integer, got: #{inspect(value)}"
  end

  defp validate_non_negative_integer!(_name, value) when is_integer(value) and value >= 0, do: :ok

  defp validate_non_negative_integer!(name, value) do
    raise ArgumentError, "#{name} must be a non-negative integer, got: #{inspect(value)}"
  end

  defp validate_non_negative_var!(_name, %Nx.Tensor{} = t) do
    if Nx.rank(t) != 0 do
      raise ArgumentError,
            "variance must be a scalar (number or rank-0 tensor), got tensor with shape #{inspect(Nx.shape(t))}"
    end

    if Nx.to_number(t) < 0,
      do: raise(ArgumentError, "variance must be non-negative"),
      else: :ok
  end

  defp validate_non_negative_var!(name, v) when is_number(v) do
    if v < 0,
      do: raise(ArgumentError, "#{name} must be non-negative, got: #{v}"),
      else: :ok
  end

  defp validate_non_negative_var!(name, v) do
    raise ArgumentError,
          "#{name} must be a non-negative number or scalar tensor, got: #{inspect(v)}"
  end

  defp validate_prior_params!(shape, scale) do
    unless is_number(shape) and shape > 0 do
      raise ArgumentError,
            "prior_shape must be a positive number, got: #{inspect(shape)}"
    end

    unless is_number(scale) and scale > 0 do
      raise ArgumentError,
            "prior_scale must be a positive number, got: #{inspect(scale)}"
    end
  end

  # -- Sum of squares helpers --------------------------------------------------

  # Process sum of squares with explicit transition parameter f
  defp process_sum_of_squares(xs, f_t) do
    t = Nx.axis_size(xs, 0)

    if t < 2 do
      0.0
    else
      prev = Nx.slice(xs, [0], [t - 1])
      curr = Nx.slice(xs, [1], [t - 1])

      curr
      |> Nx.subtract(Nx.multiply(f_t, prev))
      |> then(&Nx.multiply(&1, &1))
      |> Nx.sum()
      |> Nx.to_number()
    end
  end

  # Observation sum of squares with time-varying h
  defp obs_sum_of_squares(obs_tensor, xs, h_tensor, obs_present_mask) do
    preds = Nx.multiply(h_tensor, xs)
    diffs = Nx.subtract(obs_tensor, preds)

    diffs
    |> then(&Nx.multiply(&1, &1))
    |> then(&Nx.select(obs_present_mask, &1, 0.0))
    |> Nx.sum()
    |> Nx.to_number()
  end

  # Resample process and observation variances from inverse-gamma posteriors
  defp resample_variances(
         key,
         prior_shape,
         prior_scale,
         {process_ss, num_diffs},
         {obs_ss, t_steps}
       ) do
    shape_q = prior_shape + num_diffs / 2
    scale_q = prior_scale + process_ss / 2
    shape_r = prior_shape + t_steps / 2
    scale_r = prior_scale + obs_ss / 2

    {samples, next_key} =
      Distributions.inv_gamma_sample_defn(
        Nx.tensor([shape_q, shape_r], type: {:f, 64}),
        Nx.tensor([scale_q, scale_r], type: {:f, 64}),
        key
      )

    q_sample = samples |> Nx.slice([0], [1]) |> Nx.squeeze(axes: [0])
    r_sample = samples |> Nx.slice([1], [1]) |> Nx.squeeze(axes: [0])

    {q_sample, r_sample, next_key}
  end

  # Convert h to a list of float values for residual computation.
  # Validates that the result length matches t.
  defp h_to_number_list(h, t) when is_number(h), do: List.duplicate(h * 1.0, t)

  defp h_to_number_list(%Nx.Tensor{} = h, t) do
    if Nx.rank(h) == 0 do
      List.duplicate(Nx.to_number(h), t)
    else
      flat = Nx.to_flat_list(h)

      if length(flat) != t do
        raise ArgumentError,
              "h tensor has #{length(flat)} elements but exactly #{t} are required"
      end

      flat
    end
  end

  defp h_to_number_list(h, _t) when is_list(h) do
    Enum.map(h, fn
      %Nx.Tensor{} = v -> Nx.to_number(v)
      v when is_number(v) -> v * 1.0
    end)
  end
end
