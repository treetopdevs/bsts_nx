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

  alias BstsNx.KalmanFilter
  alias BstsNx.ModelSpec
  import BstsNx.Utils, only: [to_tensor: 1, missing_observation?: 1]
  require Logger

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
    be expressed as `nil`. These are supported but are excluded from the
    observation variance update, and a warning is logged.
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

    # Determine base seed from options or default
    base_seed = Keyword.get(opts, :seed, System.os_time())
    seeds = Keyword.get(opts, :seeds)
    key_opt = Keyword.get(opts, :key)
    timeout = Keyword.get(opts, :timeout, 300_000)

    chain_opts_list =
      cond do
        is_list(seeds) and length(seeds) == num_chains ->
          # Explicit per-chain seeds; remove :key to avoid overriding
          opts_clean = opts |> Keyword.delete(:key) |> Keyword.delete(:seeds)

          Enum.map(0..(num_chains - 1), fn idx ->
            Keyword.put(opts_clean, :seed, Enum.at(seeds, idx))
          end)

        key_opt != nil ->
          # Split the PRNG key into independent per-chain subkeys
          keys = Nx.Random.split(key_opt, parts: num_chains)

          opts_clean =
            opts |> Keyword.delete(:key) |> Keyword.delete(:seed) |> Keyword.delete(:seeds)

          Enum.map(0..(num_chains - 1), fn idx ->
            Keyword.put(opts_clean, :key, keys[idx])
          end)

        true ->
          opts_clean = opts |> Keyword.delete(:seeds)

          Enum.map(0..(num_chains - 1), fn idx ->
            Keyword.put(opts_clean, :seed, base_seed + idx)
          end)
      end

    results =
      chain_opts_list
      |> Task.async_stream(
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
        end,
        max_concurrency: System.schedulers_online(),
        timeout: timeout,
        on_timeout: :kill_task
      )
      |> Enum.with_index()
      |> Enum.reduce([], fn
        {{:ok, res}, _idx}, acc ->
          [res | acc]

        {{:exit, reason}, idx}, acc ->
          Logger.warning("Gibbs sampler chain #{idx} failed: #{inspect(reason)}")
          acc
      end)
      |> Enum.reverse()

    if results == [] do
      raise RuntimeError, "all #{num_chains} Gibbs sampler chains failed"
    end

    results
  end

  @doc """
  Generalized Gibbs sampler that accepts a time-varying observation vector `h`.

  Unlike `sample/7` which hardcodes a scalar local level model (`F=1, H=1`),
  this function accepts explicit `f` (transition matrix) and `h` (observation
  vector) arguments.  When `h` is a 1-D tensor of length `t`, element `h[i]`
  is the observation coefficient at time step `i`.  This enables seasonal
  regression models where Fourier predictions serve as the time-varying
  regressor.

  * `observations` – list of scalar observations. Missing values should be `nil`.
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
          [number | nil],
          number | Nx.t(),
          number | Nx.t(),
          pos_integer(),
          keyword()
        ) :: [sample_result]
  def sample_general(observations, f, h, num_samples, opts \\ []) do
    validate_positive!(:num_samples, num_samples)

    if Enum.any?(observations, &missing_observation?/1) do
      Logger.warning(
        "GibbsSampler received missing observations (nil or NaN); these are skipped in the observation variance update"
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

    f_val = Nx.to_number(f_t)
    t = length(observations)
    h_vals = h_to_number_list(h, t)

    if length(h_vals) != t do
      raise ArgumentError,
            "h vector length (#{length(h_vals)}) must match observations length (#{t})"
    end

    {_, _, samples_acc, _key_acc} =
      Enum.reduce(1..total_iters, {process_var_t, obs_var_t, [], key}, fn iter,
                                                                          {q_prev, r_prev, acc, k} ->
        {filtered, predicted} =
          KalmanFilter.filter_with_pred(observations, f_t, h, q_prev, r_prev, x0, p0)

        smoothed = BstsNx.Smoother.rts(filtered, predicted, f_t)

        {sampled_states, new_key_smooth} =
          BstsNx.Smoother.simulate_with_key(smoothed, filtered, predicted, f_t, key: k)

        {_means, covs} = Enum.unzip(filtered)
        # Batch conversion: stack all scalar state tensors and extract as flat list
        x_list = sampled_states |> Nx.stack() |> Nx.to_flat_list()

        {process_ss, num_diffs} = process_sum_of_squares(x_list, f_val)
        {obs_ss, t_steps} = obs_sum_of_squares(observations, x_list, h_vals)

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
              states: sampled_states,
              state_covs: covs,
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
          obs_var: Nx.t()
        }

  @doc """
  Gibbs sampler for structured multi-dimensional state-space models.

  Unlike `sample_general/5` which works with scalar state and a single
  process variance, this function accepts a `%ModelSpec{}` that fully
  describes a multi-dimensional model.  Each diagonal entry of the
  process covariance matrix Q is resampled independently from its own
  inverse-gamma posterior.

  ## Arguments

    * `observations` — list of scalar observations. Missing values as `nil`.
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
  @spec sample_structured([number | nil], ModelSpec.t(), pos_integer(), keyword()) ::
          [structured_result()]
  def sample_structured(observations, %ModelSpec{} = spec, num_samples, opts \\ []) do
    validate_positive!(:num_samples, num_samples)

    if Enum.any?(observations, &missing_observation?/1) do
      Logger.warning(
        "GibbsSampler.sample_structured received missing observations (nil or NaN); " <>
          "these are skipped in the observation variance update"
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

    # Build initial Q matrix from q_specs
    q_matrix = build_initial_q(spec.q_specs, Nx.axis_size(spec.f, 0))
    r_var = to_tensor(spec.obs_var)

    # Normalize H to a per-timestep list for residual computation
    h_list = normalize_h_for_structured(spec.h, t)

    {_, _, samples_acc, _key_acc} =
      Enum.reduce(1..total_iters, {q_matrix, r_var, [], key}, fn iter, {q_prev, r_prev, acc, k} ->
        # 1. Kalman filter
        {filtered, predicted} =
          KalmanFilter.filter_with_pred(
            observations,
            spec.f,
            spec.h,
            q_prev,
            r_prev,
            spec.x0,
            spec.p0
          )

        # 2. RTS smoother
        smoothed = BstsNx.Smoother.rts(filtered, predicted, spec.f)

        # 3. Carter-Kohn simulation smoother
        {sampled_states, new_key} =
          BstsNx.Smoother.simulate_with_key(smoothed, filtered, predicted, spec.f, key: k)

        # 4. Extract filtered covariances
        {_means, covs} = Enum.unzip(filtered)

        # 5. Process residuals: e_t = x_t - F * x_{t-1}
        per_dim_ss = process_residuals_structured(sampled_states, spec.f, spec.q_specs)

        # 6. Observation residuals: SS_obs = Σ (y_t - H_t · x_t)²
        {obs_ss, t_obs} = obs_residuals_structured(observations, sampled_states, h_list)

        # 7. Resample each Q diagonal entry independently
        {q_new, key_after_q} = resample_q_components(spec.q_specs, per_dim_ss, t, new_key)

        # 8. Rebuild Q matrix
        q_matrix_new = rebuild_q(q_prev, q_new)

        # 9. Resample observation variance
        shape_r = spec.obs_prior_shape + t_obs / 2
        scale_r = spec.obs_prior_scale + obs_ss / 2
        keys_r = Nx.Random.split(key_after_q, parts: 2)
        r_sample = BstsNx.Distributions.inv_gamma_sample(shape_r, scale_r, key: keys_r[0])
        next_key = keys_r[1]

        acc2 =
          if iter > burn_in and rem(iter - burn_in, thin) == 0 do
            sample_map = %{
              states: sampled_states,
              state_covs: covs,
              q_matrix: q_matrix_new,
              obs_var: r_sample
            }

            [sample_map | acc]
          else
            acc
          end

        {q_matrix_new, r_sample, acc2, next_key}
      end)

    Enum.reverse(samples_acc)
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
          [number | nil],
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

    base_seed = Keyword.get(opts, :seed, System.os_time())
    seeds = Keyword.get(opts, :seeds)
    key_opt = Keyword.get(opts, :key)
    timeout = Keyword.get(opts, :timeout, 300_000)

    chain_opts_list =
      cond do
        is_list(seeds) and length(seeds) == num_chains ->
          opts_clean = opts |> Keyword.delete(:key) |> Keyword.delete(:seeds)

          Enum.map(0..(num_chains - 1), fn idx ->
            Keyword.put(opts_clean, :seed, Enum.at(seeds, idx))
          end)

        key_opt != nil ->
          keys = Nx.Random.split(key_opt, parts: num_chains)

          opts_clean =
            opts |> Keyword.delete(:key) |> Keyword.delete(:seed) |> Keyword.delete(:seeds)

          Enum.map(0..(num_chains - 1), fn idx ->
            Keyword.put(opts_clean, :key, keys[idx])
          end)

        true ->
          opts_clean = opts |> Keyword.delete(:seeds)

          Enum.map(0..(num_chains - 1), fn idx ->
            Keyword.put(opts_clean, :seed, base_seed + idx)
          end)
      end

    results =
      chain_opts_list
      |> Task.async_stream(
        fn chain_opts ->
          sample_structured(observations, spec, num_samples, chain_opts)
        end,
        max_concurrency: System.schedulers_online(),
        timeout: timeout,
        on_timeout: :kill_task
      )
      |> Enum.with_index()
      |> Enum.reduce([], fn
        {{:ok, res}, _idx}, acc ->
          [res | acc]

        {{:exit, reason}, idx}, acc ->
          Logger.warning("Structured Gibbs sampler chain #{idx} failed: #{inspect(reason)}")
          acc
      end)
      |> Enum.reverse()

    if results == [] do
      raise RuntimeError, "all #{num_chains} structured Gibbs sampler chains failed"
    end

    results
  end

  # -- Structured sampler helpers ----------------------------------------------

  # Builds the initial diagonal Q matrix from q_specs.
  defp build_initial_q(q_specs, n) do
    diag = List.duplicate(0.0, n)

    diag =
      Enum.reduce(q_specs, diag, fn qs, d ->
        List.replace_at(d, qs.dim_index, qs.initial)
      end)

    Nx.tensor(diag) |> Nx.make_diagonal()
  end

  # Computes per-dimension sum of squares for process residuals.
  # Returns a map %{dim_index => sum_of_squares}.
  defp process_residuals_structured(sampled_states, f_t, q_specs) do
    # sampled_states is a list of state tensors (vectors for multi-dim)
    t = length(sampled_states)

    if t < 2 do
      Map.new(q_specs, fn qs -> {qs.dim_index, 0.0} end)
    else
      # Compute residuals: e_t = x_t - F * x_{t-1}
      residuals =
        sampled_states
        |> Enum.chunk_every(2, 1, :discard)
        |> Enum.map(fn [x_prev, x_curr] ->
          predicted = Nx.dot(f_t, x_prev)
          Nx.subtract(x_curr, predicted)
        end)

      # Sum of squares per dimension
      Map.new(q_specs, fn qs ->
        d = qs.dim_index

        ss =
          Enum.reduce(residuals, 0.0, fn e_t, acc ->
            val = e_t |> Nx.flatten() |> then(&Nx.to_number(&1[d]))
            acc + val * val
          end)

        {d, ss}
      end)
    end
  end

  # Computes observation sum of squares with multi-dimensional state support.
  # h_list is a list of per-timestep observation tensors.
  defp obs_residuals_structured(observations, sampled_states, h_list) do
    residuals =
      observations
      |> Enum.zip(sampled_states)
      |> Enum.zip(h_list)
      |> Enum.flat_map(fn {{y, x_t}, h_t} ->
        if missing_observation?(y) do
          []
        else
          y_val = if is_number(y), do: y, else: Nx.to_number(y)
          h_row = structured_h_row(h_t)
          pred = Nx.to_number(Nx.dot(h_row, Nx.flatten(x_t)))
          diff = y_val - pred
          [diff * diff]
        end
      end)

    t_obs = Enum.count(observations, &(not missing_observation?(&1)))
    {Enum.sum(residuals), t_obs}
  end

  # Resamples each Q diagonal entry from its inverse-gamma posterior.
  # Returns {list_of_{dim_index, new_value}, next_key}.
  defp resample_q_components(q_specs, per_dim_ss, t, key) do
    num_diffs = max(t - 1, 0)
    num_q = length(q_specs)
    keys = Nx.Random.split(key, parts: num_q + 1)

    new_vals =
      q_specs
      |> Enum.with_index()
      |> Enum.map(fn {qs, idx} ->
        ss = Map.get(per_dim_ss, qs.dim_index, 0.0)
        shape = qs.prior_shape + num_diffs / 2
        scale = qs.prior_scale + ss / 2
        sample = BstsNx.Distributions.inv_gamma_sample(shape, scale, key: keys[idx])
        {qs.dim_index, Nx.to_number(sample)}
      end)

    {new_vals, keys[num_q]}
  end

  # Updates the diagonal entries of Q matrix after resampling.
  defp rebuild_q(q_prev, new_vals) do
    Enum.reduce(new_vals, q_prev, fn {dim_idx, val}, q ->
      # Create index tensor and update the diagonal entry
      indices = Nx.tensor([[dim_idx, dim_idx]])
      updates = Nx.tensor([val])
      Nx.indexed_put(q, indices, updates)
    end)
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
  defp process_sum_of_squares(x_list, f_val) do
    diffs =
      x_list
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.map(fn [x_prev, x_curr] -> :math.pow(x_curr - f_val * x_prev, 2) end)

    {Enum.sum(diffs), length(diffs)}
  end

  # Observation sum of squares with time-varying h
  defp obs_sum_of_squares(observations, x_list, h_vals) do
    residuals =
      observations
      |> Enum.zip(x_list)
      |> Enum.zip(h_vals)
      |> Enum.flat_map(fn {{y, x_t}, h_i} ->
        if missing_observation?(y) do
          []
        else
          y_val = if is_number(y), do: y, else: Nx.to_number(y)
          [:math.pow(y_val - h_i * x_t, 2)]
        end
      end)

    {Enum.sum(residuals), Enum.count(observations, &(not missing_observation?(&1)))}
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

    keys_split = Nx.Random.split(key, parts: 3)
    q_sample = BstsNx.Distributions.inv_gamma_sample(shape_q, scale_q, key: keys_split[0])
    r_sample = BstsNx.Distributions.inv_gamma_sample(shape_r, scale_r, key: keys_split[1])
    {q_sample, r_sample, keys_split[2]}
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
