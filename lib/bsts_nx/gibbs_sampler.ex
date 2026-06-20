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
  alias BstsNx.GibbsSampler.Residuals
  alias BstsNx.GibbsSampler.SpikeSlab
  alias BstsNx.GibbsSampler.Structured
  alias BstsNx.KalmanFilter
  alias BstsNx.ModelSpec

  import BstsNx.Utils,
    only: [to_tensor: 1, missing_observation?: 1, split_key_at: 2]

  require Logger
  require Nx.Defn

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
    * `:fused` – when `true`, the entire MCMC chain runs inside a single
      `Nx.Defn` while-loop, so a compiled backend executes the whole run as
      one program (roughly an order of magnitude faster under EXLA).
      Defaults to `true` when a global defn compiler is configured via
      `Nx.Defn.global_default_options(compiler: ...)` and `false` under the
      interpreting evaluator, where stepwise execution is faster.  Draws are
      identical between both paths for the same key.

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
  * `:allow_partial` – when `true`, return successful chains if some chains fail
    (default: `false`, all-or-fail).

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
    validate_non_empty_observations!(observations)

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
    validate_non_empty_observations!(observations)

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

    obs_tensor = Structured.observations_to_filter_tensor(observations)
    h_tensor = Nx.tensor(h_vals, type: {:f, 64})
    obs_present_mask = Nx.equal(obs_tensor, obs_tensor)
    t_steps = obs_present_mask |> Nx.sum() |> Nx.to_number() |> round()
    num_diffs = Structured.latent_transition_count(observations)

    if Keyword.get(opts, :fused, fused_default()) and t >= 2 do
      sample_general_fused(
        obs_tensor,
        h_tensor,
        f_t,
        process_var_t,
        obs_var_t,
        x0,
        p0,
        prior_shape,
        prior_scale,
        num_diffs,
        t_steps,
        obs_present_mask,
        burn_in,
        thin,
        num_samples,
        total_iters,
        key
      )
    else
      sample_general_stepwise(
        obs_tensor,
        h_tensor,
        f_t,
        process_var_t,
        obs_var_t,
        x0,
        p0,
        prior_shape,
        prior_scale,
        num_diffs,
        t_steps,
        obs_present_mask,
        burn_in,
        thin,
        total_iters,
        key
      )
    end
  end

  defp sample_general_stepwise(
         obs_tensor,
         h_tensor,
         f_t,
         process_var_t,
         obs_var_t,
         x0,
         p0,
         prior_shape,
         prior_scale,
         num_diffs,
         t_steps,
         obs_present_mask,
         burn_in,
         thin,
         total_iters,
         key
       ) do
    {_, _, samples_acc, _key_acc} =
      Enum.reduce(1..total_iters, {process_var_t, obs_var_t, [], key}, fn iter,
                                                                          {q_prev, r_prev, acc, k} ->
        {xs, ps} = KalmanFilter.filter_defn(obs_tensor, f_t, h_tensor, q_prev, r_prev, x0, p0)

        {_smoothed_xs, _smoothed_ps, sampled_xs, new_key_smooth} =
          BstsNx.Smoother.rts_and_simulate_defn(xs, ps, f_t, q_prev, k)

        process_ss = Residuals.process_sum_of_squares(sampled_xs, f_t)
        obs_ss = Residuals.obs_sum_of_squares(obs_tensor, sampled_xs, h_tensor, obs_present_mask)

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
              states: Structured.scalar_tensor_list(sampled_xs),
              state_covs: Structured.scalar_tensor_list(ps),
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

  # Runs the whole scalar Gibbs chain inside a single compiled defn so the
  # filter, smoother, simulation smoother and variance resampling execute as
  # one backend program with no per-iteration host round-trips.  Draws are
  # bitwise-identical to the stepwise path for the same PRNG key because the
  # same defn kernels are invoked in the same order.
  defp sample_general_fused(
         obs_tensor,
         h_tensor,
         f_t,
         process_var_t,
         obs_var_t,
         x0,
         p0,
         prior_shape,
         prior_scale,
         num_diffs,
         t_steps,
         obs_present_mask,
         burn_in,
         thin,
         num_samples,
         total_iters,
         key
       ) do
    f64 = {:f, 64}
    t = Nx.axis_size(obs_tensor, 0)

    shape_qr =
      Nx.tensor([prior_shape + num_diffs / 2, prior_shape + t_steps / 2], type: f64)

    loop_meta = Nx.tensor([burn_in, thin, total_iters], type: {:s, 64})
    zero = Nx.tensor(0.0, type: f64)

    {states_out, covs_out, qr_out} =
      scalar_chain_defn(
        obs_tensor,
        h_tensor,
        Nx.as_type(f_t, f64),
        Nx.as_type(process_var_t, f64),
        Nx.as_type(obs_var_t, f64),
        Nx.as_type(x0, f64),
        Nx.as_type(p0, f64),
        shape_qr,
        Nx.tensor(prior_scale, type: f64),
        obs_present_mask,
        loop_meta,
        key,
        Nx.broadcast(zero, {num_samples, t}),
        Nx.broadcast(zero, {num_samples, t}),
        Nx.broadcast(zero, {num_samples, 2})
      )

    # Pull chain outputs to the host once so the per-sample slicing below does
    # not issue thousands of small device ops on compiled backends.
    states_rows =
      states_out |> to_host() |> Nx.to_batched(1) |> Enum.map(&Nx.squeeze(&1, axes: [0]))

    covs_rows = covs_out |> to_host() |> Nx.to_batched(1) |> Enum.map(&Nx.squeeze(&1, axes: [0]))
    qr_rows = qr_out |> to_host() |> Nx.to_batched(1) |> Enum.map(&Nx.squeeze(&1, axes: [0]))

    [states_rows, covs_rows, qr_rows]
    |> Enum.zip()
    |> Enum.map(fn {states_row, covs_row, qr_row} ->
      %{
        states: Structured.scalar_tensor_list(states_row),
        state_covs: Structured.scalar_tensor_list(covs_row),
        process_var: qr_row[0],
        obs_var: qr_row[1]
      }
    end)
  end

  # Compiled scalar Gibbs chain.  `loop_meta` is `[burn_in, thin, total_iters]`
  # and `shape_qr` holds the (iteration-invariant) inverse-gamma posterior
  # shapes for Q and R.  Retained draws are written to the preallocated
  # accumulators: non-retained iterations overwrite the pending slot, which is
  # finalized by the next retained iteration (the final iteration is always
  # retained because total_iters = burn_in + num_samples * thin).
  Nx.Defn.defn scalar_chain_defn(
                 obs,
                 h_vec,
                 f_t,
                 q0,
                 r0,
                 x0,
                 p0,
                 shape_qr,
                 prior_scale,
                 obs_mask,
                 loop_meta,
                 key,
                 states_acc0,
                 covs_acc0,
                 qr_acc0
               ) do
    t = Nx.axis_size(obs, 0)

    {_, _, _, _, _, states_out, covs_out, qr_out, _, _, _, _, _, _, _, _, _} =
      while {iter = Nx.tensor(1, type: {:s, 64}), q = q0, r = r0, k = key,
             kept = Nx.tensor(0, type: {:s, 64}), states_acc = states_acc0, covs_acc = covs_acc0,
             qr_acc = qr_acc0, obs_in = obs, h_in = h_vec, f_in = f_t, x0_in = x0, p0_in = p0,
             shape_in = shape_qr, prior_scale_in = prior_scale, mask_in = obs_mask,
             meta = loop_meta},
            iter <= meta[2] do
        {xs, ps} = KalmanFilter.filter_defn_vec(obs_in, f_in, h_in, q, r, x0_in, p0_in)
        {sxs, sps} = BstsNx.Smoother.rts_defn_impl(xs, ps, f_in, q)
        {states_raw, k2} = BstsNx.Smoother.simulate_defn_impl(sxs, sps, xs, ps, f_in, q, k)
        states = Nx.as_type(states_raw, {:f, 64})

        prev = Nx.slice(states, [0], [t - 1])
        curr = Nx.slice(states, [1], [t - 1])
        pdiff = curr - f_in * prev
        process_ss = Nx.sum(pdiff * pdiff)

        odiff = obs_in - h_in * states
        obs_ss = Nx.sum(Nx.select(mask_in, odiff * odiff, 0.0))

        scales = Nx.stack([prior_scale_in + process_ss / 2.0, prior_scale_in + obs_ss / 2.0])

        {qr_draw, k3} =
          Distributions.inv_gamma_sample_defn_impl(
            shape_in,
            scales,
            k2,
            Nx.tensor(0.0, type: {:f, 64}),
            Nx.tensor(0, type: {:u, 8})
          )

        keep =
          Nx.logical_and(iter > meta[0], Nx.remainder(iter - meta[0], meta[1]) == 0)
          |> Nx.as_type({:s, 64})

        states_acc = Nx.put_slice(states_acc, [kept, 0], Nx.new_axis(states, 0))

        covs_acc =
          Nx.put_slice(covs_acc, [kept, 0], Nx.new_axis(Nx.as_type(ps, {:f, 64}), 0))

        qr_acc = Nx.put_slice(qr_acc, [kept, 0], Nx.reshape(qr_draw, {1, 2}))

        {iter + 1, qr_draw[0], qr_draw[1], k3, kept + keep, states_acc, covs_acc, qr_acc, obs_in,
         h_in, f_in, x0_in, p0_in, shape_in, prior_scale_in, mask_in, meta}
      end

    {states_out, covs_out, qr_out}
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
    * `:fused` — when `true`, the standard (non-spike-and-slab) sampler runs
      the entire MCMC chain inside a single `Nx.Defn` while-loop, so a
      compiled backend executes the whole run as one program.  Defaults to
      `true` when a global defn compiler is configured via
      `Nx.Defn.global_default_options(compiler: ...)` and `false` under the
      interpreting evaluator.  Draws are identical between both paths for
      the same key.

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
    validate_non_empty_observations!(observations)
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
    q_matrix = Structured.build_initial_q(spec.q_specs, state_dim)
    r_var = to_tensor(spec.obs_var)

    # Normalize H to a per-timestep list for residual computation
    h_list = Structured.normalize_h(spec.h, t)

    case spec.regression do
      %{mode: :spike_and_slab} = regression ->
        SpikeSlab.sample(
          observations,
          spec,
          regression,
          q_matrix,
          r_var,
          h_list,
          burn_in,
          thin,
          key,
          total_iters
        )

      _ ->
        Structured.sample_standard(
          observations,
          spec,
          q_matrix,
          r_var,
          h_list,
          burn_in,
          thin,
          key,
          total_iters,
          Keyword.get(opts, :fused, fused_default())
        )
    end
  end

  # Whole-chain fusion pays off when defn graphs are compiled (e.g. EXLA),
  # where it collapses thousands of per-iteration dispatches into one
  # program.  Under the interpreting evaluator the fused while-loop copies
  # its sample accumulators every iteration and is slower than the stepwise
  # path, so fusion is off unless a real compiler is configured.
  defp fused_default do
    compiler = Nx.Defn.default_options() |> Keyword.get(:compiler, Nx.Defn.Evaluator)
    compiler != Nx.Defn.Evaluator
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
    * `:allow_partial` — when `true`, return successful chains if some chains
      fail (default: `false`, all-or-fail)
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
    validate_non_empty_observations!(observations)

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
    allow_partial? = Keyword.get(opts, :allow_partial, false)

    outcomes =
      num_chains
      |> chain_opts_list(opts)
      |> Task.async_stream(&capture_chain_result(fun, &1),
        max_concurrency: System.schedulers_online(),
        timeout: timeout,
        on_timeout: :kill_task
      )
      |> Enum.with_index()
      |> Enum.map(fn
        {{:ok, {:chain_ok, res}}, idx} ->
          {:ok, idx, res}

        {{:ok, {:chain_error, reason}}, idx} ->
          {:error, idx, reason}

        {{:exit, reason}, idx} ->
          {:error, idx, reason}
      end)

    failures =
      for {:error, idx, reason} <- outcomes do
        Logger.warning("#{warning_label} chain #{idx} failed: #{inspect(reason)}")
        {idx, reason}
      end

    results = for {:ok, _idx, res} <- outcomes, do: res

    cond do
      failures == [] ->
        results

      results == [] ->
        raise RuntimeError, "all #{num_chains} #{error_label} chains failed"

      allow_partial? ->
        results

      true ->
        raise RuntimeError,
              "#{length(failures)} of #{num_chains} #{error_label} chains failed"
    end
  end

  defp capture_chain_result(fun, chain_opts) do
    {:chain_ok, fun.(chain_opts)}
  rescue
    exception ->
      {:chain_error, exception}
  catch
    kind, reason ->
      {:chain_error, {kind, reason}}
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

  # Moves a (potentially device-backed) tensor to the host BinaryBackend so
  # repeated small slices stay off the device dispatch path.  Cross-backend
  # arithmetic remains supported by Nx, so downstream consumers are
  # unaffected.
  defp to_host(tensor), do: Nx.backend_copy(tensor, Nx.BinaryBackend)

  # -- Input validation helpers ------------------------------------------------

  defp validate_non_empty_observations!(observations)
       when is_list(observations) and observations != [],
       do: :ok

  defp validate_non_empty_observations!([]) do
    raise ArgumentError, "observations must contain at least one value"
  end

  defp validate_non_empty_observations!(observations) do
    raise ArgumentError, "observations must be a non-empty list, got: #{inspect(observations)}"
  end

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
