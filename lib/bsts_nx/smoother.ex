defmodule BstsNx.Smoother do
  @moduledoc """
  Kalman and simulation smoothing for linear Gaussian state-space models.

  This module implements the Rauch–Tung–Striebel (RTS) smoother for
  producing smoothed state estimates from the outputs of a Kalman
  filter.  It also provides a simulation smoother (Carter–Kohn
  algorithm) that generates sample latent state trajectories from
  their posterior distribution.  These routines operate on the
  sequences of filtered and predicted states returned by
  `BstsNx.KalmanFilter.filter_with_pred/7` and require the state
  transition matrix `F` as input.
  """

  import Nx, only: [dot: 2, transpose: 1, subtract: 2, add: 2]
  import BstsNx.Utils, only: [to_tensor: 1]
  require Nx.Defn
  require Logger

  @doc """
  Compiled RTS backward smoother for one-dimensional systems.

  Takes the flat `{xs, ps}` tensors returned by
  `BstsNx.KalmanFilter.filter_defn/7` and the scalar transition
  parameter `f` and process variance `q`, and returns
  `{smoothed_xs, smoothed_ps}` as tensors of shape `{t}`.

  Unlike `rts/3`, this function works directly on tensors and can
  be JIT-compiled via `Nx.Defn` for improved performance.  The
  predicted state and covariance at each step are recomputed from
  the filtered outputs, avoiding the need for the caller to store
  them separately.

  ## Examples

      iex> obs = Nx.tensor([1.0, 2.0, 3.0])
      iex> {xs, ps} = BstsNx.KalmanFilter.filter_defn(obs, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0)
      iex> {sxs, _sps} = BstsNx.Smoother.rts_defn(xs, ps, 1.0, 1.0)
      iex> Nx.shape(sxs)
      {3}
  """
  @spec rts_defn(Nx.t(), Nx.t(), number | Nx.t(), number | Nx.t()) :: {Nx.t(), Nx.t()}
  def rts_defn(xs, ps, f, q) do
    f_t = to_tensor(f)
    q_t = to_tensor(q)
    rts_defn_impl(xs, ps, f_t, q_t)
  end

  Nx.Defn.defn rts_defn_impl(xs, ps, f, q) do
    t = Nx.axis_size(xs, 0)
    # Initialize output accumulators
    sxs = Nx.broadcast(Nx.tensor(0.0), {t})
    sps = Nx.broadcast(Nx.tensor(0.0), {t})
    # Set the last element to the filtered value (smoother boundary condition)
    last_idx = t - 1
    sxs = Nx.put_slice(sxs, [last_idx], Nx.reshape(xs[last_idx], {1}))
    sps = Nx.put_slice(sps, [last_idx], Nx.reshape(ps[last_idx], {1}))

    # Backward pass: iterate from t-2 down to 0
    # Use an ascending counter k from 0..(t-2) and compute i = t - 2 - k
    # to avoid signed integer issues with decrementing Nx while loops
    num_steps = t - 1

    {_, sxs_out, sps_out, _, _, _, _} =
      while {k = Nx.tensor(0), sxs_acc = sxs, sps_acc = sps, xs_in = xs, ps_in = ps, f_in = f,
             q_in = q},
            k < num_steps do
        i = last_idx - 1 - k
        x_filt = xs_in[i]
        p_filt = ps_in[i]
        # Recompute predicted state/cov at i+1 from filtered at i
        x_pred_next = f_in * x_filt
        p_pred_next = f_in * p_filt * f_in + q_in
        # Smoothing gain: C_k = P_filt * F / P_pred_next
        # Guard against zero predicted covariance (perfect state knowledge)
        near_zero_p = Nx.abs(p_pred_next) < 1.0e-15
        safe_pred = Nx.select(near_zero_p, 1.0, p_pred_next)
        c = Nx.select(near_zero_p, 0.0, p_filt * f_in / safe_pred)
        # Smoothed state from previously stored smooth_{i+1}
        x_smooth_next = sxs_acc[i + 1]
        p_smooth_next = sps_acc[i + 1]
        x_smooth = x_filt + c * (x_smooth_next - x_pred_next)
        p_smooth = p_filt + c * (p_smooth_next - p_pred_next) * c
        sxs_new = Nx.put_slice(sxs_acc, [i], Nx.reshape(x_smooth, {1}))
        sps_new = Nx.put_slice(sps_acc, [i], Nx.reshape(p_smooth, {1}))
        {k + 1, sxs_new, sps_new, xs_in, ps_in, f_in, q_in}
      end

    {sxs_out, sps_out}
  end

  @doc """
  Compiled RTS backward smoother that also returns lag-one cross-covariances.

  Identical to `rts_defn/4` but additionally computes the lag-one smoother
  cross-covariance `P_{t+1,t|T} = P_{t+1|T} * G_t` at each step, where
  `G_t` is the RTS smoother gain.  These cross-covariances are needed for
  computing the variance of a block sum of smoothed states (e.g., for
  confidence intervals on cumulative lift).

  Returns `{smoothed_xs, smoothed_ps, lag1_covs}` where `lag1_covs` is a
  tensor of shape `{max(t-1, 1)}`.  `lag1_covs[i]` = `Cov(x_{i+1}, x_i | all data)`
  for `i` in `0..t-2`.  For a series of length 1, `lag1_covs` is `{1}` with
  a single zero element (no meaningful cross-covariances exist).

  ## Examples

      iex> obs = Nx.tensor([1.0, 2.0, 3.0])
      iex> {xs, ps} = BstsNx.KalmanFilter.filter_defn(obs, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0)
      iex> {sxs, sps, lag1} = BstsNx.Smoother.rts_defn_with_lag1(xs, ps, 1.0, 1.0)
      iex> Nx.shape(lag1)
      {2}
  """
  @spec rts_defn_with_lag1(Nx.t(), Nx.t(), number | Nx.t(), number | Nx.t()) ::
          {Nx.t(), Nx.t(), Nx.t()}
  def rts_defn_with_lag1(xs, ps, f, q) do
    f_t = to_tensor(f)
    q_t = to_tensor(q)
    {sxs, sps, lag1_padded} = rts_defn_with_lag1_impl(xs, ps, f_t, q_t)
    # The defn returns lag1 of shape {t} for shape compatibility.
    # Slice to {t-1} since lag1[t-1] is unused padding.
    t = Nx.axis_size(xs, 0)

    lag1 =
      if t > 1,
        do: Nx.slice(lag1_padded, [0], [t - 1]),
        else: Nx.reshape(Nx.tensor(0.0), {1})

    {sxs, sps, lag1}
  end

  Nx.Defn.defn rts_defn_with_lag1_impl(xs, ps, f, q) do
    t = Nx.axis_size(xs, 0)
    # Initialize output accumulators
    sxs = Nx.broadcast(Nx.tensor(0.0), {t})
    sps = Nx.broadcast(Nx.tensor(0.0), {t})
    # Allocate lag1 at size {t} for shape compatibility in defn.
    # Only indices 0..t-2 are meaningful; index t-1 is zero padding.
    lag1 = Nx.broadcast(Nx.tensor(0.0), {t})

    # Set the last element to the filtered value (smoother boundary condition)
    last_idx = t - 1
    sxs = Nx.put_slice(sxs, [last_idx], Nx.reshape(xs[last_idx], {1}))
    sps = Nx.put_slice(sps, [last_idx], Nx.reshape(ps[last_idx], {1}))

    # Backward pass: iterate from t-2 down to 0
    num_steps = t - 1

    {_, sxs_out, sps_out, _, _, _, _, lag1_out} =
      while {k = Nx.tensor(0), sxs_acc = sxs, sps_acc = sps, xs_in = xs, ps_in = ps, f_in = f,
             q_in = q, lag1_acc = lag1},
            k < num_steps do
        i = last_idx - 1 - k
        x_filt = xs_in[i]
        p_filt = ps_in[i]
        # Recompute predicted state/cov at i+1 from filtered at i
        x_pred_next = f_in * x_filt
        p_pred_next = f_in * p_filt * f_in + q_in
        # Smoothing gain: C_k = P_filt * F / P_pred_next
        near_zero_p = Nx.abs(p_pred_next) < 1.0e-15
        safe_pred = Nx.select(near_zero_p, 1.0, p_pred_next)
        c = Nx.select(near_zero_p, 0.0, p_filt * f_in / safe_pred)
        # Smoothed state from previously stored smooth_{i+1}
        x_smooth_next = sxs_acc[i + 1]
        p_smooth_next = sps_acc[i + 1]
        x_smooth = x_filt + c * (x_smooth_next - x_pred_next)
        p_smooth = p_filt + c * (p_smooth_next - p_pred_next) * c
        sxs_new = Nx.put_slice(sxs_acc, [i], Nx.reshape(x_smooth, {1}))
        sps_new = Nx.put_slice(sps_acc, [i], Nx.reshape(p_smooth, {1}))
        # Lag-one cross-covariance: P_{i+1,i|T} = P_{i+1|T} * G_i
        lag1_val = p_smooth_next * c
        lag1_new = Nx.put_slice(lag1_acc, [i], Nx.reshape(lag1_val, {1}))
        {k + 1, sxs_new, sps_new, xs_in, ps_in, f_in, q_in, lag1_new}
      end

    {sxs_out, sps_out, lag1_out}
  end

  @doc """
  Performs the Rauch–Tung–Striebel (RTS) smoother.

  Takes lists of filtered and predicted states as returned by
  `BstsNx.KalmanFilter.filter_with_pred/7` along with the state
  transition matrix `f` and returns a list of smoothed states
  `(mean, covariance)` for each time step.  The lists should be
  ordered such that the i‑th element corresponds to the i‑th
  observation.  The returned list has the same ordering.

  **Trust boundary**: This function does NOT receive `Q` (process
  covariance) and trusts that `predicted` is consistent with `f` and
  the filter output.  Always pass the `predicted` list directly from
  `filter_with_pred/7`; reconstructing or modifying it may produce
  incorrect smoother gains.  For scalar systems where `Q` is available,
  prefer `rts_defn/4` which recomputes predictions internally.
  """
  @spec rts([{Nx.t(), Nx.t()}], [{Nx.t(), Nx.t()}], number | list | Nx.t()) :: [{Nx.t(), Nx.t()}]
  def rts(filtered, predicted, f) do
    f_t = to_tensor(f)
    n = length(filtered)

    if length(predicted) != n do
      raise ArgumentError,
            "filtered (length #{n}) and predicted (length #{length(predicted)}) must have the same length"
    end

    cond do
      n == 0 ->
        []

      n == 1 ->
        filtered

      true ->
        # Convert filtered and predicted to :array for O(1) access
        filt_arr = :array.from_list(filtered)
        pred_arr = :array.from_list(predicted)
        # Preallocate smoothed array with nils
        smooth_arr = :array.new(n, default: nil)
        # Set last entry to last filtered estimate
        {x_last, p_last} = :array.get(n - 1, filt_arr)
        smooth_arr = :array.set(n - 1, {x_last, p_last}, smooth_arr)
        # Iterate backwards to fill smoothed array
        smooth_arr =
          Enum.reduce((n - 2)..0//-1, smooth_arr, fn idx, acc ->
            {x_filt, p_filt} = :array.get(idx, filt_arr)
            {x_pred_next, p_pred_next} = :array.get(idx + 1, pred_arr)
            {x_smooth_next, p_smooth_next} = :array.get(idx + 1, acc)
            c = smoother_gain(p_filt, f_t, p_pred_next)

            # state estimate
            x_smooth = add(x_filt, dot(c, subtract(x_smooth_next, x_pred_next)))
            # covariance estimate
            p_smooth =
              add(p_filt, dot(c, dot(subtract(p_smooth_next, p_pred_next), transpose(c))))

            :array.set(idx, {x_smooth, p_smooth}, acc)
          end)

        # Convert array to list
        Enum.map(0..(n - 1), fn i -> :array.get(i, smooth_arr) end)
    end
  end

  @doc """
  Runs the Carter–Kohn simulation smoother to draw a sample of the
  latent state trajectory from its posterior distribution.

  The function takes the smoothed, filtered and predicted state
  sequences (as returned by the Kalman filter and the RTS smoother)
  along with the state transition matrix `f` and returns a list of
  sampled states.  An optional `noise_list` may be provided to
  override the internally generated random draws.  When supplied,
  `noise_list` must be a list of standard normal tensors (one per
  time step) matching the state dimension.  This allows
  reproducible comparisons against external implementations such as
  Python.  If `noise_list` is not provided, fresh random values are
  drawn via `Nx.Random` for each step.
  """
  @spec simulate(
          [{Nx.t(), Nx.t()}],
          [{Nx.t(), Nx.t()}],
          [{Nx.t(), Nx.t()}],
          number | list | Nx.t(),
          [Nx.t()] | nil
        ) :: [Nx.t()]
  def simulate(smoothed, filtered, predicted, f, noise_list \\ nil) do
    # Delegate to `simulate_with_key/5` for improved reproducibility and performance.
    {states, _key_out} =
      simulate_with_key(smoothed, filtered, predicted, f, noise_list: noise_list)

    states
  end

  @doc """
  Runs the Carter–Kohn simulation smoother with explicit random number control.

  This function generalises `simulate/5` by accepting a keyword list of options.

  Supported options are:

    * `:noise_list` – a list of pre‑sampled standard normal tensors, one per
      time step. When provided, these values are used in place of internally
      generated random draws. The list length must equal the number of time
      steps.

    * `:key` – a PRNG key (`Nx` tensor of shape `{2}`) used to generate random
      draws. When supplied, the key is split into a sequence of subkeys via
      `Nx.Random.split/2`. The final key is returned as part of the result to
      facilitate reproducible sampling across calls.

  The return value is a tuple `{states, new_key}` where `states` is the
  sampled trajectory (list of tensors) and `new_key` is the final PRNG key after
  consumption of random draws. When neither `:noise_list` nor `:key` is provided,
  a fresh key is generated using a unique integer and split into per-step
  subkeys. The final key is returned to support reproducibility across calls.

  ## Examples

      iex> key = Nx.Random.key(123)
      iex> {states, key2} = BstsNx.Smoother.simulate_with_key(smoothed, filtered, predicted, 1.0, key: key)
      iex> is_list(states)
      true
  """
  @spec simulate_with_key(
          [{Nx.t(), Nx.t()}],
          [{Nx.t(), Nx.t()}],
          [{Nx.t(), Nx.t()}],
          number | list | Nx.t(),
          keyword()
        ) :: {[Nx.t()], Nx.t()}
  def simulate_with_key(smoothed, filtered, predicted, f, opts \\ [])

  def simulate_with_key([], _filtered, _predicted, _f, _opts) do
    {[], Nx.Random.key(:erlang.unique_integer([:positive]))}
  end

  def simulate_with_key(smoothed, filtered, predicted, f, opts) do
    f_t = to_tensor(f)
    t = length(smoothed)
    # Convert inputs to :array for O(1) access
    filt_arr = :array.from_list(filtered)
    pred_arr = :array.from_list(predicted)
    smooth_arr = :array.from_list(smoothed)
    # Determine state shape from last smoothed mean
    {x_T_mean, _p_T_cov} = List.last(smoothed)
    state_shape = Nx.shape(x_T_mean)
    # Extract options
    noise_list_opt = Keyword.get(opts, :noise_list, nil)
    key_opt = Keyword.get(opts, :key, nil)
    # Generate noise draws and new PRNG key
    {noises, key_out} =
      cond do
        is_list(noise_list_opt) ->
          # Return provided key or generate a fresh one so callers always get a valid key
          out_key = key_opt || Nx.Random.key(:erlang.unique_integer([:positive]))
          {noise_list_opt, out_key}

        key_opt != nil ->
          # Split key into t+1 subkeys: t for noise draws and 1 for the returned key
          keys_tensor = Nx.Random.split(key_opt, parts: t + 1)
          # first t keys for noise, last key for next_key
          noise_samples =
            Enum.map(0..(t - 1), fn idx ->
              subkey = keys_tensor[idx]
              {sample, _unused} = Nx.Random.normal(subkey, 0.0, 1.0, shape: state_shape)
              sample
            end)

          next_key = keys_tensor[t]
          {noise_samples, next_key}

        true ->
          # Generate random draws using a single base key split across steps
          seed = :erlang.unique_integer([:positive])
          base_key = Nx.Random.key(seed)
          keys_tensor = Nx.Random.split(base_key, parts: t + 1)

          noise_samples =
            Enum.map(0..(t - 1), fn idx ->
              subkey = keys_tensor[idx]
              {sample, _unused} = Nx.Random.normal(subkey, 0.0, 1.0, shape: state_shape)
              sample
            end)

          next_key = keys_tensor[t]
          {noise_samples, next_key}
      end

    # Convert noises to :array for O(1) access in backward pass
    noises_arr = :array.from_list(noises)
    # Allocate array for sampled states
    states_arr = :array.new(t, default: nil)
    # Sample final state (index t-1)
    {_x_T, p_T_cov} = :array.get(t - 1, smooth_arr)
    eps_T = :array.get(t - 1, noises_arr)

    x_sample_T =
      if Nx.rank(p_T_cov) == 0 do
        p_T_val = Nx.to_number(p_T_cov)

        safe_var =
          if p_T_val < 0.0 do
            Logger.warning(
              "Simulation smoother: negative terminal variance #{p_T_val}, clamping to 1.0e-12"
            )

            Nx.tensor(1.0e-12)
          else
            Nx.max(p_T_cov, 1.0e-12)
          end

        chol_T = Nx.sqrt(safe_var)
        Nx.add(x_T_mean, Nx.multiply(chol_T, eps_T))
      else
        chol_T = BstsNx.Utils.safe_cholesky(p_T_cov)

        add(x_T_mean, dot(chol_T, eps_T))
      end

    states_arr = :array.set(t - 1, x_sample_T, states_arr)
    # Iterate backwards for k = t-2..0 when t > 1
    states_arr =
      if t > 1 do
        Enum.reduce((t - 2)..0//-1, states_arr, fn idx, st_acc ->
          {x_filt, p_filt} = :array.get(idx, filt_arr)
          {x_pred_next, p_pred_next} = :array.get(idx + 1, pred_arr)
          x_next = :array.get(idx + 1, st_acc)
          j = smoother_gain(p_filt, f_t, p_pred_next)

          # Conditional mean and covariance
          # Use the stable form: Cov = P_filt - C * F * P_filt
          # This is algebraically equivalent to P_filt - C * P_pred * C',
          # but avoids the double-inverse instability when smoother_gain
          # applies jitter to invert P_pred (the jittered P_pred used in C
          # would not cancel cleanly with the un-jittered P_pred here).
          mean = add(x_filt, dot(j, subtract(x_next, x_pred_next)))

          cov =
            if Nx.rank(p_filt) == 0 do
              Nx.multiply(p_filt, Nx.subtract(1.0, Nx.multiply(j, f_t)))
            else
              n = Nx.axis_size(p_filt, 0)
              identity = Nx.eye(n)
              i_jf = subtract(identity, dot(j, f_t))
              dot(i_jf, p_filt)
            end

          eps_k = :array.get(idx, noises_arr)

          x_k =
            if Nx.rank(cov) == 0 do
              cov_val = Nx.to_number(cov)

              safe_cov =
                if cov_val < 0.0 do
                  Logger.warning(
                    "Simulation smoother: negative conditional variance #{cov_val} at step #{idx}, clamping to 1.0e-12"
                  )

                  Nx.tensor(1.0e-12)
                else
                  Nx.max(cov, 1.0e-12)
                end

              chol_k = Nx.sqrt(safe_cov)
              Nx.add(mean, Nx.multiply(chol_k, eps_k))
            else
              chol_k = BstsNx.Utils.safe_cholesky(cov)

              add(mean, dot(chol_k, eps_k))
            end

          :array.set(idx, x_k, st_acc)
        end)
      else
        states_arr
      end

    # Convert array to list in time order
    sampled_states = Enum.map(0..(t - 1), fn i -> :array.get(i, states_arr) end)
    {sampled_states, key_out}
  end

  # Computes the smoother gain C_k = P_filt * F^T * P_pred^{-1}.
  # For scalar systems uses division; for matrix systems uses solve+transpose.
  defp smoother_gain(p_filt, f_t, p_pred_next) do
    if Nx.rank(p_pred_next) == 0 do
      if abs(Nx.to_number(p_pred_next)) < 1.0e-15 do
        Nx.tensor(0.0)
      else
        p_filt |> Nx.multiply(f_t) |> Nx.divide(p_pred_next)
      end
    else
      rhs = dot(f_t, p_filt)
      safe_solve(p_pred_next, rhs) |> transpose()
    end
  end

  # Attempts `Nx.LinAlg.solve(a, b)` with progressive diagonal jitter
  # when the system matrix is near-singular. Returns a zero matrix matching
  # the expected solution shape when all retries fail.
  defp safe_solve(a, b) do
    result =
      try do
        Nx.LinAlg.solve(a, b)
      rescue
        _ -> :failed
      end

    case result do
      :failed ->
        solve_with_jitter(a, b)

      tensor ->
        if has_non_finite?(tensor) do
          solve_with_jitter(a, b)
        else
          tensor
        end
    end
  end

  defp has_non_finite?(tensor) do
    Nx.any(Nx.logical_or(Nx.is_nan(tensor), Nx.is_infinity(tensor)))
    |> Nx.to_number() == 1
  end

  defp solve_with_jitter(a, b) do
    dim = Nx.shape(a) |> elem(0)

    Enum.reduce_while([1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3], nil, fn jitter_scale, _acc ->
      jitter = Nx.eye(dim) |> Nx.multiply(jitter_scale)

      result =
        try do
          Nx.LinAlg.solve(Nx.add(a, jitter), b)
        rescue
          _ -> :failed
        end

      case result do
        :failed ->
          {:cont, nil}

        tensor ->
          if has_non_finite?(tensor) do
            {:cont, nil}
          else
            {:halt, tensor}
          end
      end
    end) ||
      (
        Logger.warning(
          "Smoother.safe_solve: solve failed even with max jitter 1e-3; " <>
            "falling back to zero gain (smoothing skipped for this step)"
        )

        Nx.broadcast(Nx.tensor(0.0), Nx.shape(b))
      )
  end
end
