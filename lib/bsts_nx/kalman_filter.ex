defmodule BstsNx.KalmanFilter do
  @moduledoc """
  Implementation of a linear Kalman filter using the Nx numerical
  computing library.  The Kalman filter is a recursive estimator
  for linear Gaussian state-space models of the form:

      xₖ   = F xₖ₋₁ + wₖ,
      zₖ   = H xₖ   + vₖ,

  where *wₖ* ~ 𝒩(0, Q) is the process noise and *vₖ* ~ 𝒩(0, R)
  is the observation noise.  Here xₖ ∈ ℝⁿ is the hidden state and
  zₖ ∈ ℝᵐ is the observation.  Given a sequence of observations
  {z₁, …, z_T} the filter yields posterior state estimates and
  covariances {x̂₁, …, x̂_T}.

  The functions in this module accept scalars, lists or Nx tensors
  for the transition matrix `F`, observation matrix `H`, process
  covariance `Q` and observation covariance `R`.  Internally
  everything is converted to Nx tensors for computation.  See
  `filter/7` for the public API.
  """

  import Nx, only: [dot: 2, transpose: 1, add: 2, subtract: 2]
  import BstsNx.Utils, only: [to_tensor: 1, safe_solve: 2, missing_observation?: 1]
  # Require Nx.Defn for compiled implementations
  require Nx.Defn

  @doc """
  Runs a Kalman filter over a sequence of observations.

  * `observations` – list of observations `zₖ`.  Each element can be
    a number, list or tensor.
  * `f` – state transition matrix `F`.
  * `h` – observation matrix `H`.
  * `q` – process noise covariance `Q`.
  * `r` – observation noise covariance `R`.
  * `initial_state` – prior state mean `x₀`.
  * `initial_cov` – prior state covariance `P₀`.

  Returns a list of `{state_estimate, covariance}` pairs for each
  observation in the same order.  Both the state estimates and
  covariances are Nx tensors.  For one-dimensional systems these
  will be scalar tensors.

  ## Examples

      iex> obs = [1.0, 2.0, 3.0]
      iex> f = 1.0
      iex> h = 1.0
      iex> q = 1.0
      iex> r = 1.0
      iex> {x0, p0} = {0.0, 1.0}
      iex> estimates = BstsNx.KalmanFilter.filter(obs, f, h, q, r, x0, p0)
      iex> Enum.map(estimates, fn {x, p} ->
      ...>   {Float.round(Nx.to_number(x), 6), Float.round(Nx.to_number(p), 6)}
      ...> end)
      [{0.666667, 0.666667}, {1.5, 0.625}, {2.428571, 0.619048}]
  """
  @spec filter(
          [number | list | Nx.t()],
          number | list | Nx.t(),
          number | list | Nx.t(),
          number | list | Nx.t(),
          number | list | Nx.t(),
          number | list | Nx.t(),
          number | list | Nx.t()
        ) ::
          [{Nx.t(), Nx.t()}]
  def filter(observations, f, h, q, r, initial_state, initial_cov) do
    # Reuse the forward filter that also records predictions, but only
    # return the filtered estimates to keep backward compatibility.
    {filtered, _predicted} =
      filter_with_pred(observations, f, h, q, r, initial_state, initial_cov)

    filtered
  end

  @doc """
  Forward pass of the Kalman filter that returns both the filtered
  estimates and the one-step-ahead predictions.  This function
  performs the same computation as `filter/7` but also records
  the predicted state and covariance before incorporating each new
  observation.  The returned tuple is `{filtered, predicted}` where
  each element is a list of `{state, covariance}` pairs.

  The `filtered` list contains posterior estimates after each
  observation.  The `predicted` list contains the prior estimates
  immediately before each observation is processed.  Both lists
  are ordered in the same temporal order as the observations.
  """
  @spec filter_with_pred(
          [number | list | Nx.t()],
          number | list | Nx.t(),
          number | list | Nx.t(),
          number | list | Nx.t(),
          number | list | Nx.t(),
          number | list | Nx.t(),
          number | list | Nx.t()
        ) ::
          {[{Nx.t(), Nx.t()}], [{Nx.t(), Nx.t()}]}
  def filter_with_pred(observations, f, h, q, r, initial_state, initial_cov) do
    # Convert static parameters to tensors once.
    # Squeeze trivial {1,1} matrices to scalars so the innovation variance
    # computation stays scalar when H is a squeezed row vector.
    f_t = to_tensor(f)
    q_t = to_tensor(q)
    r_t = to_tensor(r) |> squeeze_1x1()
    obs_list = List.wrap(observations)

    if obs_list == [] do
      {[], []}
    else
      # Determine H per time step.  If `h` is scalar or a constant matrix,
      # replicate for each step.  If `h` is a tensor whose first axis matches
      # the number of observations, treat it as time-varying and slice per step.
      h_series = normalize_h_series(h, obs_list)

      # zip observations and h series
      zipped = Enum.zip(obs_list, h_series)

      {{_, _}, filtered, predicted} =
        Enum.reduce(zipped, {{to_tensor(initial_state), to_tensor(initial_cov)}, [], []}, fn {z,
                                                                                              h_i},
                                                                                             {{x_prev,
                                                                                               p_prev},
                                                                                              filt_acc,
                                                                                              pred_acc} ->
          # one-step ahead prediction
          x_pred = mul_or_dot(f_t, x_prev)
          p_pred = add(mul_or_dot(mul_or_dot(f_t, p_prev), transpose(f_t)), q_t)
          # record prediction
          pred_acc2 = [{x_pred, p_pred} | pred_acc]
          # update with observation; handle missing observations (nil)
          if missing_observation?(z) do
            # No observation (nil or NaN): skip update
            x_new = x_pred
            p_new = p_pred
            filt_acc2 = [{x_new, p_new} | filt_acc]
            {{x_new, p_new}, filt_acc2, pred_acc2}
          else
            # update with actual observation
            z_t = to_tensor(z)
            y = subtract(z_t, mul_or_dot(h_i, x_pred))
            s = add(mul_or_dot(mul_or_dot(h_i, p_pred), transpose(h_i)), r_t)
            # Compute Kalman gain K = P * H^T * S^{-1}.
            # For scalar S use dot+division; otherwise use solve for stability.
            # dot(p_pred, h_i) correctly handles both scalar (multiply) and
            # multi-dim state (matrix-vector product giving K as a vector).
            k =
              if Nx.rank(s) == 0 do
                if abs(Nx.to_number(s)) < 1.0e-15 do
                  Nx.broadcast(0.0, Nx.shape(x_pred))
                else
                  mul_or_dot(p_pred, h_i) |> Nx.divide(s)
                end
              else
                # Solve S * X = H * P_pred for X; the result has shape (m x n)'
                rhs =
                  if Nx.rank(p_pred) == 0 do
                    Nx.multiply(h_i, p_pred)
                  else
                    mul_or_dot(h_i, p_pred)
                  end

                k_t = safe_solve(s, rhs)
                transpose(k_t)
              end

            # State update: use dot for matrix K (multi-dim obs), multiply for vector/scalar K
            k_times_y = if Nx.rank(k) >= 2, do: mul_or_dot(k, y), else: Nx.multiply(k, y)
            x_new = add(x_pred, k_times_y)
            # Joseph form covariance update: (I - K H) P (I - K H)' + K R K'
            p_new =
              if Nx.rank(p_pred) == 0 do
                i_kh = Nx.subtract(1.0, Nx.multiply(k, h_i))
                i_term = Nx.multiply(Nx.multiply(i_kh, p_pred), i_kh)
                k_term = k |> Nx.multiply(r_t) |> Nx.multiply(k)
                Nx.add(i_term, k_term)
              else
                n = Nx.axis_size(p_pred, 0)
                identity = Nx.eye(n)
                # When K is a vector (multi-dim state, scalar obs), K*H is an
                # outer product; when K is a matrix, use standard dot.
                kh =
                  if Nx.rank(k) < 2 do
                    Nx.outer(Nx.flatten(k), Nx.flatten(h_i))
                  else
                    mul_or_dot(k, h_i)
                  end

                i_kh = subtract(identity, kh)

                kr_kt =
                  if Nx.rank(k) < 2 do
                    Nx.outer(Nx.flatten(k), Nx.flatten(k)) |> Nx.multiply(r_t)
                  else
                    mul_or_dot(mul_or_dot(k, r_t), transpose(k))
                  end

                add(mul_or_dot(mul_or_dot(i_kh, p_pred), transpose(i_kh)), kr_kt)
              end

            # accumulate filtered estimates
            filt_acc2 = [{x_new, p_new} | filt_acc]
            {{x_new, p_new}, filt_acc2, pred_acc2}
          end
        end)

      {Enum.reverse(filtered), Enum.reverse(predicted)}
    end
  end

  @doc """
  Compiled Kalman filter implementation for one‑dimensional systems.

  This function uses `Nx.Defn` to JIT compile the Kalman filter for
  improved performance.  It supports both constant and time‑varying
  observation matrices in the scalar case, as well as missing data.

  * `observations` – an `Nx` tensor of shape `{t}` containing the
    observations.  Missing observations should be encoded as
    `Nx.Constants.nan/0` in the tensor.  When an observation is
    `NaN` the filter performs only the prediction step and skips the
    update.
  * `f` – the scalar state transition parameter.  If an `Nx` tensor is
    supplied, it must be 0‑dimensional.
  * `h` – the observation parameter, either a scalar or a length‑`t`
    vector tensor.  When a vector is provided, its `i`‑th element is
    used as the observation matrix at time step `i` (time‑varying
    regressor).  For a scalar, it is broadcast to all time steps.
  * `q` – process noise variance (scalar).
  * `r` – observation noise variance (scalar).
  * `x0` – initial state mean.
  * `p0` – initial state variance.

  Returns a tuple `{states, variances}` where both elements are
  tensors of shape `{t}` containing the posterior state means and
  variances, respectively.

  Example:

  ```elixir
  nan = Nx.Constants.nan()
  obs = Nx.concatenate([Nx.tensor([1.0]), Nx.broadcast(nan, {1}), Nx.tensor([3.0])])
  h_vec = Nx.tensor([1.0, 2.0, 1.0])
  {xs, _ps} = BstsNx.KalmanFilter.filter_defn(obs, 1.0, h_vec, 0.1, 0.5, 0.0, 1.0)
  # The second observation is missing (NaN); state prediction is used without update.
  Nx.shape(xs)
  #=> {3}
  ```
  """
  @spec filter_defn(
          Nx.t(),
          number | Nx.t(),
          number | Nx.t(),
          number | Nx.t(),
          number | Nx.t(),
          number,
          number
        ) :: {Nx.t(), Nx.t()}
  def filter_defn(observations, f, h, q, r, x0, p0) do
    # wrapper to ensure h is a vector of same length as observations.  Allows constant or vector h.
    t = Nx.axis_size(observations, 0)
    f_t = to_tensor(f)
    # convert h to vector: if scalar, broadcast; otherwise accept a length-t vector
    # or a {t, 1} column vector for convenience.
    h_vec =
      cond do
        is_number(h) ->
          Nx.broadcast(to_tensor(h), {t})

        match?(%Nx.Tensor{}, h) and Nx.rank(h) == 0 ->
          Nx.broadcast(h, {t})

        match?(%Nx.Tensor{}, h) and Nx.rank(h) == 1 and Nx.axis_size(h, 0) == t ->
          h

        match?(%Nx.Tensor{}, h) and Nx.rank(h) == 2 and Nx.shape(h) == {t, 1} ->
          Nx.squeeze(h, axes: [1])

        match?(%Nx.Tensor{}, h) ->
          raise ArgumentError,
                "h must be scalar, rank-1 length #{t}, or rank-2 shape {#{t}, 1} for filter_defn/7"

        true ->
          raise ArgumentError, "h must be a scalar or a tensor of length equal to observations"
      end

    # convert q, r, x0, p0 to scalar tensors outside defn
    q_t = to_tensor(q)
    r_t = to_tensor(r)
    x0_t = to_tensor(x0)
    p0_t = to_tensor(p0)
    # call compiled function with vector h
    filter_defn_vec(observations, f_t, h_vec, q_t, r_t, x0_t, p0_t)
  end

  # compiled defn that accepts vector h for each time step and supports missing data via NaN sentinel.
  Nx.Defn.defn filter_defn_vec(observations, f_t, h_vec, q_t, r_t, x0, p0) do
    t = Nx.axis_size(observations, 0)
    # initialise accumulators
    xs = Nx.broadcast(0.0, {t})
    ps = Nx.broadcast(0.0, {t})
    # loop over observations
    #
    # Note: in newer Nx versions, values referenced inside `while` must be part
    # of the while state to avoid defn context mismatches.
    {_, _, _, xs_out, ps_out, _, _, _, _, _} =
      while {i = 0, x = x0, p = p0, xs_acc = xs, ps_acc = ps, obs = observations, h = h_vec,
             f = f_t, q = q_t, r = r_t},
            i < t do
        # get current observation and regressor at time i
        z = Nx.squeeze(Nx.slice(obs, [i], [1]))
        h_i = Nx.squeeze(Nx.slice(h, [i], [1]))
        # prediction step
        x_pred = f * x
        p_pred = f * p * f + q
        # compute update quantities
        y = z - h_i * x_pred
        s = h_i * p_pred * h_i + r
        # Guard against s==0 (zero innovation variance) to prevent division by zero
        near_zero_s = Nx.abs(s) < 1.0e-15
        safe_s = Nx.select(near_zero_s, 1.0, s)
        k = Nx.select(near_zero_s, 0.0, p_pred * h_i / safe_s)
        x_updated = x_pred + k * y
        # Joseph form covariance update
        i_kh = 1.0 - k * h_i
        p_updated = i_kh * p_pred * i_kh + k * r * k
        # detect missing observation: z is NaN => missing
        update_flag = Nx.equal(z, z)
        # choose between prediction and update based on presence of observation
        x_new = Nx.select(update_flag, x_updated, x_pred)
        p_new = Nx.select(update_flag, p_updated, p_pred)
        # store results
        xs_updated = Nx.put_slice(xs_acc, [i], Nx.reshape(x_new, {1}))
        ps_updated = Nx.put_slice(ps_acc, [i], Nx.reshape(p_new, {1}))
        {i + 1, x_new, p_new, xs_updated, ps_updated, obs, h, f, q, r}
      end

    {xs_out, ps_out}
  end

  # Normalizes the observation matrix `h` into a list of per-time-step tensors.
  #
  # Supports:
  #   * scalar h (broadcast)
  #   * list of per-step h entries
  #   * tensor h:
  #       - rank 1: time-varying scalar/row coefficients of length T
  #       - rank 2:
  #           - static matrix (default)
  #           - time-varying rows when observations are scalar and shape {T, n}
  #       - rank >= 3: time-varying matrices with first axis T
  #
  # Row matrices ({1, n}) are squeezed to vectors ({n}) for scalar-observation
  # paths, preserving compatibility with scalar update code.
  defp normalize_h_series(h, obs_list) do
    t_len = length(obs_list)
    obs_dim = observation_dim(obs_list)

    h_series =
      cond do
        is_number(h) or (match?(%Nx.Tensor{}, h) and Nx.rank(h) == 0) ->
          Enum.map(0..(t_len - 1), fn _ -> to_tensor(h) end)

        is_list(h) ->
          h_list = Enum.map(h, &to_tensor/1)

          if length(h_list) != t_len do
            raise ArgumentError,
                  "observation matrix length must match observations length"
          end

          h_list

        match?(%Nx.Tensor{}, h) ->
          rank = Nx.rank(h)
          len = Nx.axis_size(h, 0)

          cond do
            rank == 1 ->
              if obs_dim != 1 do
                raise ArgumentError,
                      "rank-1 observation matrix h implies scalar observations, got observation dimension #{obs_dim}"
              end

              if len != t_len do
                raise ArgumentError,
                      "observation matrix length must match observations length"
              end

              # Time-varying scalar coefficients.
              Enum.map(0..(len - 1), fn idx ->
                Nx.slice(h, [idx], [1]) |> Nx.squeeze()
              end)

            rank == 2 and obs_dim == 1 and len == t_len ->
              # Scalar-observation time-varying H encoded as {T, n}
              Enum.map(0..(len - 1), fn idx ->
                start_indices = [idx | List.duplicate(0, rank - 1)]
                lengths = [1 | Enum.map(1..(rank - 1)//1, &Nx.axis_size(h, &1))]
                Nx.slice(h, start_indices, lengths) |> Nx.squeeze()
              end)

            rank == 2 ->
              # Rank-2 tensors are static observation matrices by default.
              # This avoids misclassifying static multivariate H matrices when
              # rows(H) == T by coincidence.
              if obs_dim != 1 and len != obs_dim do
                raise ArgumentError,
                      "static observation matrix row count #{len} must match observation dimension #{obs_dim}"
              end

              h_t = to_tensor(h)
              Enum.map(0..(t_len - 1), fn _ -> h_t end)

            len == t_len ->
              # Rank-3+ time-varying matrix H_t over first axis
              Enum.map(0..(len - 1), fn idx ->
                start_indices = [idx | List.duplicate(0, rank - 1)]
                lengths = [1 | Enum.map(1..(rank - 1)//1, &Nx.axis_size(h, &1))]
                Nx.slice(h, start_indices, lengths) |> Nx.squeeze()
              end)

            true ->
              raise ArgumentError,
                    "observation matrix H with rank #{rank} and first axis #{len} " <>
                      "is ambiguous for #{t_len} observations; pass a rank-2 matrix " <>
                      "for a static H, or ensure the first axis equals the number of observations"
          end

        true ->
          raise ArgumentError, "invalid h: must be number, list or Nx tensor"
      end

    Enum.map(h_series, fn h_i ->
      if Nx.rank(h_i) == 2 and Nx.axis_size(h_i, 0) == 1 do
        Nx.squeeze(h_i, axes: [0])
      else
        h_i
      end
    end)
  end

  # Squeezes trivial {1,1} matrices or {1} vectors to scalars
  defp squeeze_1x1(%Nx.Tensor{} = t) do
    case Nx.shape(t) do
      {1, 1} -> Nx.squeeze(t)
      {1} -> Nx.squeeze(t)
      _ -> t
    end
  end

  defp squeeze_1x1(t), do: t

  # Infers observation dimension from the first non-missing observation.
  # Scalars -> 1, vectors -> length, matrices -> row count.
  defp observation_dim([]), do: 1

  defp observation_dim(obs_list) do
    obs_list
    |> Enum.find(&(not missing_observation?(&1)))
    |> case do
      nil ->
        1

      %Nx.Tensor{} = t ->
        case Nx.rank(t) do
          0 -> 1
          1 -> Nx.axis_size(t, 0)
          _ -> Nx.axis_size(t, 0)
        end

      v when is_number(v) ->
        1

      list when is_list(list) ->
        length(list)
    end
  end

  # Nx 0.6 emits warnings for scalar dot paths in some Elixir/OTP combos.
  # Also avoid rank-2/rank-1 dot calls (matrix-vector and vector-matrix),
  # which can trigger noisy range warnings in older Nx versions.
  defp mul_or_dot(a, b) do
    case {Nx.rank(a), Nx.rank(b)} do
      {0, 0} ->
        Nx.multiply(a, b)

      {2, 1} ->
        b_row = Nx.reshape(b, {1, Nx.axis_size(b, 0)})
        Nx.multiply(a, b_row) |> Nx.sum(axes: [1])

      {2, 2} ->
        {m, n} = Nx.shape(a)
        {n_b, p} = Nx.shape(b)

        if n != n_b do
          raise ArgumentError,
                "incompatible matrix shapes for multiplication: #{inspect(Nx.shape(a))} and #{inspect(Nx.shape(b))}"
        end

        a_expanded = Nx.reshape(a, {m, n, 1})
        b_expanded = Nx.reshape(b, {1, n, p})
        Nx.multiply(a_expanded, b_expanded) |> Nx.sum(axes: [1])

      {1, 2} ->
        a_col = Nx.reshape(a, {Nx.axis_size(a, 0), 1})
        Nx.multiply(a_col, b) |> Nx.sum(axes: [0])

      {1, 1} ->
        Nx.multiply(a, b) |> Nx.sum()

      _ ->
        dot(a, b)
    end
  end
end
