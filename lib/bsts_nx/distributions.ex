defmodule BstsNx.Distributions do
  @moduledoc """
  Probability distribution helpers for BSTS models.

  This module provides a sampler for the inverse–gamma distribution.
  The inverse–gamma distribution with shape parameter `alpha` and
  scale parameter `beta` has probability density function

      f(x; α, β) = β^α / Γ(α) * x^(−α−1) * exp(−β/x),   x > 0.

  To draw a sample from this distribution we exploit the relation
  that if `G ∼ Gamma(α, 1)` then `X = β / G` has the above
  inverse–gamma distribution.  Erlang's random gamma sampler
  (`:rand` module, shape + scale)
  samples from a gamma distribution with given shape and scale
  parameters, which we use internally.  The resulting value is
  wrapped in an `Nx` tensor to integrate with the rest of this
  library.  Currently only scalar and like‑shaped inputs are
  supported; broadcasting is not implemented.
  """

  import BstsNx.Utils, only: [to_tensor: 1, derive_exsss_seed: 1]
  require Nx.Defn
  @gamma_rejection_max_iters 10_000

  @doc """
  Draws a sample from the inverse–gamma distribution with shape parameter
  `alpha` and scale parameter `beta`.

  Both `alpha` and `beta` may be numbers, lists or `Nx.Tensor`s. For
  tensors or lists, the shapes of `alpha` and `beta` must match exactly;
  no implicit broadcasting is performed.

  Supported options:

    * `:key` – an `Nx.Random` PRNG key for deterministic sampling. This API
      preserves backward compatibility and returns only the sample. For
      functional PRNG flows that also return the next key, prefer
      `inv_gamma_sample_with_key/4`.
    * `:max_value` – optional upper cap for samples. Defaults to `:infinity`
      (no truncation, statistically correct inverse-gamma draw). Passing a
      finite positive number truncates the sampled value and changes the
      implied posterior target.

  ## Examples

      iex> x = BstsNx.Distributions.inv_gamma_sample(2.0, 3.0)
      iex> Nx.to_number(x) > 0
      true

      iex> key = Nx.Random.key(123)
      iex> _x = BstsNx.Distributions.inv_gamma_sample(2.0, 3.0, key: key)
  """
  @spec inv_gamma_sample(number | list | Nx.t(), number | list | Nx.t(), keyword()) :: Nx.t()
  def inv_gamma_sample(alpha, beta, opts \\ []) do
    max_value = opts |> Keyword.get(:max_value, :infinity) |> parse_max_value()

    cond do
      is_number(alpha) and is_number(beta) ->
        alpha_f = alpha * 1.0
        beta_f = beta * 1.0

        case Keyword.get(opts, :key) do
          nil ->
            rand_state = :rand.seed_s(:exsss, :erlang.unique_integer([:positive]))
            {sample, _next_state} = inv_gamma_draw(alpha_f, beta_f, max_value, rand_state)
            Nx.tensor(sample)

          key ->
            {sample, _next_key} = sample_with_key_scalar(alpha_f, beta_f, max_value, key)
            sample
        end

      true ->
        {a, b, _max} = prepare_inv_gamma_inputs(alpha, beta, opts)

        case Keyword.get(opts, :key) do
          nil ->
            rand_state = :rand.seed_s(:exsss, :erlang.unique_integer([:positive]))
            sample_with_rand_state(a, b, max_value, rand_state)

          key ->
            {sample, _next_key} = sample_with_key(a, b, max_value, key)
            sample
        end
    end
  end

  @doc """
  Draws an inverse-gamma sample and returns `{sample, next_key}`.

  This is the preferred API for deterministic, functional PRNG workflows.
  The next key is produced deterministically from the input key via
  `Nx.Random.split/2`.
  """
  @spec inv_gamma_sample_with_key(
          number | list | Nx.t(),
          number | list | Nx.t(),
          Nx.t(),
          keyword()
        ) ::
          {Nx.t(), Nx.t()}
  def inv_gamma_sample_with_key(alpha, beta, key, opts \\ []) do
    if Keyword.has_key?(opts, :key) do
      raise ArgumentError,
            "pass the PRNG key as the third argument to inv_gamma_sample_with_key/4"
    end

    max_value = opts |> Keyword.get(:max_value, :infinity) |> parse_max_value()

    if is_number(alpha) and is_number(beta) do
      sample_with_key_scalar(alpha * 1.0, beta * 1.0, max_value, key)
    else
      {a, b, _max} = prepare_inv_gamma_inputs(alpha, beta, opts)
      sample_with_key(a, b, max_value, key)
    end
  end

  @doc """
  Draws inverse-gamma sample(s) using a pure `Nx.Defn` Marsaglia-Tsang path.

  This API is additive and is intended for compiled/vectorized call sites.
  It returns `{sample, next_key}` to support functional PRNG chaining.
  """
  @spec inv_gamma_sample_defn(
          number | list | Nx.t(),
          number | list | Nx.t(),
          Nx.t(),
          keyword()
        ) :: {Nx.t(), Nx.t()}
  def inv_gamma_sample_defn(alpha, beta, key, opts \\ []) do
    if Keyword.has_key?(opts, :key) do
      raise ArgumentError, "pass the PRNG key as the third argument to inv_gamma_sample_defn/4"
    end

    validate_prng_key_shape!(key)
    max_value = opts |> Keyword.get(:max_value, :infinity) |> parse_max_value()
    a = to_tensor(alpha) |> Nx.as_type({:f, 64})
    b = to_tensor(beta) |> Nx.as_type({:f, 64})

    unless Nx.shape(a) == Nx.shape(b) do
      raise ArgumentError, "alpha and beta must have the same shape; broadcasting not supported"
    end

    if Nx.any(Nx.less_equal(a, 0.0)) |> Nx.to_number() == 1 do
      raise ArgumentError, "alpha must be strictly positive for inverse-gamma sampling"
    end

    if Nx.any(Nx.less_equal(b, 0.0)) |> Nx.to_number() == 1 do
      raise ArgumentError, "beta must be strictly positive for inverse-gamma sampling"
    end

    flat_size = Nx.size(a)
    a_flat = Nx.reshape(a, {flat_size})
    b_flat = Nx.reshape(b, {flat_size})

    {sample_flat, next_key} =
      case max_value do
        :infinity ->
          inv_gamma_sample_defn_impl(
            a_flat,
            b_flat,
            key,
            Nx.tensor(0.0, type: {:f, 64}),
            Nx.tensor(0, type: {:u, 8})
          )

        cap ->
          inv_gamma_sample_defn_impl(
            a_flat,
            b_flat,
            key,
            Nx.tensor(cap, type: {:f, 64}),
            Nx.tensor(1, type: {:u, 8})
          )
      end

    {Nx.reshape(sample_flat, Nx.shape(a)), next_key}
  end

  @doc false
  @spec inv_gamma_sample_with_rand_state(number(), number(), :rand.state()) ::
          {float(), :rand.state()}
  def inv_gamma_sample_with_rand_state(alpha, beta, rand_state)
      when is_number(alpha) and is_number(beta) do
    inv_gamma_draw(alpha * 1.0, beta * 1.0, :infinity, rand_state)
  end

  Nx.Defn.defn inv_gamma_sample_defn_impl(alpha, beta, key, cap_value, apply_cap) do
    n = Nx.axis_size(alpha, 0)
    eps = Nx.tensor(1.0e-12, type: {:f, 64})
    gamma_floor = Nx.tensor(1.0e-300, type: {:f, 64})
    alpha_lt_one = Nx.less(alpha, 1.0)
    alpha_base = Nx.select(alpha_lt_one, Nx.add(alpha, 1.0), alpha)
    d = Nx.subtract(alpha_base, 1.0 / 3.0)
    c = Nx.divide(1.0, Nx.sqrt(Nx.multiply(9.0, d)))

    accepted0 = Nx.broadcast(Nx.tensor(0, type: {:u, 8}), {n})
    gamma0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 64}), {n})
    max_iters = Nx.tensor(@gamma_rejection_max_iters, type: {:s, 64})

    {_, accepted_out, gamma_out, key_out, _, _, _} =
      while {iter = Nx.tensor(0, type: {:s, 64}), accepted = accepted0, gamma = gamma0,
             key_acc = key, d_in = d, c_in = c, eps_in = eps},
            Nx.logical_and(Nx.less(iter, max_iters), Nx.any(Nx.equal(accepted, 0))) do
        {x_raw, key_after_x} = Nx.Random.normal(key_acc, 0.0, 1.0, shape: {n})
        x = Nx.as_type(x_raw, {:f, 64})
        {u_raw, key_after_u} = Nx.Random.uniform(key_after_x, 0.0, 1.0, shape: {n})
        u = Nx.as_type(u_raw, {:f, 64}) |> Nx.max(eps_in)

        v = Nx.add(1.0, Nx.multiply(c_in, x))
        valid_v = Nx.greater(v, 0.0)
        v2 = Nx.multiply(v, v)
        v3 = Nx.multiply(v2, v)
        x2 = Nx.multiply(x, x)
        x4 = Nx.multiply(x2, x2)

        quick_accept = Nx.less(u, Nx.subtract(1.0, Nx.multiply(0.0331, x4)))

        log_accept =
          Nx.less(
            Nx.log(u),
            Nx.add(
              Nx.multiply(0.5, x2),
              Nx.multiply(d_in, Nx.add(Nx.subtract(1.0, v3), Nx.log(Nx.max(v3, eps_in))))
            )
          )

        accept = Nx.logical_and(valid_v, Nx.logical_or(quick_accept, log_accept))
        pending = Nx.equal(accepted, 0)
        take = Nx.logical_and(pending, accept)
        gamma_candidate = Nx.multiply(d_in, v3)
        gamma_new = Nx.select(take, gamma_candidate, gamma)

        accepted_new =
          Nx.select(take, Nx.broadcast(Nx.tensor(1, type: {:u, 8}), {n}), accepted)

        {iter + 1, accepted_new, gamma_new, key_after_u, d_in, c_in, eps_in}
      end

    fallback = Nx.max(d, eps)
    gamma_accepted = Nx.select(Nx.equal(accepted_out, 1), gamma_out, fallback)

    {u_small_raw, key_after_small} = Nx.Random.uniform(key_out, 0.0, 1.0, shape: {n})
    u_small = Nx.as_type(u_small_raw, {:f, 64}) |> Nx.max(eps)
    adjust = Nx.pow(u_small, Nx.divide(1.0, alpha))
    gamma_final = Nx.select(alpha_lt_one, Nx.multiply(gamma_accepted, adjust), gamma_accepted)
    safe_gamma = Nx.max(gamma_final, gamma_floor)
    sample = Nx.divide(beta, safe_gamma)
    sample_capped = Nx.min(sample, Nx.broadcast(cap_value, {n}))
    cap_enabled = Nx.equal(apply_cap, Nx.tensor(1, type: {:u, 8}))

    {Nx.select(cap_enabled, sample_capped, sample), key_after_small}
  end

  @doc """
  Draws a sample from a univariate normal distribution with the given mean
  and standard deviation, using an `Nx.Random` key for reproducibility.

  The function splits the provided key to obtain a fresh subkey for the
  sample and returns a tuple `{sample, new_key}`.  You may optionally
  specify `:mean` (default 0.0) and `:stddev` (default 1.0) via the
  options keyword list.
  """
  @spec normal_sample(Nx.t(), keyword()) :: {Nx.t(), Nx.t()}
  def normal_sample(key, opts \\ []) do
    mean = Keyword.get(opts, :mean, 0.0)
    stddev = Keyword.get(opts, :stddev, 1.0)
    [key_draw_row, key_next_row] = Nx.Random.split(key, parts: 2) |> Nx.to_list()
    key_draw = Nx.tensor(key_draw_row, type: Nx.type(key))
    key_next = Nx.tensor(key_next_row, type: Nx.type(key))
    {sample, _unused} = Nx.Random.normal(key_draw, mean, stddev)
    {sample, key_next}
  end

  @doc """
  Draws a sample from a multivariate normal distribution with mean `mean`
  and covariance matrix `cov` using a PRNG key.

  The function splits the provided key to obtain a subkey for the noise
  draw and then computes a Cholesky factorisation of `cov` to transform
  a standard normal vector.  If the Cholesky fails due to numerical
  issues, a small diagonal jitter is added.  Returns `{sample, new_key}`.
  """
  @spec mv_normal_sample(Nx.t(), Nx.t(), Nx.t()) :: {Nx.t(), Nx.t()}
  def mv_normal_sample(key, mean, cov) do
    # Compute Cholesky; add jitter if needed.
    # Uses NaN detection to handle backends (e.g. EXLA) that return NaN
    # instead of raising on non-positive-definite input.
    chol = BstsNx.Utils.safe_cholesky(cov)
    mv_normal_sample_with_chol(key, mean, cov, chol)
  end

  @doc """
  Draws a multivariate normal sample using a precomputed Cholesky factor.

  This avoids recomputing `chol(cov)` when sampling repeatedly from the same
  covariance matrix.
  """
  @spec mv_normal_sample_with_chol(Nx.t(), Nx.t(), Nx.t(), Nx.t()) :: {Nx.t(), Nx.t()}
  def mv_normal_sample_with_chol(key, mean, _cov, chol) do
    dim = Nx.axis_size(mean, 0)
    [key_draw_row, key_next_row] = Nx.Random.split(key, parts: 2) |> Nx.to_list()
    key_draw = Nx.tensor(key_draw_row, type: Nx.type(key))
    next_key = Nx.tensor(key_next_row, type: Nx.type(key))
    {noise, _unused} = Nx.Random.normal(key_draw, 0.0, 1.0, shape: {dim})

    sample = Nx.add(mean, Nx.dot(chol, noise))
    {sample, next_key}
  end

  defp prepare_inv_gamma_inputs(alpha, beta, opts) do
    a = to_tensor(alpha)
    b = to_tensor(beta)
    max_value = opts |> Keyword.get(:max_value, :infinity) |> parse_max_value()

    unless Nx.shape(a) == Nx.shape(b) do
      raise ArgumentError, "alpha and beta must have the same shape; broadcasting not supported"
    end

    {a, b, max_value}
  end

  defp sample_with_rand_state(a, b, max_value, rand_state) do
    a_list = Nx.to_flat_list(a)
    b_list = Nx.to_flat_list(b)

    {samples, _final_rand_state} =
      Enum.zip(a_list, b_list)
      |> Enum.map_reduce(rand_state, fn {alpha_i, beta_i}, rs ->
        {sample, rs2} = inv_gamma_draw(alpha_i, beta_i, max_value, rs)
        {sample, rs2}
      end)

    Nx.tensor(samples) |> Nx.reshape(Nx.shape(a))
  end

  defp sample_with_key(a, b, max_value, key) do
    validate_prng_key_shape!(key)
    a_list = Nx.to_flat_list(a)
    b_list = Nx.to_flat_list(b)
    num_draws = length(a_list)
    split_key_rows = Nx.Random.split(key, parts: num_draws + 1) |> Nx.to_list()
    sample_keys = Enum.take(split_key_rows, num_draws)

    samples =
      Enum.zip(Enum.zip(a_list, b_list), sample_keys)
      |> Enum.map(fn {{alpha_i, beta_i}, subkey} ->
        rand_state = rand_state_from_key(subkey)
        {sample, _rs2} = inv_gamma_draw(alpha_i, beta_i, max_value, rand_state)
        sample
      end)

    sample_tensor = Nx.tensor(samples) |> Nx.reshape(Nx.shape(a))
    next_key = split_key_rows |> List.last() |> Nx.tensor(type: Nx.type(key))
    {sample_tensor, next_key}
  end

  defp sample_with_key_scalar(alpha, beta, max_value, key) do
    validate_prng_key_shape!(key)
    [draw_key_row, next_key_row] = Nx.Random.split(key, parts: 2) |> Nx.to_list()
    draw_key = Nx.tensor(draw_key_row, type: Nx.type(key))
    next_key = Nx.tensor(next_key_row, type: Nx.type(key))
    rand_state = rand_state_from_key(draw_key)
    {sample, _rs2} = inv_gamma_draw(alpha, beta, max_value, rand_state)
    {Nx.tensor(sample), next_key}
  end

  defp inv_gamma_draw(alpha_i, beta_i, max_value, rand_state) do
    # Draw gamma(α, 1) and invert; clamp gamma to avoid division by zero on underflow.
    {gamma, rs2} = gamma_sample(alpha_i, 1.0, rand_state)
    gamma_safe = max(gamma, 1.0e-300)
    sample = maybe_cap(beta_i / gamma_safe, max_value)
    {sample, rs2}
  end

  defp rand_state_from_key(key) do
    ints =
      case key do
        %Nx.Tensor{} = t ->
          if Nx.shape(t) != {2} do
            raise ArgumentError,
                  "Expected Nx.Random key with shape {2}, got: #{inspect(Nx.shape(t))}"
          end

          Nx.to_flat_list(t)

        [i1, i2] when is_integer(i1) and is_integer(i2) ->
          [i1, i2]

        other ->
          raise ArgumentError, "Expected Nx.Random key with shape {2}, got: #{inspect(other)}"
      end

    case ints do
      [_i1, _i2] ->
        {a58, b58, c58} = derive_exsss_seed(ints)
        :rand.seed_s(:exsss, {a58, b58, c58})

      _ ->
        raise ArgumentError, "Expected Nx.Random key with shape {2}, got: #{inspect(ints)}"
    end
  end

  defp validate_prng_key_shape!(%Nx.Tensor{} = key) do
    if Nx.shape(key) != {2} do
      raise ArgumentError, "Expected Nx.Random key with shape {2}, got: #{inspect(Nx.shape(key))}"
    end

    :ok
  end

  defp validate_prng_key_shape!(other) do
    raise ArgumentError, "Expected Nx.Random key with shape {2}, got: #{inspect(other)}"
  end

  defp maybe_cap(sample, :infinity), do: sample
  defp maybe_cap(sample, max_value), do: min(sample, max_value)

  defp parse_max_value(:infinity), do: :infinity

  defp parse_max_value(v) when is_number(v) and v > 0 do
    v * 1.0
  end

  defp parse_max_value(v) do
    raise ArgumentError,
          "max_value must be :infinity or a positive number, got: #{inspect(v)}"
  end

  # Marsaglia and Tsang's method for Gamma(alpha, scale) sampling.
  # Uses an explicit `:rand` state to avoid mutating the process dictionary.
  defp gamma_sample(alpha, _scale, _rand_state) when is_number(alpha) and alpha <= 0 do
    raise ArgumentError, "gamma_sample requires alpha > 0, got: #{alpha}"
  end

  defp gamma_sample(alpha, scale, rand_state) when is_number(alpha) and alpha > 0 do
    if alpha < 1.0 do
      {u, rs2} = :rand.uniform_s(rand_state)
      {sample, rs3} = gamma_sample(alpha + 1.0, scale, rs2)
      {sample * :math.pow(u, 1.0 / alpha), rs3}
    else
      d = alpha - 1.0 / 3.0
      c = 1.0 / :math.sqrt(9.0 * d)
      gamma_sample_loop(d, c, scale, rand_state)
    end
  end

  defp gamma_sample_loop(d, c, scale, rand_state) do
    gamma_sample_loop(d, c, scale, rand_state, 10_000)
  end

  defp gamma_sample_loop(_d, _c, _scale, _rand_state, 0) do
    raise RuntimeError, "Gamma sampling failed to converge after 10,000 iterations"
  end

  defp gamma_sample_loop(d, c, scale, rand_state, remaining) do
    {x, rs2} = :rand.normal_s(rand_state)
    v = 1.0 + c * x

    if v <= 0.0 do
      gamma_sample_loop(d, c, scale, rs2, remaining - 1)
    else
      v3 = v * v * v
      {u, rs3} = :rand.uniform_s(rs2)

      accept =
        u < 1.0 - 0.0331 * x * x * x * x or
          :math.log(u) < 0.5 * x * x + d * (1.0 - v3 + :math.log(v3))

      if accept do
        {d * v3 * scale, rs3}
      else
        gamma_sample_loop(d, c, scale, rs3, remaining - 1)
      end
    end
  end
end
