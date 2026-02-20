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

  @doc """
  Draws a sample from the inverse–gamma distribution with shape parameter
  `alpha` and scale parameter `beta`.

  Both `alpha` and `beta` may be numbers, lists or `Nx.Tensor`s.  For
  tensors or lists, the shapes of `alpha` and `beta` must match exactly;
  no implicit broadcasting is performed.  An optional keyword list can be
  provided as a third argument to control the random seed.

  Supported options:

    * `:key` – an `Nx.Random` PRNG key used to seed Erlang's `:rand` generator
      for reproducible sampling.  When supplied, the key is converted into
      three 32‑bit integers and used to seed `:rand.exsss`.  Note that
      the key is **not** returned by this function; callers should manage
      key splitting externally if deterministic sequences are required.

    * `:max_value` – optional upper bound for samples.  When set to a number,
      each sample is clamped to `min(sample, max_value)`.  Defaults to
      `:infinity` (no clamping).

  ## Examples

      iex> x = BstsNx.Distributions.inv_gamma_sample(2.0, 3.0)
      iex> Nx.to_number(x) > 0
      true

      iex> key = Nx.Random.key(123)
      iex> _x = BstsNx.Distributions.inv_gamma_sample(2.0, 3.0, key: key)
  """
  @spec inv_gamma_sample(number | list | Nx.t(), number | list | Nx.t(), keyword()) :: Nx.t()
  def inv_gamma_sample(alpha, beta, opts \\ []) do
    a = to_tensor(alpha)
    b = to_tensor(beta)
    # Ensure shapes match
    unless Nx.shape(a) == Nx.shape(b) do
      raise ArgumentError, "alpha and beta must have the same shape; broadcasting not supported"
    end

    # Create explicit random state without mutating the process dictionary.
    # This avoids interference when inv_gamma_sample is called concurrently
    # within the same process.
    max_value = Keyword.get(opts, :max_value, :infinity)

    rand_state =
      case Keyword.get(opts, :key) do
        nil ->
          :rand.seed_s(:exsss, :erlang.unique_integer([:positive]))

        key ->
          ints = Nx.to_flat_list(key)

          case ints do
            [_i1, _i2] ->
              {a58, b58, c58} = derive_exsss_seed(ints)
              :rand.seed_s(:exsss, {a58, b58, c58})

            _ ->
              raise ArgumentError,
                    "Expected Nx.Random key with shape {2}, got: #{inspect(ints)}"
          end
      end

    # Flatten to list for iteration
    a_list = Nx.to_flat_list(a)
    b_list = Nx.to_flat_list(b)

    {samples, _final_rand_state} =
      Enum.zip(a_list, b_list)
      |> Enum.map_reduce(rand_state, fn {alpha_i, beta_i}, rs ->
        # Draw gamma(α, 1) and invert; clamp to avoid division by zero on underflow.
        {gamma, rs2} = gamma_sample(alpha_i, 1.0, rs)
        gamma_safe = max(gamma, 1.0e-300)
        sample = beta_i / gamma_safe
        capped = if max_value == :infinity, do: sample, else: min(sample, max_value)
        {capped, rs2}
      end)

    # Reshape back to original shape
    Nx.tensor(samples) |> Nx.reshape(Nx.shape(a))
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
    # split key for sample
    keys = Nx.Random.split(key, parts: 2)
    key1 = keys[0]
    key2 = keys[1]
    {sample, _unused} = Nx.Random.normal(key1, mean, stddev)
    {sample, key2}
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
    # Determine dimensionality from mean
    dim = Nx.axis_size(mean, 0)
    # Split key for noise vector and return key
    keys = Nx.Random.split(key, parts: 2)
    key_draw = keys[0]
    next_key = keys[1]
    # Draw standard normal vector of dimension dim
    {noise, _unused} = Nx.Random.normal(key_draw, 0.0, 1.0, shape: {dim})
    # Compute Cholesky; add jitter if needed.
    # Uses NaN detection to handle backends (e.g. EXLA) that return NaN
    # instead of raising on non-positive-definite input.
    chol = BstsNx.Utils.safe_cholesky(cov)

    sample = Nx.add(mean, Nx.dot(chol, noise))
    {sample, next_key}
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
