defmodule BstsNx.Utils do
  @moduledoc false
  import Bitwise

  @doc """
  Converts numbers or lists into Nx tensors; leaves tensors unchanged.

  Scalars are wrapped in a 1-element tensor and squeezed to a
  0-dimensional tensor.  Lists are converted directly via `Nx.tensor/1`.
  """
  @spec to_tensor(number | list | Nx.t()) :: Nx.t()
  def to_tensor(%Nx.Tensor{} = t), do: t
  def to_tensor(v) when is_number(v), do: Nx.tensor([v]) |> Nx.squeeze()
  def to_tensor(list) when is_list(list), do: Nx.tensor(list)

  @doc """
  Cholesky factorisation with exponentially increasing jitter.

  Handles both raising backends (BinaryBackend) and NaN-returning
  backends (EXLA).  Tries the raw Cholesky first; if it raises or
  produces NaN, retries with exponentially increasing diagonal
  jitter (1e-6, 1e-5, 1e-4, 1e-3).
  """
  @spec safe_cholesky(Nx.t()) :: Nx.t()
  def safe_cholesky(mat) do
    chol =
      try do
        Nx.LinAlg.cholesky(mat)
      rescue
        _ -> :failed
      end

    case chol do
      :failed ->
        cholesky_with_jitter(mat)

      tensor ->
        if Nx.any(Nx.is_nan(tensor)) |> Nx.to_number() == 1 do
          cholesky_with_jitter(mat)
        else
          tensor
        end
    end
  end

  defp cholesky_with_jitter(mat) do
    dim = Nx.shape(mat) |> elem(0)

    Enum.reduce_while([1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3], nil, fn jitter_scale, _acc ->
      jitter = Nx.eye(dim) |> Nx.multiply(jitter_scale)

      chol =
        try do
          Nx.LinAlg.cholesky(Nx.add(mat, jitter))
        rescue
          _ -> :failed
        end

      case chol do
        :failed ->
          {:cont, nil}

        tensor ->
          if Nx.any(Nx.is_nan(tensor)) |> Nx.to_number() == 1 do
            {:cont, nil}
          else
            {:halt, tensor}
          end
      end
    end) ||
      raise ArgumentError, "Cholesky factorisation failed even with jitter up to 1e-3"
  end

  @doc """
  Computes a two-tailed credible/confidence interval from a pre-sorted list
  using the nearest-rank percentile method.

  `sorted` is a pre-sorted (ascending) list of samples, `n` is its length,
  and `alpha` is the significance level (e.g. 0.05 for a 95% interval).

  Returns `{lower, upper}`. When `n < 2`, returns `{val, val}` where `val`
  is the single element.
  """
  @spec percentile_interval([number()], non_neg_integer(), float()) :: {number(), number()}
  def percentile_interval([], 0, _alpha), do: {0.0, 0.0}

  def percentile_interval(sorted, n, _alpha) when n < 2 do
    val = hd(sorted)
    {val, val}
  end

  def percentile_interval(sorted, n, alpha) do
    last = n - 1
    lower_idx = trunc(Float.floor(alpha / 2.0 * last))
    upper_idx = trunc(Float.ceil((1.0 - alpha / 2.0) * last))
    {Enum.at(sorted, max(lower_idx, 0)), Enum.at(sorted, min(upper_idx, last))}
  end

  # Precomputed two-tailed z-scores for common significance levels.
  @z_scores %{
    0.001 => 3.290527,
    0.005 => 2.807034,
    0.01 => 2.575829,
    0.02 => 2.326348,
    0.05 => 1.959964,
    0.10 => 1.644854,
    0.20 => 1.281552
  }

  @doc """
  Two-tailed z-score for significance level alpha.

  Returns `z` such that `P(-z < Z < z) = 1 - alpha` for a standard normal `Z`.
  Uses a lookup table for common values and falls back to `erfinv` + Halley
  refinement (~1e-12 precision) for non-standard alpha.

  ## Examples

      iex> BstsNx.Utils.z_score(0.05)
      1.959964
  """
  @spec z_score(float()) :: float()
  def z_score(alpha) do
    case Map.fetch(@z_scores, alpha) do
      {:ok, z} -> z
      :error -> :math.sqrt(2.0) * erfinv(1.0 - alpha)
    end
  end

  # Inverse error function: initial A&S approximation + one Halley refinement step.
  defp erfinv(+0.0), do: 0.0
  defp erfinv(x) when x < 0.0, do: -erfinv(-x)

  defp erfinv(x) when x >= 1.0 do
    raise ArgumentError, "erfinv(x) is undefined for x >= 1.0, got: #{x}"
  end

  defp erfinv(x) do
    a = 0.147
    ln_part = :math.log(1.0 - x * x)
    b = 2.0 / (:math.pi() * a) + ln_part / 2.0
    y0 = :math.sqrt(:math.sqrt(b * b - ln_part / a) - b)

    # Halley refinement: one step on erf(y) - x = 0
    ey = :math.erf(y0) - x
    two_over_sqrt_pi = 2.0 / :math.sqrt(:math.pi())
    exp_neg_y2 = :math.exp(-y0 * y0)
    fp = two_over_sqrt_pi * exp_neg_y2
    fpp = -2.0 * y0 * fp
    denom = 2.0 * fp * fp - ey * fpp

    if abs(denom) < 1.0e-30, do: y0, else: y0 - 2.0 * ey * fp / denom
  end

  @doc """
  Derives three independent 58-bit seed values from a two-element Nx PRNG key
  for use with Erlang's `:exsss` PRNG algorithm.

  All three components are mixed via `phash2` to ensure independence even when
  the raw key integers have structured bit patterns (e.g., Threefry keys).
  """
  @spec derive_exsss_seed([integer()]) :: {integer(), integer(), integer()}
  def derive_exsss_seed([i1, i2]) do
    # Mix all components through phash2 to break any structure in raw key bits
    a_upper = :erlang.phash2({i1, :bsts_nx_a_upper})
    a_lower = :erlang.phash2({i1, i2, :bsts_nx_a_lower})
    a58 = :erlang.band(Bitwise.bsl(a_upper, 31) ||| a_lower, 0x3FFFFFFFFFFFFFF)

    b_upper = :erlang.phash2({i2, :bsts_nx_b_upper})
    b_lower = :erlang.phash2({i2, i1, :bsts_nx_b_lower})
    b58 = :erlang.band(Bitwise.bsl(b_upper, 31) ||| b_lower, 0x3FFFFFFFFFFFFFF)

    c_upper = :erlang.phash2({i1, i2, :bsts_nx_c_upper})
    c_lower = :erlang.phash2({i2, i1, :bsts_nx_c_lower})
    c58 = :erlang.band(Bitwise.bsl(c_upper, 31) ||| c_lower, 0x3FFFFFFFFFFFFFF)

    {a58, b58, c58}
  end
end
