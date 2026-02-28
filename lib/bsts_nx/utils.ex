defmodule BstsNx.Utils do
  @moduledoc false
  import Bitwise
  require Logger

  @doc """
  Converts numbers or lists into Nx tensors; leaves tensors unchanged.

  Scalars are converted directly to 0-dimensional tensors. Lists are
  converted directly via `Nx.tensor/1`.
  """
  @spec to_tensor(number | list | Nx.t()) :: Nx.t()
  def to_tensor(%Nx.Tensor{} = t), do: t
  def to_tensor(v) when is_number(v), do: Nx.tensor(v)
  def to_tensor(list) when is_list(list), do: Nx.tensor(list)

  @doc """
  Returns `true` if the observation should be treated as missing.

  Missing observations are `nil` values or tensors containing NaN.
  """
  @spec missing_observation?(any()) :: boolean()
  def missing_observation?(nil), do: true

  def missing_observation?(%Nx.Tensor{} = t) do
    Nx.any(Nx.is_nan(t)) |> Nx.to_number() == 1
  end

  def missing_observation?(_), do: false

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
    eye = Nx.eye(dim)

    Enum.reduce_while([1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3], nil, fn jitter_scale, _acc ->
      jitter = Nx.multiply(eye, jitter_scale)

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
  Returns `true` if the observation should be treated as missing.

  Missing observations are `nil`, the atom `:nan`, tensors containing NaN,
  or float NaN values.
  """
  @spec missing_observation?(any()) :: boolean()
  def missing_observation?(nil), do: true
  def missing_observation?(:nan), do: true

  def missing_observation?(%Nx.Tensor{} = t) do
    Nx.any(Nx.is_nan(t)) |> Nx.to_number() == 1
  end

  def missing_observation?(v) when is_float(v), do: v != v
  def missing_observation?(_), do: false

  @doc """
  Solves the linear system `a * x = b` with progressive diagonal jitter
  when `a` is near-singular. Falls back to pseudoinverse then zero matrix.

  Returns a tensor matching the shape of `b`.
  """
  @spec safe_solve(Nx.t(), Nx.t()) :: Nx.t()
  def safe_solve(a, b) do
    case scalar_or_1x1_solve(a, b) do
      {:ok, tensor} ->
        tensor

      :not_applicable ->
        result =
          try do
            linalg_solve_compat(a, b)
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
  end

  @doc """
  Returns `true` if the tensor contains any NaN or Inf values.
  """
  @spec has_non_finite?(Nx.t()) :: boolean()
  def has_non_finite?(tensor) do
    tensor
    |> Nx.is_nan()
    |> Nx.logical_or(Nx.is_infinity(tensor))
    |> Nx.any()
    |> Nx.to_number()
    |> Kernel.==(1)
  end

  defp solve_with_jitter(a, b) do
    dim = Nx.axis_size(a, 0)
    a_sym = Nx.multiply(Nx.add(a, Nx.transpose(a)), 0.5)

    if dim == 1 do
      scalar_divide_with_jitter(Nx.to_number(Nx.squeeze(a_sym)), b)
    else
      Enum.reduce_while(
        [1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3],
        nil,
        fn jitter_scale, _acc ->
          jitter = Nx.eye(dim) |> Nx.multiply(jitter_scale)

          result =
            try do
              linalg_solve_compat(Nx.add(a_sym, jitter), b)
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
        end
      ) ||
        pinv_fallback(a_sym, b) ||
        (
          Logger.warning(
            "Utils.safe_solve: solve failed even with max jitter 1e-3; " <>
              "falling back to zero matrix"
          )

          Nx.broadcast(Nx.tensor(0.0), Nx.shape(b))
        )
    end
  end

  # On Nx 0.11+, direct solve is stable and fastest for the rank patterns used
  # in this project.
  defp linalg_solve_compat(a, b) do
    Nx.LinAlg.solve(a, b)
  end

  defp scalar_or_1x1_solve(a, b) do
    cond do
      Nx.rank(a) == 0 ->
        {:ok, scalar_divide_with_jitter(Nx.to_number(a), b)}

      Nx.rank(a) == 2 and Nx.shape(a) == {1, 1} ->
        {:ok, scalar_divide_with_jitter(Nx.to_number(Nx.squeeze(a)), b)}

      true ->
        :not_applicable
    end
  end

  defp scalar_divide_with_jitter(denom, b) do
    if abs(denom) >= 1.0e-15 do
      direct = Nx.divide(b, denom)

      if has_non_finite?(direct) do
        scalar_divide_with_jitter_fallback(denom, b)
      else
        direct
      end
    else
      scalar_divide_with_jitter_fallback(denom, b)
    end
  end

  defp scalar_divide_with_jitter_fallback(denom, b) do
    Enum.reduce_while([1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3], nil, fn jitter, _acc ->
      adjusted = denom + jitter

      if abs(adjusted) < 1.0e-15 do
        {:cont, nil}
      else
        candidate = Nx.divide(b, adjusted)

        if has_non_finite?(candidate) do
          {:cont, nil}
        else
          {:halt, candidate}
        end
      end
    end) ||
      (
        Logger.warning(
          "Utils.safe_solve: scalar solve failed even with max jitter 1e-3; " <>
            "falling back to zero matrix"
        )

        Nx.broadcast(Nx.tensor(0.0), Nx.shape(b))
      )
  end

  defp pinv_fallback(a, b) do
    result =
      try do
        compat_dot(Nx.LinAlg.pinv(a), b)
      rescue
        _ -> :failed
      end

    case result do
      :failed ->
        nil

      tensor ->
        if has_non_finite?(tensor), do: nil, else: tensor
    end
  end

  # Nx.dot handles all rank combinations we use; keep scalar/scalar explicit.
  defp compat_dot(a, b) do
    case {Nx.rank(a), Nx.rank(b)} do
      {0, 0} ->
        Nx.multiply(a, b)

      _ ->
        Nx.dot(a, b)
    end
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
    arr = List.to_tuple(sorted)
    {elem(arr, max(lower_idx, 0)), elem(arr, min(upper_idx, last))}
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
