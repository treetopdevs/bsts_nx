defmodule BstsNx.Utils do
  @moduledoc false
  import Bitwise
  require Logger

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

  @doc """
  Returns `true` if the tensor contains any NaN or Inf values.
  """
  @spec has_non_finite?(Nx.t()) :: boolean()
  def has_non_finite?(tensor) do
    Nx.any(Nx.logical_or(Nx.is_nan(tensor), Nx.is_infinity(tensor)))
    |> Nx.to_number() == 1
  end

  defp solve_with_jitter(a, b) do
    dim = Nx.axis_size(a, 0)
    a_sym = Nx.multiply(Nx.add(a, Nx.transpose(a)), 0.5)

    Enum.reduce_while(
      [1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3],
      nil,
      fn jitter_scale, _acc ->
        jitter = Nx.eye(dim) |> Nx.multiply(jitter_scale)

        result =
          try do
            Nx.LinAlg.solve(Nx.add(a_sym, jitter), b)
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

  # Rank-safe dot for Nx 0.6 compatibility
  defp compat_dot(a, b) do
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
