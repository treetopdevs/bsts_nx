defmodule BstsNx.StateSpace do
  @moduledoc """
  Utilities for composing state‑space models.

  Many BSTS models consist of multiple independent components,
  such as local level and seasonal cycles.  When combining these
  components into a single model it is useful to form block
  diagonal matrices and concatenated observation matrices.  This
  module provides helper functions to build such composite
  structures.

  The functions expect `Nx` tensors or plain numbers/lists and
  return `Nx` tensors.  The shapes of the input matrices should
  correspond to valid state‑space components.
  """

  @doc """
  Constructs a block diagonal matrix from a list of square matrices.

  Each element of `matrices` should be a square matrix (or a
  scalar, which is treated as a 1×1 matrix).  The result is a
  larger matrix with the inputs placed along the diagonal and
  zeros elsewhere.  For example:

      iex> f1 = Nx.tensor([[1.0, 0.1], [0.0, 1.0]])
      iex> f2 = Nx.tensor([[0.9]])
      iex> block = BstsNx.StateSpace.block_diag([f1, f2])
      iex> Nx.shape(block)
      {3, 3}

      iex> f1 = Nx.tensor([[1, 2], [3, 4]])
      iex> f2 = Nx.tensor([[5]])
      iex> block = BstsNx.StateSpace.block_diag([f1, f2])
      iex> Nx.to_flat_list(block)
      [1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 5.0]
  """
  @spec block_diag([number | list | Nx.t()]) :: Nx.t()
  def block_diag([]), do: raise(ArgumentError, "cannot build block diagonal of empty list")

  def block_diag([single]) do
    to_square_matrix(single)
  end

  def block_diag(matrices) when is_list(matrices) do
    if Enum.all?(matrices, &is_number/1) do
      matrices
      |> Nx.tensor(type: {:f, 64})
      |> Nx.make_diagonal()
    else
      block_diag_matrix_impl(matrices)
    end
  end

  defp block_diag_matrix_impl([a, b]) do
    a_t = to_square_matrix(a)
    b_t = to_square_matrix(b)
    {n1, _} = Nx.shape(a_t)
    {n2, _} = Nx.shape(b_t)

    top = Nx.concatenate([a_t, Nx.broadcast(0.0, {n1, n2})], axis: 1)
    bottom = Nx.concatenate([Nx.broadcast(0.0, {n2, n1}), b_t], axis: 1)
    Nx.concatenate([top, bottom], axis: 0)
  end

  defp block_diag_matrix_impl(matrices) do
    mats = Enum.map(matrices, &to_square_matrix/1)
    sizes = Enum.map(mats, fn m -> Nx.shape(m) |> elem(0) end)
    total = Enum.sum(sizes)
    # start with zero matrix
    zero = Nx.broadcast(0.0, {total, total})
    # place each block
    {result, _offset} =
      Enum.reduce(mats, {zero, 0}, fn m, {acc, offset} ->
        {rows, _cols} = Nx.shape(m)
        # insert m into acc at position offset
        acc = Nx.put_slice(acc, [offset, offset], m)
        {acc, offset + rows}
      end)

    result
  end

  @doc """
  Composes two state‑space components into a single component.

  Given two components `comp1` and `comp2`, each defined as a map
  with keys `:f`, `:q` and `:h` (transition matrix, process
  covariance and observation matrix respectively), this function
  returns a new component where the transition and process
  covariance matrices are placed on the block diagonal and the
  observation matrices are concatenated horizontally.  The
  observation dimensions of the two components must match.
  """
  @spec compose(
          %{f: number | list | Nx.t(), q: number | list | Nx.t(), h: number | list | Nx.t()},
          %{f: number | list | Nx.t(), q: number | list | Nx.t(), h: number | list | Nx.t()}
        ) ::
          %{f: Nx.t(), q: Nx.t(), h: Nx.t()}
  def compose(%{f: f1, q: q1, h: h1}, %{f: f2, q: q2, h: h2}) do
    f = block_diag([f1, f2])
    q = block_diag([q1, q2])
    # observation matrices must have same row dimension
    h1_t = to_matrix(h1)
    h2_t = to_matrix(h2)
    {r1, _c1} = Nx.shape(h1_t)
    {r2, _c2} = Nx.shape(h2_t)

    unless r1 == r2 do
      raise ArgumentError, "observation matrices must have same number of rows"
    end

    h = Nx.concatenate([h1_t, h2_t], axis: 1)
    %{f: f, q: q, h: h}
  end

  # Ensure the value is a square matrix tensor
  defp to_square_matrix(%Nx.Tensor{} = t) do
    case Nx.rank(t) do
      0 ->
        # Scalar tensor (shape {}) — treat as 1×1 matrix
        Nx.reshape(t, {1, 1})

      1 ->
        # Vector tensor (shape {n}) — treat as 1×n? No, only valid if n=1
        if Nx.axis_size(t, 0) == 1 do
          Nx.reshape(t, {1, 1})
        else
          raise ArgumentError, "rank-1 tensor with size > 1 is not a square matrix"
        end

      2 ->
        {rows, cols} = Nx.shape(t)

        if rows == cols do
          t
        else
          raise ArgumentError, "matrix must be square"
        end

      _ ->
        raise ArgumentError, "tensor rank must be 0, 1, or 2 for square matrix conversion"
    end
  end

  defp to_square_matrix(v) when is_number(v) do
    Nx.tensor([[v]])
  end

  defp to_square_matrix(list) when is_list(list) do
    # Delegate to the tensor clause which handles all ranks
    list |> Nx.tensor() |> to_square_matrix()
  end

  # Converts numbers/lists into matrices; leaves matrices unchanged
  defp to_matrix(%Nx.Tensor{} = t) do
    case Nx.rank(t) do
      0 -> Nx.reshape(t, {1, 1})
      1 -> Nx.reshape(t, {1, Nx.axis_size(t, 0)})
      2 -> t
      _ -> raise ArgumentError, "observation matrix must have rank 0, 1, or 2"
    end
  end

  defp to_matrix(v) when is_number(v), do: Nx.tensor([[v]])
  defp to_matrix(list) when is_list(list), do: list |> Nx.tensor() |> to_matrix()
end
