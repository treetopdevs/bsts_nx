defmodule BstsNx.ModelSpec do
  @moduledoc """
  Struct defining a state-space model specification for the structured Gibbs sampler.

  A `ModelSpec` encodes the complete parametric form of a linear Gaussian
  state-space model:

      x_t = F x_{t-1} + w_t,   w_t ~ N(0, Q)
      y_t = H_t x_t  + v_t,    v_t ~ N(0, R)

  where Q is a diagonal matrix whose entries are independently resampled
  from inverse-gamma posteriors during Gibbs sampling.

  ## Fields

    * `:f` — transition matrix, `{n, n}` tensor
    * `:h` — observation matrix. Either a single `{1, n}` tensor (static)
      or a list of `{1, n}` tensors (time-varying, one per observation)
    * `:x0` — initial state mean, `{n}` vector
    * `:p0` — initial state covariance, `{n, n}` matrix
    * `:obs_var` — initial observation variance (positive float)
    * `:q_specs` — list of maps, one per diagonal Q entry to resample.
      Each map has keys: `:dim_index` (int), `:initial` (float),
      `:prior_shape` (float), `:prior_scale` (float)
    * `:regression` — optional regression metadata map used by advanced
      in-loop samplers (for example spike-and-slab priors). `nil` by default.
    * `:obs_prior_shape` — shape parameter of the inverse-gamma prior on
      observation variance (default: 1.0)
    * `:obs_prior_scale` — scale parameter of the inverse-gamma prior on
      observation variance (default: 1.0)
  """

  @enforce_keys [:f, :h, :x0, :p0, :obs_var, :q_specs]
  defstruct [
    :f,
    :h,
    :x0,
    :p0,
    :obs_var,
    :q_specs,
    regression: nil,
    obs_prior_shape: 1.0,
    obs_prior_scale: 1.0
  ]

  @type q_spec :: %{
          dim_index: non_neg_integer(),
          initial: float(),
          prior_shape: float(),
          prior_scale: float()
        }

  @type t :: %__MODULE__{
          f: Nx.t(),
          h: Nx.t() | [Nx.t()],
          x0: Nx.t(),
          p0: Nx.t(),
          obs_var: float(),
          q_specs: [q_spec()],
          regression: map() | nil,
          obs_prior_shape: float(),
          obs_prior_scale: float()
        }

  @doc """
  Validates a model specification and returns it unchanged when valid.

  Raises `ArgumentError` with a descriptive message on invalid dimensions
  or prior parameters.
  """
  @spec validate!(t()) :: t()
  def validate!(%__MODULE__{} = spec) do
    n = validate_state_matrices!(spec)
    validate_h!(spec.h, n)
    validate_q_specs!(spec.q_specs, n)
    validate_positive_number!(spec.obs_var, :obs_var)
    validate_positive_number!(spec.obs_prior_shape, :obs_prior_shape)
    validate_positive_number!(spec.obs_prior_scale, :obs_prior_scale)
    spec
  end

  defp validate_state_matrices!(%__MODULE__{f: f, x0: x0, p0: p0}) do
    if Nx.rank(f) != 2 do
      raise ArgumentError, "f must be rank-2, got rank #{Nx.rank(f)}"
    end

    {n_rows, n_cols} = Nx.shape(f)

    if n_rows != n_cols do
      raise ArgumentError, "f must be square, got shape #{inspect(Nx.shape(f))}"
    end

    if Nx.rank(x0) != 1 or Nx.axis_size(x0, 0) != n_rows do
      raise ArgumentError, "x0 must have shape {#{n_rows}}, got #{inspect(Nx.shape(x0))}"
    end

    if Nx.rank(p0) != 2 or Nx.shape(p0) != {n_rows, n_rows} do
      raise ArgumentError,
            "p0 must have shape {#{n_rows}, #{n_rows}}, got #{inspect(Nx.shape(p0))}"
    end

    n_rows
  end

  defp validate_h!(h, n) when is_list(h) do
    if h == [] do
      raise ArgumentError, "h list must be non-empty"
    end

    Enum.each(h, &validate_h_row!(&1, n))
  end

  defp validate_h!(%Nx.Tensor{} = h, n) do
    case Nx.rank(h) do
      1 ->
        if Nx.axis_size(h, 0) != n do
          raise ArgumentError,
                "h rank-1 tensor must have length #{n}, got #{inspect(Nx.shape(h))}"
        end

      2 ->
        {_rows, cols} = Nx.shape(h)

        if cols != n do
          raise ArgumentError,
                "h rank-2 tensor must have last dimension #{n}, got #{inspect(Nx.shape(h))}"
        end

      rank ->
        raise ArgumentError, "h must be rank 1, rank 2, or list of row tensors, got rank #{rank}"
    end
  end

  defp validate_h!(other, _n) do
    raise ArgumentError, "h must be an Nx tensor or list of tensors, got: #{inspect(other)}"
  end

  defp validate_h_row!(%Nx.Tensor{} = h_row, n) do
    cond do
      Nx.rank(h_row) == 1 and Nx.axis_size(h_row, 0) == n ->
        :ok

      Nx.rank(h_row) == 2 and Nx.shape(h_row) == {1, n} ->
        :ok

      true ->
        raise ArgumentError,
              "h row must have shape {#{n}} or {1, #{n}}, got #{inspect(Nx.shape(h_row))}"
    end
  end

  defp validate_h_row!(other, _n) do
    raise ArgumentError, "h list entries must be Nx tensors, got: #{inspect(other)}"
  end

  defp validate_q_specs!(q_specs, n) when is_list(q_specs) do
    dim_indices =
      Enum.map(q_specs, fn qs ->
        dim_index = Map.get(qs, :dim_index)
        prior_shape = Map.get(qs, :prior_shape)
        prior_scale = Map.get(qs, :prior_scale)
        initial = Map.get(qs, :initial)

        if not is_integer(dim_index) or dim_index < 0 or dim_index >= n do
          raise ArgumentError,
                "q_specs dim_index must be in [0, #{n - 1}], got: #{inspect(dim_index)}"
        end

        validate_non_negative_number!(initial, :initial)
        validate_positive_number!(prior_shape, :prior_shape)
        validate_positive_number!(prior_scale, :prior_scale)
        dim_index
      end)

    if length(dim_indices) != MapSet.size(MapSet.new(dim_indices)) do
      raise ArgumentError, "q_specs dim_index values must be unique, got: #{inspect(dim_indices)}"
    end
  end

  defp validate_q_specs!(other, _n) do
    raise ArgumentError, "q_specs must be a list, got: #{inspect(other)}"
  end

  defp validate_positive_number!(value, field) do
    if not is_number(value) or value <= 0 do
      raise ArgumentError, "#{field} must be > 0, got: #{inspect(value)}"
    end
  end

  defp validate_non_negative_number!(value, field) do
    if not is_number(value) or value < 0 do
      raise ArgumentError, "#{field} must be >= 0, got: #{inspect(value)}"
    end
  end
end
