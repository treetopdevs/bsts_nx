defmodule BstsNx.ObservationWeights do
  @moduledoc """
  Known relative observation-variance weights for Gaussian state-space models.

  The weights represent a heteroskedastic observation model:

      y_t = H_t x_t + v_t
      v_t ~ Normal(0, sigma_r^2 * w_t)

  where every `w_t` is finite and strictly positive. Larger weights mean a
  less precise observation. The common scale `sigma_r^2` is still estimated by
  the existing Gibbs sampler.

  BstsNx implements this model through exact Gaussian prewhitening:

      y'_t = y_t / sqrt(w_t)
      H'_t = H_t / sqrt(w_t)

  The transformed model has constant observation variance `sigma_r^2` and can
  use the existing Kalman and Gibbs paths without changing their posterior.
  Forecast observations are transformed back to the original response scale
  after posterior simulation.
  """

  alias BstsNx.ModelSpec
  alias BstsNx.Utils

  @type weights :: [float()]

  @doc """
  Normalizes a list or rank-1 tensor of variance weights.

  Returns `nil` when the input is `nil`. Otherwise returns a float list whose
  length is `expected_length`. Every value must be finite and greater than
  zero.
  """
  @spec normalize!(nil | [number()] | Nx.t(), non_neg_integer(), atom()) :: nil | weights()
  def normalize!(nil, _expected_length, _name), do: nil

  def normalize!(values, expected_length, name)
      when is_integer(expected_length) and expected_length >= 0 and is_atom(name) do
    weights = to_list!(values, name)

    if length(weights) != expected_length do
      raise ArgumentError,
            "#{name} length (#{length(weights)}) must match expected length (#{expected_length})"
    end

    Enum.map(weights, fn value -> normalize_value!(value, name) end)
  end

  @doc """
  Prewhitens observations and a model specification for known variance weights.

  The returned specification differs only in its observation rows (`h`). State
  transition dynamics, priors, and regression metadata are preserved. Missing
  observations are retained so the downstream sampler can skip their update.
  """
  @spec prewhiten([number() | atom()], ModelSpec.t(), weights()) ::
          {[number() | atom()], ModelSpec.t()}
  def prewhiten(observations, %ModelSpec{} = spec, weights)
      when is_list(observations) and is_list(weights) do
    normalized_weights =
      normalize!(weights, length(observations), :observation_variance_weights)

    transformed_observations =
      observations
      |> Enum.zip(normalized_weights)
      |> Enum.map(fn {observation, weight} ->
        if Utils.missing_observation?(observation) do
          observation
        else
          observation / :math.sqrt(weight)
        end
      end)

    {transformed_observations, %{spec | h: prewhiten_h(spec.h, normalized_weights)}}
  end

  @doc """
  Scales static or time-varying observation rows by `1 / sqrt(w_t)`.

  The return value is always a list of `{1, state_dim}` tensors, one per
  weight.
  """
  @spec prewhiten_h(Nx.t() | [Nx.t()], weights()) :: [Nx.t()]
  def prewhiten_h(h, weights) when is_list(weights) do
    normalized_weights =
      normalize!(weights, length(weights), :observation_variance_weights)

    h
    |> observation_rows!(length(normalized_weights))
    |> Enum.zip(normalized_weights)
    |> Enum.map(fn {row, weight} -> Nx.divide(row, :math.sqrt(weight)) end)
  end

  @doc """
  Restores simulated prewhitened observations to their original response scale.
  """
  @spec restore_trajectories([[number()]], weights()) :: [[float()]]
  def restore_trajectories(trajectories, weights)
      when is_list(trajectories) and is_list(weights) do
    normalized_weights =
      normalize!(weights, length(weights), :future_observation_variance_weights)

    Enum.map(trajectories, fn trajectory ->
      validate_same_length!(
        trajectory,
        normalized_weights,
        :future_observation_variance_weights
      )

      trajectory
      |> Enum.zip(normalized_weights)
      |> Enum.map(fn {value, weight} -> value * :math.sqrt(weight) end)
    end)
  end

  defp to_list!(%Nx.Tensor{} = tensor, name) do
    if Nx.rank(tensor) != 1 do
      raise ArgumentError,
            "#{name} must be a list or rank-1 tensor, got shape #{inspect(Nx.shape(tensor))}"
    end

    Nx.to_flat_list(tensor)
  end

  defp to_list!(values, _name) when is_list(values), do: values

  defp to_list!(value, name) do
    raise ArgumentError,
          "#{name} must be a list or rank-1 tensor, got: #{inspect(value)}"
  end

  defp normalize_value!(value, name) when is_number(value) do
    normalized = value * 1.0

    if finite_number?(normalized) and normalized > 0.0 do
      normalized
    else
      raise ArgumentError,
            "#{name} values must be finite and greater than zero, got: #{inspect(value)}"
    end
  end

  defp normalize_value!(value, name) do
    raise ArgumentError,
          "#{name} values must be finite numbers greater than zero, got: #{inspect(value)}"
  end

  defp observation_rows!(rows, expected_length) when is_list(rows) do
    if length(rows) != expected_length do
      raise ArgumentError,
            "time-varying observation rows length (#{length(rows)}) must match weights length (#{expected_length})"
    end

    Enum.map(rows, &row_matrix!/1)
  end

  defp observation_rows!(%Nx.Tensor{} = h, expected_length) do
    case Nx.rank(h) do
      1 ->
        row = row_matrix!(h)
        List.duplicate(row, expected_length)

      2 ->
        rows = Nx.axis_size(h, 0)

        cond do
          rows == 1 ->
            List.duplicate(row_matrix!(h), expected_length)

          rows == expected_length ->
            h
            |> Utils.tensor_rows_to_row_matrices()
            |> Enum.map(&row_matrix!/1)

          true ->
            raise ArgumentError,
                  "observation matrix rows (#{rows}) must be 1 or match weights length (#{expected_length})"
        end

      rank ->
        raise ArgumentError,
              "observation matrix must be rank 1 or rank 2, got rank #{rank}"
    end
  end

  defp observation_rows!(value, _expected_length) do
    raise ArgumentError,
          "observation matrix must be an Nx tensor or list of tensors, got: #{inspect(value)}"
  end

  defp row_matrix!(%Nx.Tensor{} = row) do
    case Nx.rank(row) do
      1 ->
        Nx.reshape(row, {1, Nx.axis_size(row, 0)})

      2 ->
        case Nx.shape(row) do
          {1, _columns} ->
            row

          shape ->
            raise ArgumentError,
                  "observation row must have shape {n} or {1, n}, got #{inspect(shape)}"
        end

      _rank ->
        raise ArgumentError,
              "observation row must have shape {n} or {1, n}, got #{inspect(Nx.shape(row))}"
    end
  end

  defp row_matrix!(value) do
    raise ArgumentError, "observation rows must be Nx tensors, got: #{inspect(value)}"
  end

  defp validate_same_length!(left, right, name) do
    if length(left) != length(right) do
      raise ArgumentError,
            "#{name} length (#{length(right)}) must match series length (#{length(left)})"
    end
  end

  defp finite_number?(value), do: value == value and abs(value) < 1.0e300
end
