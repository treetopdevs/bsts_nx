defmodule BstsNx.Forecast do
  @moduledoc """
  Joint posterior forecast draws with reusable summaries and risk operations.

  A forecast stores draws as a rank-2 tensor with shape `{draw, horizon}`.
  Keeping the complete draw matrix preserves dependence across future periods,
  which is required for operations such as flight-level delivery, threshold
  probabilities, and draw-by-draw composition of related forecasts.

  Use `new/2` to construct a forecast from a tensor or nested lists. A flat
  list is interpreted as draws for a one-period forecast.
  """

  alias BstsNx.Validation

  @enforce_keys [
    :draws,
    :mean,
    :median,
    :sd,
    :lower,
    :upper,
    :horizon,
    :num_draws,
    :alpha
  ]
  defstruct [
    :draws,
    :mean,
    :median,
    :sd,
    :lower,
    :upper,
    :horizon,
    :num_draws,
    :alpha,
    metadata: %{}
  ]

  @type t :: %__MODULE__{
          draws: Nx.t(),
          mean: Nx.t(),
          median: Nx.t(),
          sd: Nx.t(),
          lower: Nx.t(),
          upper: Nx.t(),
          horizon: pos_integer(),
          num_draws: pos_integer(),
          alpha: float(),
          metadata: map()
        }

  @doc """
  Builds a forecast and computes its marginal summaries.

  ## Options

    * `:alpha` - significance level for `lower` and `upper` (default: `0.05`)
    * `:metadata` - arbitrary map retained on the forecast
  """
  @spec new(Nx.t() | [number()] | [[number()]], keyword()) :: t()
  def new(draws, opts \\ []) do
    alpha = Keyword.get(opts, :alpha, 0.05)
    Validation.validate_alpha!(alpha)
    metadata = metadata_option!(opts)

    draws_t = normalize_draws!(draws)
    {num_draws, horizon} = Nx.shape(draws_t)

    if num_draws == 0 or horizon == 0 do
      raise ArgumentError, "forecast draws must have non-zero draw and horizon dimensions"
    end

    sorted = Nx.sort(draws_t, axis: 0)
    mean = Nx.mean(draws_t, axes: [0])

    sd =
      if num_draws < 2 do
        Nx.broadcast(Nx.tensor(0.0, type: Nx.type(draws_t)), {horizon})
      else
        Nx.standard_deviation(draws_t, axes: [0], ddof: 1)
      end

    %__MODULE__{
      draws: draws_t,
      mean: mean,
      median: quantile_from_sorted(sorted, num_draws, 0.5),
      sd: sd,
      lower: quantile_from_sorted(sorted, num_draws, alpha / 2.0),
      upper: quantile_from_sorted(sorted, num_draws, 1.0 - alpha / 2.0),
      horizon: horizon,
      num_draws: num_draws,
      alpha: alpha,
      metadata: metadata
    }
  end

  @doc """
  Returns a summary map in list or tensor form.

  The default format is `:lists`; pass `format: :tensors` to retain Nx tensors.
  """
  @spec summary(t(), keyword()) :: map()
  def summary(%__MODULE__{} = forecast, opts \\ []) do
    case Keyword.get(opts, :format, :lists) do
      :lists ->
        %{
          mean: Nx.to_flat_list(forecast.mean),
          median: Nx.to_flat_list(forecast.median),
          sd: Nx.to_flat_list(forecast.sd),
          lower: Nx.to_flat_list(forecast.lower),
          upper: Nx.to_flat_list(forecast.upper),
          horizon: forecast.horizon,
          num_draws: forecast.num_draws,
          alpha: forecast.alpha,
          metadata: forecast.metadata
        }

      :tensors ->
        %{
          mean: forecast.mean,
          median: forecast.median,
          sd: forecast.sd,
          lower: forecast.lower,
          upper: forecast.upper,
          horizon: forecast.horizon,
          num_draws: forecast.num_draws,
          alpha: forecast.alpha,
          metadata: forecast.metadata
        }

      other ->
        raise ArgumentError, "format must be :lists or :tensors, got: #{inspect(other)}"
    end
  end

  @doc "Returns the joint draw matrix as nested lists."
  @spec draws_to_lists(t()) :: [[float()]]
  def draws_to_lists(%__MODULE__{draws: draws, horizon: horizon}) do
    draws
    |> Nx.to_flat_list()
    |> Enum.chunk_every(horizon)
  end

  @doc """
  Computes a marginal quantile at each forecast horizon.

  Quantiles use the nearest indexed order statistic over the retained draws.
  """
  @spec quantile(t(), float()) :: [float()]
  def quantile(%__MODULE__{} = forecast, probability) do
    validate_probability!(probability)

    forecast.draws
    |> Nx.sort(axis: 0)
    |> quantile_from_sorted(forecast.num_draws, probability)
    |> Nx.to_flat_list()
  end

  @doc """
  Applies a scalar function to every retained draw and rebuilds the summary.
  """
  @spec map(t(), (float() -> number()), keyword()) :: t()
  def map(%__MODULE__{} = forecast, fun, opts \\ []) when is_function(fun, 1) do
    rows =
      forecast
      |> draws_to_lists()
      |> Enum.map(fn row -> Enum.map(row, fun) end)

    rebuild(forecast, rows, opts)
  end

  @doc """
  Combines two shape-compatible forecasts draw by draw.

  The forecasts must have the same number of draws and horizon. This explicit
  alignment prevents accidental destruction of dependence between forecasts.
  """
  @spec combine(t(), t(), (float(), float() -> number()), keyword()) :: t()
  def combine(%__MODULE__{} = left, %__MODULE__{} = right, fun, opts \\ [])
      when is_function(fun, 2) do
    validate_same_shape!(left, right)

    rows =
      left
      |> draws_to_lists()
      |> Enum.zip(draws_to_lists(right))
      |> Enum.map(fn {left_row, right_row} ->
        left_row
        |> Enum.zip(right_row)
        |> Enum.map(fn {left_value, right_value} -> fun.(left_value, right_value) end)
      end)

    metadata =
      Map.merge(left.metadata, right.metadata)
      |> Map.merge(metadata_option!(opts))

    new(rows, alpha: Keyword.get(opts, :alpha, left.alpha), metadata: metadata)
  end

  @doc """
  Multiplies every horizon by a scalar or horizon-length vector.
  """
  @spec scale(t(), number() | [number()] | Nx.t(), keyword()) :: t()
  def scale(%__MODULE__{} = forecast, values, opts \\ []) do
    scale_t = normalize_horizon_vector!(values, forecast.horizon, :scale)
    draws = Nx.multiply(forecast.draws, scale_t)
    rebuild(forecast, draws, opts)
  end

  @doc """
  Sums each draw across the forecast horizon and returns a one-period forecast.
  """
  @spec sum(t(), keyword()) :: t()
  def sum(%__MODULE__{} = forecast, opts \\ []) do
    weights = Nx.broadcast(Nx.tensor(1.0, type: Nx.type(forecast.draws)), {forecast.horizon})
    weighted_sum(forecast, weights, opts)
  end

  @doc """
  Computes one weighted total for each retained trajectory.

  This is suitable for aggregating forecast periods into contract or flight
  delivery while retaining uncertainty and cross-period dependence.
  """
  @spec weighted_sum(t(), [number()] | Nx.t(), keyword()) :: t()
  def weighted_sum(%__MODULE__{} = forecast, weights, opts \\ []) do
    weights_t = normalize_horizon_vector!(weights, forecast.horizon, :weights)

    totals =
      forecast.draws
      |> Nx.dot(weights_t)
      |> Nx.reshape({forecast.num_draws, 1})

    rebuild(forecast, totals, opts)
  end

  @doc "Returns the posterior probability that each horizon is below `threshold`."
  @spec probability_below(t(), number()) :: [float()]
  def probability_below(%__MODULE__{} = forecast, threshold) do
    threshold = normalize_finite_number!(threshold, :threshold)
    threshold_t = Nx.tensor(threshold, type: Nx.type(forecast.draws))

    forecast.draws
    |> Nx.less(threshold_t)
    |> Nx.as_type({:f, 64})
    |> Nx.mean(axes: [0])
    |> Nx.to_flat_list()
  end

  @doc "Returns expected positive shortfall `max(threshold - draw, 0)` by horizon."
  @spec expected_shortfall(t(), number()) :: [float()]
  def expected_shortfall(%__MODULE__{} = forecast, threshold) do
    threshold = normalize_finite_number!(threshold, :threshold)
    type = Nx.type(forecast.draws)
    threshold_t = Nx.tensor(threshold, type: type)
    zero = Nx.tensor(0.0, type: type)

    threshold_t
    |> Nx.subtract(forecast.draws)
    |> Nx.max(zero)
    |> Nx.mean(axes: [0])
    |> Nx.to_flat_list()
  end

  @doc """
  Returns expected shortfall conditional on being below `threshold` by horizon.

  Horizons with no under-threshold draws return `0.0`.
  """
  @spec conditional_expected_shortfall(t(), number()) :: [float()]
  def conditional_expected_shortfall(%__MODULE__{} = forecast, threshold) do
    threshold = normalize_finite_number!(threshold, :threshold)

    forecast
    |> draws_to_lists()
    |> BstsNx.Utils.transpose_rows()
    |> Enum.map(fn values ->
      shortfalls =
        for value <- values, value < threshold do
          threshold - value
        end

      case shortfalls do
        [] -> 0.0
        _ -> Enum.sum(shortfalls) / length(shortfalls)
      end
    end)
  end

  defp rebuild(%__MODULE__{} = source, draws, opts) do
    metadata = Map.merge(source.metadata, metadata_option!(opts))
    new(draws, alpha: Keyword.get(opts, :alpha, source.alpha), metadata: metadata)
  end

  defp normalize_draws!(%Nx.Tensor{} = draws) do
    case Nx.rank(draws) do
      1 ->
        draws
        |> Nx.as_type({:f, 64})
        |> Nx.reshape({Nx.axis_size(draws, 0), 1})
        |> validate_finite_draws!()

      2 ->
        draws
        |> Nx.as_type({:f, 64})
        |> validate_finite_draws!()

      rank ->
        raise ArgumentError, "forecast draws must be rank 1 or 2, got rank #{rank}"
    end
  end

  defp normalize_draws!(draws) when is_list(draws) do
    if draws == [] do
      raise ArgumentError, "forecast draws must be non-empty"
    end

    draws
    |> Nx.tensor(type: {:f, 64})
    |> normalize_draws!()
  rescue
    e in ArgumentError ->
      raise ArgumentError,
            "forecast draws must be a rectangular numeric list: #{Exception.message(e)}"
  end

  defp normalize_draws!(other) do
    raise ArgumentError,
          "forecast draws must be an Nx tensor or numeric list, got: #{inspect(other)}"
  end

  defp validate_finite_draws!(draws) do
    if BstsNx.Utils.has_non_finite?(draws) do
      raise ArgumentError, "forecast draws must contain only finite values"
    end

    draws
  end

  defp normalize_horizon_vector!(value, horizon, _name) when is_number(value) do
    value
    |> normalize_finite_number!(:value)
    |> Nx.tensor(type: {:f, 64})
    |> Nx.broadcast({horizon})
  end

  defp normalize_horizon_vector!(%Nx.Tensor{} = values, horizon, name) do
    values_t = Nx.as_type(values, {:f, 64})

    if Nx.rank(values_t) != 1 or Nx.axis_size(values_t, 0) != horizon do
      raise ArgumentError,
            "#{name} must be a scalar or length-#{horizon} vector, got shape #{inspect(Nx.shape(values_t))}"
    end

    if BstsNx.Utils.has_non_finite?(values_t) do
      raise ArgumentError, "#{name} must contain only finite values"
    end

    values_t
  end

  defp normalize_horizon_vector!(values, horizon, name) when is_list(values) do
    values
    |> Nx.tensor(type: {:f, 64})
    |> normalize_horizon_vector!(horizon, name)
  rescue
    e in ArgumentError ->
      raise ArgumentError, "#{name} must be numeric: #{Exception.message(e)}"
  end

  defp normalize_horizon_vector!(value, _horizon, name) do
    raise ArgumentError, "#{name} must be a number, list, or Nx tensor, got: #{inspect(value)}"
  end

  defp validate_same_shape!(left, right) do
    if left.num_draws != right.num_draws or left.horizon != right.horizon do
      raise ArgumentError,
            "forecasts must have the same draw and horizon dimensions, got " <>
              "#{inspect(Nx.shape(left.draws))} and #{inspect(Nx.shape(right.draws))}"
    end
  end

  defp validate_probability!(probability) do
    if not finite_number?(probability) or probability < 0.0 or probability > 1.0 do
      raise ArgumentError,
            "probability must be a finite value in [0, 1], got: #{inspect(probability)}"
    end
  end

  defp normalize_finite_number!(value, name)
       when value in [:nan, :infinity, :neg_infinity] do
    raise ArgumentError, "#{name} must be finite, got: #{inspect(value)}"
  end

  defp normalize_finite_number!(value, name) when is_number(value) do
    normalized = value * 1.0

    if finite_number?(normalized) do
      normalized
    else
      raise ArgumentError, "#{name} must be finite, got: #{inspect(value)}"
    end
  end

  defp normalize_finite_number!(value, name) do
    raise ArgumentError, "#{name} must be numeric, got: #{inspect(value)}"
  end

  defp metadata_option!(opts) do
    case Keyword.get(opts, :metadata, %{}) do
      metadata when is_map(metadata) -> metadata
      metadata -> raise ArgumentError, "metadata must be a map, got: #{inspect(metadata)}"
    end
  end

  defp finite_number?(value) when is_number(value) do
    value == value and abs(value) < 1.0e300
  end

  defp finite_number?(_value), do: false

  defp quantile_from_sorted(sorted, num_draws, probability) do
    index = round(probability * (num_draws - 1))

    sorted
    |> Nx.slice([index, 0], [1, Nx.axis_size(sorted, 1)])
    |> Nx.squeeze(axes: [0])
  end
end
