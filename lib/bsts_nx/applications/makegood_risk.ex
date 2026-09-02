defmodule BstsNx.Applications.MakegoodRisk do
  @moduledoc """
  Posterior delivery and under-delivery risk calculations.

  The input forecast must retain joint trajectories. Each forecast period is
  multiplied by its scheduled exposure weight and summed within each draw,
  preserving cross-period uncertainty before contract-level risk is measured.
  """

  alias BstsNx.Forecast
  alias BstsNx.Utils

  @type result :: %{
          delivery_forecast: Forecast.t(),
          expected_delivery: float(),
          median_delivery: float(),
          lower_delivery: float(),
          upper_delivery: float(),
          conservative_delivery: float(),
          reserve_quantile: float(),
          guarantee: float(),
          underdelivery_probability: float(),
          expected_shortfall: float(),
          conditional_expected_shortfall: float()
        }

  @doc """
  Evaluates delivery risk against a guarantee.

  `exposure_weights` must contain one finite, non-negative value per forecast
  horizon. It can encode spot counts, impression multipliers, or any other
  non-negative linear exposure. The result includes the full one-period
  delivery forecast so downstream callers can compute additional posterior
  risk measures.

  ## Options

    * `:reserve_quantile` - lower delivery quantile used as a conservative
      inventory estimate (default: `0.10`)
    * `:alpha` - interval significance level for the delivery forecast;
      defaults to the source forecast's alpha
    * `:metadata` - metadata merged into the delivery forecast
  """
  @spec evaluate(Forecast.t(), [number()] | Nx.t(), number(), keyword()) :: result()
  def evaluate(forecast, exposure_weights, guarantee, opts \\ [])

  def evaluate(
        %Forecast{} = audience_forecast,
        exposure_weights,
        guarantee,
        opts
      ) do
    guarantee = normalize_non_negative_finite!(guarantee, :guarantee)
    exposure_weights = normalize_exposure_weights!(exposure_weights, audience_forecast.horizon)
    reserve_quantile = Keyword.get(opts, :reserve_quantile, 0.10)
    validate_quantile!(reserve_quantile)

    metadata =
      %{aggregation: :weighted_delivery, guarantee: guarantee}
      |> Map.merge(metadata_option!(opts))

    delivery_forecast =
      Forecast.weighted_sum(
        audience_forecast,
        exposure_weights,
        alpha: Keyword.get(opts, :alpha, audience_forecast.alpha),
        metadata: metadata
      )

    summary = Forecast.summary(delivery_forecast)

    %{
      delivery_forecast: delivery_forecast,
      expected_delivery: only(summary.mean),
      median_delivery: only(summary.median),
      lower_delivery: only(summary.lower),
      upper_delivery: only(summary.upper),
      conservative_delivery: delivery_forecast |> Forecast.quantile(reserve_quantile) |> only(),
      reserve_quantile: reserve_quantile,
      guarantee: guarantee,
      underdelivery_probability:
        delivery_forecast |> Forecast.probability_below(guarantee) |> only(),
      expected_shortfall: delivery_forecast |> Forecast.expected_shortfall(guarantee) |> only(),
      conditional_expected_shortfall:
        delivery_forecast
        |> Forecast.conditional_expected_shortfall(guarantee)
        |> only()
    }
  end

  def evaluate(forecast, _weights, _guarantee, _opts) do
    raise ArgumentError, "forecast must be a BstsNx.Forecast, got: #{inspect(forecast)}"
  end

  @doc """
  Returns expected makegood units for a positive impressions-per-unit value.

  The calculation uses expected shortfall rather than shortfall at the mean.
  """
  @spec expected_makegood_units(result(), number()) :: float()
  def expected_makegood_units(%{expected_shortfall: shortfall}, impressions_per_unit)
      when is_number(impressions_per_unit) do
    if finite_number?(impressions_per_unit) and impressions_per_unit > 0.0 do
      shortfall / impressions_per_unit
    else
      raise ArgumentError,
            "impressions_per_unit must be a positive finite number, got: " <>
              inspect(impressions_per_unit)
    end
  end

  def expected_makegood_units(_result, impressions_per_unit) do
    raise ArgumentError,
          "impressions_per_unit must be a positive finite number, got: " <>
            inspect(impressions_per_unit)
  end

  defp normalize_exposure_weights!(%Nx.Tensor{} = weights, horizon) do
    weights = Nx.as_type(weights, {:f, 64})

    if Nx.rank(weights) != 1 or Nx.axis_size(weights, 0) != horizon do
      raise ArgumentError,
            "exposure_weights must be a length-#{horizon} vector, got shape " <>
              inspect(Nx.shape(weights))
    end

    cond do
      Utils.has_non_finite?(weights) ->
        raise ArgumentError, "exposure_weights must contain only finite values"

      weights |> Nx.less(0.0) |> Nx.any() |> Nx.to_number() == 1 ->
        raise ArgumentError, "exposure_weights must contain only non-negative values"

      true ->
        weights
    end
  end

  defp normalize_exposure_weights!(weights, horizon) when is_list(weights) do
    weights
    |> Nx.tensor(type: {:f, 64})
    |> normalize_exposure_weights!(horizon)
  rescue
    error in ArgumentError ->
      raise ArgumentError, "exposure_weights must be numeric: #{Exception.message(error)}"
  end

  defp normalize_exposure_weights!(weights, horizon) do
    raise ArgumentError,
          "exposure_weights must be a length-#{horizon} list or Nx tensor, got: " <>
            inspect(weights)
  end

  defp normalize_non_negative_finite!(value, name)
       when value in [:nan, :infinity, :neg_infinity] do
    raise ArgumentError, "#{name} must be finite, got: #{inspect(value)}"
  end

  defp normalize_non_negative_finite!(value, name) when is_number(value) do
    normalized = value * 1.0

    cond do
      not finite_number?(normalized) ->
        raise ArgumentError, "#{name} must be finite, got: #{inspect(value)}"

      normalized < 0.0 ->
        raise ArgumentError, "#{name} must be non-negative, got: #{inspect(value)}"

      true ->
        normalized
    end
  end

  defp normalize_non_negative_finite!(value, name) do
    raise ArgumentError, "#{name} must be numeric, got: #{inspect(value)}"
  end

  defp validate_quantile!(quantile) do
    if not finite_number?(quantile) or quantile < 0.0 or quantile > 1.0 do
      raise ArgumentError,
            "reserve_quantile must be a finite value in [0, 1], got: #{inspect(quantile)}"
    end

    :ok
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

  defp only([value]), do: value
end
