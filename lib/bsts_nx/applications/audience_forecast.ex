defmodule BstsNx.Applications.AudienceForecast do
  @moduledoc """
  Draw-preserving composition of viewing-level and share forecasts.

  Audience is commonly decomposed as

      audience = universe * PUT_or_HUT * share

  This module performs that multiplication for every posterior draw and future
  period. It deliberately does not multiply marginal means, which would lose
  nonlinear uncertainty and dependence needed for delivery-risk calculations.
  """

  alias BstsNx.Forecast
  alias BstsNx.Utils

  @doc """
  Combines compatible PUT/HUT and share forecasts with an audience universe.

  `universe` may be:

    * a non-negative scalar,
    * a non-negative horizon-length vector,
    * a `{draw, horizon}` tensor/list aligned to the two forecasts, or
    * another shape-compatible `BstsNx.Forecast`.

  The PUT/HUT and share forecasts must have the same draw and horizon shapes.
  """
  @spec combine(
          Forecast.t(),
          Forecast.t(),
          number() | list() | Nx.t() | Forecast.t(),
          keyword()
        ) :: Forecast.t()
  def combine(%Forecast{} = viewing, %Forecast{} = share, universe, opts \\ []) do
    validate_forecast_shapes!(viewing, share)
    universe_draws = normalize_universe!(universe, viewing.num_draws, viewing.horizon)

    draws =
      viewing.draws
      |> Nx.multiply(share.draws)
      |> Nx.multiply(universe_draws)

    metadata =
      %{
        decomposition: :universe_viewing_share,
        viewing_metadata: viewing.metadata,
        share_metadata: share.metadata
      }
      |> Map.merge(metadata_option!(opts))

    Forecast.new(
      draws,
      alpha: Keyword.get(opts, :alpha, viewing.alpha),
      metadata: metadata
    )
  end

  @doc "Alias for `combine/4` using HUT/PUT terminology."
  @spec from_hut_share(
          Forecast.t(),
          Forecast.t(),
          number() | list() | Nx.t() | Forecast.t(),
          keyword()
        ) :: Forecast.t()
  def from_hut_share(viewing, share, universe, opts \\ []) do
    combine(viewing, share, universe, opts)
  end

  defp normalize_universe!(%Forecast{} = universe, num_draws, horizon) do
    if universe.num_draws != num_draws or universe.horizon != horizon do
      raise ArgumentError,
            "universe forecast must have shape {#{num_draws}, #{horizon}}, got " <>
              inspect(Nx.shape(universe.draws))
    end

    validate_universe_values!(universe.draws)
  end

  defp normalize_universe!(universe, num_draws, horizon) when is_number(universe) do
    universe
    |> Nx.tensor(type: {:f, 64})
    |> Nx.broadcast({num_draws, horizon})
    |> validate_universe_values!()
  end

  defp normalize_universe!(%Nx.Tensor{} = universe, num_draws, horizon) do
    universe = Nx.as_type(universe, {:f, 64})

    normalized =
      cond do
        Nx.rank(universe) == 0 ->
          Nx.broadcast(universe, {num_draws, horizon})

        Nx.rank(universe) == 1 and Nx.axis_size(universe, 0) == horizon ->
          Nx.broadcast(universe, {num_draws, horizon})

        Nx.rank(universe) == 2 and Nx.shape(universe) == {num_draws, horizon} ->
          universe

        true ->
          raise ArgumentError,
                "universe must be scalar, length #{horizon}, or shape " <>
                  "{#{num_draws}, #{horizon}}; got #{inspect(Nx.shape(universe))}"
      end

    validate_universe_values!(normalized)
  end

  defp normalize_universe!(universe, num_draws, horizon) when is_list(universe) do
    universe_tensor =
      try do
        Nx.tensor(universe, type: {:f, 64})
      rescue
        error in ArgumentError ->
          raise ArgumentError, "universe must be numeric: #{Exception.message(error)}"
      end

    normalize_universe!(universe_tensor, num_draws, horizon)
  end

  defp normalize_universe!(other, _num_draws, _horizon) do
    raise ArgumentError,
          "universe must be a number, list, Nx tensor, or Forecast, got: #{inspect(other)}"
  end

  defp validate_universe_values!(universe) do
    non_finite? = Utils.has_non_finite?(universe)
    negative? = universe |> Nx.less(0.0) |> Nx.any() |> Nx.to_number() == 1

    cond do
      non_finite? -> raise ArgumentError, "universe must contain only finite values"
      negative? -> raise ArgumentError, "universe must contain only non-negative values"
      true -> universe
    end
  end

  defp validate_forecast_shapes!(left, right) do
    if left.num_draws != right.num_draws or left.horizon != right.horizon do
      raise ArgumentError,
            "viewing and share forecasts must have the same draw and horizon dimensions, got " <>
              "#{inspect(Nx.shape(left.draws))} and #{inspect(Nx.shape(right.draws))}"
    end
  end

  defp metadata_option!(opts) do
    case Keyword.get(opts, :metadata, %{}) do
      metadata when is_map(metadata) -> metadata
      metadata -> raise ArgumentError, "metadata must be a map, got: #{inspect(metadata)}"
    end
  end
end
