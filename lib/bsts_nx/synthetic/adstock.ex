defmodule BstsNx.Synthetic.Adstock do
  @moduledoc """
  Adstock and saturation transforms for synthetic data generation.

  These pure-Elixir functions model two key advertising phenomena:

  1. **Geometric adstock** — carry-over effect where a TV spot's impact
     decays geometrically over subsequent time steps.
  2. **Hill saturation** — diminishing returns where additional advertising
     intensity yields progressively smaller incremental effects.

  Both operate on plain Elixir lists (not Nx tensors) since synthetic data
  generation is a one-time operation that doesn't benefit from JIT compilation.
  """

  @doc """
  Applies geometric adstock (carry-over) transformation to a signal.

  Each output value is: `A(t) = signal(t) + lambda * A(t-1)`

  This models the lagged effect of advertising — a TV spot at time t
  continues to influence behavior at t+1, t+2, etc., with exponentially
  decaying weight controlled by `lambda`.

  ## Parameters

  - `signal` — list of floats representing the raw impulse signal
  - `lambda` — decay rate in [0, 1) (number). 0 = no carry-over, approaches 1 = near-infinite memory

  ## Examples

      iex> BstsNx.Synthetic.Adstock.geometric_adstock([1.0, 0.0, 0.0, 0.0], 0.5)
      [1.0, 0.5, 0.25, 0.125]

      iex> BstsNx.Synthetic.Adstock.geometric_adstock([1.0, 0.0, 0.0], 0.0)
      [1.0, 0.0, 0.0]

      iex> BstsNx.Synthetic.Adstock.geometric_adstock([], 0.5)
      []
  """
  @spec geometric_adstock([float()], number()) :: [float()]
  def geometric_adstock([], _lambda), do: []

  def geometric_adstock(signal, lambda) when is_list(signal) and is_number(lambda) do
    {result, _} =
      Enum.map_reduce(signal, 0.0, fn s, prev ->
        current = s + lambda * prev
        {current, current}
      end)

    result
  end

  @doc """
  Applies Hill saturation (diminishing returns) transformation.

  Each output value is: `H(x) = 1 / (1 + (x / ec)^(-slope))`

  This is a sigmoid-like function that maps input intensity to a [0, 1]
  response curve. `ec` (effective concentration / half-saturation) controls
  the inflection point, and `slope` controls the steepness.

  Values at or below zero map to 0.0 (no negative saturation).

  ## Parameters

  - `signal` — list of non-negative floats
  - `ec` — half-saturation point (positive float). At `x = ec`, output = 0.5
  - `slope` — steepness of the saturation curve (positive float)

  ## Examples

      iex> BstsNx.Synthetic.Adstock.hill_saturation([0.0, 1.0, 10.0], 1.0, 1.0)
      [0.0, 0.5, 0.9090909090909091]

      iex> BstsNx.Synthetic.Adstock.hill_saturation([], 1.0, 1.0)
      []
  """
  @spec hill_saturation([float()], number(), number()) :: [float()]
  def hill_saturation([], _ec, _slope), do: []

  def hill_saturation(signal, ec, slope)
      when is_list(signal) and is_number(ec) and ec > 0.0 and is_number(slope) and slope > 0.0 do
    # Pre-compute ec^slope once
    ec_s = :math.pow(ec, slope)

    Enum.map(signal, fn x ->
      if x <= 0.0 do
        0.0
      else
        # Equivalent to 1/(1+(x/ec)^(-slope)) but avoids overflow for small x.
        # Rewritten as: x^s / (x^s + ec^s)
        x_s = :math.pow(x, slope)
        x_s / (x_s + ec_s)
      end
    end)
  end
end
