defmodule BstsNx.TestHelpers do
  @moduledoc false

  alias BstsNx.KalmanFilter

  @spec uplift_series(pos_integer(), [non_neg_integer()], number(), number()) :: [float()]
  def uplift_series(total_steps, uplift_indices, baseline \\ 10.0, uplift \\ 5.0) do
    uplift_set = MapSet.new(uplift_indices)

    Enum.map(0..(total_steps - 1), fn idx ->
      if MapSet.member?(uplift_set, idx), do: baseline + uplift, else: baseline * 1.0
    end)
  end

  @spec scalar_filter_outputs([number()], keyword()) :: {Nx.t(), Nx.t()}
  def scalar_filter_outputs(observations, opts \\ []) do
    f = Keyword.get(opts, :f, 1.0)
    h = Keyword.get(opts, :h, 1.0)
    q = Keyword.get(opts, :q, 0.2)
    r = Keyword.get(opts, :r, 0.5)
    x0 = Keyword.get(opts, :x0, 0.0)
    p0 = Keyword.get(opts, :p0, 1.0)

    observations
    |> Nx.tensor(type: {:f, 32})
    |> KalmanFilter.filter_defn(f, h, q, r, x0, p0)
  end
end
