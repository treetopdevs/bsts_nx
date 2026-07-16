defmodule BstsSiteWeb.Format do
  @moduledoc false

  @spec decimal1(number()) :: binary()
  def decimal1(value) when is_number(value) do
    :erlang.float_to_binary(value * 1.0, decimals: 1)
  end

  @spec decimal2(number()) :: binary()
  def decimal2(value) when is_number(value) do
    :erlang.float_to_binary(value * 1.0, decimals: 2)
  end
end
