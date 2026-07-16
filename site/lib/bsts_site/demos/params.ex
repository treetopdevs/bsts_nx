defmodule BstsSite.Demos.Params do
  @moduledoc false

  @spec clamp(integer() | binary() | term(), Range.t()) :: integer()
  def clamp(value, range) when is_integer(value) do
    value
    |> max(range.first)
    |> min(range.last)
  end

  def clamp(value, range) when is_binary(value) do
    case Integer.parse(value) do
      {integer, _rest} -> clamp(integer, range)
      :error -> range.first
    end
  end

  def clamp(_value, range), do: range.first
end
