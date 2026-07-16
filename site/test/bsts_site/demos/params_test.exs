defmodule BstsSite.Demos.ParamsTest do
  use ExUnit.Case, async: true

  alias BstsSite.Demos.Params

  test "clamps integer and string slider values to the range" do
    range = 2..8

    assert Params.clamp(1, range) == 2
    assert Params.clamp(5, range) == 5
    assert Params.clamp("12", range) == 8
  end

  test "falls back to the first range value for invalid input" do
    assert Params.clamp("invalid", 2..8) == 2
    assert Params.clamp(nil, 2..8) == 2
  end
end
