defmodule BstsSite.Demos.ScenariosTest do
  use ExUnit.Case, async: true

  alias BstsSite.Demos.Scenarios

  test "noise returns exactly the requested number of deterministic samples" do
    for count <- [0, 1, 2, 5, 6] do
      samples = Scenarios.noise(count, 2.0, 42)

      assert length(samples) == count
      assert samples == Scenarios.noise(count, 2.0, 42)
    end
  end
end
