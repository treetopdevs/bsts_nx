defmodule BstsSite.Demos.DefaultCacheTest do
  use ExUnit.Case, async: true

  alias BstsSite.Demos.DefaultCache
  alias BstsSite.Demos.Scenarios

  test "computes on first call and returns the stored value on later calls" do
    key = {:test, make_ref()}
    counter = :counters.new(1, [])

    fun = fn ->
      :counters.add(counter, 1, 1)
      :computed
    end

    assert DefaultCache.get(key, fun) == :computed
    assert DefaultCache.get(key, fun) == :computed
    assert DefaultCache.get(key, fun) == :computed

    assert :counters.get(counter, 1) == 1
  end

  test "distinct keys don't collide" do
    key_a = {:test, make_ref()}
    key_b = {:test, make_ref()}

    assert DefaultCache.get(key_a, fn -> :value_a end) == :value_a
    assert DefaultCache.get(key_b, fn -> :value_b end) == :value_b

    # Re-reading each key still returns its own value, not the other's.
    assert DefaultCache.get(key_a, fn -> :other end) == :value_a
    assert DefaultCache.get(key_b, fn -> :other end) == :value_b
  end

  test "Scenarios.hero/2 is deterministic across calls (the cache's core assumption)" do
    first = Scenarios.hero(12, 4)
    second = Scenarios.hero(12, 4)

    assert first == second
  end
end
