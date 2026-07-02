defmodule BstsSite.Demos.LimiterTest do
  use ExUnit.Case, async: false

  alias BstsSite.Demos.Limiter

  # Each test gets its own single-slot Limiter instance so it cannot interfere
  # with the app-supervised singleton (or other tests).
  setup do
    name = :"limiter_test_#{System.unique_integer([:positive])}"
    start_supervised!({Limiter, name: name, max: 1})
    %{server: name}
  end

  defp used(server), do: :sys.get_state(server).used

  defp eventually(fun, attempts \\ 100) do
    if fun.() do
      :ok
    else
      if attempts == 0, do: flunk("condition never became true")
      Process.sleep(10)
      eventually(fun, attempts - 1)
    end
  end

  test "slot is released on normal completion", %{server: server} do
    assert {:ok, 42} = Limiter.run(fn -> 42 end, server)
    eventually(fn -> used(server) == 0 end)

    # The freed slot is immediately usable again.
    assert {:ok, :again} = Limiter.run(fn -> :again end, server)
    eventually(fn -> used(server) == 0 end)
  end

  test "slot is reclaimed when the calling process is killed mid-run", %{server: server} do
    parent = self()

    pid =
      spawn(fn ->
        Limiter.run(
          fn ->
            send(parent, :acquired)
            Process.sleep(:infinity)
          end,
          server
        )
      end)

    assert_receive :acquired, 1_000
    assert used(server) == 1
    assert Limiter.run(fn -> :nope end, server) == :busy

    Process.exit(pid, :kill)
    eventually(fn -> used(server) == 0 end)

    assert {:ok, :works} = Limiter.run(fn -> :works end, server)
  end

  test "explicit release then holder death does not double-release", %{server: server} do
    parent = self()

    pid =
      spawn(fn ->
        assert {:ok, :done} = Limiter.run(fn -> :done end, server)
        send(parent, :released)
        Process.sleep(:infinity)
      end)

    assert_receive :released, 1_000
    eventually(fn -> used(server) == 0 end)

    # The holder dies AFTER its slot was explicitly released. Demonitor with
    # :flush means no stray :DOWN may decrement the counter a second time.
    Process.exit(pid, :kill)
    Process.sleep(50)
    assert used(server) == 0

    # Occupy the single slot; if the kill had double-released, a second run
    # would sneak in here instead of getting :busy.
    holder =
      spawn(fn ->
        Limiter.run(
          fn ->
            send(parent, :held)
            Process.sleep(:infinity)
          end,
          server
        )
      end)

    assert_receive :held, 1_000
    assert used(server) == 1
    assert Limiter.run(fn -> :nope end, server) == :busy

    Process.exit(holder, :kill)
    eventually(fn -> used(server) == 0 end)
  end
end
