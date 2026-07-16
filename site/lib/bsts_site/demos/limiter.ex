defmodule BstsSite.Demos.Limiter do
  @moduledoc """
  A small counting semaphore around the MCMC demo tiers.

  Instant-tier demos (filter lane, milliseconds) run unguarded. Anything that
  samples (hundreds of milliseconds to seconds) must pass through here so a
  burst of visitors degrades to a polite "busy" message instead of a pegged
  machine.

  Slot lifetime is owned by the Limiter itself: every acquire monitors the
  calling process, so a slot is reclaimed even when a LiveView (and the
  `start_async` task linked to it) dies mid-fit. The explicit release in
  `run/2` stays as the fast path for prompt turnaround; the monitor is the
  safety net.
  """

  use GenServer

  @max_concurrent 3

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: Keyword.get(opts, :name, __MODULE__))
  end

  @doc """
  Runs `fun` if a slot is free, returning `{:ok, result}`; otherwise `:busy`.
  The slot is released even if `fun` raises, and reclaimed via monitor if the
  calling process dies mid-run.
  """
  def run(fun, server \\ __MODULE__) when is_function(fun, 0) do
    case GenServer.call(server, :acquire) do
      :ok ->
        try do
          {:ok, fun.()}
        after
          GenServer.cast(server, {:release, self()})
        end

      :busy ->
        :busy
    end
  end

  @impl true
  def init(opts) do
    {:ok, %{used: 0, max: Keyword.get(opts, :max, @max_concurrent), holders: %{}}}
  end

  @impl true
  def handle_call(:acquire, {pid, _tag}, %{used: used, max: max} = state) when used < max do
    ref = Process.monitor(pid)
    {:reply, :ok, %{state | used: used + 1, holders: Map.put(state.holders, ref, pid)}}
  end

  def handle_call(:acquire, _from, state), do: {:reply, :busy, state}

  @impl true
  def handle_cast({:release, pid}, state) do
    case Enum.find(state.holders, fn {_ref, holder} -> holder == pid end) do
      {ref, ^pid} ->
        # Demonitor with :flush BEFORE decrementing so a late :DOWN for this
        # ref can never double-release the slot.
        Process.demonitor(ref, [:flush])
        {:noreply, %{state | used: state.used - 1, holders: Map.delete(state.holders, ref)}}

      nil ->
        {:noreply, state}
    end
  end

  @impl true
  def handle_info({:DOWN, ref, :process, _pid, _reason}, state) do
    case Map.pop(state.holders, ref) do
      {nil, _holders} -> {:noreply, state}
      {_pid, holders} -> {:noreply, %{state | used: state.used - 1, holders: holders}}
    end
  end
end
