# Plan 026: Per-client fairness for the demo compute Limiter

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**:
> `git diff --stat 9b7cb8d..HEAD -- site/lib/bsts_site/demos/limiter.ex site/lib/bsts_site_web/live/async_demo.ex site/lib/bsts_site_web/endpoint.ex site/test/bsts_site/demos/limiter_test.exs`
> Plan 024 is EXPECTED to have created `async_demo.ex` (the single
> `Limiter.run` call site). If it does not exist, see STOP conditions.

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: MED (touches the WebSocket connect path and the semaphore; gated by the existing limiter tests + new ones)
- **Depends on**: plans/024-async-scaffold-and-error-states.md (one call site
  to change). Soft: 019/021 change the load math — retune constants after
  they land.
- **Category**: security (availability)
- **Planned at**: commit `9b7cb8d`, 2026-07-08

## Why this matters

Audit finding SEC-01 (residual): `BstsSite.Demos.Limiter` is a single global
3-slot counting semaphore. It correctly prevents unbounded concurrency, but
has **no notion of who is asking**: one client that re-fires MCMC requests
back-to-back holds slots continuously (each fit releases and is instantly
reacquired by the same client), pinning both vCPUs and returning `:busy` to
every other visitor indefinitely. Plan 019's debounce removes the
instant-lane flood and plan 021 shrinks fit durations, but neither creates
fairness on the heavy tier. The cheap, high-leverage fix: cap concurrent
slots **per client** (1 of 3), so a greedy client competes with itself, not
with everyone.

## Current state

- `site/lib/bsts_site/demos/limiter.ex` — full module read at planning; the
  relevant mechanics:

  ```elixir
  @max_concurrent 3

  def run(fun, server \\ __MODULE__) when is_function(fun, 0) do
    case GenServer.call(server, :acquire) do
      :ok ->
        try do
          {:ok, fun.()}
        after
          GenServer.cast(server, {:release, self()})
        end
      :busy -> :busy
    end
  end

  def handle_call(:acquire, {pid, _tag}, %{used: used, max: max} = state) when used < max do
    ref = Process.monitor(pid)
    {:reply, :ok, %{state | used: used + 1, holders: Map.put(state.holders, ref, pid)}}
  end
  def handle_call(:acquire, _from, state), do: {:reply, :busy, state}
  ```

  `init` builds `%{used: 0, max: ..., holders: %{}}` (holders maps monitor
  ref → pid); release demonitors with `:flush` before decrementing; `:DOWN`
  reclaims. This design is correct — extend it, don't rewrite it.

- After plan 024, the ONLY caller is
  `site/lib/bsts_site_web/live/async_demo.ex` (`Limiter.run(work)` inside
  `run_guarded`).

- Client identity: the site runs behind Fly's proxy. The endpoint socket
  (`site/lib/bsts_site_web/endpoint.ex`) currently passes
  `connect_info: [session: @session_options]` — no peer/header info. Fly sets
  the `fly-client-ip` header on proxied requests;
  `connect_info: [:peer_data, :x_headers, session: ...]` exposes it to
  `mount` via `get_connect_info(socket, :x_headers)`. On localhost there is
  no such header — fall back to `peer_data.address`.

- Existing tests: `site/test/bsts_site/demos/limiter_test.exs` (normal
  release, kill-mid-run reclamation, double-release guard) — extend, keep all
  passing.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Limiter tests | `cd site && mix test test/bsts_site/demos/limiter_test.exs` | 0 failures |
| Full site suite | `cd site && mix test` | 0 failures |
| Compile/format | `cd site && mix compile --warnings-as-errors && mix format --check-formatted` | exit 0 |

Tooling note: `mix` is a `mise` shim; prefix `mise exec -- ` if needed.

## Scope

**In scope**:
- `site/lib/bsts_site/demos/limiter.ex`
- `site/lib/bsts_site_web/live/async_demo.ex` (thread the client id)
- `site/lib/bsts_site_web/endpoint.ex` (extend `connect_info`)
- The 8 MCMC LiveViews ONLY if the client id can't be derived inside
  `AsyncDemo` generically (prefer a shared `client_id(socket)` in
  `AsyncDemo`; mounts pass nothing)
- `site/test/bsts_site/demos/limiter_test.exs` (extend)

**Out of scope**:
- Global rate limiting / token buckets on instant-lane events (debounce in
  plan 019 covers the realistic case; a full rate-limit layer is a separate
  decision).
- `fly.toml` `hard_limit` tuning.
- Queueing/waiting semantics — the Limiter stays immediate-reject (`:busy`),
  by design (the UX copy expects it).

## Git workflow

- Branch: `advisor/026-limiter-per-client-fairness` (from `execute-plans`,
  after 024).
- Commit style: `fix(site): per-client slot cap in the demo Limiter`
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Extend the Limiter with a per-client cap

Backward-compatible API: `run(fun, opts_or_server \\ [])` accepting
`client: term()` (default `:anonymous`) — or add `run/3`; keep the existing
`run/2` (fun, server) contract working so current tests pass unmodified.
State grows a `clients: %{client_id => count}` map:

- `handle_call({:acquire, client}, ...)`: reply `:busy` when `used >= max`
  **or** `Map.get(clients, client, 0) >= @max_per_client` (`@max_per_client 1`;
  make it an init opt like `max`). `:anonymous` client ids are exempt from
  the per-client check ONLY if you cannot derive an id (see Step 3) — count
  them under the global cap as today.
- Holders store `{pid, client}` per monitor ref so both release paths
  (explicit cast and `:DOWN`) decrement the right client count and drop
  zero-count entries (no unbounded map growth).

**Verify**: `cd site && mix test test/bsts_site/demos/limiter_test.exs` →
existing tests still pass (they use the old API).

### Step 2: New Limiter tests

Extend `limiter_test.exs` (same style — start a private named instance with
small `max`):

1. Same client, second concurrent acquire → `:busy` even with global slots
   free.
2. Two different clients → both get slots (up to global max).
3. Per-client count is released on normal completion AND on kill-mid-run
   (mirror the existing `:DOWN` test but assert the same client can acquire
   again afterwards).
4. Zero-count cleanup: after release, the internal clients map does not
   retain the key (use `:sys.get_state/1` as the existing tests' style
   allows, or assert behaviorally).

**Verify**: `cd site && mix test test/bsts_site/demos/limiter_test.exs` → 0
failures, ≥4 new tests.

### Step 3: Derive a client id and thread it through

1. `endpoint.ex`: extend the LiveView socket to
   `connect_info: [:peer_data, :x_headers, session: @session_options]`
   (both `websocket:` and `longpoll:`).
2. In `AsyncDemo` add:

   ```elixir
   def client_id(socket) do
     with headers when is_list(headers) <- Phoenix.LiveView.get_connect_info(socket, :x_headers),
          {_, ip} <- List.keyfind(headers, "fly-client-ip", 0) do
       ip
     else
       _ ->
         case Phoenix.LiveView.get_connect_info(socket, :peer_data) do
           %{address: addr} -> :inet.ntoa(addr) |> to_string()
           _ -> :anonymous
         end
     end
   end
   ```

   Call it inside `run_guarded` (connect info is available post-connect,
   which is the only time events fire) and pass `client:` to `Limiter.run`.
   Cache it in an assign on first use — `get_connect_info` is cheap but
   there's no reason to re-derive per click.

**Verify**: `cd site && mix compile --warnings-as-errors` → exit 0;
`cd site && mix test` → 0 failures (plan 023/024 tests exercise the pages;
in the test transport `get_connect_info` returns what the test passes —
absent info falls back to `:anonymous`, which keeps LiveView tests working).

### Step 4: Manual verification

`cd site && mix phx.server`; open an MCMC page in two browser tabs. Fire a
fit in tab 1 and immediately in tab 2.

**Verify**: locally both tabs share one client ip (127.0.0.1), so tab 2 gets
the busy card while tab 1 runs — which IS the per-client cap working
(previously tab 2 would have taken a second global slot). Note this
observation in the report.

## Test plan

Step 2's four Limiter tests (concurrency semantics) + the existing three;
plan 023/024's LiveView tests as the integration net; the two-tab manual
check for the end-to-end path (automated two-client LiveView simulation is
possible with two `live/2` conns but connect-info injection is fiddly — do it
only if `Phoenix.LiveViewTest.put_connect_info/2` makes it trivial, else
manual + note).

## Done criteria

- [ ] `Limiter` enforces a per-client concurrent cap (default 1) with the
      global cap unchanged (default 3); old API still works
- [ ] Monitor-reclaim decrements per-client counts (test proves it)
- [ ] `endpoint.ex` exposes `:x_headers` + `:peer_data`; `AsyncDemo` derives
      `fly-client-ip` with peer fallback
- [ ] `cd site && mix test` → 0 failures, ≥4 new tests
- [ ] Compile + format checks clean
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report back (do not improvise) if:

- `async_demo.ex` does not exist (plan 024 not landed) — doing this against 8
  copy-pasted call sites doubles the diff for no reason.
- The existing limiter tests fail under the extended state shape — the
  backward-compat requirement is being violated; rethink, don't patch tests.
- `get_connect_info/2` returns nothing post-connect on LiveView 1.2 in a way
  the fallback can't handle (would mean the connect_info plumbing differs
  from expectation — report).
- You find yourself adding a queue, TTLs, or token buckets — scope creep;
  the plan is a concurrent-slot cap only.

## Maintenance notes

- Constants (`max: 3`, `max_per_client: 1`) were sized for BinaryBackend fit
  times; after plan 021 (EXLA) they deserve a revisit with
  `mix bench.structured_backends` data — cheaper fits may justify more slots.
- The client id is an IP: NAT'd offices share one. With `max_per_client: 1`
  that's an acceptable trade for a demo site; if complaints arrive, session
  ids (from `connect_info` session) are the finer-grained alternative.
- If a real rate-limiting layer (per-IP event budgets) is ever wanted, put it
  at connect/plug level, not inside the Limiter — keep the semaphore simple.
