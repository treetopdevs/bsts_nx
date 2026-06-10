requested_backend =
  System.get_env("BSTS_NX_TEST_BACKEND")
  |> case do
    nil ->
      if Code.ensure_loaded?(EMLX.Backend), do: "emlx", else: "native"

    value ->
      String.downcase(value)
  end

case requested_backend do
  "exla" ->
    unless Code.ensure_loaded?(EXLA.Backend) do
      raise "BSTS_NX_TEST_BACKEND=exla requested, but EXLA.Backend is unavailable"
    end

    Application.put_env(:nx, :default_backend, EXLA.Backend)
    Nx.global_default_backend(EXLA.Backend)

  "emlx" ->
    unless Code.ensure_loaded?(EMLX.Backend) do
      raise "BSTS_NX_TEST_BACKEND=emlx requested, but EMLX.Backend is unavailable"
    end

    Application.put_env(:nx, :default_backend, EMLX.Backend)
    Nx.global_default_backend(EMLX.Backend)

  "native" ->
    :ok

  other ->
    raise "Unsupported BSTS_NX_TEST_BACKEND=#{inspect(other)} (expected exla, emlx, or native)"
end

if requested_backend in ["exla", "emlx"] do
  ExUnit.start(exclude: [:slow], max_cases: 1, timeout: 300_000)
else
  # The BinaryBackend is the slowest execution path; MCMC-heavy tests need
  # more than the default 60s per test on shared CI hardware.
  ExUnit.start(exclude: [:slow], timeout: 300_000)
end
