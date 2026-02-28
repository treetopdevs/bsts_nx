exla_test_backend? = System.get_env("BSTS_NX_TEST_BACKEND") == "exla"

if exla_test_backend? do
  unless Code.ensure_loaded?(EXLA.Backend) do
    raise "BSTS_NX_TEST_BACKEND=exla requested, but EXLA.Backend is unavailable"
  end

  Application.put_env(:nx, :default_backend, EXLA.Backend)
  Nx.global_default_backend(EXLA.Backend)
end

if exla_test_backend? do
  ExUnit.start(exclude: [:slow], max_cases: 1)
else
  ExUnit.start(exclude: [:slow])
end
