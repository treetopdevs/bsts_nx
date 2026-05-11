defmodule BstsNx.Execution do
  @moduledoc """
  Shared execution metadata helpers for high-level workflows.

  The high-level APIs expose this metadata so callers can distinguish fast
  operational estimates from slower Bayesian/offline analyses.
  """

  @type mode :: :operational | :bayesian
  @type engine :: :elixir_nx | :r_sidecar

  @type method_used ::
          :scalar_filter
          | :structured_filter
          | :scalar_forecast_filter
          | :structured_forecast_filter
          | :scalar_smooth_filter
          | :structured_smooth_filter
          | :elixir_mcmc
          | :r_causal_impact

  @type t :: %{
          mode: mode(),
          engine: engine(),
          method_used: method_used(),
          elapsed_ms: non_neg_integer(),
          backend: String.t(),
          baseline: :forecast | :smooth | nil,
          prepared?: boolean(),
          fallback?: boolean(),
          fallback_reason: String.t() | nil,
          warnings: [String.t()]
        }

  @doc """
  Resolves the requested execution mode.

  `:mode` is authoritative. When omitted, legacy `:method` values are mapped
  for compatibility (`:filter` -> `:operational`, `:mcmc` -> `:bayesian`).
  """
  @spec resolve_mode!(keyword(), mode()) :: mode()
  def resolve_mode!(opts, default \\ :operational) do
    cond do
      Keyword.has_key?(opts, :mode) ->
        validate_mode!(Keyword.fetch!(opts, :mode))

      Keyword.get(opts, :method) == :filter ->
        :operational

      Keyword.get(opts, :method) == :mcmc ->
        :bayesian

      Keyword.has_key?(opts, :method) ->
        raise ArgumentError,
              "method must be :filter or :mcmc, got: #{inspect(Keyword.fetch!(opts, :method))}"

      true ->
        default
    end
  end

  @doc """
  Measures a function with monotonic time and returns `{elapsed_us, result}`.
  """
  @spec measure((-> result)) :: {non_neg_integer(), result} when result: term()
  def measure(fun) when is_function(fun, 0) do
    start = System.monotonic_time()
    result = fun.()
    elapsed_us = System.convert_time_unit(System.monotonic_time() - start, :native, :microsecond)
    {elapsed_us, result}
  end

  @doc """
  Builds canonical execution metadata.
  """
  @spec metadata(mode(), engine(), method_used(), non_neg_integer(), keyword()) :: t()
  def metadata(mode, engine, method_used, elapsed_us, opts \\ []) do
    %{
      mode: validate_mode!(mode),
      engine: validate_engine!(engine),
      method_used: validate_method_used!(method_used),
      elapsed_ms: div(max(elapsed_us, 0) + 999, 1000),
      backend: inspect(Nx.default_backend()),
      baseline: Keyword.get(opts, :baseline),
      prepared?: Keyword.get(opts, :prepared?, false),
      fallback?: Keyword.get(opts, :fallback?, false),
      fallback_reason: Keyword.get(opts, :fallback_reason),
      warnings: Keyword.get(opts, :warnings, [])
    }
  end

  @doc """
  Returns true when the caller explicitly opted into MCMC fallback.
  """
  @spec explicit_mcmc_fallback?(keyword()) :: boolean()
  def explicit_mcmc_fallback?(opts) do
    Keyword.get(opts, :fallback) == :mcmc or Keyword.get(opts, :allow_mcmc_fallback, false)
  end

  @doc """
  Recognizes the current Nx missing-LU backend error.
  """
  @spec backend_lu_missing?(Exception.t()) :: boolean()
  def backend_lu_missing?(%RuntimeError{message: message}) do
    String.contains?(message, "Nx.Backend.lu/3 not implemented")
  end

  def backend_lu_missing?(_), do: false

  @doc """
  Raises a clear error when the operational structured filter is unsupported.
  """
  @spec raise_structured_filter_unsupported!(Exception.t()) :: no_return()
  def raise_structured_filter_unsupported!(error) do
    raise RuntimeError,
          "structured operational filter is unsupported on the current Nx backend " <>
            "(#{inspect(Nx.default_backend())}): #{Exception.message(error)}. " <>
            "No MCMC fallback was run. Pass fallback: :mcmc for an explicit slow fallback, " <>
            "or run with mode: :bayesian for an offline MCMC analysis."
  end

  defp validate_mode!(mode) when mode in [:operational, :bayesian], do: mode

  defp validate_mode!(mode) do
    raise ArgumentError, "mode must be :operational or :bayesian, got: #{inspect(mode)}"
  end

  defp validate_engine!(engine) when engine in [:elixir_nx, :r_sidecar], do: engine

  defp validate_engine!(engine) do
    raise ArgumentError, "engine must be :elixir_nx or :r_sidecar, got: #{inspect(engine)}"
  end

  defp validate_method_used!(method)
       when method in [
              :scalar_filter,
              :structured_filter,
              :scalar_forecast_filter,
              :structured_forecast_filter,
              :scalar_smooth_filter,
              :structured_smooth_filter,
              :elixir_mcmc,
              :r_causal_impact
            ],
       do: method

  defp validate_method_used!(method) do
    raise ArgumentError,
          "method_used must be a known operational, bayesian, or R method, got: #{inspect(method)}"
  end
end
