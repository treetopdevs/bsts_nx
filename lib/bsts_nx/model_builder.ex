defmodule BstsNx.ModelBuilder do
  @moduledoc """
  Shared model building utilities for BSTS application modules.

  Centralizes the common pattern of "resolve model spec from options" that
  appears across `InterventionAnalysis`, `Forecaster`, `AnomalyDetector`,
  `DemandForecaster`, `TVAttribution`, `MarketingLift`, and `PolicyEvaluator`.

  Also provides shared formatting and coercion utilities used across modules.
  """

  alias BstsNx.Components
  alias BstsNx.CovariateSelection
  alias BstsNx.ModelSpec

  @default_initial_cov 10.0
  @analysis_passthrough_opts [
    :alpha,
    :num_samples,
    :burn_in,
    :seed,
    :seasonality,
    :model_spec,
    :method,
    :mode,
    :fallback,
    :allow_mcmc_fallback,
    :return,
    :baseline,
    :key,
    :thin,
    :n_samples,
    :initial_state,
    :initial_cov,
    :process_var,
    :obs_var,
    :x0,
    :p0,
    :q,
    :r,
    :regression_mode,
    :regression_opts,
    :control_regression_mode,
    :control_regression_opts
  ]

  @doc """
  Builds a model spec from options.

  Returns `{spec, :structured}` when a `ModelSpec`, seasonality, or regressors
  are provided, or `{nil, :scalar}` for the simple local-level model.

  ## Options

    * `:model_spec` - a `%BstsNx.ModelSpec{}` (takes precedence over other options)
    * `:seasonality` - integer >= 2 for seasonal component
    * `:regressors` - a `{T, p}` Nx tensor of regressor values (one row per
      observation, one column per covariate). When provided, produces a composed
      model with trend + optional seasonal + regression components.
    * `:regression_mode` - `:dynamic` (default) or `:spike_and_slab` when
      building the regression component.
    * `:regression_opts` - keyword options forwarded to
      `Components.regression_spec/2`.

  ## Examples

      iex> {nil, :scalar} = BstsNx.ModelBuilder.build_spec([1.0, 2.0, 3.0])
      iex> {%BstsNx.ModelSpec{}, :structured} = BstsNx.ModelBuilder.build_spec([1.0, 2.0], seasonality: 7)
      iex> regressors = Nx.tensor([[1.0], [2.0], [3.0]])
      iex> {%BstsNx.ModelSpec{}, :structured} = BstsNx.ModelBuilder.build_spec([1.0, 2.0, 3.0], regressors: regressors)
  """
  @spec build_spec([number()], keyword()) :: {ModelSpec.t() | nil, :structured | :scalar}
  def build_spec(observations, opts \\ []) do
    case Keyword.get(opts, :model_spec) do
      %ModelSpec{} = spec ->
        {spec, :structured}

      nil ->
        seasonality = Keyword.get(opts, :seasonality)
        regressors = Keyword.get(opts, :regressors)

        base = build_base_spec(observations, seasonality)

        case {base, regressors} do
          {nil, nil} ->
            {nil, :scalar}

          {nil, reg} ->
            trend = default_trend_spec(observations)
            reg_spec = Components.regression_spec(ensure_tensor(reg), regression_spec_opts(opts))
            {Components.compose_specs(trend, reg_spec), :structured}

          {spec, nil} ->
            {spec, :structured}

          {spec, reg} ->
            reg_spec = Components.regression_spec(ensure_tensor(reg), regression_spec_opts(opts))
            {Components.compose_specs(spec, reg_spec), :structured}
        end
    end
  end

  @doc """
  Builds an intervention-analysis model spec from full observations.

  The pre-period controls initial-state choices for default trend components,
  while full-series regressors are preserved so post-period counterfactuals have
  the observation rows they need.
  """
  @spec build_intervention_spec([number()], {pos_integer(), pos_integer()}, keyword()) ::
          {ModelSpec.t() | nil, :structured | :scalar}
  def build_intervention_spec(observations, pre_period, opts \\ []) do
    validate_pre_period!(observations, pre_period)

    case Keyword.get(opts, :model_spec) do
      %ModelSpec{} = spec ->
        {spec, :structured}

      nil ->
        {pre_start, pre_end} = pre_period
        pre_count = pre_end - pre_start + 1
        pre_obs = Enum.slice(observations, pre_start - 1, pre_count)

        opts =
          case Keyword.fetch(opts, :regressors) do
            {:ok, regressors} ->
              Keyword.put(
                opts,
                :regressors,
                validate_regressor_rows!(regressors, length(observations))
              )

            :error ->
              opts
          end

        build_spec(pre_obs, opts)
    end
  end

  @doc """
  Builds a model spec that includes control series as regression covariates.

  Used by `MarketingLift` and `PolicyEvaluator` when `:control_series` is provided.
  Falls back to `build_spec/2` when no controls are given.

  ## Parameters

    * `observations` - the response time series
    * `control_series` - list of control time series (same length as observations),
      or `nil` for no controls
    * `opts` - keyword options (same as `build_spec/2`, plus `:seasonality`)

  ## Control-selection options

    * `:control_selection` - optional covariate-selection strategy for controls.
      When `true`, uses default Pearson screening. When a keyword list, options
      are passed to `BstsNx.CovariateSelection.select/3` (for example:
      `control_selection: [method: :spike_and_slab]`).
    * `:control_selection_pre_period` - optional `{start, end}` (1-based
      inclusive) window used to fit control selection. Defaults to the full
      series when omitted.
    * `:control_regression_mode` - `:dynamic` (default) or `:spike_and_slab`.
      In `:spike_and_slab` mode, coefficients are sampled in-loop with
      Zellner's g-prior.
    * `:control_regression_opts` - keyword options forwarded to
      `Components.regression_spec/2` (for example `prior_inclusion`, `g`).

  Returns a keyword list suitable for passing to `InterventionAnalysis.analyze/3`.
  """
  @spec build_opts_with_controls([number()], [[number()]] | nil, keyword()) :: keyword()
  def build_opts_with_controls(_observations, nil, opts) do
    Keyword.take(opts, @analysis_passthrough_opts)
  end

  def build_opts_with_controls(observations, [], opts) do
    build_opts_with_controls(observations, nil, opts)
  end

  def build_opts_with_controls(observations, controls, opts) when is_list(controls) do
    n = length(observations)

    regressors_all =
      controls
      |> Enum.map(fn series ->
        if length(series) != n do
          raise ArgumentError,
                "control series length (#{length(series)}) must match observations (#{n})"
        end

        series
      end)
      |> Nx.tensor()
      |> Nx.transpose()

    regressors = maybe_select_controls(observations, regressors_all, opts)

    spec =
      case {Keyword.get(opts, :model_spec), regressors} do
        {%ModelSpec{} = base_spec, nil} ->
          base_spec

        {%ModelSpec{} = base_spec, %Nx.Tensor{} = selected_regressors} ->
          reg_spec =
            Components.regression_spec(
              selected_regressors,
              control_regression_spec_opts(opts)
            )

          # Preserve explicit model_spec precedence while still adding controls.
          Components.compose_specs(base_spec, reg_spec)

        {nil, selected_regressors} ->
          build_opts =
            opts
            |> Keyword.take([:seasonality])
            |> maybe_put_regressors(selected_regressors)
            |> put_control_regression_config(opts)

          case build_spec(observations, build_opts) do
            {nil, :scalar} -> nil
            {built_spec, :structured} -> built_spec
          end
      end

    result = Keyword.take(opts, @analysis_passthrough_opts)

    if spec == nil do
      result
    else
      Keyword.put(result, :model_spec, spec)
    end
  end

  @doc """
  Coerces a list of observations to floats.

  Handles Nx tensors (extracts scalar), integers (converts to float), and
  missing observations (`nil`, `:nan`, or NaN values) by preserving them as a
  NaN sentinel for downstream missing-data handling.
  """
  @spec coerce_obs([number() | nil | atom() | Nx.t()]) :: [float()]
  def coerce_obs(observations) do
    Enum.map(observations, &coerce_observation/1)
  end

  @doc """
  Returns the first observed value as a float, defaulting to 0.0.
  """
  @spec first_obs([number() | nil | atom() | Nx.t()]) :: float()
  def first_obs([]), do: 0.0

  def first_obs(observations) do
    Enum.find_value(observations, fn obs ->
      if BstsNx.Utils.missing_observation?(obs) do
        nil
      else
        coerce_observation(obs)
      end
    end) || 0.0
  end

  @doc """
  Safely converts potentially non-numeric values to floats.

  Returns 0.0 for `:nan`, atoms, and other non-numbers.
  """
  @spec safe_number(number() | atom()) :: float()
  def safe_number(:nan), do: 0.0

  def safe_number(n) do
    case normalize_number(n) do
      {:ok, f} -> f
      :error -> 0.0
    end
  end

  @doc """
  Formats a number with 2 decimal places, or "N/A" for atoms.
  """
  @spec format_num(number() | atom()) :: String.t()
  def format_num(n) do
    case normalize_number(n) do
      {:ok, f} -> :erlang.float_to_binary(f, decimals: 2)
      :error -> "N/A"
    end
  end

  @doc """
  Formats a number as a percentage with 1 decimal place, or "N/A" for atoms.
  """
  @spec format_pct(number() | atom()) :: String.t()
  def format_pct(n) do
    case normalize_number(n) do
      {:ok, f} -> :erlang.float_to_binary(f * 100.0, decimals: 1) <> "%"
      :error -> "N/A"
    end
  end

  @typedoc """
  Standardized effect measurement returned by application modules.

  All modules that measure causal effects (`MarketingLift`, `PolicyEvaluator`)
  return this same structure, making downstream code portable across modules.

  Fields:
    * `cumulative` - total effect over the post-period
    * `cumulative_sd` - standard deviation of cumulative effect
    * `cumulative_lower` / `cumulative_upper` - credible interval bounds
    * `per_period` - average effect per post-period time step
    * `per_period_sd` - standard deviation of per-period effect
    * `relative` - relative (%) change vs baseline
    * `relative_sd` - standard deviation of relative change
  """
  @type effect :: %{
          cumulative: float(),
          cumulative_sd: float(),
          cumulative_lower: float(),
          cumulative_upper: float(),
          per_period: float(),
          per_period_sd: float(),
          relative: float(),
          relative_sd: float()
        }

  @doc """
  Builds a standardized effect map from a `CausalImpact.summary()`.

  ## Parameters

    * `summary` - a summary map from `CausalImpact.summary/1` with
      `:cumulative_effect` and `:relative_effect` sub-maps
    * `n_post` - number of post-period time steps (for per-period averaging)

  ## Examples

      iex> summary = %{
      ...>   cumulative_effect: %{mean: 50.0, sd: 10.0, lower: 30.0, upper: 70.0},
      ...>   relative_effect: %{mean: 0.25, sd: 0.05, lower: 0.15, upper: 0.35}
      ...> }
      iex> effect = BstsNx.ModelBuilder.build_effect(summary, 5)
      iex> effect.cumulative
      50.0
      iex> effect.per_period
      10.0
      iex> effect.relative
      0.25
  """
  @spec build_effect(map(), non_neg_integer()) :: effect()
  def build_effect(summary, n_post) do
    cum = summary.cumulative_effect
    rel = summary.relative_effect

    %{
      cumulative: safe_number(cum.mean),
      cumulative_sd: safe_number(cum.sd),
      cumulative_lower: safe_number(cum.lower),
      cumulative_upper: safe_number(cum.upper),
      per_period: safe_divide(safe_number(cum.mean), n_post),
      per_period_sd: safe_divide(safe_number(cum.sd), n_post),
      relative: safe_number(rel.mean),
      relative_sd: safe_number(rel.sd)
    }
  end

  @doc """
  Builds a list of future H matrices for forecast horizons with regressors.

  Given a composed model spec (trend + optional seasonal + regression) and
  a `{horizon, p}` tensor of future regressor values, constructs per-step
  observation matrices by combining the static H (for trend/seasonal) with
  the future regressor row.

  ## Parameters

    * `spec` - the composed `ModelSpec` (must include regression dimensions)
    * `future_regressors` - `{horizon, p}` tensor of future regressor values
    * `n_regression_dims` - number of regression dimensions in the state vector

  ## Returns

  A list of `horizon` observation matrices (each `{1, n_state}`).
  """
  @spec build_future_h(ModelSpec.t(), Nx.t(), non_neg_integer()) :: [Nx.t()]
  def build_future_h(spec, future_regressors, n_regression_dims) do
    future_t = ensure_tensor(future_regressors)
    horizon = Nx.axis_size(future_t, 0)
    p = Nx.axis_size(future_t, 1)
    n_state = Nx.axis_size(spec.f, 0)
    n_non_reg = n_state - n_regression_dims

    if p != n_regression_dims do
      raise ArgumentError,
            "future_regressors columns (#{p}) must match training regressors (#{n_regression_dims})"
    end

    static_h =
      case spec.h do
        list when is_list(list) ->
          rows = Enum.map(list, &BstsNx.Utils.h_to_row_tensor/1)

          static_rows =
            Enum.map(rows, fn h_row ->
              Nx.slice(h_row, [0], [n_non_reg])
            end)

          [first_static | rest_static] = static_rows

          varying_non_reg? =
            Enum.any?(rest_static, fn h_row ->
              Nx.all_close(h_row, first_static, atol: 1.0e-10, rtol: 1.0e-10)
              |> Nx.to_number() == 0
            end)

          if varying_non_reg? do
            raise ArgumentError,
                  "build_future_h/3 requires a time-invariant non-regression H portion; " <>
                    "pass a model spec with static trend/seasonal observation mapping"
          end

          Nx.reshape(first_static, {1, n_non_reg})

        %Nx.Tensor{} = h ->
          h_row = BstsNx.Utils.h_to_row_tensor(h)
          Nx.reshape(Nx.slice(h_row, [0], [n_non_reg]), {1, n_non_reg})
      end

    static_h_batched = Nx.broadcast(static_h, {horizon, n_non_reg})
    combined = Nx.concatenate([static_h_batched, future_t], axis: 1)

    BstsNx.Utils.tensor_rows_to_row_matrices(combined)
  end

  defp build_base_spec(_observations, nil), do: nil

  defp build_base_spec(observations, seasonality) do
    trend = default_trend_spec(observations)
    seasonal = Components.seasonal_spec(seasonality)
    Components.compose_specs(trend, seasonal)
  end

  defp default_trend_spec(observations) do
    Components.local_linear_trend_spec(
      initial_level: first_obs(observations),
      initial_cov_level: @default_initial_cov
    )
  end

  @doc false
  @spec ensure_tensor(Nx.t() | list()) :: Nx.t()
  def ensure_tensor(%Nx.Tensor{} = t), do: t
  def ensure_tensor(list) when is_list(list), do: Nx.tensor(list)

  defp safe_divide(_, 0), do: 0.0
  defp safe_divide(_, d) when d == 0.0, do: 0.0
  defp safe_divide(n, d), do: n / d

  defp regression_spec_opts(opts) do
    mode = Keyword.get(opts, :regression_mode, :dynamic)
    extra_opts = Keyword.get(opts, :regression_opts, [])
    validate_keyword_opts!(:regression_opts, extra_opts)
    [mode: mode] ++ extra_opts
  end

  defp control_regression_spec_opts(opts) do
    mode =
      Keyword.get(opts, :control_regression_mode, Keyword.get(opts, :regression_mode, :dynamic))

    extra_opts =
      Keyword.get(opts, :control_regression_opts, Keyword.get(opts, :regression_opts, []))

    validate_keyword_opts!(:control_regression_opts, extra_opts)
    [mode: mode] ++ extra_opts
  end

  defp put_control_regression_config(build_opts, source_opts) do
    spec_opts = control_regression_spec_opts(source_opts)
    {mode, reg_opts} = Keyword.pop(spec_opts, :mode, :dynamic)
    build_opts |> Keyword.put(:regression_mode, mode) |> Keyword.put(:regression_opts, reg_opts)
  end

  defp validate_keyword_opts!(_name, []), do: :ok

  defp validate_keyword_opts!(name, opts) when is_list(opts) do
    if not Keyword.keyword?(opts) do
      raise ArgumentError, "#{name} must be a keyword list, got: #{inspect(opts)}"
    end
  end

  defp validate_keyword_opts!(name, opts) do
    raise ArgumentError, "#{name} must be a keyword list, got: #{inspect(opts)}"
  end

  defp maybe_put_regressors(opts, nil), do: opts
  defp maybe_put_regressors(opts, regressors), do: Keyword.put(opts, :regressors, regressors)

  defp validate_pre_period!(observations, {pre_start, pre_end})
       when is_integer(pre_start) and is_integer(pre_end) do
    n = length(observations)

    if pre_start < 1 or pre_end < pre_start or pre_end > n do
      raise ArgumentError,
            "pre_period must be {start, end} within [1, #{n}], got: #{inspect({pre_start, pre_end})}"
    end
  end

  defp validate_pre_period!(observations, pre_period) do
    raise ArgumentError,
          "pre_period must be {start, end} within [1, #{length(observations)}], got: #{inspect(pre_period)}"
  end

  defp validate_regressor_rows!(regressors, expected_rows) do
    tensor = ensure_tensor(regressors)

    if Nx.rank(tensor) != 2 do
      raise ArgumentError,
            "regressors must be a rank-2 tensor/list with #{expected_rows} rows, got shape #{inspect(Nx.shape(tensor))}"
    end

    rows = Nx.axis_size(tensor, 0)

    if rows != expected_rows do
      raise ArgumentError,
            "regressors row count (#{rows}) must match observations (#{expected_rows})"
    end

    tensor
  end

  defp maybe_select_controls(observations, regressors, opts) do
    case normalize_control_selection(Keyword.get(opts, :control_selection)) do
      :disabled ->
        regressors

      selection_opts ->
        n_obs = length(observations)

        {start_idx, end_idx} =
          select_window(Keyword.get(opts, :control_selection_pre_period), n_obs)

        pre_count = end_idx - start_idx + 1
        p = Nx.axis_size(regressors, 1)

        target_pre = Enum.slice(observations, (start_idx - 1)..(end_idx - 1))
        regressors_pre = Nx.slice(regressors, [start_idx - 1, 0], [pre_count, p])

        selected =
          CovariateSelection.select(target_pre, regressors_pre, selection_opts).selected_indices

        build_selected_regressors(regressors, selected)
    end
  end

  defp normalize_control_selection(nil), do: :disabled
  defp normalize_control_selection(false), do: :disabled
  defp normalize_control_selection(true), do: []

  defp normalize_control_selection(selection_opts) when is_list(selection_opts),
    do: selection_opts

  defp normalize_control_selection(other) do
    raise ArgumentError,
          ":control_selection must be true, false/nil, or a keyword list; got: #{inspect(other)}"
  end

  defp select_window(nil, n_obs), do: {1, n_obs}

  defp select_window({start_idx, end_idx}, n_obs)
       when is_integer(start_idx) and is_integer(end_idx) and start_idx >= 1 and
              end_idx >= start_idx and end_idx <= n_obs do
    {start_idx, end_idx}
  end

  defp select_window(other, n_obs) do
    raise ArgumentError,
          ":control_selection_pre_period must be {start, end} within [1, #{n_obs}], got: #{inspect(other)}"
  end

  defp build_selected_regressors(_regressors, []), do: nil

  defp build_selected_regressors(regressors, selected_indices) do
    Nx.take(regressors, Nx.tensor(selected_indices, type: {:s, 64}), axis: 1)
  end

  defp coerce_observation(obs) do
    cond do
      BstsNx.Utils.missing_observation?(obs) ->
        missing_observation_sentinel()

      match?(%Nx.Tensor{}, obs) ->
        Nx.to_number(obs) + 0.0

      is_number(obs) ->
        obs + 0.0

      true ->
        raise ArgumentError,
              "observations must be numbers, scalar Nx tensors, or missing values; got: #{inspect(obs)}"
    end
  end

  defp missing_observation_sentinel do
    Nx.Constants.nan() |> Nx.to_number()
  end

  defp normalize_number(n) when is_number(n) do
    f = n * 1.0
    if f == f and abs(f) < 1.0e300, do: {:ok, f}, else: :error
  end

  defp normalize_number(_), do: :error
end
