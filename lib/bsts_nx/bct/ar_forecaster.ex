defmodule BstsNx.BCT.ARForecaster do
  @moduledoc """
  Phase A scaffold for a Bayesian Context Tree AR forecaster.

  This module provides a lightweight BCT-AR interface (`fit/2`, `predict/2`,
  `fit_predict/3`) that is immediately usable for forecasting while reserving
  the tree/context internals for later phases.

  In this scaffold, forecasting is powered by a ridge-regularized AR(p) core
  with simulation-based uncertainty intervals. Context-tree fields are exposed
  as metadata (`:context_depth`, `:quantizer_bins`, `:context_tree`) so the
  public contract is stable when tree inference is introduced.
  """

  alias BstsNx.Distributions
  alias BstsNx.ModelBuilder
  alias BstsNx.Utils
  alias BstsNx.Validation

  @type fit_result :: %{
          backend: :bct_ar,
          phase: :scaffold,
          order: pos_integer(),
          context_depth: pos_integer(),
          quantizer_bins: pos_integer(),
          ridge: float(),
          coefficients: [float()],
          innovation_sd: float(),
          in_sample_rmse: float(),
          num_samples: pos_integer(),
          training_length: pos_integer(),
          observations: [float()],
          context_tree: nil,
          quantizer: %{bins: pos_integer(), edges: nil}
        }

  @type forecast_result :: %{
          backend: :bct_ar,
          phase: :scaffold,
          mean: [float()],
          lower: [float()],
          upper: [float()],
          sd: [float()],
          horizon: pos_integer(),
          alpha: float(),
          num_samples: pos_integer()
        }

  @doc """
  Fits the Phase A BCT-AR scaffold model.

  ## Options

    * `:order` - AR order `p` (default: `1`)
    * `:context_depth` - reserved context depth metadata (default: `order`)
    * `:quantizer_bins` - reserved quantizer bin count metadata (default: `8`)
    * `:ridge` - ridge penalty for AR fit stability (default: `1.0e-6`)
    * `:num_samples` - default simulation draws for `predict/2` (default: `200`)
  """
  @spec fit([number() | Nx.t()], keyword()) :: fit_result()
  def fit(observations, opts \\ []) do
    obs = ModelBuilder.coerce_obs(observations)

    if obs == [] do
      raise ArgumentError, "observations must be non-empty"
    end

    order = Keyword.get(opts, :order, 1)
    validate_positive_integer!(:order, order)

    if length(obs) <= order do
      raise ArgumentError,
            "observations length (#{length(obs)}) must be greater than order (#{order})"
    end

    context_depth = Keyword.get(opts, :context_depth, order)
    validate_positive_integer!(:context_depth, context_depth)

    quantizer_bins = Keyword.get(opts, :quantizer_bins, 8)

    if not is_integer(quantizer_bins) or quantizer_bins < 2 do
      raise ArgumentError,
            "quantizer_bins must be an integer >= 2, got: #{inspect(quantizer_bins)}"
    end

    ridge = Keyword.get(opts, :ridge, 1.0e-6)

    if not is_number(ridge) or ridge < 0.0 do
      raise ArgumentError, "ridge must be a non-negative number, got: #{inspect(ridge)}"
    end

    num_samples = Keyword.get(opts, :num_samples, 200)
    validate_positive_integer!(:num_samples, num_samples)

    {rows, targets} = build_lagged_dataset(obs, order)
    coefficients = estimate_coefficients(rows, targets, ridge)
    residuals = residuals(rows, targets, coefficients)
    innovation_sd = sample_sd(residuals) |> max(1.0e-6)
    rmse = rmse(residuals)

    %{
      backend: :bct_ar,
      phase: :scaffold,
      order: order,
      context_depth: context_depth,
      quantizer_bins: quantizer_bins,
      ridge: ridge,
      coefficients: coefficients,
      innovation_sd: innovation_sd,
      in_sample_rmse: rmse,
      num_samples: num_samples,
      training_length: length(obs),
      observations: obs,
      context_tree: nil,
      quantizer: %{bins: quantizer_bins, edges: nil}
    }
  end

  @doc """
  Predicts future values from a fitted Phase A BCT-AR scaffold model.

  ## Options

    * `:horizon` - forecast steps ahead (required)
    * `:alpha` - interval significance level (default: `0.05`)
    * `:num_samples` - number of simulation paths (default: from fit result)
    * `:seed` - deterministic seed for simulation
    * `:key` - `Nx.Random` key (overrides `:seed`)
  """
  @spec predict(fit_result(), keyword()) :: forecast_result()
  def predict(%{backend: :bct_ar, phase: :scaffold} = fit_result, opts) do
    validate_fit_result!(fit_result)

    horizon = Keyword.fetch!(opts, :horizon)
    validate_positive_integer!(:horizon, horizon)

    alpha = Keyword.get(opts, :alpha, 0.05)
    Validation.validate_alpha!(alpha, "alpha must be between 0 and 1 (exclusive)")

    num_samples = Keyword.get(opts, :num_samples, fit_result.num_samples)
    validate_positive_integer!(:num_samples, num_samples)

    seed = Keyword.get(opts, :seed, System.os_time())
    key = Keyword.get(opts, :key, Nx.Random.key(seed))

    paths = simulate_paths(fit_result, horizon, num_samples, key)
    summary = summarize_paths(paths, alpha)

    %{
      backend: :bct_ar,
      phase: :scaffold,
      mean: summary.mean,
      lower: summary.lower,
      upper: summary.upper,
      sd: summary.sd,
      horizon: horizon,
      alpha: alpha,
      num_samples: num_samples
    }
  end

  @doc """
  Fits and predicts in one call.
  """
  @spec fit_predict([number() | Nx.t()], pos_integer(), keyword()) :: forecast_result()
  def fit_predict(observations, horizon, opts \\ []) do
    fit_result = fit(observations, opts)

    predict_opts =
      opts
      |> derive_predict_prng_opts()
      |> Keyword.put(:horizon, horizon)

    predict(fit_result, predict_opts)
  end

  # -- AR scaffold internals ---------------------------------------------------

  defp build_lagged_dataset(observations, order) do
    t = length(observations)
    obs_tuple = List.to_tuple(observations)

    rows =
      Enum.map(order..(t - 1), fn idx ->
        lags =
          Enum.map(1..order, fn lag ->
            elem(obs_tuple, idx - lag)
          end)

        [1.0 | lags]
      end)

    targets = Enum.drop(observations, order)
    {rows, targets}
  end

  defp estimate_coefficients(rows, targets, ridge) do
    n_features = rows |> hd() |> length()
    x = Nx.tensor(rows, type: {:f, 64})
    y = Nx.tensor(targets, type: {:f, 64})
    xt = Nx.transpose(x)
    xtx = Nx.dot(xt, x)
    xty = Nx.dot(xt, y)
    ridge_mat = Nx.multiply(Nx.eye(n_features), ridge)

    solve_coefficients(Nx.add(xtx, ridge_mat), xty, targets)
  end

  defp solve_coefficients(xtx, xty, targets) do
    try do
      coeffs = xtx |> BstsNx.Utils.safe_solve(xty) |> Nx.to_flat_list()

      if Enum.all?(coeffs, &finite_number?/1) do
        coeffs
      else
        fallback_coefficients(Nx.axis_size(xty, 0), targets)
      end
    rescue
      _ -> fallback_coefficients(Nx.axis_size(xty, 0), targets)
    end
  end

  defp fallback_coefficients(n_features, targets) do
    intercept = if targets == [], do: 0.0, else: Enum.sum(targets) / length(targets)
    [intercept | List.duplicate(0.0, n_features - 1)]
  end

  defp derive_predict_prng_opts(opts) do
    case Keyword.get(opts, :key) do
      %Nx.Tensor{} = key ->
        opts
        |> Keyword.put(:key, split_prng_key(key, 1))
        |> Keyword.delete(:seed)

      _ ->
        case Keyword.get(opts, :seed) do
          seed when is_integer(seed) ->
            opts
            |> Keyword.put(:key, split_prng_key(Nx.Random.key(seed), 1))
            |> Keyword.delete(:seed)

          _ ->
            opts
        end
    end
  end

  defp split_prng_key(key, index) do
    key
    |> Nx.Random.split(parts: index + 1)
    |> Utils.split_key_at(index)
  end

  defp finite_number?(x) when is_number(x) do
    x == x and abs(x) < 1.0e300
  end

  defp residuals(rows, targets, coefficients) do
    Enum.zip(rows, targets)
    |> Enum.map(fn {row, y} ->
      y - dot(coefficients, row)
    end)
  end

  defp rmse(values) do
    if values == [] do
      0.0
    else
      :math.sqrt(Enum.reduce(values, 0.0, fn x, acc -> acc + x * x end) / length(values))
    end
  end

  defp sample_sd(values) do
    n = length(values)

    cond do
      n <= 1 ->
        0.0

      true ->
        mean = Enum.sum(values) / n

        ss =
          Enum.reduce(values, 0.0, fn x, acc ->
            d = x - mean
            acc + d * d
          end)

        :math.sqrt(ss / (n - 1))
    end
  end

  defp dot(a, b) do
    Enum.zip(a, b)
    |> Enum.reduce(0.0, fn {x, y}, acc -> acc + x * y end)
  end

  defp simulate_paths(fit_result, horizon, num_samples, key) do
    keys = Nx.Random.split(key, parts: num_samples) |> Nx.to_list()

    Enum.map(keys, fn key_row ->
      simulate_single_path(fit_result, horizon, Nx.tensor(key_row, type: Nx.type(key)))
    end)
  end

  defp simulate_single_path(fit_result, horizon, key) do
    step_keys = Nx.Random.split(key, parts: horizon) |> Nx.to_list()

    initial_window =
      fit_result.observations |> Enum.take(-fit_result.order) |> left_pad(fit_result.order)

    {_window, draws} =
      Enum.reduce(step_keys, {initial_window, []}, fn step_key, {window, acc} ->
        mean = ar_mean(window, fit_result.coefficients)
        {z, _} = Distributions.normal_sample(Nx.tensor(step_key, type: Nx.type(key)))
        draw = mean + Nx.to_number(z) * fit_result.innovation_sd
        next_window = window |> tl() |> Kernel.++([draw])
        {next_window, [draw | acc]}
      end)

    Enum.reverse(draws)
  end

  defp ar_mean(window, coefficients) do
    lags = Enum.reverse(window)
    dot(coefficients, [1.0 | lags])
  end

  defp left_pad(values, desired_length) do
    missing = desired_length - length(values)
    if missing > 0, do: List.duplicate(0.0, missing) ++ values, else: values
  end

  defp summarize_paths(paths, alpha) do
    lower_p = alpha / 2.0
    upper_p = 1.0 - alpha / 2.0

    columns = Enum.zip_with(paths, & &1)

    %{
      mean: Enum.map(columns, &mean/1),
      sd: Enum.map(columns, &sample_sd/1),
      lower: Enum.map(columns, &quantile(&1, lower_p)),
      upper: Enum.map(columns, &quantile(&1, upper_p))
    }
  end

  defp mean(values) do
    if values == [], do: 0.0, else: Enum.sum(values) / length(values)
  end

  defp quantile(values, p) do
    sorted = Enum.sort(values)
    sorted_t = List.to_tuple(sorted)
    n = length(sorted)

    cond do
      n == 0 ->
        0.0

      p <= 0.0 ->
        elem(sorted_t, 0)

      p >= 1.0 ->
        elem(sorted_t, n - 1)

      true ->
        pos = (n - 1) * p
        i = trunc(:math.floor(pos))
        j = trunc(:math.ceil(pos))
        low = elem(sorted_t, i)
        high = elem(sorted_t, j)
        weight = pos - i
        low + (high - low) * weight
    end
  end

  defp validate_fit_result!(fit_result) do
    required = [:order, :coefficients, :innovation_sd, :observations, :num_samples]

    missing =
      Enum.reject(required, fn key ->
        Map.has_key?(fit_result, key)
      end)

    if missing != [] do
      raise ArgumentError, "invalid fit_result: missing keys #{inspect(missing)}"
    end
  end

  defp validate_positive_integer!(_name, value) when is_integer(value) and value > 0, do: :ok

  defp validate_positive_integer!(name, value) do
    raise ArgumentError, "#{name} must be a positive integer, got: #{inspect(value)}"
  end
end
