defmodule BstsNx.Validation do
  @moduledoc """
  Statistical validation suite for BSTS model quality.

  Provides six independent checks that test different assumptions of a
  Bayesian Structural Time Series lift estimation pipeline:

  1. **Prediction error** (RMSE & MAPE) — baseline fit quality
  2. **CI coverage** — uncertainty calibration
  3. **Durbin-Watson** — residual autocorrelation
  4. **Placebo test** — false positive detection
  5. **Effect stability** — sensitivity to window perturbation
  6. **Assess** — aggregate pass/warn/fail verdicts

  The first three functions operate on Nx tensors directly. The placebo and
  stability functions accept callback functions so callers can plug in any
  estimator (Kalman filter, seasonal regression, etc.) without this module
  depending on those implementations.
  """

  # -------------------------------------------------------------------
  # 1. Prediction Error
  # -------------------------------------------------------------------

  @doc """
  Computes prediction error metrics on indexed positions.

  `actual` and `baseline` are 1-D Nx tensors of the same length.
  `indices` is a list of 0-based integer positions (typically off-air minutes).

  Returns `%{rmse: float, mape: float, n: integer}` or `nil` when `indices`
  is empty.

  Note: The MAPE denominator is floored at 1.0 to avoid division by zero.
  For time series with values predominantly in [0, 1] (proportions,
  probabilities), MAPE effectively reduces to MAE. Consider using RMSE
  as the primary metric in such cases.
  """
  @spec prediction_error(Nx.Tensor.t(), Nx.Tensor.t(), [non_neg_integer()]) ::
          %{rmse: float(), mape: float(), n: integer()} | nil
  def prediction_error(_actual, _baseline, []), do: nil

  def prediction_error(actual, baseline, indices) when is_list(indices) do
    idx = Nx.tensor(indices, type: {:s, 64})
    a = Nx.take(actual, idx)
    b = Nx.take(baseline, idx)
    errors = Nx.subtract(a, b)

    rmse = errors |> Nx.pow(2) |> Nx.mean() |> Nx.sqrt() |> Nx.to_number()

    # MAPE: floor denominator at 1.0 to avoid division by zero
    abs_actual = a |> Nx.abs() |> Nx.max(1.0)
    mape = Nx.abs(errors) |> Nx.divide(abs_actual) |> Nx.mean() |> Nx.to_number()

    %{rmse: rmse, mape: mape, n: length(indices)}
  end

  # -------------------------------------------------------------------
  # 2. CI Coverage
  # -------------------------------------------------------------------

  @doc """
  Computes what fraction of actual values fall within the model's CI.

  `actual` and `baseline` are 1-D Nx tensors. `state_variances` is a 1-D
  tensor of filtered/smoothed state covariances. `obs_variance` is a scalar
  (float or 0-D tensor). `indices` selects which positions to evaluate.
  `h` is the observation coefficient (scalar or 1-D tensor); defaults to 1.0.
  When `h` is a 1-D tensor, the values at the given `indices` are used.

  ## Options

  - `:confidence_level` — CI confidence level as a float 0-1 (default: 0.95).
    The CI half-width is `z * sqrt(h^2 * state_variance + obs_variance)` where
    `z` is derived from the confidence level via the inverse normal CDF.

  Returns `%{pct: float, n_covered: integer, n_total: integer}` or `nil`
  when `indices` is empty.
  """
  @spec coverage(
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          Nx.Tensor.t(),
          float() | Nx.Tensor.t(),
          [non_neg_integer()],
          number() | Nx.Tensor.t(),
          keyword()
        ) :: %{pct: float(), n_covered: integer(), n_total: integer()} | nil
  def coverage(actual, baseline, state_variances, obs_variance, indices, h \\ 1.0, opts \\ [])
  def coverage(_actual, _baseline, _state_variances, _obs_variance, [], _h, _opts), do: nil

  def coverage(actual, baseline, state_variances, obs_variance, indices, h, opts)
      when is_list(indices) do
    confidence_level = Keyword.get(opts, :confidence_level, 0.95)
    alpha = 1.0 - confidence_level
    z = z_score(alpha)

    idx = Nx.tensor(indices, type: {:s, 64})
    a = Nx.take(actual, idx)
    b = Nx.take(baseline, idx)

    h_vals =
      case h do
        h when is_number(h) -> h
        %Nx.Tensor{} = ht -> if Nx.rank(ht) == 0, do: Nx.to_number(ht), else: Nx.take(ht, idx)
      end

    h_sq = if is_number(h_vals), do: h_vals * h_vals, else: Nx.multiply(h_vals, h_vals)
    scaled_vars = state_variances |> Nx.take(idx) |> Nx.multiply(h_sq)
    sds = scaled_vars |> Nx.add(obs_variance) |> Nx.sqrt()

    lower = Nx.subtract(b, Nx.multiply(z, sds))
    upper = Nx.add(b, Nx.multiply(z, sds))

    within =
      Nx.logical_and(
        Nx.greater_equal(a, lower),
        Nx.less_equal(a, upper)
      )

    n_covered = within |> Nx.sum() |> Nx.to_number() |> round()
    n_total = length(indices)

    %{pct: n_covered / max(n_total, 1), n_covered: n_covered, n_total: n_total}
  end

  @doc """
  Two-tailed z-score for significance level alpha.

  Delegates to the shared z-score helper. Returns `z` such that
  `P(-z < Z < z) = 1 - alpha` for a standard normal `Z`.

  ## Examples

      iex> BstsNx.Validation.z_score(0.05)
      1.959964
  """
  @spec z_score(float()) :: float()
  defdelegate z_score(alpha), to: BstsNx.Utils

  # -------------------------------------------------------------------
  # 3. Durbin-Watson
  # -------------------------------------------------------------------

  @doc """
  Computes the Durbin-Watson statistic on residuals at the given indices.

  `residuals` is a 1-D Nx tensor. `indices` selects which positions to use
  (typically off-air minutes). Requires at least 3 indices; returns `nil`
  otherwise.

  Returns `%{statistic: float, n: integer}`.
  """
  @spec durbin_watson(Nx.Tensor.t(), [non_neg_integer()]) ::
          %{statistic: float(), n: integer()} | nil
  def durbin_watson(_residuals, indices) when length(indices) < 3, do: nil

  def durbin_watson(residuals, indices) when is_list(indices) do
    idx = Nx.tensor(indices, type: {:s, 64})
    r = Nx.take(residuals, idx)
    n = Nx.size(r)

    r_prev = Nx.slice(r, [0], [n - 1])
    r_curr = Nx.slice(r, [1], [n - 1])
    diffs = Nx.subtract(r_curr, r_prev)

    numerator = diffs |> Nx.pow(2) |> Nx.sum() |> Nx.to_number()
    denominator = r |> Nx.pow(2) |> Nx.sum() |> Nx.to_number()

    statistic = if denominator > 0, do: numerator / denominator, else: 2.0

    %{statistic: statistic, n: length(indices)}
  end

  # -------------------------------------------------------------------
  # 4. Placebo Test
  # -------------------------------------------------------------------

  @doc """
  Runs a placebo (in-time) test to detect false positives.

  Places a fake intervention window of the same size as `on_air_indices`
  in the middle of the pre-period and runs `estimate_fn` on it.

  `sessions` is a plain Elixir list of numbers (one per time step).
  `on_air_indices` is a sorted list of 0-based integer indices.
  `estimate_fn` is a callback `fn(sessions, placebo_indices) -> result`
  where `result` must contain `:lift_sessions`, `:lift_pct`, and
  `:lift_ci95` (with `:lower`, `:upper`, `:sd`).

  Returns `nil` when the pre-period is too short (< 2x on-air length).
  """
  @spec placebo_test(
          [number()],
          [non_neg_integer()],
          (list(), list() -> map())
        ) :: map() | nil
  def placebo_test(sessions, on_air_indices, estimate_fn)
      when is_list(sessions) and is_list(on_air_indices) and is_function(estimate_fn, 2) do
    n_on_air = length(on_air_indices)
    t = length(sessions)

    if n_on_air == 0 or t == 0 do
      nil
    else
      on_air_set = MapSet.new(on_air_indices)

      # Gather pre-period indices (all indices NOT in on-air set)
      pre_indices =
        0..(t - 1)
        |> Enum.filter(fn i -> not MapSet.member?(on_air_set, i) end)
        |> Enum.sort()

      if length(pre_indices) < 2 * n_on_air do
        nil
      else
        # Pick a contiguous block from the middle of the pre-period
        mid = div(length(pre_indices), 2)
        start_pos = max(0, mid - div(n_on_air, 2))
        placebo_indices = Enum.slice(pre_indices, start_pos, n_on_air)

        result = estimate_fn.(sessions, placebo_indices)

        ci = result.lift_ci95
        is_significant = ci.lower > 0.0 or ci.upper < 0.0

        %{
          lift: result.lift_sessions,
          lift_pct: result.lift_pct,
          ci95: ci,
          is_significant: is_significant
        }
      end
    end
  end

  # -------------------------------------------------------------------
  # 5. Effect Stability
  # -------------------------------------------------------------------

  @doc """
  Tests whether the lift estimate is robust to small window perturbations.

  `main_lift` is the lift (in sessions) from the primary estimation.
  `estimate_at_window_fn` is a callback `fn(window_minutes) -> lift_sessions`
  that re-runs the estimator at a different window width.
  `window` is the current window width in minutes.
  `delta` is the perturbation size (default: 5 minutes).

  Returns a map with the base, perturbed lifts, and max percentage change.
  """
  @spec effect_stability(float(), (integer() -> float()), integer(), integer()) :: map()
  def effect_stability(main_lift, estimate_at_window_fn, window, delta \\ 5)
      when is_function(estimate_at_window_fn, 1) do
    low_window = max(5, window - delta)
    high_window = window + delta

    low_lift = estimate_at_window_fn.(low_window)
    high_lift = estimate_at_window_fn.(high_window)

    max_pct_change =
      if abs(main_lift) > 0.01 do
        max(abs(low_lift - main_lift), abs(high_lift - main_lift)) / abs(main_lift)
      else
        0.0
      end

    %{
      lift_base: main_lift,
      lift_low: low_lift,
      lift_high: high_lift,
      max_pct_change: max_pct_change,
      window_low: low_window,
      window_high: high_window
    }
  end

  # -------------------------------------------------------------------
  # 6. Assess (aggregate verdicts)
  # -------------------------------------------------------------------

  @doc """
  Applies pass/warn/fail thresholds to a map of validation results.

  Expects a map with optional keys: `:prediction_error`, `:coverage`,
  `:durbin_watson`, `:placebo`, `:effect_stability`. Each may be `nil`
  (skipped) or the result map from the corresponding function above.

  Returns a map of `{check_name, :pass | :warn | :fail}`.

  Thresholds (from the BSTS validation methodology):
  - MAPE: <= 10% pass, <= 20% warn, > 20% fail
  - Coverage: 85-99% pass, 80-85% or 99-100% warn, <80% fail
  - DW: 1.5-2.5 pass, 1.0-1.5 or 2.5-3.0 warn, outside fail
  - Placebo: CI includes zero AND |lift_pct| < 5% pass
  - Stability: max_pct_change <= 10% pass, <= 25% warn, > 25% fail
  """
  @spec assess(map()) :: map()
  def assess(results) when is_map(results) do
    %{
      prediction_error: assess_prediction_error(results[:prediction_error]),
      coverage: assess_coverage(results[:coverage]),
      durbin_watson: assess_durbin_watson(results[:durbin_watson]),
      placebo: assess_placebo(results[:placebo]),
      effect_stability: assess_effect_stability(results[:effect_stability])
    }
  end

  defp assess_prediction_error(nil), do: :skip
  defp assess_prediction_error(%{mape: mape}) when mape <= 0.10, do: :pass
  defp assess_prediction_error(%{mape: mape}) when mape <= 0.20, do: :warn
  defp assess_prediction_error(%{mape: _}), do: :fail

  defp assess_coverage(nil), do: :skip
  defp assess_coverage(%{pct: pct}) when pct >= 0.85 and pct <= 0.99, do: :pass
  defp assess_coverage(%{pct: pct}) when pct >= 0.80 and pct < 0.85, do: :warn
  defp assess_coverage(%{pct: pct}) when pct > 0.99 and pct <= 1.0, do: :warn
  defp assess_coverage(%{pct: _}), do: :fail

  defp assess_durbin_watson(nil), do: :skip
  defp assess_durbin_watson(%{statistic: dw}) when dw >= 1.5 and dw <= 2.5, do: :pass
  defp assess_durbin_watson(%{statistic: dw}) when dw >= 1.0 and dw < 1.5, do: :warn
  defp assess_durbin_watson(%{statistic: dw}) when dw > 2.5 and dw <= 3.0, do: :warn
  defp assess_durbin_watson(%{statistic: _}), do: :fail

  defp assess_placebo(nil), do: :skip

  defp assess_placebo(%{is_significant: sig, lift_pct: pct}) do
    if not sig and abs(pct) < 0.05, do: :pass, else: :fail
  end

  defp assess_effect_stability(nil), do: :skip
  defp assess_effect_stability(%{max_pct_change: c}) when c <= 0.10, do: :pass
  defp assess_effect_stability(%{max_pct_change: c}) when c <= 0.25, do: :warn
  defp assess_effect_stability(%{max_pct_change: _}), do: :fail

  # -------------------------------------------------------------------
  # 7. Known Lift Injection (calibration test)
  # -------------------------------------------------------------------

  @doc """
  Calibration test: injects a known effect and checks recovery.

  Generates synthetic data with a known `true_lift` added uniformly
  to the post-period, runs CausalImpact estimation, and checks whether
  the true cumulative effect falls within the posterior credible interval.

  ## Options
    - `:n_pre` — pre-period length (default: 100)
    - `:n_post` — post-period length (default: 20)
    - `:base_level` — baseline level of the series (default: 100.0)
    - `:noise_sd` — observation noise SD (default: 2.0)
    - `:num_samples` — MCMC samples (default: 200)
    - `:burn_in` — burn-in (default: 100)
    - `:seed` — PRNG seed for both data generation and MCMC
    - `:credible_level` — CI level, 0.0 to 1.0 (default: 0.95)
  """
  @spec known_lift_injection(float(), BstsNx.ModelSpec.t(), keyword()) :: map()
  def known_lift_injection(true_lift, %BstsNx.ModelSpec{} = spec, opts \\ []) do
    n_pre = Keyword.get(opts, :n_pre, 100)
    n_post = Keyword.get(opts, :n_post, 20)
    base_level = Keyword.get(opts, :base_level, 100.0)
    noise_sd = Keyword.get(opts, :noise_sd, 2.0)
    num_samples = Keyword.get(opts, :num_samples, 200)
    burn_in = Keyword.get(opts, :burn_in, 100)
    seed = Keyword.get(opts, :seed, 42)
    credible_level = Keyword.get(opts, :credible_level, 0.95)

    if not is_integer(n_pre) or n_pre <= 0 do
      raise ArgumentError, "n_pre must be a positive integer, got: #{inspect(n_pre)}"
    end

    if not is_integer(n_post) or n_post <= 0 do
      raise ArgumentError, "n_post must be a positive integer, got: #{inspect(n_post)}"
    end

    if not is_integer(num_samples) or num_samples <= 0 do
      raise ArgumentError,
            "num_samples must be a positive integer, got: #{inspect(num_samples)}"
    end

    if not is_integer(burn_in) or burn_in < 0 do
      raise ArgumentError, "burn_in must be a non-negative integer, got: #{inspect(burn_in)}"
    end

    if not is_number(noise_sd) or noise_sd < 0.0 do
      raise ArgumentError, "noise_sd must be a non-negative number, got: #{inspect(noise_sd)}"
    end

    if not is_number(credible_level) or credible_level <= 0.0 or credible_level >= 1.0 do
      raise ArgumentError,
            "credible_level must be a number in (0, 1), got: #{inspect(credible_level)}"
    end

    # Use functional :rand.seed_s to avoid mutating the process dictionary,
    # which would cause interference in concurrent ExUnit async tests.
    rng = :rand.seed_s(:exsss, {seed, seed + 1, seed + 2})

    # Generate pre-period: y_t = base_level + noise
    {pre, rng} =
      Enum.map_reduce(1..n_pre, rng, fn _, state ->
        {n, state} = :rand.normal_s(state)
        {base_level + n * noise_sd, state}
      end)

    # Per-step lift spread uniformly across post-period
    per_step_lift = true_lift / n_post

    # Generate post-period: y_t = base_level + per_step_lift + noise
    {post, _rng} =
      Enum.map_reduce(1..n_post, rng, fn _, state ->
        {n, state} = :rand.normal_s(state)
        {base_level + per_step_lift + n * noise_sd, state}
      end)

    obs = pre ++ post

    # Run CausalImpact estimation
    mcmc_opts = [
      num_samples: num_samples,
      burn_in: burn_in,
      seed: seed
    ]

    result =
      BstsNx.CausalImpact.estimate_structured(
        obs,
        {1, n_pre},
        {n_pre + 1, n_pre + n_post},
        spec,
        mcmc_opts
      )

    summary = BstsNx.CausalImpact.summary(result)

    # Compute credible interval from cumulative_effects samples
    cum_effects = result.cumulative_effects
    estimated_mean = summary.cumulative_effect.mean
    estimated_sd = summary.cumulative_effect.sd

    {lower, upper} = credible_interval(cum_effects, credible_level)

    covered = true_lift >= lower and true_lift <= upper

    relative_error = relative_error(estimated_mean, true_lift)

    %{
      true_lift: true_lift,
      estimated_mean: estimated_mean,
      estimated_sd: estimated_sd,
      lower: lower,
      upper: upper,
      covered: covered,
      relative_error: relative_error,
      n_pre: n_pre,
      n_post: n_post,
      num_samples: num_samples
    }
  end

  defp credible_interval(samples, level) do
    sorted = Enum.sort(samples)
    BstsNx.Utils.percentile_interval(sorted, length(sorted), 1.0 - level)
  end

  # Relative error is unstable when the ground-truth effect is near zero.
  # In that regime we report absolute error instead.
  defp relative_error(estimated_mean, true_lift) do
    if abs(true_lift) > 1.0e-6 do
      (estimated_mean - true_lift) / abs(true_lift)
    else
      estimated_mean - true_lift
    end
  end
end
