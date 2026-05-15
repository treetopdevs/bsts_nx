defmodule BstsNx.Diagnostics do
  require Logger
  alias BstsNx.Utils
  alias BstsNx.Validation

  @moduledoc """
  Diagnostic statistics for assessing convergence of MCMC chains.

  This module provides functions to compute the Gelman–Rubin R̂ statistic
  and the effective sample size (ESS) for parameters sampled by the Gibbs
  sampler. It also provides single-chain diagnostics:

    * split R̂ (`split_r_hat/1`)
    * Geweke Z-score (`geweke_z/2`)
    * highest posterior density interval (`hpd_interval/2`)

  These metrics help evaluate whether Markov chains have converged to the
  target distribution and how many samples are effectively independent.

  The functions expect a list of chains, where each chain is a list of
  sampled values for a single scalar parameter.  For example, to
  compute R̂ for the process variance sampled from multiple runs of
  `BstsNx.GibbsSampler`, you can extract the `:process_var` field
  from each chain of samples and pass the resulting list to
  `r_hat/1`.
  """

  @doc """
  Computes the Gelman–Rubin potential scale reduction factor (R̂).

  Accepts a list of chains, where each chain is a list of numeric
  samples (e.g. posterior draws of a variance).  All chains must
  have the same length.  Returns a float ≥ 1.0.  Values close to 1.0
  indicate convergence.  If only one chain is provided, the chains
  have length < 2, or the within-chain variance is zero, returns `:nan`.
  """
  @spec r_hat([list(number())]) :: float() | :nan
  def r_hat(chains) do
    case chain_stats(chains) do
      :nan ->
        :nan

      {_m, _n, w, _b, var_hat} ->
        if w == 0.0, do: :nan, else: :math.sqrt(var_hat / w)
    end
  end

  @doc """
  Estimates the effective sample size (ESS) for a set of chains.

  Uses the standard Gelman–Rubin formula ESS = m × n × W / V̂⁺, where
  `m` is the number of chains, `n` the number of samples per chain,
  `W` the within-chain variance, and V̂⁺ = (n−1)/n × W + B/n is the
  estimated marginal posterior variance.  The result is capped at
  `m × n` (the total number of draws).  Returns `:nan` if both W and B
  are zero (all samples identical) or only one chain is provided.
  """
  @spec effective_sample_size([list(number())]) :: float() | :nan
  def effective_sample_size(chains) do
    case chain_stats(chains) do
      :nan ->
        :nan

      {m, n, w, _b, var_hat} ->
        cond do
          var_hat == 0.0 -> :nan
          true -> min(m * n * w / var_hat, m * n * 1.0)
        end
    end
  end

  @doc """
  Estimates the effective sample size (ESS) from a single chain using
  Geyer's initial monotone sequence estimator.

  Accepts a list of numeric samples from a single MCMC chain.  The
  algorithm computes the autocorrelation at increasing lags, groups them
  into consecutive pairs, and truncates the sequence at the first
  non-positive pair sum, enforcing monotonicity.  This yields a
  conservative ESS estimate without requiring multiple chains.

  Returns a float ≥ 1.0.  If the chain has fewer than 4 samples,
  returns `:nan`.
  """
  @spec ess_single([number()]) :: float() | :nan
  def ess_single(chain) when is_list(chain) do
    n = length(chain)

    if n < 4 do
      :nan
    else
      mean = Enum.sum(chain) / n

      # Compute variance (gamma_0) and autocovariances
      gamma_0 =
        Enum.reduce(chain, 0.0, fn x, acc -> acc + (x - mean) * (x - mean) end) / n

      if gamma_0 == 0.0 do
        :nan
      else
        # Precompute centered values for O(1) access
        centered = :array.from_list(Enum.map(chain, &(&1 - mean)))

        # Initial positive sequence: sum consecutive pairs of autocovariances
        # gamma(2k) + gamma(2k+1) for k = 0, 1, 2, ...
        max_k = div(n - 2, 2)

        # Build pair sums and apply initial positive sequence truncation
        {tau, _} =
          Enum.reduce_while(0..max_k, {0.0, :infinity}, fn k, {tau_acc, prev_pair} ->
            pair_sum =
              autocovariance_at(centered, n, 2 * k) +
                autocovariance_at(centered, n, 2 * k + 1)

            # Truncate at first non-positive pair sum
            if pair_sum <= 0.0 do
              {:halt, {tau_acc, prev_pair}}
            else
              # Enforce monotonicity
              pair_sum_mono = min(pair_sum, prev_pair)
              {:cont, {tau_acc + pair_sum_mono, pair_sum_mono}}
            end
          end)

        # ESS = n / (1 + 2 * sum_of_pair_sums / gamma_0)
        # But tau already contains gamma(0) + gamma(1) in the first pair
        # So we need: ESS = n * gamma_0 / (-gamma_0 + 2 * tau)
        denom = -gamma_0 + 2.0 * tau

        if denom <= 1.0e-15 do
          # Denominator near zero or negative means autocorrelation structure is
          # degenerate or tau didn't accumulate enough; return minimum ESS.
          1.0
        else
          ess = n * gamma_0 / denom
          min(max(ess, 1.0), n * 1.0)
        end
      end
    end
  end

  @doc """
  Computes split-chain R-hat from a single MCMC chain by splitting
  it in half and treating each half as an independent chain.

  This is a standard diagnostic from Gelman et al. that detects
  non-stationarity within a single chain.  The chain is split at
  its midpoint and R-hat is computed on the two halves using
  `r_hat/1`.

  Returns a float ≥ 1.0 or `:nan` if the chain is too short (< 4).
  """
  @spec split_r_hat([number()]) :: float() | :nan
  def split_r_hat(chain) when is_list(chain) do
    n = length(chain)

    if n < 4 do
      :nan
    else
      half = div(n, 2)
      first_half = Enum.take(chain, half)
      second_half = Enum.drop(chain, n - half)
      r_hat([first_half, second_half])
    end
  end

  @doc """
  Computes Geweke's convergence Z-score for a single chain.

  Compares the mean of the first part of the chain to the mean of the
  last part, using spectral-density-adjusted standard errors (Bartlett
  window estimate at frequency zero).

  ## Options

    * `:first_frac` - fraction of samples for the first window (default: 0.1)
    * `:last_frac` - fraction of samples for the last window (default: 0.5)
    * `:max_lag` - truncation lag for spectral variance estimate
      (default: `trunc(sqrt(window_n))`)

  Returns a Z-score (float) or `:nan` when windows are too short or
  non-finite.
  """
  @spec geweke_z([number()], keyword()) :: float() | :nan
  def geweke_z(chain, opts \\ []) when is_list(chain) do
    n = length(chain)
    first_frac = Keyword.get(opts, :first_frac, 0.1)
    last_frac = Keyword.get(opts, :last_frac, 0.5)
    max_lag_opt = Keyword.get(opts, :max_lag)

    if n < 20 do
      :nan
    else
      validate_fraction!(:first_frac, first_frac)
      validate_fraction!(:last_frac, last_frac)

      if first_frac + last_frac >= 1.0 do
        raise ArgumentError,
              "first_frac + last_frac must be < 1.0, got #{first_frac + last_frac}"
      end

      n_first = max(trunc(Float.floor(n * first_frac)), 2)
      n_last = max(trunc(Float.floor(n * last_frac)), 2)

      if n_first + n_last > n do
        :nan
      else
        first = Enum.take(chain, n_first)
        last = Enum.drop(chain, n - n_last)

        m_first = Enum.sum(first) / n_first
        m_last = Enum.sum(last) / n_last

        max_lag_first = max_lag_for_window(max_lag_opt, n_first)
        max_lag_last = max_lag_for_window(max_lag_opt, n_last)

        s0_first = spectral_density_zero(first, max_lag_first)
        s0_last = spectral_density_zero(last, max_lag_last)

        cond do
          s0_first <= 0.0 or s0_last <= 0.0 ->
            :nan

          true ->
            denom = :math.sqrt(s0_first / n_first + s0_last / n_last)
            if denom <= 1.0e-15, do: :nan, else: (m_first - m_last) / denom
        end
      end
    end
  end

  @doc """
  Returns whether a Geweke test passes at significance level `:alpha`.

  A pass means `|z| <= z_(1 - alpha/2)`.

  ## Options

    * `:alpha` - two-sided significance level (default: 0.05)

  All options accepted by `geweke_z/2` are also accepted.
  """
  @spec geweke_pass?([number()], keyword()) :: boolean() | :nan
  def geweke_pass?(chain, opts \\ []) do
    alpha = Keyword.get(opts, :alpha, 0.05)
    Validation.validate_alpha!(alpha)

    case geweke_z(chain, opts) do
      :nan -> :nan
      z -> abs(z) <= Utils.z_score(alpha)
    end
  end

  @doc """
  Computes a highest posterior density (HPD) interval from scalar samples.

  Uses the shortest interval containing at least `level` posterior mass.
  Returns `{lower, upper}` or `:nan` for empty input.
  """
  @spec hpd_interval([number()], float()) :: {float(), float()} | :nan
  def hpd_interval(samples, level \\ 0.95)

  def hpd_interval([], _level), do: :nan

  def hpd_interval(samples, level) when is_list(samples) do
    if not is_number(level) or level <= 0.0 or level >= 1.0 do
      raise ArgumentError, "level must be in (0, 1), got: #{inspect(level)}"
    end

    sorted = Enum.sort(samples)
    sorted_t = List.to_tuple(sorted)
    n = length(sorted)

    if n == 1 do
      v = elem(sorted_t, 0)
      {v, v}
    else
      window = max(trunc(Float.ceil(level * n)), 1)

      if window >= n do
        {elem(sorted_t, 0), elem(sorted_t, n - 1)}
      else
        {best_low, best_high, _best_width} =
          Enum.reduce(0..(n - window), nil, fn i, best ->
            low = elem(sorted_t, i)
            high = elem(sorted_t, i + window - 1)
            width = high - low

            case best do
              nil ->
                {low, high, width}

              {_, _, best_width} when width < best_width ->
                {low, high, width}

              _ ->
                best
            end
          end)

        {best_low, best_high}
      end
    end
  end

  # -- Shared helper -----------------------------------------------------------

  # Computes between-chain variance B, within-chain variance W, and the
  # estimated marginal posterior variance V̂⁺ from a list of chains.
  # Returns {m, n, w, b, var_hat} or :nan if chains are insufficient.
  defp chain_stats(chains) do
    m = length(chains)

    if m < 2 do
      :nan
    else
      lengths = Enum.map(chains, &length/1)
      n = Enum.min(lengths)
      max_n = Enum.max(lengths)

      if max_n > n do
        Logger.warning(
          "Diagnostics: chains have unequal lengths (min=#{n}, max=#{max_n}); " <>
            "truncating all chains to #{n} samples"
        )
      end

      if n < 2 do
        :nan
      else
        chains_n = Enum.map(chains, &Enum.take(&1, n))
        means = Enum.map(chains_n, fn c -> Enum.sum(c) / n end)
        overall_mean = Enum.sum(means) / m

        b =
          Enum.reduce(means, 0.0, fn mean_i, acc ->
            d = mean_i - overall_mean
            acc + d * d
          end) * n / (m - 1)

        w =
          (Enum.zip(chains_n, means)
           |> Enum.reduce(0.0, fn {c, mean_c}, acc ->
             var_c =
               Enum.reduce(c, 0.0, fn x, acc2 ->
                 d = x - mean_c
                 acc2 + d * d
               end) / (n - 1)

             acc + var_c
           end)) / m

        var_hat = (n - 1) / n * w + b / n
        {m, n, w, b, var_hat}
      end
    end
  end

  defp validate_fraction!(name, value) do
    if not is_number(value) or value <= 0.0 or value >= 1.0 do
      raise ArgumentError, "#{name} must be in (0, 1), got: #{inspect(value)}"
    end
  end

  defp max_lag_for_window(nil, n), do: max(trunc(:math.sqrt(n)), 1)
  defp max_lag_for_window(max_lag, _n) when is_integer(max_lag) and max_lag >= 0, do: max_lag

  defp max_lag_for_window(max_lag, _n) do
    raise ArgumentError, "max_lag must be a non-negative integer, got: #{inspect(max_lag)}"
  end

  # Bartlett-window spectral density estimate at frequency zero.
  defp spectral_density_zero(samples, max_lag) do
    n = length(samples)
    mean = Enum.sum(samples) / n
    centered = :array.from_list(Enum.map(samples, &(&1 - mean)))
    gamma_0 = autocovariance_at(centered, n, 0)

    if gamma_0 <= 0.0 do
      0.0
    else
      used_lag = min(max_lag, n - 1)

      gamma_sum =
        if used_lag <= 0 do
          0.0
        else
          Enum.reduce(1..used_lag, 0.0, fn lag, acc ->
            weight = 1.0 - lag / (used_lag + 1)
            acc + 2.0 * weight * autocovariance_at(centered, n, lag)
          end)
        end

      s0 = gamma_0 + gamma_sum
      if s0 > 0.0, do: s0, else: 0.0
    end
  end

  defp autocovariance_at(_centered, n, lag) when lag >= n, do: 0.0

  defp autocovariance_at(centered, n, lag) do
    Enum.reduce(0..(n - lag - 1), 0.0, fn i, acc ->
      acc + :array.get(i, centered) * :array.get(i + lag, centered)
    end) / n
  end
end
