defmodule BstsNx.Diagnostics do
  require Logger

  @moduledoc """
  Diagnostic statistics for assessing convergence of MCMC chains.

  This module provides functions to compute the Gelman–Rubin R̂ statistic
  and the effective sample size (ESS) for parameters sampled by the
  Gibbs sampler.  These metrics help evaluate whether Markov chains
  have converged to the target distribution and how many samples are
  effectively independent.

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

        # Compute autocovariance at lag k using precomputed centered values
        gamma = fn lag ->
          Enum.reduce(0..(n - lag - 1), 0.0, fn i, acc ->
            acc + :array.get(i, centered) * :array.get(i + lag, centered)
          end) / n
        end

        # Initial positive sequence: sum consecutive pairs of autocovariances
        # gamma(2k) + gamma(2k+1) for k = 0, 1, 2, ...
        max_k = div(n - 2, 2)

        # Build pair sums and apply initial positive sequence truncation
        {tau, _} =
          Enum.reduce_while(0..max_k, {0.0, :infinity}, fn k, {tau_acc, prev_pair} ->
            pair_sum = gamma.(2 * k) + gamma.(2 * k + 1)

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
            acc + :math.pow(mean_i - overall_mean, 2)
          end) * n / (m - 1)

        w =
          Enum.reduce(chains_n, 0.0, fn c, acc ->
            mean_c = Enum.sum(c) / n

            var_c =
              Enum.reduce(c, 0.0, fn x, acc2 -> acc2 + :math.pow(x - mean_c, 2) end) / (n - 1)

            acc + var_c
          end) / m

        var_hat = (n - 1) / n * w + b / n
        {m, n, w, b, var_hat}
      end
    end
  end
end
