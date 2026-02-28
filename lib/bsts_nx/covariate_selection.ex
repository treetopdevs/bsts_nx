defmodule BstsNx.CovariateSelection do
  @moduledoc """
  Covariate selection for BSTS models.

  Two methods are supported:

    * `:pearson` (default) - pre-period correlation screening
    * `:spike_and_slab` - Bayesian stochastic-search variable selection
      returning posterior inclusion probabilities (PIP)

  Both methods return selected column indices and an optional selected
  covariate matrix.
  """

  @default_expected_controls 3.0

  @doc """
  Selects covariates from a pool.

  ## Parameters
    * `target` - list or {n} tensor of target values (pre-period only)
    * `candidates` - {n, p} tensor of candidate covariate values
    * `opts` - keyword options

  ## Options
    * `:method` - `:pearson` (default) or `:spike_and_slab`

  For `:pearson`:

    * `:threshold` - minimum absolute correlation (default: 0.1)
    * `:max_controls` - maximum number to select (default: 10)

  For `:spike_and_slab`:

    * `:num_samples` - retained posterior draws (default: 500)
    * `:burn_in` - warmup iterations (default: 500)
    * `:thin` - thinning interval (default: 1)
    * `:seed` - PRNG seed (default: 42)
    * `:prior_inclusion` - prior inclusion probability π (default: `min(0.5, 3 / p)`)
    * `:spike_sd` - spike prior SD τ0 (default: 0.05)
    * `:slab_sd` - slab prior SD τ1 (default: 1.0)
    * `:sigma_prior_shape` - inverse-gamma shape for residual variance (default: 2.0)
    * `:sigma_prior_scale` - inverse-gamma scale for residual variance (default: 1.0)
    * `:pip_threshold` - PIP threshold for inclusion (default: 0.5)
    * `:max_controls` - maximum number to select (default: all columns)

  ## Returns
    * `:selected_indices` - 0-based selected column indices
    * `:selected_matrix` - {n, k} tensor of selected covariates, or nil if none selected

  Additional output:

    * Pearson method includes `:correlations`
    * Spike-and-slab method includes `:pip`, `:posterior_mean_beta`,
      and `:num_posterior_samples`

  ## Examples

      iex> target = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      iex> cands = Nx.tensor([[2.0, 5.0], [4.0, 5.0], [6.0, 5.0], [8.0, 5.0], [10.0, 5.0]])
      iex> result = BstsNx.CovariateSelection.select(target, cands, threshold: 0.5)
      iex> result.selected_indices
      [0]
  """
  @spec select([number()] | Nx.t(), Nx.t(), keyword()) :: map()
  def select(target, candidates, opts \\ [])

  def select(target, candidates, opts) do
    case Keyword.get(opts, :method, :pearson) do
      :pearson ->
        select_pearson(target, candidates, opts)

      :spike_and_slab ->
        select_spike_slab(target, candidates, opts)

      other ->
        raise ArgumentError, "unknown selection method #{inspect(other)}"
    end
  end

  @doc """
  Bayesian spike-and-slab covariate selection.

  Uses a Gibbs sampler for a sparse linear model:

      y = Xβ + ε, ε ~ N(0, σ²)
      β_j | γ_j ~ N(0, τ_γ²),  γ_j ~ Bernoulli(π)

  where `τ_γ` is `spike_sd` for excluded variables and `slab_sd` for
  included variables.

  Returns posterior inclusion probabilities (PIP) and selected indices.
  """
  @spec select_spike_slab([number()] | Nx.t(), Nx.t(), keyword()) :: map()
  def select_spike_slab(target, candidates, opts \\ []) do
    target_t = normalize_target(target)
    n = Nx.axis_size(target_t, 0)
    {n_cand, p} = Nx.shape(candidates)

    if n_cand != n do
      raise ArgumentError,
            "candidates rows (#{n_cand}) must match target length (#{n})"
    end

    if p == 0 do
      %{
        method: :spike_and_slab,
        selected_indices: [],
        pip: [],
        posterior_mean_beta: [],
        selected_matrix: nil,
        num_posterior_samples: 0
      }
    else
      num_samples = Keyword.get(opts, :num_samples, 500)
      burn_in = Keyword.get(opts, :burn_in, 500)
      thin = Keyword.get(opts, :thin, 1)
      seed = Keyword.get(opts, :seed, 42)
      spike_sd = Keyword.get(opts, :spike_sd, 0.05)
      slab_sd = Keyword.get(opts, :slab_sd, 1.0)
      sigma_prior_shape = Keyword.get(opts, :sigma_prior_shape, 2.0)
      sigma_prior_scale = Keyword.get(opts, :sigma_prior_scale, 1.0)

      prior_inclusion =
        Keyword.get(opts, :prior_inclusion, min(0.5, @default_expected_controls / p))

      pip_threshold = Keyword.get(opts, :pip_threshold, 0.5)
      max_controls = Keyword.get(opts, :max_controls, p)

      validate_spike_and_slab_opts!(
        num_samples,
        burn_in,
        thin,
        prior_inclusion,
        spike_sd,
        slab_sd,
        sigma_prior_shape,
        sigma_prior_scale,
        pip_threshold,
        max_controls
      )

      y_raw = Nx.to_flat_list(target_t)
      x_rows = Nx.to_flat_list(candidates) |> Enum.chunk_every(p) |> Enum.map(&List.to_tuple/1)
      {y, _mean_y, _sd_y} = standardize(y_raw)
      y_var = variance(y)

      if y_var <= 1.0e-12 do
        pip = Enum.map(0..(p - 1), fn j -> {j, 0.0} end)

        %{
          method: :spike_and_slab,
          selected_indices: [],
          pip: pip,
          posterior_mean_beta: List.duplicate(0.0, p),
          selected_matrix: nil,
          num_posterior_samples: 0
        }
      else
        x_cols =
          Enum.map(0..(p - 1), fn j ->
            col = Enum.map(x_rows, &elem(&1, j))
            {scaled, _mean, _sd} = standardize(col)
            scaled
          end)

        ss = Enum.map(x_cols, &dot(&1, &1))

        expected_controls = max(round(prior_inclusion * p), 1)

        active =
          x_cols
          |> Enum.with_index()
          |> Enum.map(fn {col, j} -> {j, abs(dot(col, y))} end)
          |> Enum.sort_by(fn {_j, score} -> score end, :desc)
          |> Enum.take(expected_controls)
          |> Enum.map(&elem(&1, 0))
          |> MapSet.new()

        beta0 = List.duplicate(0.0, p)
        gamma0 = Enum.map(0..(p - 1), fn j -> if MapSet.member?(active, j), do: 1, else: 0 end)
        sigma2_0 = max(y_var, 1.0e-6)
        total_iters = burn_in + num_samples * thin
        rng0 = :rand.seed_s(:exsss, seed_triplet(seed))

        {_, _, _, _, counts, beta_sums, kept, _rng} =
          Enum.reduce(
            1..total_iters,
            {beta0, gamma0, sigma2_0, y, List.duplicate(0, p), List.duplicate(0.0, p), 0, rng0},
            fn iter, {beta, gamma, sigma2, residual, counts, beta_sums, kept, rng} ->
              {beta_new, gamma_new, residual_new, rng} =
                gibbs_sweep(
                  beta,
                  gamma,
                  residual,
                  sigma2,
                  x_cols,
                  ss,
                  prior_inclusion,
                  spike_sd,
                  slab_sd,
                  rng
                )

              {sigma2_new, rng} =
                resample_sigma2(
                  residual_new,
                  beta_new,
                  gamma_new,
                  spike_sd,
                  slab_sd,
                  sigma_prior_shape,
                  sigma_prior_scale,
                  n,
                  rng
                )

              if iter > burn_in and rem(iter - burn_in, thin) == 0 do
                counts_new = Enum.zip_with([counts, gamma_new], fn [a, b] -> a + b end)
                beta_sums_new = Enum.zip_with([beta_sums, beta_new], fn [a, b] -> a + b end)

                {beta_new, gamma_new, sigma2_new, residual_new, counts_new, beta_sums_new,
                 kept + 1, rng}
              else
                {beta_new, gamma_new, sigma2_new, residual_new, counts, beta_sums, kept, rng}
              end
            end
          )

        pips =
          counts
          |> Enum.with_index()
          |> Enum.map(fn {cnt, j} ->
            pip = if kept > 0, do: cnt / kept, else: 0.0
            {j, pip}
          end)
          |> Enum.sort_by(fn {_j, pip} -> pip end, :desc)

        selected_indices =
          pips
          |> Enum.filter(fn {_j, pip} -> pip >= pip_threshold end)
          |> Enum.take(max_controls)
          |> Enum.map(&elem(&1, 0))

        posterior_mean_beta =
          if kept > 0 do
            Enum.map(beta_sums, &(&1 / kept))
          else
            List.duplicate(0.0, p)
          end

        %{
          method: :spike_and_slab,
          selected_indices: selected_indices,
          pip: pips,
          posterior_mean_beta: posterior_mean_beta,
          selected_matrix: build_selected_matrix(candidates, n, selected_indices),
          num_posterior_samples: kept
        }
      end
    end
  end

  defp select_pearson(target, candidates, opts) do
    threshold = Keyword.get(opts, :threshold, 0.1)
    max_controls = Keyword.get(opts, :max_controls, 10)

    # Flatten target to 1-D
    target_t = normalize_target(target)
    target_t = Nx.flatten(target_t)
    n = Nx.axis_size(target_t, 0)

    {n_cand, p} = Nx.shape(candidates)

    if n_cand != n do
      raise ArgumentError,
            "candidates rows (#{n_cand}) must match target length (#{n})"
    end

    if p == 0 do
      %{
        selected_indices: [],
        correlations: [],
        selected_matrix: nil
      }
    else
      t_centered = Nx.subtract(target_t, Nx.mean(target_t))
      c_centered = Nx.subtract(candidates, Nx.mean(candidates, axes: [0]))

      numerators = Nx.dot(t_centered, c_centered)
      t_ss = Nx.sum(Nx.multiply(t_centered, t_centered))
      col_ss = Nx.sum(Nx.multiply(c_centered, c_centered), axes: [0])
      denom = Nx.sqrt(Nx.multiply(t_ss, col_ss))
      near_zero = Nx.less(denom, 1.0e-15)
      corrs = Nx.select(near_zero, Nx.broadcast(0.0, {p}), Nx.divide(numerators, denom))

      correlations =
        corrs
        |> Nx.to_flat_list()
        |> Enum.with_index()
        |> Enum.map(fn {corr, j} -> {j, corr} end)

      # Sort by absolute correlation descending
      sorted = Enum.sort_by(correlations, fn {_j, corr} -> abs(corr) end, :desc)

      # Filter by threshold and take top max_controls
      selected =
        sorted
        |> Enum.filter(fn {_j, corr} -> abs(corr) >= threshold end)
        |> Enum.take(max_controls)

      selected_indices = Enum.map(selected, fn {j, _corr} -> j end)

      %{
        method: :pearson,
        selected_indices: selected_indices,
        correlations: sorted,
        selected_matrix: build_selected_matrix(candidates, n, selected_indices)
      }
    end
  end

  defp normalize_target(target) do
    case target do
      %Nx.Tensor{} -> target
      list when is_list(list) -> Nx.tensor(list)
    end
  end

  defp build_selected_matrix(_candidates, _n, []), do: nil

  defp build_selected_matrix(candidates, n, selected_indices) do
    _ = n
    Nx.take(candidates, Nx.tensor(selected_indices, type: {:s, 64}), axis: 1)
  end

  @doc """
  Computes Pearson correlation between two equal-length series.

  Returns 0.0 if either series has zero variance.

  ## Examples

      iex> x = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      iex> y = Nx.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
      iex> BstsNx.CovariateSelection.pearson_correlation(x, y)
      1.0

      iex> x = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      iex> y = Nx.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
      iex> BstsNx.CovariateSelection.pearson_correlation(x, y)
      -1.0
  """
  @spec pearson_correlation(Nx.t(), Nx.t()) :: float()
  def pearson_correlation(x, y) do
    x_mean = Nx.mean(x)
    y_mean = Nx.mean(y)

    x_centered = Nx.subtract(x, x_mean)
    y_centered = Nx.subtract(y, y_mean)

    numerator = Nx.multiply(x_centered, y_centered) |> Nx.sum() |> Nx.to_number()

    x_ss = x_centered |> Nx.multiply(x_centered) |> Nx.sum() |> Nx.to_number()
    y_ss = y_centered |> Nx.multiply(y_centered) |> Nx.sum() |> Nx.to_number()

    denominator = :math.sqrt(x_ss * y_ss)

    if denominator < 1.0e-15 do
      0.0
    else
      numerator / denominator
    end
  end

  defp validate_spike_and_slab_opts!(
         num_samples,
         burn_in,
         thin,
         prior_inclusion,
         spike_sd,
         slab_sd,
         sigma_prior_shape,
         sigma_prior_scale,
         pip_threshold,
         max_controls
       ) do
    if not is_integer(num_samples) or num_samples <= 0 do
      raise ArgumentError, "num_samples must be a positive integer, got: #{inspect(num_samples)}"
    end

    if not is_integer(burn_in) or burn_in < 0 do
      raise ArgumentError, "burn_in must be a non-negative integer, got: #{inspect(burn_in)}"
    end

    if not is_integer(thin) or thin <= 0 do
      raise ArgumentError, "thin must be a positive integer, got: #{inspect(thin)}"
    end

    if not is_number(prior_inclusion) or prior_inclusion <= 0.0 or prior_inclusion >= 1.0 do
      raise ArgumentError,
            "prior_inclusion must be in (0, 1), got: #{inspect(prior_inclusion)}"
    end

    if not is_number(spike_sd) or spike_sd <= 0.0 do
      raise ArgumentError, "spike_sd must be > 0, got: #{inspect(spike_sd)}"
    end

    if not is_number(slab_sd) or slab_sd <= 0.0 do
      raise ArgumentError, "slab_sd must be > 0, got: #{inspect(slab_sd)}"
    end

    if slab_sd <= spike_sd do
      raise ArgumentError, "slab_sd must be larger than spike_sd"
    end

    if not is_number(sigma_prior_shape) or sigma_prior_shape <= 0.0 do
      raise ArgumentError,
            "sigma_prior_shape must be > 0, got: #{inspect(sigma_prior_shape)}"
    end

    if not is_number(sigma_prior_scale) or sigma_prior_scale <= 0.0 do
      raise ArgumentError,
            "sigma_prior_scale must be > 0, got: #{inspect(sigma_prior_scale)}"
    end

    if not is_number(pip_threshold) or pip_threshold < 0.0 or pip_threshold > 1.0 do
      raise ArgumentError, "pip_threshold must be in [0, 1], got: #{inspect(pip_threshold)}"
    end

    if not is_integer(max_controls) or max_controls < 0 do
      raise ArgumentError,
            "max_controls must be a non-negative integer, got: #{inspect(max_controls)}"
    end
  end

  defp seed_triplet(seed) do
    seed_int =
      cond do
        is_integer(seed) -> abs(seed)
        true -> :erlang.phash2(seed)
      end

    base = max(seed_int, 1)
    {base, base + 1, base + 2}
  end

  defp gibbs_sweep(
         beta,
         gamma,
         residual,
         sigma2,
         x_cols,
         ss,
         prior_inclusion,
         spike_sd,
         slab_sd,
         rng
       ) do
    {beta_rev, gamma_rev, residual_new, rng} =
      Enum.zip([beta, gamma, x_cols, ss])
      |> Enum.reduce({[], [], residual, rng}, fn {beta_j, gamma_j, x_j, ss_j},
                                                 {beta_acc, gamma_acc, r_curr, rng_acc} ->
        r_plus = add_scaled(r_curr, x_j, beta_j)

        tau = if gamma_j == 1, do: slab_sd, else: spike_sd
        precision = ss_j + 1.0 / (tau * tau)
        var_j = sigma2 / max(precision, 1.0e-12)
        mean_j = dot(x_j, r_plus) / max(precision, 1.0e-12)

        {z, rng_acc} = :rand.normal_s(rng_acc)
        beta_new = mean_j + :math.sqrt(max(var_j, 1.0e-12)) * z

        r_new = add_scaled(r_plus, x_j, -beta_new)

        pip_j = inclusion_probability(beta_new, sigma2, prior_inclusion, spike_sd, slab_sd)
        {u, rng_acc} = :rand.uniform_s(rng_acc)
        gamma_new = if u < pip_j, do: 1, else: 0

        {[beta_new | beta_acc], [gamma_new | gamma_acc], r_new, rng_acc}
      end)

    {Enum.reverse(beta_rev), Enum.reverse(gamma_rev), residual_new, rng}
  end

  defp inclusion_probability(beta_j, sigma2, prior_inclusion, spike_sd, slab_sd) do
    log_prior_ratio = :math.log(prior_inclusion / (1.0 - prior_inclusion))
    log_sd_ratio = 0.5 * :math.log(spike_sd * spike_sd / (slab_sd * slab_sd))

    quad =
      beta_j * beta_j / (2.0 * max(sigma2, 1.0e-12)) *
        (1.0 / (spike_sd * spike_sd) - 1.0 / (slab_sd * slab_sd))

    # Stable logistic transform
    logit = log_prior_ratio + log_sd_ratio + quad

    cond do
      logit >= 35.0 -> 1.0
      logit <= -35.0 -> 0.0
      true -> 1.0 / (1.0 + :math.exp(-logit))
    end
  end

  defp resample_sigma2(
         residual,
         beta,
         gamma,
         spike_sd,
         slab_sd,
         prior_shape,
         prior_scale,
         n,
         rng
       ) do
    rss = dot(residual, residual)

    prior_quad =
      Enum.zip(beta, gamma)
      |> Enum.reduce(0.0, fn {beta_j, gamma_j}, acc ->
        tau = if gamma_j == 1, do: slab_sd, else: spike_sd
        acc + beta_j * beta_j / (tau * tau)
      end)

    p = length(beta)
    shape = prior_shape + (n + p) / 2.0
    scale = prior_scale + 0.5 * (rss + prior_quad)
    {sigma2, rng} = inv_gamma_sample(shape, scale, rng)
    {max(sigma2, 1.0e-9), rng}
  end

  defp standardize(values) do
    n = length(values)
    mean = Enum.sum(values) / max(n, 1)

    centered = Enum.map(values, &(&1 - mean))
    var = variance_from_centered(centered)
    sd = :math.sqrt(max(var, 0.0))

    if sd <= 1.0e-12 do
      {List.duplicate(0.0, n), mean, 0.0}
    else
      {Enum.map(centered, &(&1 / sd)), mean, sd}
    end
  end

  defp variance(values) do
    n = length(values)

    if n < 2 do
      0.0
    else
      mean = Enum.sum(values) / n

      Enum.reduce(values, 0.0, fn x, acc ->
        diff = x - mean
        acc + diff * diff
      end) / (n - 1)
    end
  end

  defp variance_from_centered(centered) do
    n = length(centered)
    if n < 2, do: 0.0, else: Enum.reduce(centered, 0.0, fn x, acc -> acc + x * x end) / (n - 1)
  end

  defp dot(xs, ys) do
    Enum.zip_reduce(xs, ys, 0.0, fn x, y, acc -> acc + x * y end)
  end

  defp add_scaled(vec, x, scale) do
    Enum.zip(vec, x)
    |> Enum.map(fn {v, x_i} -> v + scale * x_i end)
  end

  defp inv_gamma_sample(alpha, beta, rng) do
    {gamma, rng} = gamma_sample(alpha, 1.0, rng)
    gamma_safe = max(gamma, 1.0e-300)
    {beta / gamma_safe, rng}
  end

  # Marsaglia-Tsang gamma sampler with explicit state threading.
  defp gamma_sample(alpha, _scale, _rng) when alpha <= 0.0 do
    raise ArgumentError, "gamma_sample requires alpha > 0, got: #{alpha}"
  end

  defp gamma_sample(alpha, scale, rng) when alpha < 1.0 do
    {u, rng} = :rand.uniform_s(rng)
    {sample, rng} = gamma_sample(alpha + 1.0, scale, rng)
    {sample * :math.pow(u, 1.0 / alpha), rng}
  end

  defp gamma_sample(alpha, scale, rng) do
    d = alpha - 1.0 / 3.0
    c = 1.0 / :math.sqrt(9.0 * d)
    gamma_sample_loop(d, c, scale, rng, 10_000)
  end

  defp gamma_sample_loop(_d, _c, _scale, _rng, 0) do
    raise RuntimeError, "gamma sampling failed to converge after 10,000 iterations"
  end

  defp gamma_sample_loop(d, c, scale, rng, remaining) do
    {x, rng} = :rand.normal_s(rng)
    v = 1.0 + c * x

    if v <= 0.0 do
      gamma_sample_loop(d, c, scale, rng, remaining - 1)
    else
      v3 = v * v * v
      {u, rng} = :rand.uniform_s(rng)

      accept =
        u < 1.0 - 0.0331 * x * x * x * x or
          :math.log(u) < 0.5 * x * x + d * (1.0 - v3 + :math.log(v3))

      if accept do
        {d * v3 * scale, rng}
      else
        gamma_sample_loop(d, c, scale, rng, remaining - 1)
      end
    end
  end
end
