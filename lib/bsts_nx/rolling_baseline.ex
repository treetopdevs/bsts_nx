defmodule BstsNx.RollingBaseline do
  @moduledoc """
  Rolling-window baseline estimation for TV spot attribution.

  Wraps Components, GibbsSampler, and forward simulation into a
  fit -> predict workflow that produces counterfactual baselines
  compatible with `BstsNx.SpotAttributor.attribute/4`.

  ## Workflow

  1. `build_spec/2` -- Composes local linear trend + seasonal model
  2. `fit/3` -- Runs structured Gibbs sampler on pre-intervention data
  3. `counterfactual/3` -- Forward-simulates to produce counterfactual
  4. `fit_and_predict/3` -- Convenience chain of all three steps

  ## Warm-Start Support

  When fitting consecutive rolling windows, passing the previous fit
  result as `:warm_start` initializes the sampler from the posterior
  of the previous window.  This improves mixing and allows reducing
  the burn-in period.

  ## Design

  This module works with indexed tensors only -- no time concepts.
  Time-to-index mapping is the caller's responsibility.
  """

  alias BstsNx.Components
  alias BstsNx.Diagnostics
  alias BstsNx.GibbsSampler
  alias BstsNx.ModelSpec
  import Nx.Defn

  @default_num_seasons 96
  @default_num_samples 200
  @default_burn_in 100
  @default_warm_start_burn_in 50
  @default_obs_var 1.0

  @type convergence :: %{
          r_hat_obs_var: float() | :nan,
          ess_obs_var: float() | :nan,
          r_hat_process_var: float() | :nan,
          ess_process_var: float() | :nan,
          converged: boolean()
        }

  @type fit_result :: %{
          posterior_samples: [GibbsSampler.structured_result()],
          spec: ModelSpec.t(),
          num_observations: non_neg_integer(),
          final_states: [Nx.t()],
          final_covs: [Nx.t()],
          convergence: convergence(),
          n_regression_dims: non_neg_integer()
        }

  @doc """
  Composes a ModelSpec from local linear trend + seasonal components,
  with optional regression.

  ## Options

    * `:num_seasons` -- number of seasonal periods (default: #{@default_num_seasons})
    * `:obs_var` -- initial observation variance (default: #{@default_obs_var})
    * `:trend_var_level` -- trend level variance (default: 0.01)
    * `:trend_var_slope` -- trend slope variance (default: 0.001)
    * `:trend_prior_shape` -- inverse-gamma shape for trend (default: 1.0)
    * `:trend_prior_scale` -- inverse-gamma scale for trend (default: 0.01)
    * `:seasonal_var` -- seasonal innovation variance (default: 0.001)
    * `:seasonal_prior_shape` -- inverse-gamma shape for seasonal (default: 1.0)
    * `:seasonal_prior_scale` -- inverse-gamma scale for seasonal (default: 0.001)
    * `:regressors` -- optional `{n_pre, p}` Nx tensor of covariates
    * `:regression_var_beta` -- per-coefficient process variance (default: 0.01)
    * `:regression_prior_shape` -- inverse-gamma shape for regression (default: 1.0)
    * `:regression_prior_scale` -- inverse-gamma scale for regression (default: 0.001)
  """
  @spec build_spec(non_neg_integer(), keyword()) :: ModelSpec.t()
  def build_spec(num_observations, opts \\ []) do
    num_seasons = Keyword.get(opts, :num_seasons, @default_num_seasons)
    obs_var = Keyword.get(opts, :obs_var, @default_obs_var)

    trend_spec =
      Components.local_linear_trend_spec(
        var_level: Keyword.get(opts, :trend_var_level, 0.01),
        var_slope: Keyword.get(opts, :trend_var_slope, 0.001),
        obs_var: obs_var,
        prior_shape: Keyword.get(opts, :trend_prior_shape, 1.0),
        prior_scale: Keyword.get(opts, :trend_prior_scale, 0.01)
      )

    seasonal_spec =
      Components.seasonal_spec(num_seasons,
        process_var: Keyword.get(opts, :seasonal_var, 0.001),
        obs_var: obs_var,
        prior_shape: Keyword.get(opts, :seasonal_prior_shape, 1.0),
        prior_scale: Keyword.get(opts, :seasonal_prior_scale, 0.001)
      )

    trend_seasonal = Components.compose_specs(trend_spec, seasonal_spec)

    case Keyword.get(opts, :regressors) do
      nil ->
        trend_seasonal

      regressors ->
        {regressors_t, _p} = normalize_regressors!(regressors, num_observations)

        reg_spec =
          Components.regression_spec(regressors_t,
            var_beta: Keyword.get(opts, :regression_var_beta, 0.01),
            obs_var: obs_var,
            prior_shape: Keyword.get(opts, :regression_prior_shape, 1.0),
            prior_scale: Keyword.get(opts, :regression_prior_scale, 0.001)
          )

        Components.compose_specs(trend_seasonal, reg_spec)
    end
  end

  @doc """
  Fits the model on pre-intervention observations via structured Gibbs sampling.

  Returns a fit result containing posterior samples, the spec used,
  the final filtered states/covariances for forward simulation,
  and convergence diagnostics.

  ## Options

    * `:num_samples` -- posterior draws (default: #{@default_num_samples})
    * `:burn_in` -- burn-in iterations (default: #{@default_burn_in}, or
      #{@default_warm_start_burn_in} when `:warm_start` is provided)
    * `:seed` -- PRNG seed for reproducibility
    * `:thin` -- thinning interval (default: 1)
    * `:warm_start` -- a previous `fit_result` map. When provided, the
      spec's `obs_var` and `q_specs` initials are set to posterior means,
      and `x0`/`p0` are initialized from the previous terminal state and
      covariance when compatible. The default burn-in is also reduced
    * `:n_regression_dims` -- number of regression dimensions (default: 0).
      Stored in the fit result for downstream use by `counterfactual/3`.
  """
  @spec fit(Nx.t() | [number()], ModelSpec.t(), keyword()) :: fit_result()
  def fit(observations, %ModelSpec{} = spec, opts \\ []) do
    obs_list = to_obs_list(observations)
    warm_start = Keyword.get(opts, :warm_start)

    {effective_spec, default_burn_in} =
      if warm_start do
        {apply_warm_start(spec, warm_start), @default_warm_start_burn_in}
      else
        {spec, @default_burn_in}
      end

    num_samples = Keyword.get(opts, :num_samples, @default_num_samples)
    burn_in = Keyword.get(opts, :burn_in, default_burn_in)

    if not is_integer(num_samples) or num_samples <= 0 do
      raise ArgumentError, "num_samples must be a positive integer, got: #{inspect(num_samples)}"
    end

    if not is_integer(burn_in) or burn_in < 0 do
      raise ArgumentError, "burn_in must be a non-negative integer, got: #{inspect(burn_in)}"
    end

    sampler_opts =
      opts
      |> Keyword.take([:seed, :key, :thin])
      |> Keyword.put(:burn_in, burn_in)

    samples =
      GibbsSampler.sample_structured(obs_list, effective_spec, num_samples, sampler_opts)

    last_sample = List.last(samples)
    final_states = last_sample.states
    final_covs = last_sample.state_covs

    convergence = compute_convergence(samples, effective_spec)

    n_regression_dims =
      case {Keyword.fetch(opts, :n_regression_dims), Keyword.get(opts, :regressors)} do
        {{:ok, n_reg}, nil} when is_integer(n_reg) and n_reg >= 0 ->
          n_reg

        {{:ok, n_reg}, regressors} when is_integer(n_reg) and n_reg >= 0 ->
          {_regressors_t, p} = normalize_regressors!(regressors)

          if n_reg != p do
            raise ArgumentError,
                  ":n_regression_dims (#{n_reg}) must match regressors columns p=#{p}"
          end

          n_reg

        {{:ok, n_reg}, _} ->
          raise ArgumentError,
                ":n_regression_dims must be a non-negative integer, got: #{inspect(n_reg)}"

        {:error, nil} ->
          0

        {:error, regressors} ->
          {_regressors_t, p} = normalize_regressors!(regressors)
          p
      end

    %{
      posterior_samples: samples,
      spec: effective_spec,
      num_observations: length(obs_list),
      final_states: final_states,
      final_covs: final_covs,
      convergence: convergence,
      n_regression_dims: n_regression_dims
    }
  end

  @doc """
  Produces a counterfactual prediction from a fit result.

  Forward-simulates from the final filtered state to generate
  `horizon` steps of counterfactual baseline in the format
  expected by `SpotAttributor.attribute/4`.

  ## Options

    * `:post_regressors` -- `{horizon, p}` Nx tensor of regressor values
      for the post-period. Required when the model includes regression
      (time-varying H). Omit for non-regression models.

  Returns `%{mean: [float], variance: [float], obs_variance: float}`.
  """
  @spec counterfactual(fit_result(), non_neg_integer(), keyword()) :: %{
          mean: [float()],
          variance: [float()],
          obs_variance: float()
        }
  def counterfactual(fit_result, horizon, opts \\ [])

  def counterfactual(_fit_result, 0, _opts) do
    %{mean: [], variance: [], obs_variance: 0.0}
  end

  def counterfactual(%{posterior_samples: samples, spec: spec} = fit_result, horizon, opts)
      when horizon > 0 do
    if samples == [] do
      raise ArgumentError,
            "fit_result.posterior_samples must be non-empty; ensure fit/3 was run with num_samples > 0"
    end

    n_state = Nx.axis_size(spec.f, 0)
    post_regressors = Keyword.get(opts, :post_regressors)
    n_reg = Map.get(fit_result, :n_regression_dims, 0)

    h_series = build_h_for_counterfactual(spec.h, n_state, n_reg, post_regressors, horizon)

    per_sample =
      Enum.map(samples, fn sample ->
        final_state = List.last(sample.states)

        final_cov =
          case sample do
            %{state_covs: covs} when is_list(covs) and covs != [] ->
              List.last(covs)

            _ ->
              Nx.broadcast(0.0, {n_state, n_state})
          end

        q_matrix = sample.q_matrix
        obs_var = Nx.to_number(sample.obs_var)

        {means, vars} =
          case h_series do
            %Nx.Tensor{} = h_row ->
              forward_simulate(final_state, final_cov, spec.f, h_row, q_matrix, horizon)

            h_list when is_list(h_list) ->
              forward_simulate_h(final_state, final_cov, spec.f, h_list, q_matrix, horizon)
          end

        {Nx.as_type(means, {:f, 64}), Nx.as_type(vars, {:f, 64}), obs_var}
      end)

    n = length(samples)
    mean_tensors = Enum.map(per_sample, &elem(&1, 0))
    var_tensors = Enum.map(per_sample, &elem(&1, 1))
    obs_var_acc = Enum.reduce(per_sample, 0.0, fn {_, _, ov}, acc -> acc + ov end)

    mean_stack = Nx.stack(mean_tensors)
    var_stack = Nx.stack(var_tensors)
    mean = Nx.mean(mean_stack, axes: [0])
    mean_sq = Nx.mean(Nx.multiply(mean_stack, mean_stack), axes: [0])
    process_var = Nx.mean(var_stack, axes: [0])
    between_var = Nx.max(Nx.subtract(mean_sq, Nx.multiply(mean, mean)), 0.0)

    %{
      mean: Nx.to_flat_list(mean),
      variance: between_var |> Nx.add(process_var) |> Nx.to_flat_list(),
      obs_variance: obs_var_acc / n
    }
  end

  @doc """
  Convenience function: builds spec, fits model, and produces counterfactual.

  ## Options

  Accepts all options from `build_spec/2` and `fit/3`, including
  `:warm_start` for warm-starting from a previous fit result.

  When `:regressors` is provided as a `{T_total, p}` tensor (where
  T_total = pre_period + post_period_length), the pre-period portion
  is forwarded to `build_spec/2` and `fit/3`, and the post-period
  portion is passed to `counterfactual/3` as `:post_regressors`.

  Returns `{counterfactual, fit_result}`.
  """
  @spec fit_and_predict(Nx.t() | [number()], non_neg_integer(), keyword()) ::
          {%{mean: [float()], variance: [float()], obs_variance: float()}, fit_result()}
  def fit_and_predict(observations, post_period_length, opts \\ []) do
    obs_list = to_obs_list(observations)
    n_pre = length(obs_list)

    {build_opts, cf_opts} = split_regressor_opts(opts, n_pre, post_period_length)

    spec = build_spec(n_pre, build_opts)
    fit_result = fit(obs_list, spec, build_opts)
    cf = counterfactual(fit_result, post_period_length, cf_opts)
    {cf, fit_result}
  end

  # -- Warm-start helpers --

  # Applies warm-start initialization by extracting posterior means from
  # the previous fit and updating the spec's initial values accordingly.
  defp apply_warm_start(%ModelSpec{} = spec, %{posterior_samples: []}), do: spec

  defp apply_warm_start(%ModelSpec{} = spec, %{posterior_samples: prev_samples} = warm_start) do
    # Posterior mean of obs_var
    n = length(prev_samples)

    obs_var_mean =
      prev_samples
      |> Enum.map(fn s -> Nx.to_number(s.obs_var) end)
      |> Enum.sum()
      |> Kernel./(n)

    # Posterior mean of each Q diagonal entry
    q_means = posterior_q_means(prev_samples, spec.q_specs)

    updated_q_specs =
      Enum.map(spec.q_specs, fn qs ->
        case Map.fetch(q_means, qs.dim_index) do
          {:ok, mean_val} -> %{qs | initial: mean_val}
          :error -> qs
        end
      end)

    {x0, p0} = warm_start_state_init(warm_start, prev_samples, spec)

    %{spec | obs_var: obs_var_mean, q_specs: updated_q_specs, x0: x0, p0: p0}
  end

  # Computes the posterior mean for each Q diagonal entry across all samples.
  # Returns %{dim_index => mean_value}.
  defp posterior_q_means(_samples, []), do: %{}

  defp posterior_q_means(samples, q_specs) do
    n = length(samples)
    all_diags = Enum.map(samples, fn s -> Nx.take_diagonal(s.q_matrix) |> Nx.to_flat_list() end)

    Enum.reduce(q_specs, %{}, fn qs, acc ->
      dim = qs.dim_index

      sum =
        Enum.reduce(all_diags, 0.0, fn diag, s ->
          val = Enum.at(diag, dim, 0.0)
          s + val
        end)

      Map.put(acc, dim, sum / n)
    end)
  end

  # -- Convergence diagnostics --

  # Computes convergence diagnostics from the posterior samples.
  # Uses split R-hat and ESS for observation variance and every process
  # variance component; reports conservative summaries across Q dimensions.
  defp compute_convergence(samples, %ModelSpec{} = spec) do
    obs_var_chain = Enum.map(samples, fn s -> Nx.to_number(s.obs_var) end)
    all_diags = Enum.map(samples, fn s -> Nx.take_diagonal(s.q_matrix) |> Nx.to_flat_list() end)

    r_hat_obs = Diagnostics.split_r_hat(obs_var_chain)
    ess_obs = Diagnostics.ess_single(obs_var_chain)

    process_diags =
      Enum.map(spec.q_specs, fn qs ->
        chain = Enum.map(all_diags, &Enum.at(&1, qs.dim_index, 0.0))

        %{
          dim_index: qs.dim_index,
          r_hat: Diagnostics.split_r_hat(chain),
          ess: Diagnostics.ess_single(chain)
        }
      end)

    r_hat_proc = summarize_process_r_hat(process_diags)
    ess_proc = summarize_process_ess(process_diags)

    process_converged =
      process_diags == [] or Enum.all?(process_diags, &converged?(&1.r_hat, &1.ess))

    converged = converged?(r_hat_obs, ess_obs) and process_converged

    %{
      r_hat_obs_var: r_hat_obs,
      ess_obs_var: ess_obs,
      r_hat_process_var: r_hat_proc,
      ess_process_var: ess_proc,
      converged: converged
    }
  end

  defp summarize_process_r_hat([]), do: :nan

  defp summarize_process_r_hat(process_diags) do
    values = Enum.map(process_diags, & &1.r_hat)
    if Enum.all?(values, &is_number/1), do: Enum.max(values), else: :nan
  end

  defp summarize_process_ess([]), do: :nan

  defp summarize_process_ess(process_diags) do
    values = Enum.map(process_diags, & &1.ess)
    if Enum.all?(values, &is_number/1), do: Enum.min(values), else: :nan
  end

  defp warm_start_state_init(warm_start, prev_samples, %ModelSpec{} = spec) do
    state_candidate =
      warm_start |> Map.get(:final_states) |> last_tensor() || sample_terminal_state(prev_samples)

    cov_candidate =
      warm_start |> Map.get(:final_covs) |> last_tensor() || sample_terminal_cov(prev_samples)

    x0 = if state_shape_matches?(state_candidate, spec), do: state_candidate, else: spec.x0
    p0 = if cov_shape_matches?(cov_candidate, spec), do: cov_candidate, else: spec.p0
    {x0, p0}
  end

  defp sample_terminal_state(samples) do
    samples
    |> List.last()
    |> Map.get(:states, [])
    |> last_tensor()
  end

  defp sample_terminal_cov(samples) do
    samples
    |> List.last()
    |> Map.get(:state_covs, [])
    |> last_tensor()
  end

  defp last_tensor(list) when is_list(list) and list != [] do
    case List.last(list) do
      %Nx.Tensor{} = tensor -> tensor
      _ -> nil
    end
  end

  defp last_tensor(_), do: nil

  defp state_shape_matches?(%Nx.Tensor{} = state, %ModelSpec{} = spec) do
    Nx.shape(state) == Nx.shape(spec.x0)
  end

  defp state_shape_matches?(_, _), do: false

  defp cov_shape_matches?(%Nx.Tensor{} = cov, %ModelSpec{} = spec) do
    Nx.shape(cov) == Nx.shape(spec.p0)
  end

  defp cov_shape_matches?(_, _), do: false

  # Checks whether a single parameter has converged based on R-hat and ESS.
  defp converged?(r_hat, ess) when is_number(r_hat) and is_number(ess) do
    r_hat < 1.1 and ess > 100
  end

  defp converged?(_r_hat, _ess), do: false

  # -- Private helpers --

  defp forward_simulate(state, cov, f, h_row, q_matrix, horizon) do
    h_vec = Nx.flatten(h_row)
    h_steps = Nx.broadcast(h_vec, {horizon, Nx.axis_size(h_vec, 0)})
    forward_simulate_defn(state, cov, f, h_steps, q_matrix)
  end

  # Forward-simulates with per-step time-varying H (list of {1, n_state} tensors).
  defp forward_simulate_h(state, cov, f, h_list, q_matrix, _horizon) do
    h_steps = h_steps_tensor(h_list)
    forward_simulate_defn(state, cov, f, h_steps, q_matrix)
  end

  defp h_steps_tensor(h_list) do
    stacked = Nx.stack(h_list)
    horizon = Nx.axis_size(stacked, 0)

    case Nx.rank(stacked) do
      2 ->
        stacked

      3 ->
        Nx.reshape(stacked, {horizon, Nx.axis_size(stacked, 2)})
    end
  end

  defnp forward_simulate_defn(state, cov, f, h_steps, q_matrix) do
    x0 = Nx.flatten(state)
    type = Nx.type(x0)
    p0 = Nx.as_type(cov, type)
    f_t = Nx.as_type(f, type)
    f_t_t = Nx.transpose(f_t)
    q_t = Nx.as_type(q_matrix, type)
    h_t = Nx.as_type(h_steps, type)
    horizon = Nx.axis_size(h_t, 0)
    state_dim = Nx.axis_size(h_t, 1)
    zero = Nx.tensor(0.0, type: type)
    means = Nx.broadcast(zero, {horizon})
    variances = Nx.broadcast(zero, {horizon})

    {_, means_out, variances_out, _, _, _, _, _, _} =
      while {i = Nx.tensor(0, type: {:s, 64}), means_acc = means, vars_acc = variances, x = x0,
             p = p0, f_in = f_t, f_trans = f_t_t, q_in = q_t, h_in = h_t},
            i < horizon do
        x_pred = Nx.dot(f_in, x)
        p_pred = Nx.add(Nx.dot(Nx.dot(f_in, p), f_trans), q_in)
        h_i = Nx.take(h_in, Nx.reshape(i, {1}), axis: 0) |> Nx.reshape({state_dim})
        mean = Nx.dot(h_i, x_pred)
        raw_variance = Nx.dot(h_i, Nx.dot(p_pred, h_i))
        variance = Nx.select(Nx.less(raw_variance, zero), zero, raw_variance)

        means_new = Nx.put_slice(means_acc, [i], Nx.reshape(mean, {1}))
        vars_new = Nx.put_slice(vars_acc, [i], Nx.reshape(variance, {1}))

        {i + 1, means_new, vars_new, x_pred, p_pred, f_in, f_trans, q_in, h_in}
      end

    {means_out, variances_out}
  end

  # Builds the H observation matrix/series for counterfactual forward simulation.
  # Returns either a static {1, n_state} tensor or a list of {1, n_state} tensors.
  defp build_h_for_counterfactual(h, _n_state, 0, nil, horizon) when is_list(h) do
    # Non-regression time-varying H: replicate last entry for forecast
    List.duplicate(List.last(h), horizon)
  end

  defp build_h_for_counterfactual(h, _n_state, _n_reg, nil, _horizon) when is_list(h) do
    raise ArgumentError,
          "RollingBaseline.counterfactual/3 requires :post_regressors when the model has " <>
            "regression (n_regression_dims > 0). Provide a {horizon, p} tensor."
  end

  defp build_h_for_counterfactual(%Nx.Tensor{} = h, _n_state, 0, nil, _horizon) do
    h
  end

  defp build_h_for_counterfactual(%Nx.Tensor{} = _h, _n_state, n_reg, post_regressors, _horizon) do
    msg =
      if post_regressors != nil do
        "RollingBaseline.counterfactual/3 received :post_regressors but the model has static H. " <>
          "Build the spec with :regressors to enable regression."
      else
        "RollingBaseline.counterfactual/3: model has static H but n_regression_dims=#{n_reg}. " <>
          "This is inconsistent — regression models should have time-varying H."
      end

    raise ArgumentError, msg
  end

  defp build_h_for_counterfactual(h, n_state, n_reg, post_regressors, horizon) do
    {post_regressors_t, p_post} = normalize_regressors!(post_regressors, horizon)

    n_reg_effective =
      cond do
        is_integer(n_reg) and n_reg > 0 and n_reg == p_post ->
          n_reg

        is_integer(n_reg) and n_reg > 0 and n_reg != p_post ->
          raise ArgumentError,
                "post_regressors has p=#{p_post} columns, but fit_result.n_regression_dims=#{n_reg}"

        true ->
          p_post
      end

    if n_reg_effective > n_state do
      raise ArgumentError,
            "post_regressors has p=#{n_reg_effective} columns, but model state dimension is #{n_state}"
    end

    n_structural = n_state - n_reg_effective

    h0 =
      case h do
        list when is_list(list) -> List.last(list)
        %Nx.Tensor{} = h_tensor -> h_tensor
      end

    structural_h = Nx.slice(h0, [0, 0], [1, n_structural])
    structural_h_batched = Nx.broadcast(structural_h, {horizon, n_structural})
    combined = Nx.concatenate([structural_h_batched, post_regressors_t], axis: 1)

    BstsNx.Utils.tensor_rows_to_row_matrices(combined)
  end

  # Splits :regressors from opts into pre-period (for build_spec/fit) and
  # post-period (for counterfactual), along with n_regression_dims metadata.
  defp split_regressor_opts(opts, n_pre, post_period_length) do
    case Keyword.get(opts, :regressors) do
      nil ->
        {opts, []}

      full_regressors ->
        {full_regressors_t, p} = normalize_regressors!(full_regressors)
        {t_total, ^p} = Nx.shape(full_regressors_t)

        expected_t_total = n_pre + post_period_length

        if t_total != expected_t_total do
          raise ArgumentError,
                "full regressors must have #{expected_t_total} rows (n_pre=#{n_pre} + post_period_length=#{post_period_length}), " <>
                  "got #{t_total}"
        end

        pre_regressors = Nx.slice(full_regressors_t, [0, 0], [n_pre, p])
        post_regressors = Nx.slice(full_regressors_t, [n_pre, 0], [post_period_length, p])

        build_opts =
          opts
          |> Keyword.put(:regressors, pre_regressors)
          |> Keyword.put(:n_regression_dims, p)

        cf_opts = [post_regressors: post_regressors]

        {build_opts, cf_opts}
    end
  end

  defp to_obs_list(%Nx.Tensor{} = t), do: Nx.to_flat_list(t)
  defp to_obs_list(list) when is_list(list), do: Enum.map(list, &to_float/1)

  defp normalize_regressors!(regressors, expected_rows \\ nil) do
    regressors_t =
      case regressors do
        %Nx.Tensor{} = t -> t
        list when is_list(list) -> Nx.tensor(list)
      end

    if Nx.rank(regressors_t) != 2 do
      raise ArgumentError,
            "regressors must be a rank-2 tensor with shape {T, p}, got: #{inspect(Nx.shape(regressors_t))}"
    end

    {t_len, p} = Nx.shape(regressors_t)

    if expected_rows != nil and t_len != expected_rows do
      raise ArgumentError,
            "regressors must have #{expected_rows} rows (T), got #{t_len}"
    end

    {regressors_t, p}
  end

  defp to_float(v) when is_float(v), do: v
  defp to_float(v) when is_integer(v), do: v * 1.0
  defp to_float(%Nx.Tensor{} = t), do: Nx.to_number(t)
end
