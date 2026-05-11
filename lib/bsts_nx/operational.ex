defmodule BstsNx.Operational do
  @moduledoc """
  Forecast-first operational counterfactual and attribution engine.

  This module is the fast Elixir/Nx lane for data pipelines and interactive
  UIs. It uses fixed-variance Kalman filtering on the pre-period, forecasts a
  post-period counterfactual without observing post-period outcomes, and then
  attributes observed lift to spot windows.
  """

  import Nx.Defn

  alias BstsNx.CausalImpact
  alias BstsNx.Execution
  alias BstsNx.KalmanFilter
  alias BstsNx.ModelSpec
  alias BstsNx.SpotAttributor

  defmodule Prepared do
    @moduledoc """
    Prepared operational model artifacts for repeated low-latency runs.
    """

    @enforce_keys [
      :spec,
      :pre_period,
      :post_period,
      :baseline,
      :return,
      :f,
      :pre_h,
      :post_h,
      :q,
      :r,
      :x0,
      :p0,
      :state_dim,
      :pre_count,
      :post_count,
      :gap_count,
      :scalar?,
      :static_scalar_h?,
      :backend
    ]
    defstruct @enforce_keys
  end

  @type baseline :: :forecast | :smooth
  @type return_format :: :lists | :tensors
  @type prepared :: %Prepared{}

  @doc """
  Prepares fixed model artifacts for repeated operational runs.
  """
  @spec prepare(
          ModelSpec.t(),
          {pos_integer(), pos_integer()},
          {pos_integer(), pos_integer()},
          keyword()
        ) ::
          prepared()
  def prepare(%ModelSpec{} = spec, pre_period, post_period, opts \\ []) do
    spec = ModelSpec.validate!(spec)
    {pre_start, pre_end, pre_count} = validate_period!(pre_period, :pre_period)
    {post_start, post_end, post_count} = validate_period!(post_period, :post_period)

    if post_start <= pre_end do
      raise ArgumentError,
            "post_period must start after pre_period ends, got #{inspect(pre_period)} and #{inspect(post_period)}"
    end

    state_dim = Nx.axis_size(spec.f, 0)
    baseline = resolve_baseline!(Keyword.get(opts, :baseline, :forecast))
    return = resolve_return!(Keyword.get(opts, :return, :lists))

    %Prepared{
      spec: spec,
      pre_period: pre_period,
      post_period: post_period,
      baseline: baseline,
      return: return,
      f: spec.f |> Nx.as_type({:f, 64}),
      pre_h: h_rows_for_period!(spec.h, pre_start, pre_end, state_dim),
      post_h: h_rows_for_period!(spec.h, post_start, post_end, state_dim),
      q: q_matrix(spec, state_dim),
      r: Nx.tensor(spec.obs_var, type: {:f, 64}),
      x0: spec.x0 |> Nx.as_type({:f, 64}) |> Nx.flatten(),
      p0: spec.p0 |> Nx.as_type({:f, 64}),
      state_dim: state_dim,
      pre_count: pre_count,
      post_count: post_count,
      gap_count: post_start - pre_end - 1,
      scalar?: scalar_spec?(spec),
      static_scalar_h?: static_scalar_h?(spec.h),
      backend: inspect(Nx.default_backend())
    }
  end

  @doc """
  Runs a prepared operational counterfactual and spot attribution workflow.
  """
  @spec run(prepared(), [number()] | Nx.t(), [SpotAttributor.spot()], keyword()) :: map()
  def run(%Prepared{} = prepared, observations, spots, opts \\ []) do
    validate_spot_windows!(spots, prepared.post_count)
    return = resolve_return!(Keyword.get(opts, :return, prepared.return))
    attribution_opts = attribution_opts(opts)

    {elapsed_us, {summary, method_used}} =
      Execution.measure(fn ->
        {summary, method_used} =
          forecast_summary(prepared, observations, Keyword.put(opts, :return, return))

        {summary, method_used}
      end)

    counterfactual = counterfactual_from_summary(summary)

    attribution_result =
      summary.actual
      |> vector_to_list()
      |> SpotAttributor.attribute(
        spots,
        counterfactual_to_lists(counterfactual),
        attribution_opts
      )

    %{
      attributions: attribution_result,
      causal_impact: nil,
      summary: summary,
      counterfactual: counterfactual,
      execution:
        Execution.metadata(:operational, :elixir_nx, method_used, elapsed_us,
          baseline: prepared.baseline,
          prepared?: true
        )
    }
  end

  @doc """
  Prepares and runs an operational workflow in one call.
  """
  @spec run(
          [number()] | Nx.t(),
          {pos_integer(), pos_integer()},
          {pos_integer(), pos_integer()},
          [SpotAttributor.spot()],
          ModelSpec.t(),
          keyword()
        ) :: map()
  def run(observations, pre_period, post_period, spots, %ModelSpec{} = spec, opts \\ []) do
    validate_spot_windows!(spots, post_count(post_period))

    {elapsed_us, {summary, method_used, prepared}} =
      Execution.measure(fn ->
        prepared = prepare(spec, pre_period, post_period, opts)
        {summary, method_used} = forecast_summary(prepared, observations, opts)
        {summary, method_used, prepared}
      end)

    counterfactual = counterfactual_from_summary(summary)

    attribution_result =
      summary.actual
      |> vector_to_list()
      |> SpotAttributor.attribute(
        spots,
        counterfactual_to_lists(counterfactual),
        attribution_opts(opts)
      )

    %{
      attributions: attribution_result,
      causal_impact: nil,
      summary: summary,
      counterfactual: counterfactual,
      execution:
        Execution.metadata(:operational, :elixir_nx, method_used, elapsed_us,
          baseline: prepared.baseline,
          prepared?: false
        )
    }
  end

  @doc """
  Forecasts a prepared operational counterfactual without spot attribution.
  """
  @spec forecast(prepared(), [number()] | Nx.t(), keyword()) :: map()
  def forecast(%Prepared{} = prepared, observations, opts \\ []) do
    {elapsed_us, {summary, method_used}} =
      Execution.measure(fn -> forecast_summary(prepared, observations, opts) end)

    %{
      summary: summary,
      counterfactual: counterfactual_from_summary(summary),
      execution:
        Execution.metadata(:operational, :elixir_nx, method_used, elapsed_us,
          baseline: prepared.baseline,
          prepared?: true
        )
    }
  end

  defp forecast_summary(%Prepared{baseline: :smooth} = prepared, observations, opts) do
    return = resolve_return!(Keyword.get(opts, :return, prepared.return))
    alpha = Keyword.get(opts, :alpha, 0.05)
    indices = intervention_indices(prepared.post_period)

    {summary, method_used} =
      if prepared.scalar? and prepared.static_scalar_h? do
        opts =
          [
            f: prepared.f |> Nx.reshape({}) |> Nx.to_number(),
            h: prepared.post_h |> Nx.slice([0, 0], [1, 1]) |> Nx.reshape({}) |> Nx.to_number(),
            q: prepared.q |> Nx.slice([0, 0], [1, 1]) |> Nx.reshape({}) |> Nx.to_number(),
            r: Nx.to_number(prepared.r),
            x0: prepared.x0 |> Nx.reshape({}) |> Nx.to_number(),
            p0: prepared.p0 |> Nx.reshape({}) |> Nx.to_number(),
            alpha: alpha
          ]

        {CausalImpact.estimate_from_filter(observations, indices, opts), :scalar_smooth_filter}
      else
        {CausalImpact.estimate_structured_from_filter(observations, indices, prepared.spec,
           alpha: alpha
         ), :structured_smooth_filter}
      end

    {maybe_tensor_summary(summary, return), method_used}
  end

  defp forecast_summary(%Prepared{baseline: :forecast} = prepared, observations, opts) do
    return = resolve_return!(Keyword.get(opts, :return, prepared.return))
    alpha = Keyword.get(opts, :alpha, 0.05)
    validate_alpha!(alpha)

    obs = observations_tensor!(observations, prepared.post_period)
    {pre_start, _pre_end} = prepared.pre_period
    {post_start, _post_end} = prepared.post_period

    pre_obs = Nx.slice(obs, [pre_start - 1], [prepared.pre_count])
    actual = Nx.slice(obs, [post_start - 1], [prepared.post_count])

    {xs, ps} =
      KalmanFilter.filter_defn_multi(
        pre_obs,
        prepared.f,
        prepared.pre_h,
        prepared.q,
        prepared.r,
        prepared.x0,
        prepared.p0
      )

    last_idx = prepared.pre_count - 1
    final_x = xs |> Nx.slice([last_idx, 0], [1, prepared.state_dim]) |> Nx.squeeze(axes: [0])

    final_p =
      ps
      |> Nx.slice([last_idx, 0, 0], [1, prepared.state_dim, prepared.state_dim])
      |> Nx.squeeze(axes: [0])

    {baseline, baseline_variance} =
      forecast_multi_defn(
        final_x,
        final_p,
        prepared.f,
        prepared.q,
        prepared.r,
        prepared.post_h,
        Nx.tensor(prepared.gap_count, type: {:s, 64})
      )

    summary =
      build_summary(actual, baseline, baseline_variance, Nx.to_number(prepared.r), alpha, return)

    {summary,
     if(prepared.scalar?, do: :scalar_forecast_filter, else: :structured_forecast_filter)}
  end

  defn forecast_multi_defn(x0, p0, f, q, _r, h_post, gap_count) do
    post_count = Nx.axis_size(h_post, 0)
    state_dim = Nx.axis_size(x0, 0)

    zero = Nx.tensor(0.0, type: Nx.type(x0))
    means = Nx.broadcast(zero, {post_count})
    variances = Nx.broadcast(zero, {post_count})

    {_, x_after_gap, p_after_gap, _, _, _} =
      while {i = Nx.tensor(0), x = x0, p = p0, f_in = f, q_in = q, gap_limit = gap_count},
            i < gap_limit do
        x_pred = Nx.dot(f_in, x)
        p_pred = Nx.add(Nx.dot(Nx.dot(f_in, p), Nx.transpose(f_in)), q_in)
        {i + 1, x_pred, p_pred, f_in, q_in, gap_limit}
      end

    {_, means_out, variances_out, _, _, _, _, _, _} =
      while {i = Nx.tensor(0), means_acc = means, vars_acc = variances, x = x_after_gap,
             p = p_after_gap, f_in = f, q_in = q, h_in = h_post, zero_in = zero},
            i < post_count do
        x_pred = Nx.dot(f_in, x)
        p_pred = Nx.add(Nx.dot(Nx.dot(f_in, p), Nx.transpose(f_in)), q_in)
        h_i = Nx.take(h_in, Nx.reshape(i, {1}), axis: 0) |> Nx.reshape({state_dim})
        mean = Nx.dot(h_i, x_pred)
        ph = Nx.dot(p_pred, h_i)
        raw_variance = Nx.dot(h_i, ph)
        variance = Nx.select(Nx.less(raw_variance, 0.0), zero_in, raw_variance)

        means_new = Nx.put_slice(means_acc, [i], Nx.reshape(mean, {1}))
        vars_new = Nx.put_slice(vars_acc, [i], Nx.reshape(variance, {1}))

        {i + 1, means_new, vars_new, x_pred, p_pred, f_in, q_in, h_in, zero_in}
      end

    {means_out, variances_out}
  end

  defp build_summary(actual, baseline, baseline_variance, obs_variance, alpha, return) do
    z = BstsNx.Utils.z_score(alpha)
    obs_var_tensor = Nx.tensor(obs_variance, type: Nx.type(baseline_variance))
    point_variance = Nx.add(baseline_variance, obs_var_tensor)
    point_variance = Nx.select(Nx.less(point_variance, 0.0), 0.0, point_variance)
    point_sd = Nx.sqrt(point_variance)
    point_mean = Nx.subtract(actual, baseline)
    point_lower = Nx.subtract(point_mean, Nx.multiply(point_sd, z))
    point_upper = Nx.add(point_mean, Nx.multiply(point_sd, z))

    cum_mean = point_mean |> Nx.sum() |> Nx.to_number()
    cum_sd = point_variance |> Nx.sum() |> Nx.sqrt() |> Nx.to_number()
    baseline_sum = baseline |> Nx.sum() |> Nx.to_number()
    rel_mean = if abs(baseline_sum) < 1.0e-10, do: 0.0, else: cum_mean / baseline_sum
    rel_sd = if abs(baseline_sum) < 1.0e-10, do: 0.0, else: cum_sd / abs(baseline_sum)

    summary = %{
      point_effects: %{
        mean: point_mean,
        lower: point_lower,
        upper: point_upper
      },
      cumulative_effect: %{
        mean: cum_mean,
        sd: cum_sd,
        lower: cum_mean - z * cum_sd,
        upper: cum_mean + z * cum_sd,
        cross_cov_included: false
      },
      relative_effect: %{
        mean: rel_mean,
        sd: rel_sd,
        lower: rel_mean - z * rel_sd,
        upper: rel_mean + z * rel_sd
      },
      actual: actual,
      baseline: baseline,
      baseline_variance: baseline_variance,
      obs_variance: obs_variance
    }

    case return do
      :tensors -> summary
      :lists -> summary_to_lists(summary)
    end
  end

  defp observations_tensor!(observations, {_post_start, post_end}) do
    obs =
      case observations do
        %Nx.Tensor{} = tensor ->
          if Nx.rank(tensor) != 1 do
            raise ArgumentError,
                  "observations must be a rank-1 tensor for operational scalar-observation workflows"
          end

          tensor

        list when is_list(list) ->
          Nx.tensor(list, type: {:f, 64})
      end
      |> Nx.as_type({:f, 64})

    if Nx.axis_size(obs, 0) < post_end do
      raise ArgumentError,
            "observations length (#{Nx.axis_size(obs, 0)}) must cover post_period ending at #{post_end}"
    end

    obs
  end

  defp q_matrix(spec, state_dim) do
    q_diag_map =
      spec.q_specs
      |> Enum.map(fn q_spec -> {q_spec.dim_index, q_initial(q_spec)} end)
      |> Map.new()

    0..(state_dim - 1)
    |> Enum.map(fn dim -> Map.get(q_diag_map, dim, 0.0) end)
    |> Nx.tensor(type: {:f, 64})
    |> Nx.make_diagonal()
  end

  defp q_initial(%{initial: initial}) when is_number(initial), do: initial * 1.0

  defp q_initial(%{prior_shape: shape, prior_scale: scale}) do
    scale / max(shape - 1.0, 0.5)
  end

  defp h_rows_for_period!(h, start_idx, end_idx, state_dim) do
    count = end_idx - start_idx + 1

    rows =
      case h do
        list when is_list(list) ->
          if length(list) < end_idx do
            raise ArgumentError,
                  "time-varying H/regressors length (#{length(list)}) must cover period ending at #{end_idx}"
          end

          list
          |> Enum.slice(start_idx - 1, count)
          |> Enum.map(&h_row_tensor!(&1, state_dim))
          |> Nx.stack()

        %Nx.Tensor{} = tensor ->
          h_tensor_rows!(tensor, start_idx, end_idx, state_dim)
      end

    Nx.as_type(rows, {:f, 64})
  end

  defp h_tensor_rows!(h, start_idx, end_idx, state_dim) do
    count = end_idx - start_idx + 1

    cond do
      Nx.rank(h) == 1 and Nx.axis_size(h, 0) == state_dim ->
        h |> Nx.as_type({:f, 64}) |> Nx.broadcast({count, state_dim})

      Nx.rank(h) == 2 and Nx.shape(h) == {1, state_dim} ->
        h |> Nx.squeeze(axes: [0]) |> Nx.as_type({:f, 64}) |> Nx.broadcast({count, state_dim})

      Nx.rank(h) == 2 and Nx.axis_size(h, 1) == state_dim and Nx.axis_size(h, 0) >= end_idx ->
        Nx.slice(h, [start_idx - 1, 0], [count, state_dim])

      Nx.rank(h) == 3 and Nx.shape(h) |> elem(1) == 1 and Nx.shape(h) |> elem(2) == state_dim and
          Nx.axis_size(h, 0) >= end_idx ->
        h
        |> Nx.slice([start_idx - 1, 0, 0], [count, 1, state_dim])
        |> Nx.squeeze(axes: [1])

      Nx.rank(h) == 2 and Nx.axis_size(h, 1) == state_dim ->
        raise ArgumentError,
              "time-varying H/regressors length (#{Nx.axis_size(h, 0)}) must cover period ending at #{end_idx}"

      true ->
        raise ArgumentError,
              "h must be static shape {#{state_dim}} / {1, #{state_dim}} or time-varying rows covering the requested periods, got #{inspect(Nx.shape(h))}"
    end
  end

  defp h_row_tensor!(row, state_dim) do
    tensor = BstsNx.Utils.to_tensor(row) |> Nx.as_type({:f, 64})

    cond do
      Nx.rank(tensor) == 1 and Nx.axis_size(tensor, 0) == state_dim ->
        tensor

      Nx.rank(tensor) == 2 and Nx.shape(tensor) == {1, state_dim} ->
        Nx.squeeze(tensor, axes: [0])

      true ->
        raise ArgumentError,
              "time-varying H row must have shape {#{state_dim}} or {1, #{state_dim}}, got #{inspect(Nx.shape(tensor))}"
    end
  end

  defp maybe_tensor_summary(summary, :lists), do: summary

  defp maybe_tensor_summary(summary, :tensors) do
    %{
      summary
      | point_effects: %{
          mean: Nx.tensor(summary.point_effects.mean, type: {:f, 64}),
          lower: Nx.tensor(summary.point_effects.lower, type: {:f, 64}),
          upper: Nx.tensor(summary.point_effects.upper, type: {:f, 64})
        },
        actual: Nx.tensor(summary.actual, type: {:f, 64}),
        baseline: Nx.tensor(summary.baseline, type: {:f, 64}),
        baseline_variance: Nx.tensor(summary.baseline_variance, type: {:f, 64})
    }
  end

  defp summary_to_lists(summary) do
    %{
      summary
      | point_effects: %{
          mean: Nx.to_flat_list(summary.point_effects.mean),
          lower: Nx.to_flat_list(summary.point_effects.lower),
          upper: Nx.to_flat_list(summary.point_effects.upper)
        },
        actual: Nx.to_flat_list(summary.actual),
        baseline: Nx.to_flat_list(summary.baseline),
        baseline_variance: Nx.to_flat_list(summary.baseline_variance)
    }
  end

  defp counterfactual_from_summary(summary) do
    variance =
      case summary.baseline_variance do
        %Nx.Tensor{} = tensor -> Nx.add(tensor, summary.obs_variance)
        list when is_list(list) -> Enum.map(list, &(&1 + summary.obs_variance))
      end

    %{mean: summary.baseline, variance: variance, obs_variance: 0.0}
  end

  defp counterfactual_to_lists(counterfactual) do
    %{
      mean: vector_to_list(counterfactual.mean),
      variance: vector_to_list(counterfactual.variance),
      obs_variance: counterfactual.obs_variance
    }
  end

  defp vector_to_list(%Nx.Tensor{} = tensor), do: Nx.to_flat_list(tensor)
  defp vector_to_list(list) when is_list(list), do: list

  defp validate_period!({start_idx, end_idx}, _name)
       when is_integer(start_idx) and is_integer(end_idx) and start_idx >= 1 and
              end_idx >= start_idx do
    {start_idx, end_idx, end_idx - start_idx + 1}
  end

  defp validate_period!(period, name) do
    raise ArgumentError,
          "#{name} must be a {start, end} tuple with 1-based inclusive positive indices, got: #{inspect(period)}"
  end

  defp post_count({post_start, post_end}) when is_integer(post_start) and is_integer(post_end) do
    post_end - post_start + 1
  end

  defp intervention_indices({post_start, post_end}),
    do: Enum.to_list((post_start - 1)..(post_end - 1))

  defp validate_spot_windows!(spots, n_post) do
    Enum.each(spots, fn spot ->
      if spot.window_start < 0 or spot.window_end > n_post do
        raise ArgumentError,
              "spot #{spot.id} window [#{spot.window_start}, #{spot.window_end}) " <>
                "is outside the post period (0..#{n_post})"
      end

      if spot.window_start > spot.window_end do
        raise ArgumentError,
              "spot #{spot.id} has window_start (#{spot.window_start}) > " <>
                "window_end (#{spot.window_end})"
      end
    end)
  end

  defp attribution_opts(opts) do
    opts
    |> Keyword.take([:alpha, :shapley_samples, :n_samples, :decay, :key, :value_fn_mode])
    |> map_shapley_samples()
  end

  defp map_shapley_samples(opts) do
    case Keyword.pop(opts, :shapley_samples) do
      {nil, rest} -> rest
      {value, rest} -> Keyword.put_new(rest, :n_samples, value)
    end
  end

  defp resolve_baseline!(baseline) when baseline in [:forecast, :smooth], do: baseline

  defp resolve_baseline!(baseline) do
    raise ArgumentError, "baseline must be :forecast or :smooth, got: #{inspect(baseline)}"
  end

  defp resolve_return!(return) when return in [:lists, :tensors], do: return

  defp resolve_return!(return) do
    raise ArgumentError, "return must be :lists or :tensors, got: #{inspect(return)}"
  end

  defp validate_alpha!(alpha) do
    if not is_number(alpha) or alpha <= 0 or alpha >= 1 do
      raise ArgumentError, "alpha must be in (0, 1), got: #{inspect(alpha)}"
    end
  end

  defp scalar_spec?(%ModelSpec{} = spec) do
    Nx.axis_size(spec.f, 0) == 1 and static_scalar_h?(spec.h)
  end

  defp static_scalar_h?(%Nx.Tensor{} = h), do: Nx.size(h) == 1
  defp static_scalar_h?(_), do: false
end
