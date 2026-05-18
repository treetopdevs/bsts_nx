defmodule BstsNx.Forward do
  @moduledoc """
  Shared forward simulation and projection helpers for BSTS workflows.

  The high-level forecast, causal-impact, anomaly, demand, and operational
  modules keep their public APIs. This module centralizes the mechanics they
  share: deterministic key splitting, f64 tensor normalization, stochastic
  scalar/structured trajectories, and fixed-variance moment projection.
  """

  import Nx.Defn

  alias BstsNx.Utils

  @type sample :: map()

  @doc """
  Splits one PRNG key into independent state/process and observation keys.
  """
  @spec split_noise_keys(Nx.t()) :: {Nx.t(), Nx.t()}
  def split_noise_keys(key) do
    keys = Nx.Random.split(key, parts: 2)
    {Utils.split_key_at(keys, 0), Utils.split_key_at(keys, 1)}
  end

  @doc """
  Simulates a scalar random-walk observation trajectory.

  `:gap_steps` advances the latent state before returning the requested post
  observations, matching causal-impact forecast timing.
  """
  @spec scalar_trajectory(
          number() | Nx.t(),
          number() | Nx.t(),
          number() | Nx.t(),
          non_neg_integer(),
          Nx.t(),
          keyword()
        ) ::
          [float()]
  def scalar_trajectory(initial_state, process_var, obs_var, steps, key, opts \\ []) do
    steps = non_negative_integer!(steps, :steps)

    if steps == 0 do
      []
    else
      gap_steps = non_negative_integer!(Keyword.get(opts, :gap_steps, 0), :gap_steps)
      total_steps = gap_steps + steps
      {key_process, key_obs} = split_noise_keys(key)
      sd_process = :math.sqrt(clamped_variance_number(process_var))
      sd_obs = :math.sqrt(clamped_variance_number(obs_var))

      {z_process, _} = Nx.Random.normal(key_process, 0.0, 1.0, shape: {total_steps})
      {z_obs, _} = Nx.Random.normal(key_obs, 0.0, 1.0, shape: {steps})

      process_noise =
        z_process
        |> Nx.as_type({:f, 64})
        |> Nx.multiply(Nx.tensor(sd_process, type: {:f, 64}))

      states =
        initial_state
        |> scalar_tensor()
        |> Nx.add(Nx.cumulative_sum(process_noise))

      post_states = Nx.slice_along_axis(states, gap_steps, steps, axis: 0)

      obs_noise =
        z_obs
        |> Nx.as_type({:f, 64})
        |> Nx.multiply(Nx.tensor(sd_obs, type: {:f, 64}))

      post_states
      |> Nx.add(obs_noise)
      |> Nx.to_flat_list()
    end
  end

  @doc """
  Simulates a structured state-space observation trajectory.

  `h_steps` may be a list of observation rows or a stacked tensor whose first
  axis is time. Each returned value corresponds to one post-period H row.
  """
  @spec structured_trajectory(
          Nx.t(),
          Nx.t(),
          [Nx.t()] | Nx.t(),
          Nx.t(),
          number() | Nx.t(),
          Nx.t(),
          keyword()
        ) ::
          [float()]
  def structured_trajectory(final_state, f, h_steps, q_matrix, obs_var, key, opts \\ []) do
    gap_steps = non_negative_integer!(Keyword.get(opts, :gap_steps, 0), :gap_steps)
    h_tensor = h_steps_tensor(h_steps)
    n_steps = Nx.axis_size(h_tensor, 0)

    if n_steps == 0 do
      []
    else
      f_t = Nx.as_type(f, {:f, 64})
      state_dim = Nx.axis_size(f_t, 0)
      total_steps = gap_steps + n_steps
      {key_state, key_obs} = split_noise_keys(key)

      {z_state, _} = Nx.Random.normal(key_state, 0.0, 1.0, shape: {total_steps, state_dim})
      {z_obs, _} = Nx.Random.normal(key_obs, 0.0, 1.0, shape: {n_steps})

      final_state
      |> Nx.flatten()
      |> Nx.as_type({:f, 64})
      |> structured_trajectory_defn(
        f_t,
        h_tensor,
        Nx.as_type(q_matrix, {:f, 64}),
        Nx.tensor(clamped_variance_number(obs_var), type: {:f, 64}),
        Nx.as_type(z_state, {:f, 64}),
        Nx.as_type(z_obs, {:f, 64}),
        Nx.tensor(gap_steps, type: {:s, 64})
      )
      |> Nx.to_flat_list()
    end
  end

  @doc """
  Simulates scalar forecast trajectories for all posterior samples.
  """
  @spec posterior_scalar_trajectories([sample()], non_neg_integer(), Nx.t()) :: [[float()]]
  def posterior_scalar_trajectories(samples, horizon, base_key) do
    horizon = non_negative_integer!(horizon, :horizon)
    sample_keys = Nx.Random.split(base_key, parts: length(samples))

    samples
    |> Enum.with_index()
    |> Enum.map(fn {sample, idx} ->
      scalar_trajectory(
        List.last(sample.states),
        Map.fetch!(sample, :process_var),
        Map.fetch!(sample, :obs_var),
        horizon,
        Utils.split_key_at(sample_keys, idx)
      )
    end)
  end

  @doc """
  Simulates structured forecast trajectories for all posterior samples.
  """
  @spec posterior_structured_trajectories([sample()], map(), [Nx.t()] | Nx.t(), Nx.t()) :: [
          [float()]
        ]
  def posterior_structured_trajectories(samples, spec, h_steps, base_key) do
    h_tensor = h_steps_tensor(h_steps)
    sample_keys = Nx.Random.split(base_key, parts: length(samples))

    samples
    |> Enum.with_index()
    |> Enum.map(fn {sample, idx} ->
      structured_trajectory(
        List.last(sample.states),
        spec.f,
        h_tensor,
        Map.fetch!(sample, :q_matrix),
        Map.fetch!(sample, :obs_var),
        Utils.split_key_at(sample_keys, idx)
      )
    end)
  end

  @doc """
  Projects scalar posterior samples into predictive means and standard deviations.
  """
  @spec scalar_moments_from_samples([sample()], non_neg_integer()) :: {[float()], [float()]}
  def scalar_moments_from_samples(samples, horizon) do
    horizon = non_negative_integer!(horizon, :horizon)

    if horizon == 0 do
      {[], []}
    else
      samples
      |> Enum.map(fn sample ->
        level = sample.states |> List.last() |> scalar_number()
        q = clamped_variance_number(Map.fetch!(sample, :process_var))
        r = clamped_variance_number(Map.fetch!(sample, :obs_var))

        vars =
          Enum.map(1..horizon, fn step ->
            max(step * q + r, 1.0e-12)
          end)

        %{means: List.duplicate(level, horizon), vars: vars}
      end)
      |> aggregate_prediction_moments()
    end
  end

  @doc """
  Projects structured posterior samples into predictive means and standard deviations.
  """
  @spec structured_moments_from_samples([sample()], map(), [Nx.t()] | Nx.t()) ::
          {[float()], [float()]}
  def structured_moments_from_samples(samples, spec, h_steps) do
    h_rows = h_steps_rows(h_steps)

    if h_rows == [] do
      {[], []}
    else
      samples
      |> Enum.map(fn sample ->
        f = Nx.as_type(spec.f, {:f, 64})
        q = Nx.as_type(Map.fetch!(sample, :q_matrix), {:f, 64})
        obs_var = clamped_variance_number(Map.fetch!(sample, :obs_var))
        init_state = sample.states |> List.last() |> Nx.flatten() |> Nx.as_type({:f, 64})
        state_dim = Nx.axis_size(init_state, 0)
        zero_cov = Nx.broadcast(Nx.tensor(0.0, type: {:f, 64}), {state_dim, state_dim})

        {_final_state, _final_cov, means_rev, vars_rev} =
          Enum.reduce(h_rows, {init_state, zero_cov, [], []}, fn h_row,
                                                                 {state, cov, means_acc, vars_acc} ->
            next_state = Nx.dot(f, state)
            next_cov = f |> Nx.dot(cov) |> Nx.dot(Nx.transpose(f)) |> Nx.add(q)
            h_row_t = Nx.as_type(h_row, {:f, 64})
            y_mean = Nx.dot(h_row_t, next_state) |> Nx.to_number()
            h_mat = Nx.new_axis(h_row_t, 0)

            proc_var =
              h_mat
              |> Nx.dot(next_cov)
              |> Nx.dot(Nx.transpose(h_mat))
              |> Nx.squeeze()
              |> Nx.to_number()

            y_var = max(proc_var + obs_var, 1.0e-12)
            {next_state, next_cov, [y_mean | means_acc], [y_var | vars_acc]}
          end)

        %{means: Enum.reverse(means_rev), vars: Enum.reverse(vars_rev)}
      end)
      |> aggregate_prediction_moments()
    end
  end

  @doc """
  Projects fixed-variance state moments over a post-period.

  Returns `{means, variances, cumulative_variance}` as tensors. The cumulative
  variance includes cross-step covariance between projected baseline steps.
  """
  defn forecast_moments_defn(x0, p0, f, q, h_post, gap_count) do
    x0_t = x0 |> Nx.flatten() |> Nx.as_type({:f, 64})
    p0_t = Nx.as_type(p0, {:f, 64})
    f_t = Nx.as_type(f, {:f, 64})
    q_t = Nx.as_type(q, {:f, 64})
    h_t = Nx.as_type(h_post, {:f, 64})
    gap_limit = Nx.as_type(gap_count, {:s, 64})
    post_count = Nx.axis_size(h_t, 0)
    state_dim = Nx.axis_size(x0_t, 0)

    zero = Nx.tensor(0.0, type: {:f, 64})
    means = Nx.broadcast(zero, {post_count})
    variances = Nx.broadcast(zero, {post_count})

    {_, x_after_gap, p_after_gap, _, _, _} =
      while {i = Nx.tensor(0, type: {:s, 64}), x = x0_t, p = p0_t, f_in = f_t, q_in = q_t,
             gap = gap_limit},
            i < gap do
        x_pred = Nx.dot(f_in, x)
        p_pred = Nx.add(Nx.dot(Nx.dot(f_in, p), Nx.transpose(f_in)), q_in)
        {i + 1, x_pred, p_pred, f_in, q_in, gap}
      end

    state_sum_cov0 = Nx.broadcast(zero, {state_dim})
    cumulative_variance0 = zero

    {_, means_out, variances_out, _, _, cumulative_variance_out, _, _, _, _, _} =
      while {i = Nx.tensor(0, type: {:s, 64}), means_acc = means, vars_acc = variances,
             x = x_after_gap, p = p_after_gap, cumulative_var = cumulative_variance0,
             state_sum_cov = state_sum_cov0, f_in = f_t, q_in = q_t, h_in = h_t, zero_in = zero},
            i < post_count do
        x_pred = Nx.dot(f_in, x)
        p_pred = Nx.add(Nx.dot(Nx.dot(f_in, p), Nx.transpose(f_in)), q_in)
        state_sum_cov_pred = Nx.dot(f_in, state_sum_cov)
        h_i = Nx.take(h_in, Nx.reshape(i, {1}), axis: 0) |> Nx.reshape({state_dim})
        mean = Nx.dot(h_i, x_pred)
        ph = Nx.dot(p_pred, h_i)
        raw_variance = Nx.dot(h_i, ph)
        variance = Nx.select(Nx.less(raw_variance, zero_in), zero_in, raw_variance)
        cross_with_previous_sum = Nx.dot(h_i, state_sum_cov_pred)

        cumulative_variance_new =
          cumulative_var
          |> Nx.add(variance)
          |> Nx.add(Nx.multiply(2.0, cross_with_previous_sum))
          |> Nx.max(zero_in)

        state_sum_cov_new = Nx.add(state_sum_cov_pred, ph)

        means_new = Nx.put_slice(means_acc, [i], Nx.reshape(mean, {1}))
        vars_new = Nx.put_slice(vars_acc, [i], Nx.reshape(variance, {1}))

        {i + 1, means_new, vars_new, x_pred, p_pred, cumulative_variance_new, state_sum_cov_new,
         f_in, q_in, h_in, zero_in}
      end

    {means_out, variances_out, cumulative_variance_out}
  end

  defnp structured_trajectory_defn(
          final_state,
          f_mat,
          h_tensor,
          q_matrix,
          obs_var,
          z_state_raw,
          z_obs_raw,
          gap_steps
        ) do
    type = Nx.type(final_state)
    f_t = Nx.as_type(f_mat, type)
    h_t = Nx.as_type(h_tensor, type)
    q_t = Nx.as_type(q_matrix, type)
    obs_var_t = Nx.as_type(obs_var, type)
    {n_steps, _} = Nx.shape(h_t)
    z_state = Nx.as_type(z_state_raw, type)
    z_obs = Nx.as_type(z_obs_raw, type)

    q_diag = Nx.take_diagonal(q_t)
    q_sds = Nx.sqrt(Nx.max(q_diag, Nx.tensor(0.0, type: type)))
    obs_sd = Nx.sqrt(Nx.max(obs_var_t, Nx.tensor(0.0, type: type)))

    {_, gap_state, _, _, _, _} =
      while {i = Nx.tensor(0, type: {:s, 64}), prev_state = final_state, z_s = z_state, f = f_t,
             q_s = q_sds, gap = gap_steps},
            i < gap do
        z_s_i = take_time_slice_defn(z_s, i)
        process_noise = Nx.multiply(z_s_i, q_s)
        next_state = Nx.add(Nx.dot(f, prev_state), process_noise)
        {i + 1, next_state, z_s, f, q_s, gap}
      end

    observations = Nx.broadcast(Nx.tensor(0.0, type: type), {n_steps})

    {_, _, observations_out, _, _, _, _, _, _, _} =
      while {i = Nx.tensor(0, type: {:s, 64}), prev_state = gap_state, obs_acc = observations,
             z_s = z_state, z_o = z_obs, h_in = h_t, f = f_t, q_s = q_sds, obs_s = obs_sd,
             gap = gap_steps},
            i < n_steps do
        z_s_i = take_time_slice_defn(z_s, i + gap)
        process_noise = Nx.multiply(z_s_i, q_s)
        next_state = Nx.add(Nx.dot(f, prev_state), process_noise)
        h_row = take_time_slice_defn(h_in, i)
        predicted_y = Nx.dot(h_row, next_state)
        z_o_i = take_time_slice_defn(z_o, i)
        observed_y = Nx.add(predicted_y, Nx.multiply(z_o_i, obs_s))
        obs_new = Nx.put_slice(obs_acc, [i], Nx.reshape(observed_y, {1}))

        {i + 1, next_state, obs_new, z_s, z_o, h_in, f, q_s, obs_s, gap}
      end

    observations_out
  end

  defnp take_time_slice_defn(tensor, idx) do
    Nx.slice_along_axis(tensor, idx, 1, axis: 0) |> Nx.squeeze(axes: [0])
  end

  defp h_steps_rows(h_steps) do
    h_tensor = h_steps_tensor(h_steps)
    n_steps = Nx.axis_size(h_tensor, 0)

    if n_steps == 0 do
      []
    else
      Enum.map(0..(n_steps - 1), &Utils.take_time_slice_at(h_tensor, &1))
    end
  end

  defp h_steps_tensor([]), do: Nx.tensor([], type: {:f, 64}) |> Nx.reshape({0, 0})

  defp h_steps_tensor(h_steps) when is_list(h_steps) do
    h_steps
    |> Enum.map(&Utils.h_to_row_tensor/1)
    |> Nx.stack()
    |> Nx.as_type({:f, 64})
  end

  defp h_steps_tensor(%Nx.Tensor{} = h_steps) do
    h =
      case Nx.rank(h_steps) do
        1 ->
          Nx.new_axis(h_steps, 0)

        2 ->
          h_steps

        3 ->
          if Nx.axis_size(h_steps, 1) == 1 do
            Nx.squeeze(h_steps, axes: [1])
          else
            raise ArgumentError,
                  "h_steps rank-3 tensor must have shape {steps, 1, state_dim}, got #{inspect(Nx.shape(h_steps))}"
          end

        rank ->
          raise ArgumentError, "h_steps must be rank 1, 2, or 3, got rank #{rank}"
      end

    Nx.as_type(h, {:f, 64})
  end

  defp aggregate_prediction_moments([]), do: {[], []}

  defp aggregate_prediction_moments(per_sample) do
    n_samples = length(per_sample)

    mean_columns =
      per_sample
      |> Enum.map(& &1.means)
      |> Utils.transpose_rows()

    var_columns =
      per_sample
      |> Enum.map(& &1.vars)
      |> Utils.transpose_rows()

    means = Enum.map(mean_columns, &(Enum.sum(&1) / n_samples))

    sds =
      mean_columns
      |> Enum.zip(var_columns)
      |> Enum.map(fn {step_means, step_noise_vars} ->
        between = sample_variance(step_means)
        avg_noise = Enum.sum(step_noise_vars) / n_samples
        :math.sqrt(max(between + avg_noise, 1.0e-12))
      end)

    {means, sds}
  end

  defp sample_variance(values) do
    n = length(values)

    if n < 2 do
      0.0
    else
      mean = Enum.sum(values) / n
      Enum.reduce(values, 0.0, fn x, acc -> acc + :math.pow(x - mean, 2) end) / (n - 1)
    end
  end

  defp scalar_tensor(%Nx.Tensor{} = tensor), do: tensor |> Nx.reshape({}) |> Nx.as_type({:f, 64})
  defp scalar_tensor(number) when is_number(number), do: Nx.tensor(number, type: {:f, 64})

  defp scalar_number(%Nx.Tensor{} = tensor),
    do: tensor |> Nx.reshape({}) |> Nx.as_type({:f, 64}) |> Nx.to_number()

  defp scalar_number(number) when is_number(number), do: number * 1.0

  defp clamped_variance_number(value), do: max(scalar_number(value), 0.0)

  defp non_negative_integer!(value, _name) when is_integer(value) and value >= 0, do: value

  defp non_negative_integer!(value, name) do
    raise ArgumentError, "#{name} must be a non-negative integer, got: #{inspect(value)}"
  end
end
