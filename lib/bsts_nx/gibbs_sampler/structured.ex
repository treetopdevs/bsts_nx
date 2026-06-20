defmodule BstsNx.GibbsSampler.Structured do
  @moduledoc false

  alias BstsNx.Distributions
  alias BstsNx.GibbsSampler.Residuals
  alias BstsNx.KalmanFilter

  import BstsNx.Utils, only: [to_tensor: 1]

  require Nx.Defn

  def sample_standard(
        observations,
        spec,
        q_matrix,
        r_var,
        h_list,
        burn_in,
        thin,
        key,
        total_iters,
        fused?
      ) do
    if fused? and length(observations) >= 2 and spec.q_specs != [] do
      sample_fused(observations, spec, q_matrix, r_var, h_list, burn_in, thin, key, total_iters)
    else
      sample_stepwise(
        observations,
        spec,
        q_matrix,
        r_var,
        h_list,
        burn_in,
        thin,
        key,
        total_iters
      )
    end
  end

  defp sample_stepwise(
         observations,
         spec,
         q_matrix,
         r_var,
         h_list,
         burn_in,
         thin,
         key,
         total_iters
       ) do
    obs_tensor = observations_to_filter_tensor(observations)
    h_tensor = Nx.stack(h_list)
    h_rows_tensor = h_rows_tensor(h_list)
    num_diffs = latent_transition_count(observations)

    {_, _, samples_acc, _key_acc} =
      Enum.reduce(1..total_iters, {q_matrix, r_var, [], key}, fn iter, {q_prev, r_prev, acc, k} ->
        {filtered_xs, filtered_ps} =
          KalmanFilter.filter_defn_multi(
            obs_tensor,
            spec.f,
            h_tensor,
            q_prev,
            r_prev,
            spec.x0,
            spec.p0
          )

        {sampled_states_tensor, new_key} =
          BstsNx.Smoother.simulate_from_filtered_defn_matrix(
            filtered_xs,
            filtered_ps,
            spec.f,
            q_prev,
            k
          )

        per_dim_ss = Residuals.process_structured(sampled_states_tensor, spec.f, spec.q_specs)

        {obs_ss, t_obs} =
          Residuals.obs_structured(observations, sampled_states_tensor, h_rows_tensor)

        {q_new, key_after_q} = resample_q_components(spec.q_specs, per_dim_ss, num_diffs, new_key)
        q_matrix_new = rebuild_q(q_prev, q_new)

        shape_r = safe_to_number(spec.obs_prior_shape) + safe_to_number(t_obs) / 2
        scale_r = safe_to_number(spec.obs_prior_scale) + safe_to_number(obs_ss) / 2

        {r_sample, next_key} =
          Distributions.inv_gamma_sample_defn(shape_r, scale_r, key_after_q)

        acc2 =
          if iter > burn_in and rem(iter - burn_in, thin) == 0 do
            sample_map = %{
              states: tensor_time_slices(sampled_states_tensor),
              state_covs: tensor_time_slices(filtered_ps),
              q_matrix: q_matrix_new,
              obs_var: r_sample,
              regression_beta: nil,
              regression_gamma: nil
            }

            [sample_map | acc]
          else
            acc
          end

        {q_matrix_new, r_sample, acc2, next_key}
      end)

    Enum.reverse(samples_acc)
  end

  # Runs the whole structured Gibbs chain inside a single compiled defn (one
  # backend program for filter + simulation smoother + Q/R resampling across
  # all iterations).  Produces the same draws as the stepwise path for the
  # same PRNG key.  Requires at least one resampled Q component and t >= 2;
  # other configurations fall back to the stepwise path.
  defp sample_fused(
         observations,
         spec,
         q_matrix,
         r_var,
         h_list,
         burn_in,
         thin,
         key,
         total_iters
       ) do
    f64 = {:f, 64}
    t = length(observations)
    num_samples = div(total_iters - burn_in, thin)

    obs_tensor = observations_to_filter_tensor(observations)
    obs_present_mask = Nx.equal(obs_tensor, obs_tensor)
    t_obs = obs_present_mask |> Nx.sum() |> Nx.to_number() |> round()
    num_diffs = latent_transition_count(observations)

    h_rows =
      h_list
      |> Enum.map(&structured_h_row/1)
      |> Enum.map(&Nx.flatten/1)
      |> Nx.stack()
      |> Nx.as_type(f64)

    n = Nx.axis_size(spec.f, 0)
    f_t = spec.f |> to_tensor() |> Nx.as_type(f64)
    x0 = spec.x0 |> to_tensor() |> Nx.flatten() |> Nx.as_type(f64)
    p0 = ensure_square_f64(spec.p0, n)
    q0 = ensure_square_f64(q_matrix, n)
    r0 = r_var |> to_tensor() |> Nx.as_type(f64)

    q_dims = Nx.tensor(Enum.map(spec.q_specs, & &1.dim_index), type: {:s, 64})

    q_shapes =
      spec.q_specs
      |> Enum.map(fn qs -> safe_to_number(qs.prior_shape) + num_diffs / 2 end)
      |> Nx.tensor(type: f64)

    q_prior_scales =
      spec.q_specs
      |> Enum.map(fn qs -> safe_to_number(qs.prior_scale) end)
      |> Nx.tensor(type: f64)

    shape_r = Nx.tensor([safe_to_number(spec.obs_prior_shape) + t_obs / 2], type: f64)
    obs_prior_scale = Nx.tensor(safe_to_number(spec.obs_prior_scale), type: f64)
    loop_meta = Nx.tensor([burn_in, thin, total_iters], type: {:s, 64})
    zero = Nx.tensor(0.0, type: f64)

    chain_fun =
      if BstsNx.Smoother.backend_supports_lu?() do
        &structured_chain_defn_solve/19
      else
        &structured_chain_defn_pinv/19
      end

    {states_out, covs_out, qdiag_out, r_out} =
      chain_fun.(
        obs_tensor,
        h_rows,
        f_t,
        q0,
        r0,
        x0,
        p0,
        q_dims,
        q_shapes,
        q_prior_scales,
        shape_r,
        obs_prior_scale,
        obs_present_mask,
        loop_meta,
        key,
        Nx.broadcast(zero, {num_samples, t, n}),
        Nx.broadcast(zero, {num_samples, t, n, n}),
        Nx.broadcast(zero, {num_samples, n}),
        Nx.broadcast(zero, {num_samples})
      )

    # Pull chain outputs to the host once so the per-sample slicing below does
    # not issue thousands of small device ops on compiled backends.
    states_out = to_host(states_out)
    covs_out = to_host(covs_out)
    qdiag_out = to_host(qdiag_out)
    r_out = to_host(r_out)

    Enum.map(0..(num_samples - 1), fn i ->
      %{
        states: tensor_time_slices(states_out[i]),
        state_covs: tensor_time_slices(covs_out[i]),
        q_matrix: Nx.make_diagonal(qdiag_out[i]),
        obs_var: r_out[i],
        regression_beta: nil,
        regression_gamma: nil
      }
    end)
  end

  # Moves a (potentially device-backed) tensor to the host BinaryBackend so
  # repeated small slices stay off the device dispatch path.  Cross-backend
  # arithmetic remains supported by Nx, so downstream consumers are
  # unaffected.
  defp to_host(tensor), do: Nx.backend_copy(tensor, Nx.BinaryBackend)

  defp ensure_square_f64(matrix, n) do
    t = to_tensor(matrix)

    case Nx.shape(t) do
      {} when n == 1 ->
        Nx.reshape(t, {1, 1}) |> Nx.as_type({:f, 64})

      {^n, ^n} ->
        Nx.as_type(t, {:f, 64})

      shape ->
        raise ArgumentError,
              "expected square matrix shape {#{n}, #{n}}, got: #{inspect(shape)}"
    end
  end

  # Compiled structured Gibbs chain.  Same retention scheme as the scalar
  # chain: `loop_meta` is `[burn_in, thin, total_iters]` and the pending
  # accumulator slot is overwritten until its retaining iteration.  The
  # solve/pinv variants mirror the smoother-gain dispatch in BstsNx.Smoother
  # for backends without LU support.
  for {name, sim_fun} <- [
        structured_chain_defn_solve: :simulate_from_filtered_defn_matrix_impl_solve,
        structured_chain_defn_pinv: :simulate_from_filtered_defn_matrix_impl_pinv
      ] do
    Nx.Defn.defn unquote(name)(
                   obs,
                   h_rows,
                   f_t,
                   q0,
                   r0,
                   x0,
                   p0,
                   q_dims,
                   q_shapes,
                   q_prior_scales,
                   shape_r,
                   obs_prior_scale,
                   obs_mask,
                   loop_meta,
                   key,
                   states_acc0,
                   covs_acc0,
                   qdiag_acc0,
                   r_acc0
                 ) do
      t = Nx.axis_size(obs, 0)
      n = Nx.axis_size(f_t, 0)

      {_, _, _, _, _, states_out, covs_out, qdiag_out, r_out, _, _, _, _, _, _, _, _, _, _, _, _} =
        while {iter = Nx.tensor(1, type: {:s, 64}), q = q0, r = r0, k = key,
               kept = Nx.tensor(0, type: {:s, 64}), states_acc = states_acc0,
               covs_acc = covs_acc0, qdiag_acc = qdiag_acc0, r_acc = r_acc0, obs_in = obs,
               h_in = h_rows, f_in = f_t, x0_in = x0, p0_in = p0, q_dims_in = q_dims,
               q_shapes_in = q_shapes, q_scales_in = q_prior_scales, shape_r_in = shape_r,
               obs_scale_in = obs_prior_scale, mask_in = obs_mask, meta = loop_meta},
              iter <= meta[2] do
          {xs, ps} =
            KalmanFilter.filter_defn_multi_impl(obs_in, f_in, h_in, q, r, x0_in, p0_in)

          {states_raw, k2} = BstsNx.Smoother.unquote(sim_fun)(xs, ps, f_in, q, k)
          states = Nx.as_type(states_raw, {:f, 64})

          prev = Nx.slice(states, [0, 0], [t - 1, n])
          nxt = Nx.slice(states, [1, 0], [t - 1, n])
          resid = nxt - Nx.dot(prev, Nx.transpose(f_in))
          ss_vec = Nx.sum(resid * resid, axes: [0])

          {q_draws, k3} =
            Distributions.inv_gamma_sample_defn_impl(
              q_shapes_in,
              q_scales_in + Nx.take(ss_vec, q_dims_in) / 2.0,
              k2,
              Nx.tensor(0.0, type: {:f, 64}),
              Nx.tensor(0, type: {:u, 8})
            )

          q_diag = Nx.indexed_put(Nx.take_diagonal(q), Nx.new_axis(q_dims_in, 1), q_draws)
          q_new = Nx.make_diagonal(q_diag)

          preds = Nx.sum(h_in * states, axes: [1])
          odiff = obs_in - preds
          obs_ss = Nx.sum(Nx.select(mask_in, odiff * odiff, 0.0))

          {r_draw, k4} =
            Distributions.inv_gamma_sample_defn_impl(
              shape_r_in,
              Nx.reshape(obs_scale_in + obs_ss / 2.0, {1}),
              k3,
              Nx.tensor(0.0, type: {:f, 64}),
              Nx.tensor(0, type: {:u, 8})
            )

          r_new = Nx.squeeze(r_draw)

          keep =
            Nx.logical_and(iter > meta[0], Nx.remainder(iter - meta[0], meta[1]) == 0)
            |> Nx.as_type({:s, 64})

          states_acc = Nx.put_slice(states_acc, [kept, 0, 0], Nx.new_axis(states, 0))

          covs_acc =
            Nx.put_slice(covs_acc, [kept, 0, 0, 0], Nx.new_axis(Nx.as_type(ps, {:f, 64}), 0))

          qdiag_acc = Nx.put_slice(qdiag_acc, [kept, 0], Nx.new_axis(q_diag, 0))
          r_acc = Nx.put_slice(r_acc, [kept], Nx.reshape(r_new, {1}))

          {iter + 1, q_new, r_new, k4, kept + keep, states_acc, covs_acc, qdiag_acc, r_acc,
           obs_in, h_in, f_in, x0_in, p0_in, q_dims_in, q_shapes_in, q_scales_in, shape_r_in,
           obs_scale_in, mask_in, meta}
        end

      {states_out, covs_out, qdiag_out, r_out}
    end
  end

  def build_initial_q(q_specs, n) do
    BstsNx.Validation.fixed_q_matrix(q_specs, n)
  end

  def resample_q_components(q_specs, per_dim_ss, num_diffs, key) do
    case q_specs do
      [] ->
        {[], key}

      _ ->
        dim_indices = Enum.map(q_specs, & &1.dim_index)

        shapes =
          Enum.map(q_specs, fn qs ->
            safe_to_number(qs.prior_shape) + num_diffs / 2
          end)

        scales =
          Enum.map(q_specs, fn qs ->
            ss = Map.get(per_dim_ss, qs.dim_index, 0.0) |> safe_to_number()
            safe_to_number(qs.prior_scale) + ss / 2
          end)

        {samples, next_key} =
          Distributions.inv_gamma_sample_defn(
            Nx.tensor(shapes, type: {:f, 64}),
            Nx.tensor(scales, type: {:f, 64}),
            key
          )

        {Enum.zip(dim_indices, Nx.to_flat_list(samples)), next_key}
    end
  end

  def rebuild_q(q_prev, new_vals) do
    diag = Nx.take_diagonal(q_prev)

    case new_vals do
      [] ->
        q_prev

      _ ->
        {dim_indices, values} = Enum.unzip(new_vals)
        index_tensor = dim_indices |> Enum.map(&[&1]) |> Nx.tensor(type: {:s, 64})
        value_tensor = Nx.tensor(values, type: Nx.type(diag))
        Nx.indexed_put(diag, index_tensor, value_tensor) |> Nx.make_diagonal()
    end
  end

  def normalize_h(h, t) when is_list(h) do
    if length(h) != t do
      raise ArgumentError,
            "time-varying H list length (#{length(h)}) must match observations length (#{t})"
    end

    h
  end

  def normalize_h(%Nx.Tensor{} = h, t) do
    rank = Nx.rank(h)

    unless rank >= 1 and rank <= 2 do
      raise ArgumentError,
            "observation matrix H must be rank 1 or 2, got rank #{rank}"
    end

    cond do
      rank == 1 ->
        List.duplicate(h, t)

      rank == 2 and Nx.axis_size(h, 0) == t ->
        Enum.map(0..(t - 1), fn i ->
          Nx.slice(h, [i, 0], [1, Nx.axis_size(h, 1)])
        end)

      rank == 2 ->
        if Nx.axis_size(h, 0) != 1 and Nx.axis_size(h, 1) != 1 do
          raise ArgumentError,
                "structured sampler expects static H with a singleton axis, got shape #{inspect(Nx.shape(h))}"
        end

        List.duplicate(h, t)
    end
  end

  def observations_to_filter_tensor(observations),
    do: Residuals.observations_to_filter_tensor(observations)

  def structured_h_row(%Nx.Tensor{} = h_t), do: Residuals.structured_h_row(h_t)

  def h_rows_tensor(h_list) do
    h_list
    |> Enum.map(&structured_h_row/1)
    |> Enum.map(&Nx.flatten/1)
    |> Nx.stack()
  end

  def scalar_tensor_list(tensor), do: tensor_time_slices(tensor)

  def tensor_time_slices(tensor) do
    tensor
    |> Nx.to_batched(1)
    |> Enum.map(&Nx.squeeze(&1, axes: [0]))
  end

  def latent_transition_count(observations) do
    max(length(observations) - 1, 0)
  end

  def safe_to_number(value), do: Residuals.safe_to_number(value)
end
