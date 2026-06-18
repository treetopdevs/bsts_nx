defmodule BstsNx.GibbsSampler.Structured do
  @moduledoc false

  alias BstsNx.Distributions
  alias BstsNx.GibbsSampler.Residuals
  alias BstsNx.KalmanFilter

  def sample_standard(
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
