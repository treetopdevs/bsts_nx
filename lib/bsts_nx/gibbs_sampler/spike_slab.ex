defmodule BstsNx.GibbsSampler.SpikeSlab do
  @moduledoc false

  alias BstsNx.Distributions
  alias BstsNx.GibbsSampler.Residuals
  alias BstsNx.GibbsSampler.Structured
  alias BstsNx.KalmanFilter

  import BstsNx.Utils, only: [compat_dot: 2]

  @rank_update_pivot_floor 1.0e-12

  def sample(
        observations,
        spec,
        regression,
        q_matrix,
        r_var,
        h_list,
        burn_in,
        thin,
        key,
        total_iters
      ) do
    t = length(observations)
    num_diffs = Structured.latent_transition_count(observations)

    ctx = prepare_context(spec, regression, h_list)

    q_struct0 =
      if ctx.struct_dim == 0 do
        nil
      else
        submatrix(q_matrix, ctx.struct_full_indices, ctx.struct_full_indices)
      end

    beta0 = slice_vector(spec.x0, ctx.reg_full_indices)
    {y_obs_init, x_obs_init} = Residuals.observed_regression_pairs(observations, ctx.x_rows)
    gamma0 = init_gamma_from_data(y_obs_init, x_obs_init, ctx.prior_inclusion, ctx.reg_dim)
    gamma_xtx_stats = if x_obs_init == [], do: nil, else: spike_slab_xtx_stats(x_obs_init)

    {_, _, _, _, samples_acc, _key_acc} =
      Enum.reduce(
        1..total_iters,
        {q_struct0, r_var, beta0, gamma0, [], key},
        fn iter, {q_struct_prev, r_prev, beta_prev, gamma_prev, acc, key_prev} ->
          y_adjusted =
            Residuals.adjust_observations_for_regression(observations, ctx.x_rows, beta_prev)

          {sampled_struct_states, struct_covs, q_struct_new, key_after_struct} =
            if ctx.struct_dim == 0 do
              {List.duplicate([], t), List.duplicate(nil, t), q_struct_prev, key_prev}
            else
              y_adjusted_tensor = Structured.observations_to_filter_tensor(y_adjusted)

              {filtered_xs, filtered_ps} =
                KalmanFilter.filter_defn_multi(
                  y_adjusted_tensor,
                  ctx.f_struct,
                  ctx.h_struct_tensor,
                  q_struct_prev,
                  r_prev,
                  ctx.x0_struct,
                  ctx.p0_struct
                )

              {sampled_states_tensor, key_after_smooth} =
                BstsNx.Smoother.simulate_from_filtered_defn_matrix(
                  filtered_xs,
                  filtered_ps,
                  ctx.f_struct,
                  q_struct_prev,
                  key_prev
                )

              sampled_states = Structured.tensor_time_slices(sampled_states_tensor)
              covs = Structured.tensor_time_slices(filtered_ps)

              per_dim_ss =
                Residuals.process_structured(
                  sampled_states_tensor,
                  ctx.f_struct,
                  ctx.q_specs_struct
                )

              {q_new_vals, key_after_q} =
                Structured.resample_q_components(
                  ctx.q_specs_struct,
                  per_dim_ss,
                  num_diffs,
                  key_after_smooth
                )

              {sampled_states, covs, Structured.rebuild_q(q_struct_prev, q_new_vals), key_after_q}
            end

          {y_reg_obs, x_reg_obs} =
            Residuals.regression_residual_pairs(
              observations,
              sampled_struct_states,
              ctx.h_struct_rows,
              ctx.x_rows
            )

          sigma2 = max(Nx.to_number(r_prev), 1.0e-12)

          {gamma_new, key_after_gamma} =
            resample_gamma_g_prior(
              gamma_prev,
              y_reg_obs,
              x_reg_obs,
              sigma2,
              ctx.prior_inclusion,
              ctx.g_prior,
              key_after_struct,
              gamma_xtx_stats
            )

          {beta_new, beta_cov, key_after_beta} =
            sample_beta_g_prior(
              gamma_new,
              y_reg_obs,
              x_reg_obs,
              sigma2,
              ctx.g_prior,
              key_after_gamma
            )

          {obs_ss, t_obs} =
            Residuals.obs_spike_slab(
              observations,
              sampled_struct_states,
              ctx.h_struct_rows,
              ctx.x_rows,
              beta_new
            )

          shape_r =
            Structured.safe_to_number(spec.obs_prior_shape) + Structured.safe_to_number(t_obs) / 2

          scale_r =
            Structured.safe_to_number(spec.obs_prior_scale) +
              Structured.safe_to_number(obs_ss) / 2

          {r_sample, key_next} =
            Distributions.inv_gamma_sample_defn(shape_r, scale_r, key_after_beta)

          acc2 =
            if iter > burn_in and rem(iter - burn_in, thin) == 0 do
              q_matrix_full =
                build_full_q_matrix(
                  ctx.state_dim,
                  ctx.struct_full_indices,
                  q_struct_new
                )

              full_states =
                build_full_state_trajectory(
                  sampled_struct_states,
                  beta_new,
                  ctx.state_dim,
                  ctx.struct_full_indices,
                  ctx.reg_full_indices
                )

              full_covs =
                build_full_covariances(
                  struct_covs,
                  beta_cov,
                  ctx.state_dim,
                  ctx.struct_full_indices,
                  ctx.reg_full_indices
                )

              sample_map = %{
                states: full_states,
                state_covs: full_covs,
                q_matrix: q_matrix_full,
                obs_var: r_sample,
                regression_beta: Nx.tensor(beta_new),
                regression_gamma: gamma_new
              }

              [sample_map | acc]
            else
              acc
            end

          {q_struct_new, r_sample, beta_new, gamma_new, acc2, key_next}
        end
      )

    Enum.reverse(samples_acc)
  end

  defp prepare_context(spec, regression, h_list) do
    state_dim = Nx.axis_size(spec.f, 0)
    reg_start = Map.get(regression, :start_dim, 0)
    reg_dim = Map.get(regression, :num_dims, 0)

    if not is_integer(reg_start) or not is_integer(reg_dim) or reg_dim <= 0 do
      raise ArgumentError,
            "invalid regression metadata for spike-and-slab: #{inspect(regression)}"
    end

    reg_end = reg_start + reg_dim - 1

    if reg_start < 0 or reg_end >= state_dim do
      raise ArgumentError,
            "regression block [#{reg_start}, #{reg_end}] is out of bounds for state dim #{state_dim}"
    end

    reg_full_indices = Enum.to_list(reg_start..reg_end)
    reg_set = MapSet.new(reg_full_indices)
    struct_full_indices = Enum.reject(0..(state_dim - 1), &MapSet.member?(reg_set, &1))
    struct_dim = length(struct_full_indices)

    index_map =
      struct_full_indices
      |> Enum.with_index()
      |> Map.new(fn {full_idx, local_idx} -> {full_idx, local_idx} end)

    q_specs_struct =
      spec.q_specs
      |> Enum.reject(fn qs -> MapSet.member?(reg_set, qs.dim_index) end)
      |> Enum.map(fn qs ->
        local_dim = Map.fetch!(index_map, qs.dim_index)
        %{qs | dim_index: local_dim}
      end)

    h_struct_rows =
      if struct_dim == 0 do
        List.duplicate([], length(h_list))
      else
        Enum.map(h_list, fn h_t -> extract_row_values(h_t, struct_full_indices) end)
      end

    {h_struct_list, h_struct_tensor} =
      if struct_dim == 0 do
        {[], nil}
      else
        h_struct_list =
          Enum.map(h_struct_rows, fn row ->
            Nx.tensor([row])
          end)

        {h_struct_list, Nx.stack(h_struct_list)}
      end

    x_rows = Enum.map(h_list, &extract_row_values(&1, reg_full_indices))

    %{
      state_dim: state_dim,
      reg_dim: reg_dim,
      reg_full_indices: reg_full_indices,
      struct_dim: struct_dim,
      struct_full_indices: struct_full_indices,
      q_specs_struct: q_specs_struct,
      h_struct_rows: h_struct_rows,
      h_struct_list: h_struct_list,
      h_struct_tensor: h_struct_tensor,
      x_rows: x_rows,
      f_struct:
        if(struct_dim == 0,
          do: nil,
          else: submatrix(spec.f, struct_full_indices, struct_full_indices)
        ),
      x0_struct: if(struct_dim == 0, do: [], else: slice_vector(spec.x0, struct_full_indices)),
      p0_struct:
        if(struct_dim == 0,
          do: nil,
          else: submatrix(spec.p0, struct_full_indices, struct_full_indices)
        ),
      prior_inclusion: Map.get(regression, :prior_inclusion, min(0.5, 3.0 / reg_dim)),
      g_prior: Map.get(regression, :g, max(length(h_list), 1))
    }
    |> then(fn ctx ->
      if not is_number(ctx.prior_inclusion) or ctx.prior_inclusion <= 0.0 or
           ctx.prior_inclusion >= 1.0 do
        raise ArgumentError,
              "spike-and-slab prior_inclusion must be in (0, 1), got: #{inspect(ctx.prior_inclusion)}"
      end

      if not is_number(ctx.g_prior) or ctx.g_prior <= 0.0 do
        raise ArgumentError, "spike-and-slab g must be > 0, got: #{inspect(ctx.g_prior)}"
      end

      ctx
    end)
  end

  defp resample_gamma_g_prior(gamma, _y_obs, _x_obs, _sigma2, _pi, _g, key, _xtx_stats)
       when gamma == [] do
    {gamma, key}
  end

  defp resample_gamma_g_prior(
         gamma,
         y_obs,
         x_obs_rows,
         sigma2,
         prior_inclusion,
         g_prior,
         key,
         xtx_stats
       ) do
    if y_obs == [] do
      {gamma, key}
    else
      p = length(gamma)
      log_prior_odds = :math.log(prior_inclusion / (1.0 - prior_inclusion))
      {u_vec, key_next} = Nx.Random.uniform(key, 0.0, 1.0, shape: {p})
      uniforms = u_vec |> Nx.to_flat_list() |> List.to_tuple()
      stats = spike_slab_sufficient_stats(y_obs, x_obs_rows, xtx_stats)
      model0 = g_prior_model_from_gamma(stats, gamma)

      {gamma_new, _model_new} =
        Enum.reduce(0..(p - 1), {gamma, model0}, fn j, {gamma_curr, model_curr} ->
          {model_off, model_on} =
            if Enum.at(gamma_curr, j) == 1 do
              {remove_g_prior_column(stats, model_curr, j), model_curr}
            else
              {model_curr, add_g_prior_column(stats, model_curr, j)}
            end

          log_ml_off = log_marginal_g_prior(model_off, sigma2, g_prior)
          log_ml_on = log_marginal_g_prior(model_on, sigma2, g_prior)
          prob_on = logistic(log_prior_odds + log_ml_on - log_ml_off)
          gamma_j = if elem(uniforms, j) < prob_on, do: 1, else: 0

          {
            List.replace_at(gamma_curr, j, gamma_j),
            if(gamma_j == 1, do: model_on, else: model_off)
          }
        end)

      {gamma_new, key_next}
    end
  end

  defp spike_slab_xtx_stats(x_obs_rows) do
    p = x_obs_rows |> hd() |> length()
    x_t = Nx.tensor(x_obs_rows)
    xtx = compat_dot(Nx.transpose(x_t), x_t)

    %{
      p: p,
      x_cols: BstsNx.Utils.transpose_rows(x_obs_rows),
      xtx:
        xtx
        |> Nx.to_flat_list()
        |> Enum.chunk_every(p)
        |> to_tuple_matrix()
    }
  end

  defp spike_slab_sufficient_stats(y_obs, x_obs_rows, nil) do
    spike_slab_sufficient_stats(y_obs, x_obs_rows, spike_slab_xtx_stats(x_obs_rows))
  end

  defp spike_slab_sufficient_stats(y_obs, _x_obs_rows, xtx_stats) do
    Map.put(xtx_stats, :xty, spike_slab_xty(xtx_stats.x_cols, y_obs))
  end

  defp spike_slab_xty(x_cols, y_obs) do
    x_cols
    |> Enum.map(&Residuals.dot_list(&1, y_obs))
    |> List.to_tuple()
  end

  defp g_prior_model_from_gamma(stats, gamma) do
    gamma
    |> active_indices()
    |> g_prior_model_from_active(stats)
  end

  defp g_prior_model_from_active([], _stats), do: empty_g_prior_model()

  defp g_prior_model_from_active(active, stats) do
    k = length(active)

    xtx =
      Enum.map(active, fn row_idx ->
        Enum.map(active, &xtx_value(stats, row_idx, &1))
      end)
      |> Nx.tensor()

    inv_xtx =
      BstsNx.Utils.safe_solve(xtx, Nx.eye(k))
      |> Nx.to_flat_list()
      |> Enum.chunk_every(k)

    if finite_matrix?(inv_xtx) do
      build_g_prior_model(stats, active, inv_xtx)
    else
      empty_g_prior_model()
    end
  end

  defp empty_g_prior_model do
    %{active: [], inv_xtx: [], score: 0.0}
  end

  defp build_g_prior_model(stats, active, inv_xtx) do
    score = score_from_inverse(stats, active, inv_xtx)

    if finite_number?(score) do
      %{active: active, inv_xtx: inv_xtx, score: score}
    else
      empty_g_prior_model()
    end
  end

  defp add_g_prior_column(stats, model, j) do
    if j in model.active do
      model
    else
      do_add_g_prior_column(stats, model, j)
    end
  end

  defp do_add_g_prior_column(stats, %{active: []}, j) do
    pivot = xtx_value(stats, j, j)

    if usable_rank_pivot?(pivot) do
      build_g_prior_model(stats, [j], [[1.0 / pivot]])
    else
      g_prior_model_from_active([j], stats)
    end
  end

  defp do_add_g_prior_column(stats, model, j) do
    b = Enum.map(model.active, &xtx_value(stats, &1, j))
    inv_b = matrix_vector_product(model.inv_xtx, b)
    pivot = xtx_value(stats, j, j) - Residuals.dot_list(b, inv_b)

    if usable_rank_pivot?(pivot) do
      top_left =
        Enum.zip(model.inv_xtx, inv_b)
        |> Enum.map(fn {row, row_scale} ->
          Enum.zip(row, inv_b)
          |> Enum.map(fn {value, col_scale} ->
            value + row_scale * col_scale / pivot
          end)
        end)

      edge = Enum.map(inv_b, &(-&1 / pivot))

      inv_xtx =
        top_left
        |> Enum.zip(edge)
        |> Enum.map(fn {row, edge_value} -> row ++ [edge_value] end)
        |> Kernel.++([edge ++ [1.0 / pivot]])

      build_g_prior_model(stats, model.active ++ [j], inv_xtx)
    else
      g_prior_model_from_active(model.active ++ [j], stats)
    end
  end

  defp remove_g_prior_column(stats, model, j) do
    case Enum.find_index(model.active, &(&1 == j)) do
      nil ->
        model

      pos ->
        do_remove_g_prior_column(stats, model, pos)
    end
  end

  defp do_remove_g_prior_column(_stats, %{active: [_only]}, _pos), do: empty_g_prior_model()

  defp do_remove_g_prior_column(stats, model, pos) do
    pivot =
      model.inv_xtx
      |> Enum.at(pos)
      |> Enum.at(pos)

    active = List.delete_at(model.active, pos)

    if usable_rank_pivot?(pivot) do
      edge =
        model.inv_xtx
        |> Enum.with_index()
        |> Enum.reject(fn {_row, row_idx} -> row_idx == pos end)
        |> Enum.map(fn {row, _row_idx} -> Enum.at(row, pos) end)

      core =
        model.inv_xtx
        |> List.delete_at(pos)
        |> Enum.map(&List.delete_at(&1, pos))

      inv_xtx =
        core
        |> Enum.zip(edge)
        |> Enum.map(fn {row, row_scale} ->
          Enum.zip(row, edge)
          |> Enum.map(fn {value, col_scale} ->
            value - row_scale * col_scale / pivot
          end)
        end)

      build_g_prior_model(stats, active, inv_xtx)
    else
      g_prior_model_from_active(active, stats)
    end
  end

  defp log_marginal_g_prior(%{active: []}, _sigma2, _g_prior), do: 0.0

  defp log_marginal_g_prior(model, sigma2, g_prior) do
    k = length(model.active)
    denom = 2.0 * max(sigma2, 1.0e-12) * (1.0 + g_prior)
    -0.5 * k * :math.log(1.0 + g_prior) + g_prior * model.score / denom
  end

  defp score_from_inverse(stats, active, inv_xtx) do
    xty = Enum.map(active, &xty_value(stats, &1))
    Residuals.dot_list(xty, matrix_vector_product(inv_xtx, xty))
  end

  defp matrix_vector_product(matrix, vector) do
    Enum.map(matrix, &Residuals.dot_list(&1, vector))
  end

  defp xtx_value(stats, i, j), do: stats.xtx |> elem(i) |> elem(j)
  defp xty_value(stats, i), do: elem(stats.xty, i)

  defp usable_rank_pivot?(value) do
    finite_number?(value) and value > @rank_update_pivot_floor
  end

  defp finite_matrix?(matrix) do
    Enum.all?(matrix, fn row -> Enum.all?(row, &finite_number?/1) end)
  end

  defp finite_number?(value) do
    is_number(value) and value == value and abs(value) < 1.0e300
  end

  defp sample_beta_g_prior(gamma, y_obs, x_obs_rows, sigma2, g_prior, key) do
    p = length(gamma)
    active = active_indices(gamma)

    if active == [] or y_obs == [] do
      {List.duplicate(0.0, p), Nx.broadcast(0.0, {p, p}), key}
    else
      x_active = select_active_columns(x_obs_rows, active)
      x_t = Nx.tensor(x_active)
      y_t = Nx.tensor(y_obs)

      xtx = compat_dot(Nx.transpose(x_t), x_t)
      xty = compat_dot(Nx.transpose(x_t), y_t)
      inv_xtx = BstsNx.Utils.safe_solve(xtx, Nx.eye(length(active)))
      beta_ols = BstsNx.Utils.safe_solve(xtx, xty)

      shrink = g_prior / (1.0 + g_prior)
      mean_active = Nx.multiply(beta_ols, shrink)

      cov_active =
        inv_xtx
        |> Nx.multiply(shrink * max(sigma2, 1.0e-12))
        |> symmetrize()

      jitter = Nx.eye(length(active)) |> Nx.multiply(1.0e-9)
      chol = BstsNx.Utils.safe_cholesky(Nx.add(cov_active, jitter))
      {z, key_next} = Nx.Random.normal(key, 0.0, 1.0, shape: {length(active)})

      draw_active =
        Nx.add(mean_active, compat_dot(chol, z))
        |> then(fn draw ->
          if BstsNx.Utils.has_non_finite?(draw), do: mean_active, else: draw
        end)
        |> Nx.to_flat_list()

      cov_rows = Nx.to_flat_list(cov_active) |> Enum.chunk_every(length(active))

      beta_full =
        List.duplicate(0.0, p)
        |> put_active_values(active, draw_active)

      beta_cov =
        List.duplicate(List.duplicate(0.0, p), p)
        |> put_active_covariance(active, cov_rows)
        |> Nx.tensor()

      {beta_full, beta_cov, key_next}
    end
  end

  defp init_gamma_from_data(y_obs, x_obs_rows, prior_inclusion, p) do
    if y_obs == [] do
      List.duplicate(0, p)
    else
      expected = max(round(prior_inclusion * p), 1)
      x_cols = BstsNx.Utils.transpose_rows(x_obs_rows)

      active =
        x_cols
        |> Enum.with_index()
        |> Enum.map(fn {col, idx} -> {idx, abs(Residuals.dot_list(col, y_obs))} end)
        |> Enum.sort_by(fn {_idx, score} -> score end, :desc)
        |> Enum.take(min(expected, p))
        |> Enum.map(&elem(&1, 0))
        |> MapSet.new()

      Enum.map(0..(p - 1), fn idx -> if MapSet.member?(active, idx), do: 1, else: 0 end)
    end
  end

  defp build_full_q_matrix(state_dim, struct_full_indices, q_struct) do
    n_struct = length(struct_full_indices)

    if n_struct == 0 do
      Nx.broadcast(0.0, {state_dim, state_dim})
    else
      base = Nx.broadcast(0.0, {state_dim, state_dim})

      indices =
        for full_i <- struct_full_indices, full_j <- struct_full_indices do
          [full_i, full_j]
        end

      updates = Nx.reshape(q_struct, {n_struct * n_struct})
      Nx.indexed_put(base, Nx.tensor(indices), updates)
    end
  end

  defp build_full_state_trajectory(
         sampled_struct_states,
         beta,
         state_dim,
         struct_full_indices,
         reg_full_indices
       ) do
    Enum.map(sampled_struct_states, fn struct_state ->
      struct_vals = Residuals.state_values(struct_state)

      List.duplicate(0.0, state_dim)
      |> put_active_values(struct_full_indices, struct_vals)
      |> put_active_values(reg_full_indices, beta)
      |> Nx.tensor()
    end)
  end

  defp build_full_covariances(
         struct_covs,
         beta_cov,
         state_dim,
         struct_full_indices,
         reg_full_indices
       ) do
    if struct_full_indices == [] do
      List.duplicate(beta_cov, length(struct_covs))
    else
      do_build_full_covariances(
        struct_covs,
        beta_cov,
        state_dim,
        struct_full_indices,
        reg_full_indices
      )
    end
  end

  defp do_build_full_covariances(
         struct_covs,
         beta_cov,
         state_dim,
         struct_full_indices,
         reg_full_indices
       ) do
    struct_map =
      struct_full_indices
      |> Enum.with_index()
      |> Map.new(fn {full_idx, local_idx} -> {full_idx, local_idx} end)

    reg_map =
      reg_full_indices
      |> Enum.with_index()
      |> Map.new(fn {full_idx, local_idx} -> {full_idx, local_idx} end)

    beta_rows =
      beta_cov
      |> Nx.to_flat_list()
      |> Enum.chunk_every(max(length(reg_full_indices), 1))
      |> to_tuple_matrix()

    Enum.map(struct_covs, fn struct_cov ->
      struct_rows =
        struct_cov
        |> Nx.to_flat_list()
        |> Enum.chunk_every(max(length(struct_full_indices), 1))
        |> to_tuple_matrix()

      rows =
        Enum.map(0..(state_dim - 1), fn i ->
          Enum.map(0..(state_dim - 1), fn j ->
            cond do
              Map.get(struct_map, i) != nil and Map.get(struct_map, j) != nil ->
                i_local = Map.get(struct_map, i)
                j_local = Map.get(struct_map, j)
                struct_rows |> elem(i_local) |> elem(j_local)

              Map.get(reg_map, i) != nil and Map.get(reg_map, j) != nil ->
                i_local = Map.get(reg_map, i)
                j_local = Map.get(reg_map, j)
                beta_rows |> elem(i_local) |> elem(j_local)

              true ->
                0.0
            end
          end)
        end)

      Nx.tensor(rows)
    end)
  end

  defp extract_row_values(h_t, indices) do
    row = Structured.structured_h_row(h_t) |> Nx.to_flat_list() |> List.to_tuple()
    Enum.map(indices, &elem(row, &1))
  end

  defp active_indices(gamma) do
    gamma
    |> Enum.with_index()
    |> Enum.filter(fn {g, _idx} -> g == 1 end)
    |> Enum.map(&elem(&1, 1))
  end

  defp select_active_columns(rows, active) do
    rows_t = Enum.map(rows, &List.to_tuple/1)
    Enum.map(rows_t, fn row -> Enum.map(active, &elem(row, &1)) end)
  end

  defp put_active_values(base, indices, values) do
    Enum.zip(indices, values)
    |> Enum.reduce(base, fn {idx, val}, acc ->
      List.replace_at(acc, idx, val)
    end)
  end

  defp put_active_covariance(base_rows, active_indices, cov_rows) do
    cov_rows_t = to_tuple_matrix(cov_rows)

    active_pos =
      active_indices
      |> Enum.with_index()
      |> Map.new(fn {idx, pos} -> {idx, pos} end)

    Enum.with_index(base_rows)
    |> Enum.map(fn {row, i} ->
      Enum.with_index(row)
      |> Enum.map(fn {_v, j} ->
        i_pos = Map.get(active_pos, i)
        j_pos = Map.get(active_pos, j)

        if i_pos != nil and j_pos != nil do
          cov_rows_t |> elem(i_pos) |> elem(j_pos)
        else
          0.0
        end
      end)
    end)
  end

  defp slice_vector(_tensor, []), do: []

  defp slice_vector(tensor, indices) do
    vals = tensor |> Nx.flatten() |> Nx.to_flat_list() |> List.to_tuple()
    Enum.map(indices, &elem(vals, &1))
  end

  defp submatrix(_tensor, [], []), do: Nx.broadcast(0.0, {0, 0})

  defp submatrix(tensor, row_indices, col_indices) do
    {_rows, cols} = Nx.shape(tensor)

    row_data =
      tensor
      |> Nx.to_flat_list()
      |> Enum.chunk_every(cols)
      |> Enum.map(&List.to_tuple/1)
      |> List.to_tuple()

    row_indices
    |> Enum.map(fn r ->
      row = elem(row_data, r)
      Enum.map(col_indices, &elem(row, &1))
    end)
    |> Nx.tensor()
  end

  defp symmetrize(matrix), do: Nx.multiply(Nx.add(matrix, Nx.transpose(matrix)), 0.5)

  defp logistic(logit) when logit >= 35.0, do: 1.0
  defp logistic(logit) when logit <= -35.0, do: 0.0
  defp logistic(logit), do: 1.0 / (1.0 + :math.exp(-logit))

  defp to_tuple_matrix(rows) do
    rows
    |> Enum.map(&List.to_tuple/1)
    |> List.to_tuple()
  end
end
