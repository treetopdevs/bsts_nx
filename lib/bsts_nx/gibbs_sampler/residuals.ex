defmodule BstsNx.GibbsSampler.Residuals do
  @moduledoc false

  import BstsNx.Utils, only: [missing_observation?: 1]

  def process_sum_of_squares(xs, f_t) do
    t = Nx.axis_size(xs, 0)

    if t < 2 do
      0.0
    else
      prev = Nx.slice(xs, [0], [t - 1])
      curr = Nx.slice(xs, [1], [t - 1])

      curr
      |> Nx.subtract(Nx.multiply(f_t, prev))
      |> then(&Nx.multiply(&1, &1))
      |> Nx.sum()
      |> Nx.to_number()
    end
  end

  def obs_sum_of_squares(obs_tensor, xs, h_tensor, obs_present_mask) do
    preds = Nx.multiply(h_tensor, xs)
    diffs = Nx.subtract(obs_tensor, preds)

    diffs
    |> then(&Nx.multiply(&1, &1))
    |> then(&Nx.select(obs_present_mask, &1, 0.0))
    |> Nx.sum()
    |> Nx.to_number()
  end

  def process_structured(%Nx.Tensor{} = sampled_states, f_t, q_specs) do
    t = Nx.axis_size(sampled_states, 0)
    state_dim = Nx.axis_size(f_t, 0)

    if t < 2 do
      Map.new(q_specs, fn qs -> {qs.dim_index, 0.0} end)
    else
      prev_states = Nx.slice(sampled_states, [0, 0], [t - 1, state_dim])
      next_states = Nx.slice(sampled_states, [1, 0], [t - 1, state_dim])
      predicted = Nx.dot(prev_states, Nx.transpose(f_t))
      residuals = Nx.subtract(next_states, predicted)

      ss_vec =
        residuals
        |> Nx.multiply(residuals)
        |> Nx.sum(axes: [0])
        |> Nx.to_flat_list()
        |> List.to_tuple()

      Map.new(q_specs, fn qs ->
        ss_val =
          if qs.dim_index < tuple_size(ss_vec), do: elem(ss_vec, qs.dim_index), else: 0.0

        {qs.dim_index, ss_val |> safe_to_number()}
      end)
    end
  end

  def process_structured(sampled_states, f_t, q_specs) when is_list(sampled_states) do
    sampled_states
    |> Nx.stack()
    |> process_structured(f_t, q_specs)
  end

  def obs_structured(
        observations,
        %Nx.Tensor{} = sampled_states,
        %Nx.Tensor{} = h_rows
      ) do
    t = Nx.axis_size(sampled_states, 0)

    if t == 0 do
      {0.0, 0}
    else
      preds = Nx.sum(Nx.multiply(h_rows, sampled_states), axes: [1])

      obs_tensor = observations_to_filter_tensor(observations)
      mask = Nx.equal(obs_tensor, obs_tensor)
      diffs = Nx.subtract(obs_tensor, preds)

      ss =
        diffs
        |> Nx.multiply(diffs)
        |> then(&Nx.select(mask, &1, 0.0))
        |> Nx.sum()
        |> Nx.to_number()

      count = mask |> Nx.sum() |> Nx.to_number() |> round()
      {ss, count}
    end
  end

  def obs_structured(observations, sampled_states, h_list)
      when is_list(sampled_states) and is_list(h_list) do
    h_rows = h_list |> Enum.map(&structured_h_row/1) |> Enum.map(&Nx.flatten/1) |> Nx.stack()

    observations
    |> obs_structured(Nx.stack(sampled_states), h_rows)
  end

  def regression_residual_pairs(observations, sampled_struct_states, h_struct_rows, x_rows) do
    observations
    |> Enum.zip(sampled_struct_states)
    |> Enum.zip(h_struct_rows)
    |> Enum.zip(x_rows)
    |> Enum.reduce({[], []}, fn {{{y, state}, h_row}, x_row}, {acc_y, acc_x} ->
      if missing_observation?(y) do
        {acc_y, acc_x}
      else
        y_val = observation_to_number(y)
        struct_pred = dot_list(h_row, state_values(state))
        {[y_val - struct_pred | acc_y], [x_row | acc_x]}
      end
    end)
    |> then(fn {ys, xs} -> {Enum.reverse(ys), Enum.reverse(xs)} end)
  end

  def observed_regression_pairs(observations, x_rows) do
    observations
    |> Enum.zip(x_rows)
    |> Enum.reduce({[], []}, fn {y, x_row}, {acc_y, acc_x} ->
      if missing_observation?(y) do
        {acc_y, acc_x}
      else
        {[observation_to_number(y) | acc_y], [x_row | acc_x]}
      end
    end)
    |> then(fn {ys, xs} -> {Enum.reverse(ys), Enum.reverse(xs)} end)
  end

  def adjust_observations_for_regression(observations, x_rows, beta) do
    observations
    |> Enum.zip(x_rows)
    |> Enum.map(fn {y, x_row} ->
      if missing_observation?(y) do
        y
      else
        observation_to_number(y) - dot_list(x_row, beta)
      end
    end)
  end

  def obs_spike_slab(observations, sampled_struct_states, h_struct_rows, x_rows, beta) do
    observations
    |> Enum.zip(sampled_struct_states)
    |> Enum.zip(h_struct_rows)
    |> Enum.zip(x_rows)
    |> Enum.reduce({0.0, 0}, fn {{{y, state}, h_row}, x_row}, {acc_ss, acc_count} ->
      if missing_observation?(y) do
        {acc_ss, acc_count}
      else
        y_val = observation_to_number(y)
        struct_pred = dot_list(h_row, state_values(state))
        reg_pred = dot_list(x_row, beta)
        diff = y_val - struct_pred - reg_pred
        {acc_ss + diff * diff, acc_count + 1}
      end
    end)
  end

  def observations_to_filter_tensor(observations) do
    nan = Nx.Constants.nan() |> Nx.to_number()

    observations
    |> Enum.map(fn y ->
      if missing_observation?(y), do: nan, else: observation_to_number(y)
    end)
    |> Nx.tensor(type: {:f, 64})
  end

  def structured_h_row(%Nx.Tensor{} = h_t) do
    case Nx.rank(h_t) do
      0 ->
        Nx.reshape(h_t, {1})

      1 ->
        h_t

      2 ->
        cond do
          Nx.axis_size(h_t, 0) == 1 ->
            Nx.squeeze(h_t, axes: [0])

          Nx.axis_size(h_t, 1) == 1 ->
            Nx.squeeze(h_t, axes: [1])

          true ->
            raise ArgumentError,
                  "structured scalar-observation residual expects rank-2 H with a singleton axis, got shape #{inspect(Nx.shape(h_t))}"
        end

      _ ->
        raise ArgumentError,
              "structured scalar-observation residual expects rank <= 2 H, got rank #{Nx.rank(h_t)}"
    end
  end

  def state_values(%Nx.Tensor{} = state), do: Nx.to_flat_list(state)
  def state_values(values) when is_list(values), do: values

  def observation_to_number(%Nx.Tensor{} = v), do: Nx.to_number(v)
  def observation_to_number(v) when is_number(v), do: v * 1.0

  def safe_to_number(%Nx.Tensor{} = t), do: Nx.to_number(t) |> safe_to_number()

  def safe_to_number(val) when is_number(val) do
    if val == val and abs(val) < 1.0e300 do
      val * 1.0
    else
      raise ArgumentError, "expected a finite numeric posterior parameter, got: #{inspect(val)}"
    end
  end

  def safe_to_number(other) do
    raise ArgumentError, "expected a finite numeric posterior parameter, got: #{inspect(other)}"
  end

  def dot_list(xs, ys) do
    Enum.zip(xs, ys)
    |> Enum.reduce(0.0, fn {x, y}, acc -> acc + x * y end)
  end
end
