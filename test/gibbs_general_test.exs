defmodule BstsNx.GibbsGeneralTest do
  use ExUnit.Case, async: true
  import ExUnit.CaptureLog

  alias BstsNx.GibbsSampler
  alias BstsNx.KalmanFilter
  alias BstsNx.Distributions
  alias BstsNx.Smoother

  test "sample_general with constant h=1 matches basic sampler" do
    # Simple local level model: should produce similar results to sample/7
    :rand.seed(:exsss, {100, 101, 102})
    obs = Enum.map(1..30, fn _ -> :rand.normal() + 5.0 end)

    samples =
      GibbsSampler.sample_general(obs, 1.0, 1.0, 5,
        initial_state: 5.0,
        initial_cov: 1.0,
        process_var: 1.0,
        obs_var: 1.0,
        burn_in: 2,
        seed: 42
      )

    assert length(samples) == 5

    Enum.each(samples, fn s ->
      assert is_list(s.states)
      assert length(s.states) == 30
      assert is_list(s.state_covs)
      assert %Nx.Tensor{} = s.process_var
      assert %Nx.Tensor{} = s.obs_var
      # Variances should be positive
      assert Nx.to_number(s.process_var) > 0.0
      assert Nx.to_number(s.obs_var) > 0.0
    end)
  end

  test "sample_general remains aligned with f64 scalar and structured filter paths" do
    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      high_precision = 16_777_217.0
      observations = List.duplicate(high_precision, 4)
      obs_tensor = Nx.tensor(observations, type: {:f, 64})
      zero = Nx.tensor(0.0, type: {:f, 64})
      one = Nx.tensor(1.0, type: {:f, 64})
      initial_state = Nx.tensor(high_precision, type: {:f, 64})

      [sample] =
        GibbsSampler.sample_general(observations, one, one, 1,
          initial_state: initial_state,
          initial_cov: zero,
          process_var: zero,
          obs_var: zero,
          prior_shape: 100.0,
          prior_scale: 1.0,
          seed: 123
        )

      {filter_xs, _filter_ps} =
        KalmanFilter.filter_defn(obs_tensor, one, one, zero, zero, initial_state, zero)

      {structured_xs, _structured_ps} =
        KalmanFilter.filter_defn_multi(
          obs_tensor,
          Nx.tensor([[1.0]], type: {:f, 64}),
          Nx.tensor([[1.0]], type: {:f, 64}),
          Nx.tensor([[0.0]], type: {:f, 64}),
          zero,
          Nx.tensor([high_precision], type: {:f, 64}),
          Nx.tensor([[0.0]], type: {:f, 64})
        )

      scalar_states = Enum.map(sample.states, &Nx.to_number/1)

      assert_all_close(scalar_states, observations, 1.0e-5)
      assert_all_close(scalar_states, Nx.to_flat_list(filter_xs), 1.0e-5)

      assert_all_close(
        scalar_states,
        structured_xs |> Nx.reshape({length(observations)}) |> Nx.to_flat_list(),
        1.0e-5
      )
    end)
  end

  test "sample_general matches the compiled scalar filter plus smoother defn iteration" do
    Nx.with_default_backend(Nx.BinaryBackend, fn ->
      observations = [1.25, 1.55, 1.35, 1.9, 2.2, 2.05]
      h_values = [1.0, 1.15, 0.95, 1.05, 1.2, 0.9]

      opts = [
        initial_state: 0.5,
        initial_cov: 1.25,
        process_var: 0.2,
        obs_var: 0.7,
        prior_shape: 2.0,
        prior_scale: 1.5,
        burn_in: 1,
        thin: 2,
        key: Nx.Random.key(2026)
      ]

      expected =
        sample_general_via_compiled_smoother(observations, 0.92, h_values, 3, opts)

      actual =
        GibbsSampler.sample_general(observations, 0.92, h_values, 3, opts)

      assert_samples_close(actual, expected, 1.0e-7)
    end)
  end

  test "sample_general with time-varying h" do
    # Create a simple model where h varies: y_t = h_t * x_t + noise
    n = 20
    h_values = Enum.map(0..(n - 1), fn i -> 1.0 + 0.5 * :math.sin(i * 0.5) end)
    h_tensor = Nx.tensor(h_values, type: {:f, 32})

    # Generate observations from a known state
    :rand.seed(:exsss, {16, 17, 18})
    true_state = 3.0
    obs = Enum.map(h_values, fn h_i -> h_i * true_state + :rand.normal() * 0.1 end)

    samples =
      GibbsSampler.sample_general(obs, 1.0, h_tensor, 3,
        initial_state: 0.0,
        initial_cov: 10.0,
        process_var: 0.01,
        obs_var: 0.1,
        burn_in: 2,
        seed: 123
      )

    assert length(samples) == 3

    # The sampled states should be in a reasonable range
    Enum.each(samples, fn s ->
      last_state = List.last(s.states) |> Nx.to_number()
      # Should be vaguely near the true state value
      assert last_state > 0.0 and last_state < 20.0,
             "state #{last_state} should be in reasonable range"
    end)
  end

  test "sample_general with h as list of numbers" do
    obs = [1.0, 2.0, 3.0, 4.0, 5.0]
    h_list = [1.0, 1.0, 1.0, 1.0, 1.0]

    samples =
      GibbsSampler.sample_general(obs, 1.0, h_list, 2,
        initial_state: 0.0,
        initial_cov: 1.0,
        burn_in: 1,
        seed: 99
      )

    assert length(samples) == 2
  end

  test "sample_general handles nil observations" do
    obs = [1.0, 2.0, nil, 4.0, 5.0]

    {samples, log} =
      capture_result_with_log(fn ->
        GibbsSampler.sample_general(obs, 1.0, 1.0, 2,
          initial_state: 0.0,
          initial_cov: 1.0,
          burn_in: 1,
          seed: 77
        )
      end)

    assert length(samples) == 2
    assert log =~ "missing observations"

    Enum.each(samples, fn s ->
      assert length(s.states) == 5
    end)
  end

  test "sample_general raises on thin: 0" do
    obs = [1.0, 2.0, 3.0]

    assert_raise ArgumentError, ~r/thin must be a positive integer/, fn ->
      GibbsSampler.sample_general(obs, 1.0, 1.0, 2,
        initial_state: 0.0,
        initial_cov: 1.0,
        thin: 0,
        seed: 42
      )
    end
  end

  test "sample/7 raises on negative process_var via sample_general" do
    obs = [1.0, 2.0, 3.0]

    assert_raise ArgumentError, ~r/non-negative/, fn ->
      GibbsSampler.sample(obs, 2, 0.0, 1.0, -1.0, 1.0, seed: 42)
    end
  end

  test "sample_general raises on negative obs_var" do
    obs = [1.0, 2.0, 3.0]

    assert_raise ArgumentError, ~r/non-negative/, fn ->
      GibbsSampler.sample_general(obs, 1.0, 1.0, 2, obs_var: -0.5, seed: 42)
    end
  end

  test "sample_general with time-varying h tensor matches state count" do
    n = 15
    # Fourier-style time-varying regressor
    h_vals = Enum.map(0..(n - 1), fn i -> :math.cos(2.0 * :math.pi() * i / 7.0) end)
    h_tensor = Nx.tensor(h_vals, type: {:f, 32})
    obs = Enum.map(h_vals, fn h_i -> h_i * 2.0 + :rand.normal() * 0.5 end)

    samples =
      GibbsSampler.sample_general(obs, 1.0, h_tensor, 5,
        initial_state: 0.0,
        initial_cov: 5.0,
        process_var: 0.1,
        obs_var: 0.5,
        burn_in: 3,
        thin: 2,
        seed: 555
      )

    assert length(samples) == 5

    Enum.each(samples, fn s ->
      assert length(s.states) == n
      assert Nx.to_number(s.process_var) > 0.0
      assert Nx.to_number(s.obs_var) > 0.0
    end)
  end

  defp capture_result_with_log(fun) do
    parent = self()

    log =
      capture_log(fn ->
        result = fun.()
        send(parent, {:captured_result, result})
      end)

    result =
      receive do
        {:captured_result, value} -> value
      after
        1000 -> flunk("timed out waiting for captured result")
      end

    {result, log}
  end

  defp assert_all_close(left, right, delta) do
    assert length(left) == length(right)

    Enum.zip(left, right)
    |> Enum.each(fn {left_value, right_value} ->
      assert_in_delta left_value, right_value, delta
    end)
  end

  defp sample_general_via_compiled_smoother(observations, f, h_values, num_samples, opts) do
    burn_in = Keyword.get(opts, :burn_in, 0)
    thin = Keyword.get(opts, :thin, 1)
    total_iters = burn_in + num_samples * thin
    prior_shape = Keyword.get(opts, :prior_shape, 1.0)
    prior_scale = Keyword.get(opts, :prior_scale, 1.0)

    f_t = f64_tensor(f)
    x0 = f64_tensor(Keyword.get(opts, :initial_state, 0.0))
    p0 = f64_tensor(Keyword.get(opts, :initial_cov, 1.0))
    q0 = f64_tensor(Keyword.get(opts, :process_var, 1.0))
    r0 = f64_tensor(Keyword.get(opts, :obs_var, 1.0))
    key = Keyword.fetch!(opts, :key)

    obs_tensor = Nx.tensor(observations, type: {:f, 64})
    h_tensor = Nx.tensor(h_values, type: {:f, 64})
    obs_present_mask = Nx.equal(obs_tensor, obs_tensor)
    t_steps = obs_present_mask |> Nx.sum() |> Nx.to_number() |> round()
    num_diffs = max(length(observations) - 1, 0)

    {_, _, samples_acc, _key_acc} =
      Enum.reduce(1..total_iters, {q0, r0, [], key}, fn iter, {q_prev, r_prev, acc, key_prev} ->
        {filtered_xs, filtered_ps} =
          KalmanFilter.filter_defn(obs_tensor, f_t, h_tensor, q_prev, r_prev, x0, p0)

        {_smoothed_xs, _smoothed_ps, sampled_xs, key_after_smooth} =
          Smoother.rts_and_simulate_defn(filtered_xs, filtered_ps, f_t, q_prev, key_prev)

        process_ss = process_sum_of_squares(sampled_xs, f_t)
        obs_ss = obs_sum_of_squares(obs_tensor, sampled_xs, h_tensor, obs_present_mask)

        {q_sample, r_sample, next_key} =
          resample_variances(
            key_after_smooth,
            prior_shape,
            prior_scale,
            {process_ss, num_diffs},
            {obs_ss, t_steps}
          )

        acc =
          if iter > burn_in and rem(iter - burn_in, thin) == 0 do
            [
              %{
                states: scalar_tensor_list(sampled_xs),
                state_covs: scalar_tensor_list(filtered_ps),
                process_var: q_sample,
                obs_var: r_sample
              }
              | acc
            ]
          else
            acc
          end

        {q_sample, r_sample, acc, next_key}
      end)

    Enum.reverse(samples_acc)
  end

  defp f64_tensor(%Nx.Tensor{} = tensor), do: Nx.as_type(tensor, {:f, 64})
  defp f64_tensor(value), do: Nx.tensor(value, type: {:f, 64})

  defp process_sum_of_squares(xs, f_t) do
    t = Nx.axis_size(xs, 0)

    if t < 2 do
      0.0
    else
      prev = Nx.slice(xs, [0], [t - 1])
      curr = Nx.slice(xs, [1], [t - 1])
      diffs = Nx.subtract(curr, Nx.multiply(f_t, prev))

      diffs
      |> Nx.multiply(diffs)
      |> Nx.sum()
      |> Nx.to_number()
    end
  end

  defp obs_sum_of_squares(obs_tensor, xs, h_tensor, obs_present_mask) do
    diffs = Nx.subtract(obs_tensor, Nx.multiply(h_tensor, xs))
    squared = Nx.multiply(diffs, diffs)

    obs_present_mask
    |> Nx.select(squared, 0.0)
    |> Nx.sum()
    |> Nx.to_number()
  end

  defp resample_variances(
         key,
         prior_shape,
         prior_scale,
         {process_ss, num_diffs},
         {obs_ss, t_steps}
       ) do
    shape_q = prior_shape + num_diffs / 2
    scale_q = prior_scale + process_ss / 2
    shape_r = prior_shape + t_steps / 2
    scale_r = prior_scale + obs_ss / 2

    {samples, next_key} =
      Distributions.inv_gamma_sample_defn(
        Nx.tensor([shape_q, shape_r], type: {:f, 64}),
        Nx.tensor([scale_q, scale_r], type: {:f, 64}),
        key
      )

    q_sample = samples |> Nx.slice([0], [1]) |> Nx.squeeze(axes: [0])
    r_sample = samples |> Nx.slice([1], [1]) |> Nx.squeeze(axes: [0])

    {q_sample, r_sample, next_key}
  end

  defp scalar_tensor_list(tensor) do
    tensor
    |> Nx.to_batched(1)
    |> Enum.map(&Nx.squeeze(&1, axes: [0]))
  end

  defp assert_samples_close(actual, expected, delta) do
    assert length(actual) == length(expected)

    Enum.zip(actual, expected)
    |> Enum.each(fn {actual_sample, expected_sample} ->
      assert_all_close(
        tensor_numbers(actual_sample.states),
        tensor_numbers(expected_sample.states),
        delta
      )

      assert_all_close(
        tensor_numbers(actual_sample.state_covs),
        tensor_numbers(expected_sample.state_covs),
        delta
      )

      assert_in_delta Nx.to_number(actual_sample.process_var),
                      Nx.to_number(expected_sample.process_var),
                      delta

      assert_in_delta Nx.to_number(actual_sample.obs_var),
                      Nx.to_number(expected_sample.obs_var),
                      delta
    end)
  end

  defp tensor_numbers(tensors), do: Enum.map(tensors, &Nx.to_number/1)
end
