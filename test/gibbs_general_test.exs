defmodule BstsNx.GibbsGeneralTest do
  use ExUnit.Case, async: true
  import ExUnit.CaptureLog

  alias BstsNx.GibbsSampler

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
    assert log =~ "missing values are skipped"

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
end
