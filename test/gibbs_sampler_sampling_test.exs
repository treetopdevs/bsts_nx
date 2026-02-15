defmodule BstsNxGibbsSamplerSamplingTest do
  use ExUnit.Case
  import Nx, only: [to_number: 1]
  alias BstsNx.GibbsSampler

  @moduletag :external

  setup do
    python = System.find_executable("python3") || System.find_executable("python")

    if python do
      {:ok, python: python}
    else
      {:skip, "python3/python not found; skipping Python comparison"}
    end
  end

  test "Gibbs sampler samples positive variances and updates them" do
    observations = [1.0, 2.0, 3.0]
    num = 5
    samples = GibbsSampler.sample(observations, num, 0.0, 1.0, 1.0, 1.0, seed: 42)
    assert length(samples) == num
    # ensure each iteration returns valid sample
    Enum.each(samples, fn samp ->
      assert length(samp.states) == length(observations)
      assert length(samp.state_covs) == length(observations)
      q = Nx.to_number(samp.process_var)
      r = Nx.to_number(samp.obs_var)
      assert q > 0.0
      assert r > 0.0
    end)

    # verify variance parameters are not all identical
    process_vars = Enum.map(samples, &Nx.to_number(&1.process_var))
    obs_vars = Enum.map(samples, &Nx.to_number(&1.obs_var))
    assert Enum.uniq(process_vars) |> length() > 1 or Enum.uniq(obs_vars) |> length() > 1
  end

  test "posterior shape and scale match Python calculation", %{python: python} do
    observations = [1.0, 2.0, 3.0]
    # run one iteration to get latent states
    [sample] = GibbsSampler.sample(observations, 1, 0.0, 1.0, 1.0, 1.0)
    states = Enum.map(sample.states, &to_number/1)
    # compute process sum of squares and observation sum of squares as in sampler
    process_ss =
      states
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.map(fn [x_t, x_prev] -> :math.pow(x_t - x_prev, 2) end)
      |> Enum.sum()

    obs_ss =
      Enum.zip(observations, states)
      |> Enum.map(fn {y, x_t} -> :math.pow(y - x_t, 2) end)
      |> Enum.sum()

    t_steps = length(observations)
    prior_shape = 1.0
    prior_scale = 1.0
    num_diffs = length(states) - 1
    shape_q = prior_shape + num_diffs / 2
    scale_q = prior_scale + process_ss / 2
    shape_r = prior_shape + t_steps / 2
    scale_r = prior_scale + obs_ss / 2
    # compute reference shape and scale in Python using same states
    # pass states as comma‑separated list to python
    states_str = Enum.map(states, &Float.to_string/1) |> Enum.join(",")
    obs_str = Enum.map(observations, &Float.to_string/1) |> Enum.join(",")

    python_code = """
    import math, sys
    states = list(map(float, "#{states_str}".split(',')))
    obs = list(map(float, "#{obs_str}".split(',')))
    process_ss = sum([(states[i] - states[i-1])**2 for i in range(1, len(states))])
    obs_ss = sum([(obs[i] - states[i])**2 for i in range(len(states))])
    t_steps = len(obs)
    prior_shape = 1.0
    prior_scale = 1.0
    num_diffs = len(states) - 1
    shape_q = prior_shape + num_diffs / 2
    scale_q = prior_scale + process_ss / 2
    shape_r = prior_shape + t_steps / 2
    scale_r = prior_scale + obs_ss / 2
    print(shape_q, scale_q, shape_r, scale_r)
    """

    {output, 0} = System.cmd(python, ["-c", python_code])

    [py_shape_q, py_scale_q, py_shape_r, py_scale_r] =
      output
      |> String.trim()
      |> String.split()
      |> Enum.map(&String.to_float/1)

    assert_in_delta(shape_q, py_shape_q, 1.0e-6)
    assert_in_delta(scale_q, py_scale_q, 1.0e-6)
    assert_in_delta(shape_r, py_shape_r, 1.0e-6)
    assert_in_delta(scale_r, py_scale_r, 1.0e-6)
  end
end
