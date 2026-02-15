defmodule BstsNxDistributionsStateSpaceTest do
  use ExUnit.Case
  import Nx, only: [to_number: 1]
  alias BstsNx.Distributions
  alias BstsNx.StateSpace

  @moduletag :external

  setup do
    python = System.find_executable("python3") || System.find_executable("python")

    if python do
      {:ok, python: python}
    else
      {:skip, "python3/python not found; skipping Python comparison"}
    end
  end

  test "inverse–gamma sampler returns positive tensor and matches mean", %{python: python} do
    alpha = 2.5
    beta = 1.0
    # draw many samples to estimate mean
    n = 10_000

    samples =
      Enum.map(1..n, fn _ ->
        x = Distributions.inv_gamma_sample(alpha, beta)
        to_number(x)
      end)

    nx_mean = Enum.sum(samples) / n
    # compute reference mean via Python sampling
    python_code = """
    import numpy as np
    alpha, beta = 2.5, 1.0
    n = 10000
    gamma_samples = np.random.gamma(shape=alpha, scale=1.0, size=n)
    inv_gamma = beta / gamma_samples
    print(inv_gamma.mean())
    """

    {output, 0} = System.cmd(python, ["-c", python_code])
    py_mean = output |> String.trim() |> String.to_float()
    # theoretical mean of InvGamma(alpha,beta) = beta/(alpha-1) for alpha>1
    true_mean = beta / (alpha - 1)
    # our empirical mean should be close to theoretical and Python
    assert nx_mean > 0
    assert_in_delta nx_mean, py_mean, 0.05
    assert_in_delta nx_mean, true_mean, 0.05
  end

  test "block diagonal construction matches Python", %{python: python} do
    f1 = Nx.tensor([[1.0, 0.1], [0.0, 1.0]])
    f2 = Nx.tensor([[0.9]])
    blk = StateSpace.block_diag([f1, f2])
    nx_vals = blk |> Nx.to_flat_list()
    shape = Nx.shape(blk)
    {_rows, _cols} = shape
    # compute reference using numpy
    python_code = """
    import numpy as np
    f1 = np.array([[1.0, 0.1], [0.0, 1.0]])
    f2 = np.array([[0.9]])
    blk = np.block([
        [f1, np.zeros((2, 1))],
        [np.zeros((1, 2)), f2]
    ])
    print(blk.reshape(-1))
    """

    {output, 0} = System.cmd(python, ["-c", python_code])
    py_vals = parse_numpy_flat(output)
    # compare flattened values
    Enum.zip(nx_vals, py_vals) |> Enum.each(fn {a, b} -> assert_in_delta a, b, 1.0e-6 end)
    assert Nx.shape(blk) == {3, 3}
  end

  test "compose two components matches Python", %{python: python} do
    comp1 = %{f: Nx.tensor([[1.0]]), q: Nx.tensor([[1.0]]), h: Nx.tensor([[1.0, 0.0]])}
    comp2 = %{f: Nx.tensor([[0.5]]), q: Nx.tensor([[0.5]]), h: Nx.tensor([[0.0, 1.0]])}
    comp = StateSpace.compose(comp1, comp2)
    # flatten matrices for comparison
    {f, q, h} = {comp.f, comp.q, comp.h}
    nx_f = Nx.to_flat_list(f)
    nx_q = Nx.to_flat_list(q)
    nx_h = Nx.to_flat_list(h)

    python_code = """
    import numpy as np
    f1 = np.array([[1.0]])
    q1 = np.array([[1.0]])
    h1 = np.array([[1.0, 0.0]])
    f2 = np.array([[0.5]])
    q2 = np.array([[0.5]])
    h2 = np.array([[0.0, 1.0]])
    F = np.block([
        [f1, np.zeros((1, 1))],
        [np.zeros((1, 1)), f2]
    ])
    Q = np.block([
        [q1, np.zeros((1, 1))],
        [np.zeros((1, 1)), q2]
    ])
    H = np.concatenate([h1, h2], axis=1)
    print(F.reshape(-1))
    print(Q.reshape(-1))
    print(H.reshape(-1))
    """

    {output, 0} = System.cmd(python, ["-c", python_code])
    [f_line, q_line, h_line] = output |> String.trim() |> String.split("\n")
    py_f = parse_numpy_flat(f_line)
    py_q = parse_numpy_flat(q_line)
    py_h = parse_numpy_flat(h_line)
    Enum.zip(nx_f, py_f) |> Enum.each(fn {a, b} -> assert_in_delta a, b, 1.0e-6 end)
    Enum.zip(nx_q, py_q) |> Enum.each(fn {a, b} -> assert_in_delta a, b, 1.0e-6 end)
    Enum.zip(nx_h, py_h) |> Enum.each(fn {a, b} -> assert_in_delta a, b, 1.0e-6 end)
    assert Nx.shape(f) == {2, 2}
    assert Nx.shape(q) == {2, 2}
    assert Nx.shape(h) == {1, 4}
  end

  defp parse_numpy_flat(output) when is_binary(output) do
    output
    |> String.trim()
    |> String.replace(~r/[\[\],]/, " ")
    |> String.split()
    |> Enum.map(fn token ->
      token = if String.ends_with?(token, "."), do: token <> "0", else: token
      String.to_float(token)
    end)
  end
end
