defmodule BstsNxPythonComparisonTest do
  use ExUnit.Case
  import Nx, only: [to_number: 1]
  alias BstsNx.KalmanFilter

  @moduletag :external

  setup do
    python = System.find_executable("python3") || System.find_executable("python")

    if python do
      {:ok, python: python}
    else
      {:skip, "python3/python not found; skipping Python comparison"}
    end
  end

  test "Kalman filter matches Python implementation", %{python: python} do
    observations = [1.0, 2.0, 3.0]
    # run Nx implementation
    nx_estimates = KalmanFilter.filter(observations, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0)
    nx_vals = Enum.map(nx_estimates, fn {x, p} -> {to_number(x), to_number(p)} end)

    # run equivalent Python implementation via system call
    python_code = """
    import numpy as np
    obs = [1.0, 2.0, 3.0]
    F, H, Q, R = 1.0, 1.0, 1.0, 1.0
    x, P = 0.0, 1.0
    out = []
    for z in obs:
        # prediction
        x_pred = F * x
        P_pred = F * P * F + Q
        # update
        y = z - H * x_pred
        S = H * P_pred * H + R
        K = P_pred * H / S
        x = x_pred + K * y
        P = (1 - K * H) * P_pred
        out.append((x, P))
    for (xv, pv) in out:
        print(f"{xv},{pv}")
    """

    {output, 0} = System.cmd(python, ["-c", python_code])

    python_vals =
      output
      |> String.trim()
      |> String.split("\n")
      |> Enum.map(fn line ->
        [x_str, p_str] = String.split(line, ",")
        {String.to_float(x_str), String.to_float(p_str)}
      end)

    # compare Nx and Python outputs
    Enum.zip(nx_vals, python_vals)
    |> Enum.each(fn {{x_nx, p_nx}, {x_py, p_py}} ->
      assert_in_delta x_nx, x_py, 1.0e-6
      assert_in_delta p_nx, p_py, 1.0e-6
    end)
  end
end
