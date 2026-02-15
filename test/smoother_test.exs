defmodule BstsNxSmootherTest do
  use ExUnit.Case
  import Nx, only: [to_number: 1]
  alias BstsNx.KalmanFilter
  alias BstsNx.Smoother

  @moduletag :external

  setup do
    python = System.find_executable("python3") || System.find_executable("python")

    if python do
      {:ok, python: python}
    else
      {:skip, "python3/python not found; skipping Python comparison"}
    end
  end

  test "RTS smoother matches Python implementation", %{python: python} do
    observations = [1.0, 2.0, 3.0]
    # run Kalman filter with predictions
    {filtered, predicted} =
      KalmanFilter.filter_with_pred(observations, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0)

    smoothed = Smoother.rts(filtered, predicted, 1.0)
    nx_vals = Enum.map(smoothed, fn {x, p} -> {to_number(x), to_number(p)} end)
    # compute smoothed results in Python
    python_code = """
    import numpy as np
    obs = [1.0, 2.0, 3.0]
    F, H, Q, R = 1.0, 1.0, 1.0, 1.0
    x0, P0 = 0.0, 1.0
    x_pred_list = []
    P_pred_list = []
    x_filt = []
    P_filt = []
    x, P = x0, P0
    for z in obs:
        x_pred = F * x
        P_pred = F * P * F + Q
        x_pred_list.append(x_pred)
        P_pred_list.append(P_pred)
        y = z - H * x_pred
        S = H * P_pred * H + R
        K = P_pred * H / S
        x = x_pred + K * y
        P = (1 - K * H) * P_pred
        x_filt.append(x)
        P_filt.append(P)
    T = len(obs)
    x_smooth = [None] * T
    P_smooth = [None] * T
    x_smooth[T-1] = x_filt[T-1]
    P_smooth[T-1] = P_filt[T-1]
    for k in range(T-2, -1, -1):
        C = P_filt[k] * F / P_pred_list[k+1]
        x_smooth[k] = x_filt[k] + C * (x_smooth[k+1] - x_pred_list[k+1])
        P_smooth[k] = P_filt[k] + C * C * (P_smooth[k+1] - P_pred_list[k+1])
    for xv, pv in zip(x_smooth, P_smooth):
        print(f"{xv},{pv}")
    """

    {output, 0} = System.cmd(python, ["-c", python_code])

    py_vals =
      output
      |> String.trim()
      |> String.split("\n")
      |> Enum.map(fn line ->
        [x_str, p_str] = String.split(line, ",")
        {String.to_float(x_str), String.to_float(p_str)}
      end)

    # compare results
    Enum.zip(nx_vals, py_vals)
    |> Enum.each(fn {{x_nx, p_nx}, {x_py, p_py}} ->
      assert_in_delta x_nx, x_py, 1.0e-6
      assert_in_delta p_nx, p_py, 1.0e-6
    end)
  end

  test "simulation smoother reproduces Python trajectory with fixed noise", %{python: python} do
    observations = [1.0, 2.0, 3.0]
    # run Kalman filter with predictions
    {filtered, predicted} =
      KalmanFilter.filter_with_pred(observations, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0)

    smoothed = Smoother.rts(filtered, predicted, 1.0)
    # choose fixed noise draws (one per time step)
    # Use scalar tensors (shape {}) to match the scalar state-space model.
    noise = [Nx.tensor(0.1), Nx.tensor(-0.5), Nx.tensor(0.2)]
    # run Nx simulation smoother
    sample = Smoother.simulate(smoothed, filtered, predicted, 1.0, noise)
    nx_sample = Enum.map(sample, &to_number/1)
    # run equivalent Python simulation smoother with same noise
    python_code = """
    import numpy as np
    obs = [1.0, 2.0, 3.0]
    F, H, Q, R = 1.0, 1.0, 1.0, 1.0
    x0, P0 = 0.0, 1.0
    eps = [0.1, -0.5, 0.2]
    # forward filter
    x_pred_list = []
    P_pred_list = []
    x_filt = []
    P_filt = []
    x, P = x0, P0
    for z in obs:
        x_pred = F * x
        P_pred = F * P * F + Q
        x_pred_list.append(x_pred)
        P_pred_list.append(P_pred)
        y = z - H * x_pred
        S = H * P_pred * H + R
        K = P_pred * H / S
        x = x_pred + K * y
        P = (1 - K * H) * P_pred
        x_filt.append(x)
        P_filt.append(P)
    # RTS smoother
    T = len(obs)
    x_smooth = [None] * T
    P_smooth = [None] * T
    x_smooth[T-1] = x_filt[T-1]
    P_smooth[T-1] = P_filt[T-1]
    for k in range(T-2, -1, -1):
        C = P_filt[k] * F / P_pred_list[k+1]
        x_smooth[k] = x_filt[k] + C * (x_smooth[k+1] - x_pred_list[k+1])
        P_smooth[k] = P_filt[k] + C * C * (P_smooth[k+1] - P_pred_list[k+1])
    # simulation smoother
    x_sample = [None] * T
    # final state
    k = T-1
    chol = np.sqrt(P_smooth[k])
    x_sample[k] = x_smooth[k] + chol * eps[k]
    for k in range(T-2, -1, -1):
        j = P_filt[k] * F / P_pred_list[k+1]
        mean = x_filt[k] + j * (x_sample[k+1] - x_pred_list[k+1])
        cov = P_filt[k] - j * P_pred_list[k+1] * j
        chol = np.sqrt(cov)
        x_sample[k] = mean + chol * eps[k]
    for x in x_sample:
        print(x)
    """

    {output, 0} = System.cmd(python, ["-c", python_code])

    py_sample =
      output
      |> String.trim()
      |> String.split("\n")
      |> Enum.map(&String.to_float/1)

    Enum.zip(nx_sample, py_sample)
    |> Enum.each(fn {x_nx, x_py} ->
      assert_in_delta x_nx, x_py, 1.0e-6
    end)
  end
end
