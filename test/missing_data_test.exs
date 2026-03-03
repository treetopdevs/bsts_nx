defmodule BstsNxMissingDataTest do
  use ExUnit.Case, async: true
  import Nx, only: [to_number: 1]
  alias BstsNx.KalmanFilter

  describe "KalmanFilter.filter/7 with missing data" do
    test "skips update for nil observations" do
      # Observations with a missing value (nil) in the middle
      observations = [1.0, nil, 3.0]
      f = 1.0
      h = 1.0
      q = 0.1
      r = 0.5
      x0 = 0.0
      p0 = 1.0
      estimates = KalmanFilter.filter(observations, f, h, q, r, x0, p0)
      # For the missing observation, the estimate should equal the prediction (no update)
      # Compute expected values manually
      # t=1 update
      x_pred1 = x0 * f
      p_pred1 = f * p0 * f + q
      k1 = p_pred1 * h / (h * p_pred1 * h + r)
      x1 = x_pred1 + k1 * (1.0 - h * x_pred1)
      p1 = (1.0 - k1 * h) * p_pred1 * (1.0 - k1 * h) + k1 * r * k1
      # t=2 missing: prediction only
      x_pred2 = f * x1
      p_pred2 = f * p1 * f + q
      x2 = x_pred2
      p2 = p_pred2
      # t=3 update
      x_pred3 = f * x2
      p_pred3 = f * p2 * f + q
      k3 = p_pred3 * h / (h * p_pred3 * h + r)
      x3 = x_pred3 + k3 * (3.0 - h * x_pred3)
      p3 = (1.0 - k3 * h) * p_pred3 * (1.0 - k3 * h) + k3 * r * k3
      expected = [{x1, p1}, {x2, p2}, {x3, p3}]

      Enum.zip(estimates, expected)
      |> Enum.each(fn {{x_t, p_t}, {x_exp, p_exp}} ->
        assert_in_delta(to_number(x_t), x_exp, 1.0e-6)
        assert_in_delta(to_number(p_t), p_exp, 1.0e-6)
      end)
    end
  end
end
