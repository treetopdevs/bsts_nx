defmodule BstsNx.MultidimAndFilterImpactTest do
  use ExUnit.Case, async: true

  alias BstsNx.{KalmanFilter, Smoother, StateSpace, Components, CausalImpact}

  describe "multi-dimensional state-space pipeline (3+ dims)" do
    test "3-dim composed model through filter + smoother" do
      # Compose local_level + local_linear_trend → 3-dim state
      ll = Components.local_level(0.5)
      llt = Components.local_linear_trend(0.1, 0.01)
      comp = StateSpace.compose(ll, llt)

      assert Nx.shape(comp.f) == {3, 3}
      assert Nx.shape(comp.q) == {3, 3}
      assert Nx.shape(comp.h) == {1, 3}

      # Generate synthetic observations
      :rand.seed(:exsss, {10, 11, 12})

      obs =
        Enum.map(1..20, fn t -> t * 0.5 + :rand.normal() * 0.3 end)
        |> Enum.map(&Nx.tensor([&1]))

      x0 = Nx.tensor([0.0, 0.0, 0.0])
      p0 = Nx.eye(3) |> Nx.multiply(10.0)

      {filtered, predicted} =
        KalmanFilter.filter_with_pred(obs, comp.f, comp.h, comp.q, Nx.tensor([[1.0]]), x0, p0)

      assert length(filtered) == 20
      assert length(predicted) == 20

      # Verify shapes of filtered states and covariances
      {x_first, p_first} = List.first(filtered)
      assert Nx.shape(x_first) == {3}
      assert Nx.shape(p_first) == {3, 3}

      # Smoother should produce same-shaped outputs
      smoothed = Smoother.rts(filtered, predicted, comp.f)
      assert length(smoothed) == 20
      {sx, sp} = List.first(smoothed)
      assert Nx.shape(sx) == {3}
      assert Nx.shape(sp) == {3, 3}

      # Smoothed covariances should be <= filtered covariances (trace comparison)
      trace_filt = Nx.take_diagonal(p_first) |> Nx.sum() |> Nx.to_number()
      trace_smooth = Nx.take_diagonal(sp) |> Nx.sum() |> Nx.to_number()
      assert trace_smooth <= trace_filt + 1.0e-6
    end

    test "4-dim model: two local_linear_trend components" do
      llt1 = Components.local_linear_trend(0.2, 0.02)
      llt2 = Components.local_linear_trend(0.1, 0.01)
      comp = StateSpace.compose(llt1, llt2)

      assert Nx.shape(comp.f) == {4, 4}
      assert Nx.shape(comp.q) == {4, 4}
      assert Nx.shape(comp.h) == {1, 4}

      :rand.seed(:exsss, {13, 14, 15})

      obs =
        Enum.map(1..15, fn t -> t * 0.3 + :math.sin(t * 0.5) + :rand.normal() * 0.2 end)
        |> Enum.map(&Nx.tensor([&1]))

      x0 = Nx.broadcast(0.0, {4})
      p0 = Nx.eye(4) |> Nx.multiply(5.0)
      r = Nx.tensor([[0.5]])

      {filtered, predicted} =
        KalmanFilter.filter_with_pred(obs, comp.f, comp.h, comp.q, r, x0, p0)

      assert length(filtered) == 15

      # All state vectors should be finite
      Enum.each(filtered, fn {x, _p} ->
        refute Nx.any(Nx.is_nan(x)) |> Nx.to_number() == 1,
               "state should not contain NaN"
      end)

      smoothed = Smoother.rts(filtered, predicted, comp.f)

      Enum.each(smoothed, fn {sx, _sp} ->
        refute Nx.any(Nx.is_nan(sx)) |> Nx.to_number() == 1,
               "smoothed state should not contain NaN"
      end)
    end

    test "3-dim simulation smoother produces finite samples" do
      ll = Components.local_level(0.3)
      llt = Components.local_linear_trend(0.2, 0.05)
      comp = StateSpace.compose(ll, llt)

      obs =
        Enum.map(1..10, fn t -> t * 1.0 + :rand.normal() * 0.5 end)
        |> Enum.map(&Nx.tensor([&1]))

      x0 = Nx.tensor([0.0, 0.0, 0.0])
      p0 = Nx.eye(3) |> Nx.multiply(5.0)
      r = Nx.tensor([[1.0]])

      {filtered, predicted} =
        KalmanFilter.filter_with_pred(obs, comp.f, comp.h, comp.q, r, x0, p0)

      smoothed = Smoother.rts(filtered, predicted, comp.f)

      # Simulation smoother with fixed key for reproducibility
      key = Nx.Random.key(42)

      {states, _new_key} =
        Smoother.simulate_with_key(smoothed, filtered, predicted, comp.f, key: key)

      assert length(states) == 10

      Enum.each(states, fn s ->
        assert Nx.shape(s) == {3}

        refute Nx.any(Nx.is_nan(s)) |> Nx.to_number() == 1,
               "sampled state should not contain NaN"
      end)
    end
  end

  describe "2-dim state Kalman filter shape regression (multiply→dot fix)" do
    test "filtered states have shape {2} not {2, 1} for constant-velocity model" do
      # Constant velocity model: state = [position, velocity]
      # F = [[1, 1], [0, 1]]  (position += velocity each step)
      # H = [[1, 0]]          (observe position only, m=1)
      # Q = [[0.1, 0], [0, 0.01]]
      # R = [[1.0]]
      f = Nx.tensor([[1.0, 1.0], [0.0, 1.0]])
      h = Nx.tensor([[1.0, 0.0]])
      q = Nx.tensor([[0.1, 0.0], [0.0, 0.01]])
      r = Nx.tensor([[1.0]])
      x0 = Nx.tensor([0.0, 0.0])
      p0 = Nx.eye(2)

      # Observations: linearly increasing position, wrapped as single-element tensors
      obs = Enum.map([1.0, 2.0, 3.0, 4.0, 5.0], &Nx.tensor([&1]))

      {filtered, _predicted} = KalmanFilter.filter_with_pred(obs, f, h, q, r, x0, p0)

      assert length(filtered) == 5

      # Every filtered state must have shape {2} (not {2, 1} from bad broadcast)
      Enum.each(filtered, fn {x, p} ->
        assert Nx.shape(x) == {2},
               "expected state shape {2}, got #{inspect(Nx.shape(x))}"

        assert Nx.shape(p) == {2, 2},
               "expected covariance shape {2, 2}, got #{inspect(Nx.shape(p))}"
      end)

      # The estimated position (first state element) should track observations
      positions =
        Enum.map(filtered, fn {x, _p} ->
          Nx.to_number(Nx.slice(x, [0], [1]) |> Nx.squeeze())
        end)

      # Position estimates should be increasing and reasonably close to observations
      Enum.zip([1.0, 2.0, 3.0, 4.0, 5.0], positions)
      |> Enum.each(fn {obs_val, est_val} ->
        assert abs(est_val - obs_val) < 2.0,
               "position estimate #{est_val} too far from observation #{obs_val}"
      end)

      # Verify monotonically increasing position estimates
      positions
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.each(fn [a, b] ->
        assert b > a, "position estimates should be increasing, got #{a} then #{b}"
      end)

      # All states must be finite (no NaN from bad broadcasting)
      Enum.each(filtered, fn {x, _p} ->
        refute Nx.any(Nx.is_nan(x)) |> Nx.to_number() == 1,
               "state should not contain NaN"
      end)
    end
  end

  describe "estimate_from_filter with time-varying h" do
    test "computes causal impact with a time-varying h vector" do
      t = 30
      # Simulate seasonal-like h values
      h_vals = Enum.map(0..(t - 1), fn i -> 1.0 + 0.3 * :math.sin(2.0 * :math.pi() * i / 7.0) end)
      h_tensor = Nx.tensor(h_vals, type: {:f, 32})

      # Generate data: pre-period stable, post-period shifted
      true_state = 5.0
      :rand.seed(:exsss, {200, 201, 202})

      obs_list =
        Enum.map(0..(t - 1), fn i ->
          base = Enum.at(h_vals, i) * true_state + :rand.normal() * 0.5
          # Add effect in post period (last 10 steps)
          if i >= 20, do: base + 3.0, else: base
        end)

      obs = Nx.tensor(obs_list, type: {:f, 32})
      intervention_indices = Enum.to_list(20..29)

      result =
        CausalImpact.estimate_from_filter(obs, intervention_indices,
          f: 1.0,
          h: h_tensor,
          q: 0.5,
          r: 0.5,
          x0: 5.0,
          p0: 2.0
        )

      # Should have point effects for each intervention index
      assert length(result.point_effects.mean) == 10
      assert length(result.actual) == 10
      assert length(result.baseline) == 10

      # Cumulative effect should be positive (we added +3.0)
      assert result.cumulative_effect.mean > 0.0,
             "cumulative effect should be positive, got #{result.cumulative_effect.mean}"

      # Baseline should reflect the h-scaled state
      Enum.each(result.baseline, fn b ->
        assert is_float(b)
        refute b != b, "baseline should not be NaN"
      end)
    end

    test "time-varying h produces different baselines than constant h" do
      t = 20
      obs_list = Enum.map(1..t, fn i -> i * 1.0 end)
      obs = Nx.tensor(obs_list, type: {:f, 32})
      intervention_indices = Enum.to_list(15..19)

      # Constant h = 1.0
      result_const =
        CausalImpact.estimate_from_filter(obs, intervention_indices,
          f: 1.0,
          h: 1.0,
          q: 1.0,
          r: 1.0,
          x0: 0.0,
          p0: 1.0
        )

      # Time-varying h
      h_varying =
        Nx.tensor(
          Enum.map(0..(t - 1), fn i -> 1.0 + 0.5 * :math.sin(i * 0.3) end),
          type: {:f, 32}
        )

      result_varying =
        CausalImpact.estimate_from_filter(obs, intervention_indices,
          f: 1.0,
          h: h_varying,
          q: 1.0,
          r: 1.0,
          x0: 0.0,
          p0: 1.0
        )

      # Baselines should differ between constant and time-varying h
      assert result_const.baseline != result_varying.baseline,
             "time-varying h should produce different baselines"
    end
  end
end
