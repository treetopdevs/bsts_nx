defmodule BstsNx.GibbsFusedChainTest do
  use ExUnit.Case, async: true

  alias BstsNx.Components
  alias BstsNx.GibbsSampler

  @moduledoc """
  Parity coverage for the fused (single-defn) Gibbs chains against the
  stepwise per-iteration paths.  Both paths invoke the same compiled kernels
  in the same order with the same PRNG key, so draws must match exactly up
  to floating-point noise on the same backend.
  """

  @tolerance 1.0e-12

  defp assert_scalar_close(a, b, label) do
    diff = abs(Nx.to_number(a) - Nx.to_number(b))
    assert diff <= @tolerance, "#{label} differs by #{diff}"
  end

  defp assert_tensor_close(a, b, label) do
    diff = Nx.subtract(a, b) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    assert diff <= @tolerance, "#{label} differs by #{diff}"
  end

  defp observations(n) do
    Enum.map(0..(n - 1), fn t -> 10.0 + 0.05 * t + :math.sin(t / 3) end)
  end

  describe "scalar fused chain" do
    test "matches the stepwise path draw-for-draw" do
      obs = observations(40)
      key = Nx.Random.key(101)
      opts = [key: key, burn_in: 4, thin: 2]

      fused = GibbsSampler.sample(obs, 6, 0.0, 1.0, 0.1, 0.1, opts ++ [fused: true])
      stepwise = GibbsSampler.sample(obs, 6, 0.0, 1.0, 0.1, 0.1, opts ++ [fused: false])

      assert length(fused) == 6
      assert length(stepwise) == 6

      assert length(fused) == length(stepwise)

      Enum.zip(fused, stepwise)
      |> Enum.with_index()
      |> Enum.each(fn {{f, s}, idx} ->
        assert_scalar_close(f.process_var, s.process_var, "process_var[#{idx}]")
        assert_scalar_close(f.obs_var, s.obs_var, "obs_var[#{idx}]")

        assert length(f.states) == length(s.states)

        Enum.zip(f.states, s.states)
        |> Enum.each(fn {a, b} -> assert_scalar_close(a, b, "state[#{idx}]") end)

        assert length(f.state_covs) == length(s.state_covs)

        Enum.zip(f.state_covs, s.state_covs)
        |> Enum.each(fn {a, b} -> assert_scalar_close(a, b, "state_cov[#{idx}]") end)
      end)
    end

    test "matches the stepwise path with missing observations" do
      obs = observations(30) |> List.replace_at(5, nil) |> List.replace_at(17, nil)
      key = Nx.Random.key(77)

      ExUnit.CaptureLog.capture_log(fn ->
        fused = GibbsSampler.sample_general(obs, 1.0, 1.0, 4, key: key, burn_in: 2, fused: true)

        stepwise =
          GibbsSampler.sample_general(obs, 1.0, 1.0, 4, key: key, burn_in: 2, fused: false)

        assert length(fused) == length(stepwise)

        Enum.zip(fused, stepwise)
        |> Enum.each(fn {f, s} ->
          assert_scalar_close(f.process_var, s.process_var, "process_var")
          assert_scalar_close(f.obs_var, s.obs_var, "obs_var")

          assert length(f.states) == length(s.states)

          Enum.zip(f.states, s.states)
          |> Enum.each(fn {a, b} -> assert_scalar_close(a, b, "state") end)

          assert length(f.state_covs) == length(s.state_covs)

          Enum.zip(f.state_covs, s.state_covs)
          |> Enum.each(fn {a, b} -> assert_scalar_close(a, b, "state_cov") end)
        end)
      end)
    end

    test "matches the stepwise path with time-varying h" do
      obs = observations(24)
      h = Nx.tensor(Enum.map(0..23, fn t -> 1.0 + 0.1 * :math.cos(t / 5) end), type: {:f, 64})
      key = Nx.Random.key(33)

      fused = GibbsSampler.sample_general(obs, 1.0, h, 3, key: key, fused: true)
      stepwise = GibbsSampler.sample_general(obs, 1.0, h, 3, key: key, fused: false)

      assert length(fused) == length(stepwise)

      Enum.zip(fused, stepwise)
      |> Enum.each(fn {f, s} ->
        assert_scalar_close(f.process_var, s.process_var, "process_var")
        assert_scalar_close(f.obs_var, s.obs_var, "obs_var")

        assert length(f.states) == length(s.states)

        Enum.zip(f.states, s.states)
        |> Enum.each(fn {a, b} -> assert_scalar_close(a, b, "state") end)
      end)
    end

    test "single observation falls back to the stepwise path" do
      fused =
        GibbsSampler.sample([4.2], 2, 0.0, 1.0, 0.1, 0.1, key: Nx.Random.key(5), fused: true)

      assert length(fused) == 2
      assert length(hd(fused).states) == 1
    end
  end

  describe "structured fused chain" do
    test "matches the stepwise path for a local linear trend model" do
      obs = observations(48)

      spec =
        Components.local_linear_trend_spec(
          initial_level: hd(obs),
          var_level: 0.1,
          var_slope: 0.01,
          obs_var: 1.0
        )

      key = Nx.Random.key(202)
      opts = [key: key, burn_in: 3, thin: 2]

      fused = GibbsSampler.sample_structured(obs, spec, 5, opts ++ [fused: true])
      stepwise = GibbsSampler.sample_structured(obs, spec, 5, opts ++ [fused: false])

      assert length(fused) == 5

      assert length(fused) == length(stepwise)

      Enum.zip(fused, stepwise)
      |> Enum.with_index()
      |> Enum.each(fn {{f, s}, idx} ->
        assert_tensor_close(f.q_matrix, s.q_matrix, "q_matrix[#{idx}]")
        assert_scalar_close(f.obs_var, s.obs_var, "obs_var[#{idx}]")
        assert f.regression_beta == nil
        assert f.regression_gamma == nil

        assert length(f.states) == length(s.states)

        Enum.zip(f.states, s.states)
        |> Enum.each(fn {a, b} -> assert_tensor_close(a, b, "states[#{idx}]") end)

        assert length(f.state_covs) == length(s.state_covs)

        Enum.zip(f.state_covs, s.state_covs)
        |> Enum.each(fn {a, b} -> assert_tensor_close(a, b, "state_covs[#{idx}]") end)
      end)
    end

    test "matches the stepwise path with missing observations" do
      obs = observations(36) |> List.replace_at(8, nil) |> List.replace_at(20, nil)

      spec =
        Components.local_level_spec(
          process_var: 0.5,
          obs_var: 1.0,
          initial_state: 10.0,
          initial_cov: 1.0
        )

      key = Nx.Random.key(404)

      ExUnit.CaptureLog.capture_log(fn ->
        fused = GibbsSampler.sample_structured(obs, spec, 4, key: key, burn_in: 2, fused: true)

        stepwise =
          GibbsSampler.sample_structured(obs, spec, 4, key: key, burn_in: 2, fused: false)

        assert length(fused) == length(stepwise)

        Enum.zip(fused, stepwise)
        |> Enum.each(fn {f, s} ->
          assert_tensor_close(f.q_matrix, s.q_matrix, "q_matrix")
          assert_scalar_close(f.obs_var, s.obs_var, "obs_var")

          assert length(f.states) == length(s.states)

          Enum.zip(f.states, s.states)
          |> Enum.each(fn {a, b} -> assert_tensor_close(a, b, "states") end)

          assert length(f.state_covs) == length(s.state_covs)

          Enum.zip(f.state_covs, s.state_covs)
          |> Enum.each(fn {a, b} -> assert_tensor_close(a, b, "state_covs") end)
        end)
      end)
    end

    test "matches the stepwise path for a composed seasonal model" do
      obs = Enum.map(0..35, fn t -> 50.0 + 3.0 * :math.sin(2.0 * :math.pi() * t / 6) end)

      spec =
        Components.compose_specs(
          Components.local_level_spec(process_var: 0.1, obs_var: 1.0, initial_state: 50.0),
          Components.seasonal_spec(6, process_var: 0.05)
        )

      key = Nx.Random.key(909)

      fused = GibbsSampler.sample_structured(obs, spec, 2, key: key, burn_in: 1, fused: true)
      stepwise = GibbsSampler.sample_structured(obs, spec, 2, key: key, burn_in: 1, fused: false)

      assert length(fused) == length(stepwise)

      Enum.zip(fused, stepwise)
      |> Enum.each(fn {f, s} ->
        assert_tensor_close(f.q_matrix, s.q_matrix, "q_matrix")
        assert_scalar_close(f.obs_var, s.obs_var, "obs_var")

        assert length(f.states) == length(s.states)

        Enum.zip(f.states, s.states)
        |> Enum.each(fn {a, b} -> assert_tensor_close(a, b, "states") end)
      end)
    end
  end
end
