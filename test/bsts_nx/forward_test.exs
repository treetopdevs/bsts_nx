defmodule BstsNx.ForwardTest do
  use ExUnit.Case, async: true

  alias BstsNx.{Components, Forward}

  describe "split_noise_keys/1" do
    test "returns the same deterministic subkeys as the shared utility helper" do
      key = Nx.Random.key(42)
      split = Nx.Random.split(key, parts: 2)
      {process_key, obs_key} = Forward.split_noise_keys(key)

      assert Nx.to_flat_list(process_key) ==
               Nx.to_flat_list(BstsNx.Utils.split_key_at(split, 0))

      assert Nx.to_flat_list(obs_key) ==
               Nx.to_flat_list(BstsNx.Utils.split_key_at(split, 1))
    end
  end

  describe "scalar_trajectory/6" do
    test "keeps a zero-noise random walk flat across gaps and post steps" do
      assert Forward.scalar_trajectory(10.0, 0.0, 0.0, 3, Nx.Random.key(1), gap_steps: 2) ==
               [10.0, 10.0, 10.0]
    end

    test "is reproducible by key and changes with different keys" do
      opts = [gap_steps: 1]
      first = Forward.scalar_trajectory(2.0, 0.25, 0.5, 4, Nx.Random.key(123), opts)
      repeat = Forward.scalar_trajectory(2.0, 0.25, 0.5, 4, Nx.Random.key(123), opts)
      different = Forward.scalar_trajectory(2.0, 0.25, 0.5, 4, Nx.Random.key(124), opts)

      assert first == repeat
      refute first == different
      assert length(first) == 4
    end
  end

  describe "structured_trajectory/7" do
    test "projects a deterministic structured state with time-varying H rows" do
      f = Nx.tensor([[1.0, 1.0], [0.0, 1.0]], type: {:f, 32})
      final_state = Nx.tensor([1.0, 2.0], type: {:f, 32})
      q = Nx.broadcast(0.0, {2, 2})
      h_steps = [Nx.tensor([[1.0, 0.0]]), Nx.tensor([1.0, 0.0]), Nx.tensor([[0.5, 0.5]])]

      assert Forward.structured_trajectory(final_state, f, h_steps, q, 0.0, Nx.Random.key(7)) ==
               [3.0, 5.0, 4.5]
    end

    test "advances through unobserved gap steps before post-period observations" do
      f = Nx.tensor([[1.0, 1.0], [0.0, 1.0]])
      final_state = Nx.tensor([1.0, 2.0])
      q = Nx.broadcast(0.0, {2, 2})
      h_steps = [Nx.tensor([[1.0, 0.0]]), Nx.tensor([[1.0, 0.0]])]

      assert Forward.structured_trajectory(final_state, f, h_steps, q, 0.0, Nx.Random.key(8),
               gap_steps: 1
             ) == [5.0, 7.0]
    end
  end

  describe "posterior trajectory helpers" do
    test "generates scalar trajectories for every posterior sample" do
      samples = [
        %{
          states: [Nx.tensor(1.0), Nx.tensor(2.0)],
          process_var: Nx.tensor(0.0),
          obs_var: Nx.tensor(0.0)
        },
        %{
          states: [Nx.tensor(3.0), Nx.tensor(4.0)],
          process_var: Nx.tensor(0.0),
          obs_var: Nx.tensor(0.0)
        }
      ]

      assert Forward.posterior_scalar_trajectories(samples, 2, Nx.Random.key(9)) ==
               [[2.0, 2.0], [4.0, 4.0]]
    end

    test "generates structured trajectories for every posterior sample" do
      spec = Components.local_level_spec(process_var: 0.0, obs_var: 0.0)

      samples = [
        %{states: [Nx.tensor([2.0])], q_matrix: Nx.tensor([[0.0]]), obs_var: Nx.tensor(0.0)},
        %{states: [Nx.tensor([4.0])], q_matrix: Nx.tensor([[0.0]]), obs_var: Nx.tensor(0.0)}
      ]

      assert Forward.posterior_structured_trajectories(
               samples,
               spec,
               [Nx.tensor([[1.0]]), Nx.tensor([[1.0]])],
               Nx.Random.key(10)
             ) == [[2.0, 2.0], [4.0, 4.0]]
    end
  end

  describe "moment helpers" do
    test "aggregates scalar posterior means and process-observation variance" do
      samples = [
        %{states: [Nx.tensor(10.0)], process_var: Nx.tensor(1.0), obs_var: Nx.tensor(0.5)},
        %{states: [Nx.tensor(20.0)], process_var: Nx.tensor(4.0), obs_var: Nx.tensor(1.0)}
      ]

      {means, sds} = Forward.scalar_moments_from_samples(samples, 2)

      assert means == [15.0, 15.0]
      assert_in_delta Enum.at(sds, 0), :math.sqrt(53.25), 1.0e-12
      assert_in_delta Enum.at(sds, 1), :math.sqrt(55.75), 1.0e-12
    end

    test "aggregates structured posterior moments with the same scalar contract" do
      spec = Components.local_level_spec(process_var: 0.0, obs_var: 0.0)

      samples = [
        %{states: [Nx.tensor([10.0])], q_matrix: Nx.tensor([[1.0]]), obs_var: Nx.tensor(0.5)},
        %{states: [Nx.tensor([20.0])], q_matrix: Nx.tensor([[4.0]]), obs_var: Nx.tensor(1.0)}
      ]

      {means, sds} =
        Forward.structured_moments_from_samples(samples, spec, [
          Nx.tensor([[1.0]]),
          Nx.tensor([[1.0]])
        ])

      assert means == [15.0, 15.0]
      assert_in_delta Enum.at(sds, 0), :math.sqrt(53.25), 1.0e-12
      assert_in_delta Enum.at(sds, 1), :math.sqrt(55.75), 1.0e-12
    end

    test "baseline moments accept explicit row forms and validate retained samples" do
      spec = %{f: Nx.tensor([[1.0]])}

      sample = %{
        states: [Nx.tensor([2.0])],
        q_matrix: Nx.tensor([[0.0]]),
        obs_var: Nx.tensor(0.0)
      }

      for rows <- [
            [Nx.tensor([[1.0]])],
            Nx.tensor([1.0]),
            Nx.tensor([[1.0]]),
            Nx.tensor([[[1.0]]])
          ] do
        assert Forward.baseline_moments_from_samples([sample], spec, rows) ==
                 %{mean: [2.0], variance: [0.0], obs_variance: 0.0}

        assert Forward.structured_moments_from_samples([sample], spec, rows) ==
                 {[2.0], [1.0e-6]}
      end

      assert Forward.baseline_moments_from_samples([], %{}, []) ==
               %{mean: [], variance: [], obs_variance: 0.0}

      assert_raise ArgumentError, ~r/non-empty samples/, fn ->
        Forward.baseline_moments_from_samples([], spec, [Nx.tensor([[1.0]])])
      end
    end

    test "baseline preserves terminal state precision while predictive propagation uses f64" do
      spec = %{f: Nx.tensor([[1.0, 1.0], [0.0, 1.0]])}

      sample = %{
        states: [Nx.tensor([16_777_216.0, 1.0], type: :f32)],
        q_matrix: Nx.broadcast(0.0, {2, 2}),
        obs_var: Nx.tensor(0.0)
      }

      rows = [Nx.tensor([[1.0, 0.0]]), Nx.tensor([[1.0, 0.0]])]

      baseline = Forward.baseline_moments_from_samples([sample], spec, rows)
      {predictive, _sds} = Forward.structured_moments_from_samples([sample], spec, rows)
      assert baseline.mean == [16_777_216.0, 16_777_216.0]
      assert predictive == [16_777_217.0, 16_777_218.0]
    end

    test "baseline clamps state variance separately from predictive observation noise" do
      # Deliberate numerical stress: keep the legacy handling of negative variance.
      spec = %{f: Nx.tensor([[1.0]])}

      sample = %{
        states: [Nx.tensor([2.0])],
        q_matrix: Nx.tensor([[-0.25]]),
        obs_var: Nx.tensor(1.0)
      }

      rows = [Nx.tensor([[1.0]])]

      assert Forward.baseline_moments_from_samples([sample], spec, rows) ==
               %{mean: [2.0], variance: [0.0], obs_variance: 1.0}

      {means, [sd]} = Forward.structured_moments_from_samples([sample], spec, rows)
      assert means == [2.0]
      assert_in_delta sd, :math.sqrt(0.75), 1.0e-12
    end

    test "projects operational moments in f64 and preserves cross-step covariance" do
      {means, vars, cumulative_var} =
        Forward.forecast_moments_defn(
          Nx.tensor([0.0], type: {:f, 32}),
          Nx.tensor([[1.0]], type: {:f, 32}),
          Nx.tensor([[1.0]], type: {:f, 32}),
          Nx.tensor([[1.0]], type: {:f, 32}),
          Nx.tensor([[1.0], [1.0], [1.0]], type: {:f, 32}),
          Nx.tensor(1, type: {:s, 64})
        )

      assert Nx.type(means) == {:f, 64}
      assert Nx.to_flat_list(means) == [0.0, 0.0, 0.0]
      assert Nx.to_flat_list(vars) == [3.0, 4.0, 5.0]
      assert Nx.to_number(cumulative_var) == 32.0
    end
  end
end
