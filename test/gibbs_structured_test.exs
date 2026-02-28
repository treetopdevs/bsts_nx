defmodule BstsNx.GibbsStructuredTest do
  use ExUnit.Case, async: true
  import ExUnit.CaptureLog

  alias BstsNx.GibbsSampler
  alias BstsNx.Components
  alias BstsNx.ModelSpec

  # ── Local Level Spec via Structured Sampler ──────────────────────────────

  describe "local_level_spec through sample_structured" do
    test "produces valid samples with correct structure" do
      :rand.seed(:exsss, {100, 101, 102})
      obs = Enum.map(1..30, fn _ -> :rand.normal() + 5.0 end)

      spec =
        Components.local_level_spec(
          process_var: 1.0,
          obs_var: 1.0,
          initial_state: 5.0,
          initial_cov: 1.0
        )

      samples =
        GibbsSampler.sample_structured(obs, spec, 5, burn_in: 2, seed: 42)

      assert length(samples) == 5

      Enum.each(samples, fn s ->
        # States: list of tensors, one per observation
        assert is_list(s.states)
        assert length(s.states) == 30

        # Each state should be a vector (dim 1)
        Enum.each(s.states, fn st ->
          assert Nx.rank(st) >= 0
        end)

        # Covariances: list of tensors
        assert is_list(s.state_covs)
        assert length(s.state_covs) == 30

        # Q matrix: {1, 1}
        assert Nx.shape(s.q_matrix) == {1, 1}
        q_val = Nx.to_number(s.q_matrix[0][0])
        assert q_val > 0.0, "Q diagonal must be positive, got #{q_val}"

        # Obs var: positive scalar
        assert Nx.to_number(s.obs_var) > 0.0
      end)
    end

    test "Q and obs_var vary across samples" do
      :rand.seed(:exsss, {200, 201, 202})
      obs = Enum.map(1..50, fn _ -> :rand.normal() + 3.0 end)

      spec = Components.local_level_spec(process_var: 1.0, obs_var: 1.0)

      samples =
        GibbsSampler.sample_structured(obs, spec, 10, seed: 99)

      q_vals = Enum.map(samples, fn s -> Nx.to_number(s.q_matrix[0][0]) end)
      r_vals = Enum.map(samples, fn s -> Nx.to_number(s.obs_var) end)

      # Variances should not all be identical
      assert Enum.uniq(q_vals) |> length() > 1 or Enum.uniq(r_vals) |> length() > 1
    end
  end

  # ── Local Linear Trend ───────────────────────────────────────────────────

  describe "local_linear_trend_spec" do
    test "spec has correct dimensions" do
      spec = Components.local_linear_trend_spec(var_level: 0.5, var_slope: 0.05)

      assert Nx.shape(spec.f) == {2, 2}
      assert Nx.shape(spec.p0) == {2, 2}
      assert Nx.shape(spec.x0) == {2}
      assert length(spec.q_specs) == 2
      assert Enum.at(spec.q_specs, 0).dim_index == 0
      assert Enum.at(spec.q_specs, 1).dim_index == 1
    end

    test "samples from synthetic local-linear-trend data" do
      # Generate synthetic data: level with slope
      :rand.seed(:exsss, {300, 301, 302})
      n = 50
      sigma_level = 0.3
      sigma_slope = 0.05
      sigma_obs = 0.5

      {_level, _slope, obs} =
        Enum.reduce(1..n, {0.0, 0.1, []}, fn _i, {level, slope, acc} ->
          new_slope = slope + :rand.normal() * sigma_slope
          new_level = level + slope + :rand.normal() * sigma_level
          y = new_level + :rand.normal() * sigma_obs
          {new_level, new_slope, acc ++ [y]}
        end)

      spec =
        Components.local_linear_trend_spec(
          var_level: 1.0,
          var_slope: 0.1,
          obs_var: 1.0,
          initial_cov_level: 10.0,
          initial_cov_slope: 1.0
        )

      samples =
        GibbsSampler.sample_structured(obs, spec, 10, burn_in: 5, seed: 42)

      assert length(samples) == 10

      Enum.each(samples, fn s ->
        # States are 2D vectors
        assert length(s.states) == n

        # Q matrix should be {2, 2} diagonal
        assert Nx.shape(s.q_matrix) == {2, 2}
        q_level = Nx.to_number(s.q_matrix[0][0])
        q_slope = Nx.to_number(s.q_matrix[1][1])
        assert q_level > 0.0, "Level variance must be positive"
        assert q_slope > 0.0, "Slope variance must be positive"

        # Off-diagonal should be zero (or very near)
        assert abs(Nx.to_number(s.q_matrix[0][1])) < 1.0e-10
        assert abs(Nx.to_number(s.q_matrix[1][0])) < 1.0e-10

        # Obs var positive
        assert Nx.to_number(s.obs_var) > 0.0
      end)

      # The two Q diagonal entries should generally be different
      last_sample = List.last(samples)
      q_l = Nx.to_number(last_sample.q_matrix[0][0])
      q_s = Nx.to_number(last_sample.q_matrix[1][1])
      # They're very unlikely to be identical with different prior data
      assert q_l != q_s
    end

    test "sampled states roughly track the data" do
      :rand.seed(:exsss, {400, 401, 402})
      # Linear data: y_t ≈ 0.5 * t
      obs = Enum.map(1..30, fn i -> 0.5 * i + :rand.normal() * 0.2 end)

      spec =
        Components.local_linear_trend_spec(
          var_level: 0.1,
          var_slope: 0.01,
          obs_var: 0.5,
          initial_cov_level: 10.0,
          initial_cov_slope: 1.0
        )

      samples =
        GibbsSampler.sample_structured(obs, spec, 5, burn_in: 10, seed: 123)

      # Check the last sample's level states track the upward trend
      last = List.last(samples)
      first_level = last.states |> List.first() |> Nx.flatten() |> then(&Nx.to_number(&1[0]))
      last_level = last.states |> List.last() |> Nx.flatten() |> then(&Nx.to_number(&1[0]))
      assert last_level > first_level, "Level should increase for upward-trending data"
    end
  end

  # ── Regression Spec ──────────────────────────────────────────────────────

  describe "regression_spec" do
    test "spec has correct dimensions for 2 regressors" do
      x = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
      spec = Components.regression_spec(x)

      assert Nx.shape(spec.f) == {2, 2}
      assert is_list(spec.h)
      assert length(spec.h) == 3
      assert length(spec.q_specs) == 2
      assert Nx.shape(spec.x0) == {2}
      assert Nx.shape(spec.p0) == {2, 2}
    end

    test "samples from synthetic regression data" do
      :rand.seed(:exsss, {500, 501, 502})
      n = 60
      true_beta1 = 2.0
      true_beta2 = -1.0
      sigma_obs = 0.3

      # Generate regressors and observations
      x1 = Enum.map(1..n, fn _ -> :rand.normal() end)
      x2 = Enum.map(1..n, fn _ -> :rand.normal() end)

      obs =
        Enum.zip(x1, x2)
        |> Enum.map(fn {xi1, xi2} ->
          true_beta1 * xi1 + true_beta2 * xi2 + :rand.normal() * sigma_obs
        end)

      regressors =
        Enum.zip(x1, x2)
        |> Enum.map(fn {xi1, xi2} -> [xi1, xi2] end)
        |> Nx.tensor()

      spec =
        Components.regression_spec(regressors,
          var_beta: 0.1,
          obs_var: 1.0,
          initial_cov_beta: 10.0
        )

      samples =
        GibbsSampler.sample_structured(obs, spec, 10, burn_in: 20, seed: 42)

      assert length(samples) == 10

      # Check structure
      Enum.each(samples, fn s ->
        assert length(s.states) == n
        assert Nx.shape(s.q_matrix) == {2, 2}
        assert Nx.to_number(s.q_matrix[0][0]) > 0.0
        assert Nx.to_number(s.q_matrix[1][1]) > 0.0
        assert Nx.to_number(s.obs_var) > 0.0
      end)

      # Check that coefficient estimates are in a reasonable range near true values
      last = List.last(samples)
      final_state = List.last(last.states) |> Nx.to_flat_list()
      [beta1_est, beta2_est] = final_state

      # With 60 observations and burn-in, estimates should be somewhat close
      assert abs(beta1_est - true_beta1) < 3.0,
             "β₁ estimate #{beta1_est} should be near #{true_beta1}"

      assert abs(beta2_est - true_beta2) < 3.0,
             "β₂ estimate #{beta2_est} should be near #{true_beta2}"
    end

    test "supports spike-and-slab regression mode metadata" do
      x = Nx.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

      spec =
        Components.regression_spec(x,
          mode: :spike_and_slab,
          prior_inclusion: 0.2,
          g: 10.0
        )

      assert spec.regression.mode == :spike_and_slab
      assert spec.regression.start_dim == 0
      assert spec.regression.num_dims == 2
      assert spec.regression.prior_inclusion == 0.2
      assert spec.regression.g == 10.0
      assert spec.q_specs == []
    end

    test "in-loop spike-and-slab returns regression beta and gamma samples" do
      :rand.seed(:exsss, {510, 511, 512})
      n = 60

      x_rows =
        Enum.map(1..n, fn _ ->
          [:rand.normal(), :rand.normal(), :rand.normal()]
        end)

      beta_true = [2.5, 0.0, -2.0]

      obs =
        Enum.map(x_rows, fn row ->
          Enum.zip(beta_true, row)
          |> Enum.reduce(0.0, fn {b, x}, acc -> acc + b * x end)
          |> Kernel.+(:rand.normal() * 0.4)
        end)

      regressors = Nx.tensor(x_rows)
      level = Components.local_level_spec(process_var: 0.05, obs_var: 1.0, initial_state: 0.0)

      reg =
        Components.regression_spec(regressors,
          mode: :spike_and_slab,
          prior_inclusion: 0.3,
          g: n
        )

      spec = Components.compose_specs(level, reg)

      samples =
        GibbsSampler.sample_structured(obs, spec, 30, burn_in: 30, seed: 4242)

      assert length(samples) == 30

      Enum.each(samples, fn sample ->
        assert is_list(sample.regression_gamma)
        assert length(sample.regression_gamma) == 3
        assert %Nx.Tensor{} = sample.regression_beta
        assert Nx.shape(sample.q_matrix) == {4, 4}
        # Regression block is static (Q = 0 for those dimensions)
        assert Nx.to_number(sample.q_matrix[1][1]) == 0.0
        assert Nx.to_number(sample.q_matrix[2][2]) == 0.0
        assert Nx.to_number(sample.q_matrix[3][3]) == 0.0
      end)
    end

    @tag :external
    @tag timeout: 180_000
    test "in-loop spike-and-slab recovers sparse signals with high PIP" do
      :rand.seed(:exsss, {520, 521, 522})
      n = 160
      p = 100
      true_idxs = [5, 23, 77]
      true_betas = %{5 => 4.0, 23 => -3.5, 77 => 3.0}

      x_rows =
        Enum.map(1..n, fn _ ->
          Enum.map(1..p, fn _ -> :rand.normal() end)
        end)

      obs =
        Enum.map(x_rows, fn row ->
          signal =
            Enum.reduce(true_idxs, 0.0, fn j, acc ->
              acc + Map.fetch!(true_betas, j) * Enum.at(row, j)
            end)

          signal + :rand.normal() * 0.5
        end)

      regressors = Nx.tensor(x_rows)
      level = Components.local_level_spec(process_var: 0.05, obs_var: 1.0, initial_state: 0.0)

      reg =
        Components.regression_spec(regressors,
          mode: :spike_and_slab,
          prior_inclusion: 0.03,
          g: n
        )

      spec = Components.compose_specs(level, reg)

      samples =
        GibbsSampler.sample_structured(obs, spec, 50, burn_in: 50, seed: 424_242)

      counts =
        Enum.reduce(samples, List.duplicate(0, p), fn sample, acc ->
          Enum.zip_with([acc, sample.regression_gamma], fn [a, g] -> a + g end)
        end)

      pip =
        counts
        |> Enum.with_index()
        |> Map.new(fn {cnt, idx} -> {idx, cnt / length(samples)} end)

      Enum.each(true_idxs, fn j ->
        assert pip[j] > 0.95,
               "true regressor #{j} should have high PIP; got #{pip[j]}"
      end)

      max_noise_pip =
        pip
        |> Enum.reject(fn {j, _pip_j} -> j in true_idxs end)
        |> Enum.map(&elem(&1, 1))
        |> Enum.max()

      assert max_noise_pip < 0.25,
             "noise regressors should have low PIP; max was #{max_noise_pip}"
    end
  end

  # ── Composed Model ──────────────────────────────────────────────────────

  describe "compose_specs" do
    test "composes local_level + regression correctly" do
      x = Nx.tensor([[1.0], [2.0], [3.0]])
      s1 = Components.local_level_spec()
      s2 = Components.regression_spec(x)

      combined = Components.compose_specs(s1, s2)

      # State dim = 1 (level) + 1 (regressor) = 2
      assert Nx.shape(combined.f) == {2, 2}
      assert Nx.shape(combined.p0) == {2, 2}
      assert Nx.shape(combined.x0) == {2}

      # 2 q_specs: dim 0 from level, dim 1 from regression
      assert length(combined.q_specs) == 2
      assert Enum.at(combined.q_specs, 0).dim_index == 0
      assert Enum.at(combined.q_specs, 1).dim_index == 1

      # H should be time-varying (list) since regression is time-varying
      assert is_list(combined.h)
      assert length(combined.h) == 3
    end

    test "shifts spike-and-slab regression metadata when composing" do
      x = Nx.tensor([[1.0], [2.0], [3.0]])
      s1 = Components.local_linear_trend_spec()
      s2 = Components.regression_spec(x, mode: :spike_and_slab, g: 3.0)

      combined = Components.compose_specs(s1, s2)

      assert combined.regression.mode == :spike_and_slab
      # local_linear_trend has 2 state dimensions, so regression starts at 2
      assert combined.regression.start_dim == 2
      assert combined.regression.num_dims == 1
    end

    test "composes two static-H specs" do
      s1 = Components.local_level_spec()
      s2 = Components.local_linear_trend_spec()

      combined = Components.compose_specs(s1, s2)

      # State dim = 1 + 2 = 3
      assert Nx.shape(combined.f) == {3, 3}
      assert length(combined.q_specs) == 3
      assert Enum.at(combined.q_specs, 0).dim_index == 0
      assert Enum.at(combined.q_specs, 1).dim_index == 1
      assert Enum.at(combined.q_specs, 2).dim_index == 2

      # H is static {1, 3}
      assert %Nx.Tensor{} = combined.h
      assert Nx.shape(combined.h) == {1, 3}
    end

    test "composed model runs through sample_structured" do
      :rand.seed(:exsss, {600, 601, 602})
      n = 20

      x_data = Enum.map(1..n, fn _ -> [:rand.normal()] end) |> Nx.tensor()
      obs = Enum.map(1..n, fn _ -> :rand.normal() + 2.0 end)

      s1 = Components.local_level_spec(process_var: 0.5, obs_var: 1.0, initial_state: 2.0)
      s2 = Components.regression_spec(x_data, var_beta: 0.01, obs_var: 1.0)
      combined = Components.compose_specs(s1, s2)

      samples =
        GibbsSampler.sample_structured(obs, combined, 5, burn_in: 3, seed: 77)

      assert length(samples) == 5

      Enum.each(samples, fn s ->
        assert length(s.states) == n
        # State dim = 2 (1 level + 1 regressor)
        assert Nx.shape(s.q_matrix) == {2, 2}
        assert Nx.to_number(s.q_matrix[0][0]) > 0.0
        assert Nx.to_number(s.q_matrix[1][1]) > 0.0
        assert Nx.to_number(s.obs_var) > 0.0
      end)
    end
  end

  # ── Edge Cases ──────────────────────────────────────────────────────────

  describe "edge cases" do
    test "handles nil observations in multi-dim model" do
      spec = Components.local_linear_trend_spec(obs_var: 1.0)
      obs = [1.0, 2.0, nil, 4.0, 5.0]

      {samples, log} =
        capture_result_with_log(fn ->
          GibbsSampler.sample_structured(obs, spec, 3, seed: 42)
        end)

      assert length(samples) == 3
      assert log =~ "missing values are skipped"

      Enum.each(samples, fn s ->
        assert length(s.states) == 5
      end)
    end

    test "single observation" do
      spec = Components.local_level_spec()
      obs = [5.0]

      samples = GibbsSampler.sample_structured(obs, spec, 3, seed: 42)

      assert length(samples) == 3

      Enum.each(samples, fn s ->
        assert length(s.states) == 1
        assert Nx.to_number(s.obs_var) > 0.0
      end)
    end

    test "supports rank-2 time-varying H encoded as {T, n}" do
      obs = [1.0, 2.0, 2.5, 3.0]

      spec = %ModelSpec{
        f: Nx.tensor([[1.0]]),
        h: Nx.tensor([[1.0], [0.8], [1.2], [1.0]]),
        x0: Nx.tensor([0.0]),
        p0: Nx.tensor([[1.0]]),
        obs_var: 1.0,
        q_specs: [%{dim_index: 0, initial: 0.5, prior_shape: 1.0, prior_scale: 1.0}],
        obs_prior_shape: 1.0,
        obs_prior_scale: 1.0
      }

      samples = GibbsSampler.sample_structured(obs, spec, 2, seed: 42)

      assert length(samples) == 2

      Enum.each(samples, fn s ->
        assert length(s.states) == length(obs)
        assert Nx.shape(s.q_matrix) == {1, 1}
        assert Nx.to_number(s.obs_var) > 0.0
      end)
    end

    test "supports static rank-2 row H matrices" do
      obs = [1.0, 2.0, 3.0]

      spec = %ModelSpec{
        f: Nx.eye(2),
        h: Nx.tensor([[1.0, 0.0]]),
        x0: Nx.tensor([0.0, 0.0]),
        p0: Nx.eye(2),
        obs_var: 1.0,
        q_specs: [
          %{dim_index: 0, initial: 0.2, prior_shape: 1.0, prior_scale: 1.0},
          %{dim_index: 1, initial: 0.2, prior_shape: 1.0, prior_scale: 1.0}
        ],
        obs_prior_shape: 1.0,
        obs_prior_scale: 1.0
      }

      samples = GibbsSampler.sample_structured(obs, spec, 1, seed: 77)

      assert length(samples) == 1
      [sample] = samples
      assert length(sample.states) == length(obs)
      assert Nx.shape(sample.q_matrix) == {2, 2}
    end

    test "raises for invalid static rank-2 H without singleton axis" do
      obs = [1.0, 2.0, 3.0]

      spec = %ModelSpec{
        f: Nx.tensor([[1.0, 0.0], [0.0, 1.0]]),
        h: Nx.tensor([[1.0, 0.0], [0.0, 1.0]]),
        x0: Nx.tensor([0.0, 0.0]),
        p0: Nx.eye(2),
        obs_var: 1.0,
        q_specs: [
          %{dim_index: 0, initial: 0.5, prior_shape: 1.0, prior_scale: 1.0},
          %{dim_index: 1, initial: 0.5, prior_shape: 1.0, prior_scale: 1.0}
        ],
        obs_prior_shape: 1.0,
        obs_prior_scale: 1.0
      }

      assert_raise ArgumentError, ~r/singleton axis/, fn ->
        GibbsSampler.sample_structured(obs, spec, 1, seed: 42)
      end
    end

    test "burn_in and thinning" do
      :rand.seed(:exsss, {700, 701, 702})
      obs = Enum.map(1..20, fn _ -> :rand.normal() end)
      spec = Components.local_level_spec()

      samples =
        GibbsSampler.sample_structured(obs, spec, 3, burn_in: 5, thin: 2, seed: 42)

      assert length(samples) == 3
    end

    test "seed reproducibility" do
      :rand.seed(:exsss, {800, 801, 802})
      obs = Enum.map(1..15, fn _ -> :rand.normal() + 1.0 end)
      spec = Components.local_level_spec()

      samples1 = GibbsSampler.sample_structured(obs, spec, 3, seed: 42)
      samples2 = GibbsSampler.sample_structured(obs, spec, 3, seed: 42)

      # Obs variance should be identical across runs with same seed
      Enum.zip(samples1, samples2)
      |> Enum.each(fn {s1, s2} ->
        assert Nx.to_number(s1.obs_var) == Nx.to_number(s2.obs_var)
        assert Nx.to_number(s1.q_matrix[0][0]) == Nx.to_number(s2.q_matrix[0][0])
      end)
    end

    test "raises on invalid num_samples" do
      spec = Components.local_level_spec()

      assert_raise ArgumentError, ~r/num_samples must be a positive integer/, fn ->
        GibbsSampler.sample_structured([1.0], spec, 0)
      end
    end

    test "raises on negative burn_in" do
      spec = Components.local_level_spec()

      assert_raise ArgumentError, ~r/burn_in must be a non-negative integer/, fn ->
        GibbsSampler.sample_structured([1.0], spec, 1, burn_in: -1)
      end
    end
  end

  # ── Multi-Chain ─────────────────────────────────────────────────────────

  describe "sample_structured_chains" do
    test "runs multiple chains in parallel" do
      :rand.seed(:exsss, {900, 901, 902})
      obs = Enum.map(1..20, fn _ -> :rand.normal() + 3.0 end)
      spec = Components.local_level_spec(obs_var: 1.0)

      chains =
        GibbsSampler.sample_structured_chains(obs, spec, 3, 5, seed: 42)

      assert length(chains) == 3

      Enum.each(chains, fn chain ->
        assert length(chain) == 5

        Enum.each(chain, fn s ->
          assert is_list(s.states)
          assert Nx.to_number(s.obs_var) > 0.0
        end)
      end)
    end

    test "chains with different seeds produce different results" do
      :rand.seed(:exsss, {1000, 1001, 1002})
      obs = Enum.map(1..30, fn _ -> :rand.normal() + 2.0 end)
      spec = Components.local_level_spec()

      chains =
        GibbsSampler.sample_structured_chains(obs, spec, 2, 5, seed: 42)

      # Last sample from each chain should have different obs_var
      r1 = Nx.to_number(List.last(Enum.at(chains, 0)).obs_var)
      r2 = Nx.to_number(List.last(Enum.at(chains, 1)).obs_var)
      assert r1 != r2, "Different chains should produce different variance samples"
    end

    test "chains with explicit seeds" do
      :rand.seed(:exsss, {1100, 1101, 1102})
      obs = Enum.map(1..15, fn _ -> :rand.normal() end)
      spec = Components.local_level_spec()

      chains =
        GibbsSampler.sample_structured_chains(obs, spec, 2, 3, seeds: [100, 200])

      assert length(chains) == 2
    end

    test "chains with PRNG key" do
      :rand.seed(:exsss, {1200, 1201, 1202})
      obs = Enum.map(1..15, fn _ -> :rand.normal() end)
      spec = Components.local_level_spec()

      key = Nx.Random.key(42)

      chains =
        GibbsSampler.sample_structured_chains(obs, spec, 2, 3, key: key)

      assert length(chains) == 2
    end
  end

  # ── ModelSpec validation ────────────────────────────────────────────────

  describe "ModelSpec struct" do
    test "enforces required keys" do
      assert_raise ArgumentError, fn ->
        struct!(ModelSpec, %{f: Nx.tensor([[1.0]])})
      end
    end

    test "local_level_spec defaults" do
      spec = Components.local_level_spec()
      assert spec.obs_var == 1.0
      assert spec.obs_prior_shape == 1.0
      assert spec.obs_prior_scale == 1.0
      assert length(spec.q_specs) == 1
      assert Enum.at(spec.q_specs, 0).initial == 1.0
    end
  end

  # ── Helper ──────────────────────────────────────────────────────────────

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
        10_000 -> flunk("timed out waiting for captured result")
      end

    {result, log}
  end
end
