defmodule BstsNx.RollingBaselineWarmStartTest do
  use ExUnit.Case, async: true

  alias BstsNx.RollingBaseline

  @moduletag :external

  # -- Helpers --

  defp synthetic_data(n, opts \\ []) do
    num_seasons = Keyword.get(opts, :num_seasons, 4)
    base = Keyword.get(opts, :base, 100.0)
    trend = Keyword.get(opts, :trend, 0.1)
    amplitude = Keyword.get(opts, :amplitude, 10.0)

    Enum.map(0..(n - 1), fn t ->
      base + trend * t + amplitude * :math.sin(2 * :math.pi() * t / num_seasons)
    end)
  end

  defp fit_opts, do: [num_samples: 10, burn_in: 5, seed: 42, num_seasons: 4]

  # ---------------------------------------------------------------
  # 1. Warm-start reduces effective burn-in
  # ---------------------------------------------------------------

  describe "warm-start burn-in reduction" do
    test "warm_start option uses reduced default burn-in" do
      data = synthetic_data(20)
      spec = RollingBaseline.build_spec(20, num_seasons: 4)

      # First fit (cold start)
      first_result =
        RollingBaseline.fit(data, spec, num_samples: 10, burn_in: 5, seed: 42)

      # Second fit with warm_start -- do NOT pass :burn_in so the default
      # warm-start burn-in (50) applies instead of the cold default (100).
      # We use a small num_samples to keep this fast; the key assertion is
      # that this completes successfully with the warm-start path active.
      warm_result =
        RollingBaseline.fit(data, spec,
          num_samples: 10,
          seed: 99,
          warm_start: first_result
        )

      assert length(warm_result.posterior_samples) == 10
      assert warm_result.num_observations == 20

      # The warm-started spec should have obs_var updated from the posterior
      warm_obs_var = warm_result.spec.obs_var

      # The warm-start obs_var should differ from the original spec default
      refute warm_obs_var == spec.obs_var,
             "warm-started obs_var should differ from cold-start default"
    end

    test "warm_start updates q_specs initials from posterior" do
      data = synthetic_data(20)
      spec = RollingBaseline.build_spec(20, num_seasons: 4)

      first_result =
        RollingBaseline.fit(data, spec, num_samples: 10, burn_in: 5, seed: 42)

      warm_result =
        RollingBaseline.fit(data, spec,
          num_samples: 10,
          burn_in: 5,
          seed: 99,
          warm_start: first_result
        )

      # Each q_spec initial in the warm-started spec should be updated
      # to the posterior mean from the first fit
      original_initials = Enum.map(spec.q_specs, & &1.initial)
      warm_initials = Enum.map(warm_result.spec.q_specs, & &1.initial)

      # At least one initial should differ (the sampler will have moved
      # away from the prior initial values)
      differs =
        Enum.zip(original_initials, warm_initials)
        |> Enum.any?(fn {orig, warm} -> orig != warm end)

      assert differs,
             "warm-started q_spec initials should differ from cold-start defaults"
    end

    test "warm_start initializes x0 and p0 from previous terminal state" do
      data = synthetic_data(20)
      spec = RollingBaseline.build_spec(20, num_seasons: 4)

      first_result =
        RollingBaseline.fit(data, spec, num_samples: 10, burn_in: 5, seed: 42)

      warm_result =
        RollingBaseline.fit(data, spec,
          num_samples: 10,
          burn_in: 5,
          seed: 99,
          warm_start: first_result
        )

      assert Nx.all_close(warm_result.spec.x0, List.last(first_result.final_states))
      assert Nx.all_close(warm_result.spec.p0, List.last(first_result.final_covs))
    end

    test "explicit burn_in overrides warm-start default" do
      data = synthetic_data(20)
      spec = RollingBaseline.build_spec(20, num_seasons: 4)

      first_result =
        RollingBaseline.fit(data, spec, num_samples: 10, burn_in: 5, seed: 42)

      # Pass explicit burn_in=3 with warm_start
      warm_result =
        RollingBaseline.fit(data, spec,
          num_samples: 10,
          burn_in: 3,
          seed: 99,
          warm_start: first_result
        )

      # Should still produce the requested number of samples
      assert length(warm_result.posterior_samples) == 10
    end
  end

  # ---------------------------------------------------------------
  # 2. Convergence diagnostics present in fit_result
  # ---------------------------------------------------------------

  describe "convergence diagnostics" do
    test "fit_result includes convergence map" do
      data = synthetic_data(20)
      spec = RollingBaseline.build_spec(20, num_seasons: 4)

      result =
        RollingBaseline.fit(data, spec, num_samples: 10, burn_in: 5, seed: 42)

      assert Map.has_key?(result, :convergence)
      assert is_map(result.convergence)
    end

    test "convergence map has all required keys" do
      data = synthetic_data(20)
      spec = RollingBaseline.build_spec(20, num_seasons: 4)

      result =
        RollingBaseline.fit(data, spec, num_samples: 10, burn_in: 5, seed: 42)

      convergence = result.convergence

      assert Map.has_key?(convergence, :r_hat_obs_var)
      assert Map.has_key?(convergence, :ess_obs_var)
      assert Map.has_key?(convergence, :r_hat_process_var)
      assert Map.has_key?(convergence, :ess_process_var)
      assert Map.has_key?(convergence, :converged)
    end

    test "convergence values have expected types" do
      data = synthetic_data(20)
      spec = RollingBaseline.build_spec(20, num_seasons: 4)

      result =
        RollingBaseline.fit(data, spec, num_samples: 10, burn_in: 5, seed: 42)

      convergence = result.convergence

      # R-hat and ESS are either floats or :nan
      assert is_float(convergence.r_hat_obs_var) or convergence.r_hat_obs_var == :nan
      assert is_float(convergence.ess_obs_var) or convergence.ess_obs_var == :nan
      assert is_float(convergence.r_hat_process_var) or convergence.r_hat_process_var == :nan
      assert is_float(convergence.ess_process_var) or convergence.ess_process_var == :nan
      assert is_boolean(convergence.converged)
    end

    test "converged is false when sample count is too low for ESS threshold" do
      data = synthetic_data(20)
      spec = RollingBaseline.build_spec(20, num_seasons: 4)

      # With only 10 samples, ESS cannot exceed 100 (the convergence threshold)
      result =
        RollingBaseline.fit(data, spec, num_samples: 10, burn_in: 5, seed: 42)

      refute result.convergence.converged,
             "converged should be false with only 10 posterior samples"
    end
  end

  # ---------------------------------------------------------------
  # 3. Warm-start with fit_and_predict
  # ---------------------------------------------------------------

  describe "fit_and_predict warm-start" do
    test "accepts and forwards warm_start option" do
      data = synthetic_data(20)

      # First fit
      {_cf1, fit1} =
        RollingBaseline.fit_and_predict(data, 5, fit_opts())

      # Second fit with warm_start
      {cf2, fit2} =
        RollingBaseline.fit_and_predict(
          data,
          5,
          Keyword.merge(fit_opts(), warm_start: fit1, seed: 99)
        )

      assert length(cf2.mean) == 5
      assert length(fit2.posterior_samples) == 10
      assert Map.has_key?(fit2, :convergence)
    end
  end
end
