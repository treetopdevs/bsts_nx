defmodule BstsNx.CalibrationTest do
  use ExUnit.Case, async: true

  alias BstsNx.Validation
  alias BstsNx.Components

  # Shared spec and options for faster tests
  @base_opts [
    n_pre: 80,
    n_post: 20,
    base_level: 100.0,
    noise_sd: 2.0,
    num_samples: 40,
    burn_in: 20,
    seed: 12345,
    credible_level: 0.95
  ]

  # -------------------------------------------------------------------
  # 1. Result structure
  # -------------------------------------------------------------------

  describe "known_lift_injection/3 result structure" do
    test "returns all expected keys" do
      spec = Components.local_level_spec(initial_state: 100.0, initial_cov: 10.0)
      result = Validation.known_lift_injection(10.0, spec, @base_opts)

      assert is_map(result)
      assert Map.has_key?(result, :true_lift)
      assert Map.has_key?(result, :estimated_mean)
      assert Map.has_key?(result, :estimated_sd)
      assert Map.has_key?(result, :lower)
      assert Map.has_key?(result, :upper)
      assert Map.has_key?(result, :covered)
      assert Map.has_key?(result, :relative_error)
      assert Map.has_key?(result, :n_pre)
      assert Map.has_key?(result, :n_post)
      assert Map.has_key?(result, :num_samples)
    end

    test "values have correct types" do
      spec = Components.local_level_spec(initial_state: 100.0, initial_cov: 10.0)
      result = Validation.known_lift_injection(10.0, spec, @base_opts)

      assert is_float(result.true_lift) or is_number(result.true_lift)
      assert is_float(result.estimated_mean) or is_number(result.estimated_mean)
      assert is_float(result.estimated_sd) or is_number(result.estimated_sd)
      assert is_float(result.lower) or is_number(result.lower)
      assert is_float(result.upper) or is_number(result.upper)
      assert is_boolean(result.covered)
      assert is_float(result.relative_error) or is_number(result.relative_error)
      assert is_integer(result.n_pre)
      assert is_integer(result.n_post)
      assert is_integer(result.num_samples)
    end

    test "lower <= upper" do
      spec = Components.local_level_spec(initial_state: 100.0, initial_cov: 10.0)
      result = Validation.known_lift_injection(10.0, spec, @base_opts)

      assert result.lower <= result.upper
    end
  end

  # -------------------------------------------------------------------
  # 2. Known positive lift recovery
  # -------------------------------------------------------------------

  describe "positive lift recovery" do
    test "known positive lift (10.0) is covered" do
      spec =
        Components.local_level_spec(
          initial_state: 100.0,
          initial_cov: 10.0,
          process_var: 1.0,
          obs_var: 4.0
        )

      result =
        Validation.known_lift_injection(10.0, spec,
          # Keep calibration setup high signal to avoid cross-version flakes.
          n_pre: 200,
          n_post: 30,
          base_level: 100.0,
          noise_sd: 0.5,
          num_samples: 120,
          burn_in: 60,
          seed: 42,
          credible_level: 0.99
        )

      assert result.covered == true,
             "True lift 10.0 not in CI [#{result.lower}, #{result.upper}], " <>
               "estimated_mean=#{result.estimated_mean}"
    end
  end

  # -------------------------------------------------------------------
  # 3. Zero lift coverage
  # -------------------------------------------------------------------

  describe "zero lift" do
    test "zero lift (0.0) is covered" do
      spec =
        Components.local_level_spec(
          initial_state: 100.0,
          initial_cov: 10.0,
          process_var: 1.0,
          obs_var: 4.0
        )

      result =
        Validation.known_lift_injection(0.0, spec,
          n_pre: 100,
          n_post: 20,
          base_level: 100.0,
          noise_sd: 2.0,
          num_samples: 50,
          burn_in: 25,
          seed: 77,
          credible_level: 0.95
        )

      assert result.covered == true,
             "Zero lift not in CI [#{result.lower}, #{result.upper}], " <>
               "estimated_mean=#{result.estimated_mean}"
    end

    test "relative_error uses absolute error for near-zero true lift" do
      spec =
        Components.local_level_spec(
          initial_state: 100.0,
          initial_cov: 10.0,
          process_var: 1.0,
          obs_var: 4.0
        )

      result =
        Validation.known_lift_injection(0.0, spec,
          n_pre: 80,
          n_post: 20,
          base_level: 100.0,
          noise_sd: 2.0,
          num_samples: 40,
          burn_in: 20,
          seed: 4242
        )

      assert_in_delta result.relative_error, result.estimated_mean - result.true_lift, 1.0e-10
    end
  end

  # -------------------------------------------------------------------
  # 4. Large lift — estimated_mean in right direction
  # -------------------------------------------------------------------

  describe "large lift" do
    test "large lift (50.0) — estimated_mean is positive" do
      spec =
        Components.local_level_spec(
          initial_state: 100.0,
          initial_cov: 10.0,
          process_var: 1.0,
          obs_var: 4.0
        )

      result =
        Validation.known_lift_injection(50.0, spec,
          n_pre: 100,
          n_post: 20,
          base_level: 100.0,
          noise_sd: 2.0,
          num_samples: 50,
          burn_in: 25,
          seed: 99,
          credible_level: 0.95
        )

      assert result.estimated_mean > 0,
             "Expected positive estimated_mean for large positive lift, got #{result.estimated_mean}"
    end
  end

  # -------------------------------------------------------------------
  # 5. Relative error is reasonable
  # -------------------------------------------------------------------

  describe "relative error" do
    @tag timeout: 120_000
    test "relative error is less than 1.0 for moderate lift with enough samples" do
      spec =
        Components.local_level_spec(
          initial_state: 100.0,
          initial_cov: 10.0,
          process_var: 1.0,
          obs_var: 4.0
        )

      result =
        Validation.known_lift_injection(10.0, spec,
          n_pre: 100,
          n_post: 20,
          base_level: 100.0,
          noise_sd: 2.0,
          num_samples: 50,
          burn_in: 25,
          seed: 42,
          credible_level: 0.95
        )

      assert abs(result.relative_error) < 1.0,
             "Relative error #{result.relative_error} is too large (expected < 1.0)"
    end
  end

  # -------------------------------------------------------------------
  # 6. Works with local_level_spec (default opts)
  # -------------------------------------------------------------------

  describe "local_level_spec compatibility" do
    test "works with default local_level_spec" do
      spec = Components.local_level_spec()

      result =
        Validation.known_lift_injection(5.0, spec,
          n_pre: 60,
          n_post: 15,
          num_samples: 30,
          burn_in: 15,
          seed: 555
        )

      assert is_map(result)
      assert is_number(result.estimated_mean)
      assert is_boolean(result.covered)
      assert result.n_pre == 60
      assert result.n_post == 15
      assert result.num_samples == 30
    end
  end

  # -------------------------------------------------------------------
  # 7. Seed produces deterministic results
  # -------------------------------------------------------------------

  describe "determinism" do
    test "same seed produces identical results" do
      spec =
        Components.local_level_spec(
          initial_state: 100.0,
          initial_cov: 10.0
        )

      opts = [
        n_pre: 60,
        n_post: 15,
        base_level: 100.0,
        noise_sd: 2.0,
        num_samples: 30,
        burn_in: 15,
        seed: 12345
      ]

      r1 = Validation.known_lift_injection(10.0, spec, opts)
      r2 = Validation.known_lift_injection(10.0, spec, opts)

      assert r1.estimated_mean == r2.estimated_mean
      assert r1.estimated_sd == r2.estimated_sd
      assert r1.lower == r2.lower
      assert r1.upper == r2.upper
      assert r1.covered == r2.covered
      assert r1.relative_error == r2.relative_error
    end
  end

  describe "input validation" do
    test "rejects non-positive n_pre" do
      spec = Components.local_level_spec(initial_state: 100.0, initial_cov: 10.0)

      assert_raise ArgumentError, ~r/n_pre/, fn ->
        Validation.known_lift_injection(10.0, spec, Keyword.put(@base_opts, :n_pre, 0))
      end
    end

    test "rejects non-positive n_post" do
      spec = Components.local_level_spec(initial_state: 100.0, initial_cov: 10.0)

      assert_raise ArgumentError, ~r/n_post/, fn ->
        Validation.known_lift_injection(10.0, spec, Keyword.put(@base_opts, :n_post, 0))
      end
    end
  end
end
