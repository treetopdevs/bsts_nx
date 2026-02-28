defmodule BstsNx.SpotAttributorPosteriorTest do
  use ExUnit.Case, async: true

  alias BstsNx.SpotAttributor

  @aggregation_delta_tol if(
                           match?(EMLX.Backend, Nx.default_backend()) or
                             match?({EMLX.Backend, _}, Nx.default_backend()),
                           do: 1.0e-5,
                           else: 1.0e-10
                         )

  # ---------------------------------------------------------------
  # Helpers
  # ---------------------------------------------------------------

  defp make_spot(id, ws, we), do: %{id: id, window_start: ws, window_end: we}

  # Generates `n_draws` counterfactual draws of length `t`, each
  # sampled from N(base, noise_sd^2).
  defp generate_draws(base, noise_sd, t, n_draws, seed) do
    :rand.seed(:exsss, {seed, seed + 1, seed + 2})

    Enum.map(1..n_draws, fn _ ->
      Enum.map(1..t, fn _ -> base + :rand.normal() * noise_sd end)
    end)
  end

  # ---------------------------------------------------------------
  # 1. Basic result structure
  # ---------------------------------------------------------------

  describe "attribute_posterior result structure" do
    test "has correct keys and correct number of attributions" do
      obs = [110.0, 112.0, 108.0, 115.0, 109.0]
      spots = [make_spot("s1", 0, 3), make_spot("s2", 3, 5)]
      draws = generate_draws(100.0, 2.0, 5, 10, 42)

      result = SpotAttributor.attribute_posterior(obs, spots, draws, 0.0)

      assert Map.has_key?(result, :attributions)
      assert Map.has_key?(result, :total_lift)
      assert Map.has_key?(result, :total_lift_sd)
      assert Map.has_key?(result, :overlap_groups)

      assert length(result.attributions) == 2
      assert is_number(result.total_lift)
      assert is_number(result.total_lift_sd)
      assert is_list(result.overlap_groups)

      Enum.each(result.attributions, fn attr ->
        assert Map.has_key?(attr, :spot_id)
        assert Map.has_key?(attr, :lift)
        assert Map.has_key?(attr, :lift_sd)
        assert Map.has_key?(attr, :lift_lower)
        assert Map.has_key?(attr, :lift_upper)
        assert Map.has_key?(attr, :p_positive)
        assert Map.has_key?(attr, :window_start)
        assert Map.has_key?(attr, :window_end)
      end)
    end
  end

  # ---------------------------------------------------------------
  # 2. Single spot positive lift detected
  # ---------------------------------------------------------------

  describe "single spot positive lift" do
    test "detects positive lift when observations exceed counterfactual" do
      # Observations consistently above counterfactual baseline
      obs = [120.0, 118.0, 122.0, 119.0, 121.0]
      spots = [make_spot("s1", 0, 5)]

      # Counterfactual draws centered around 100
      draws = generate_draws(100.0, 3.0, 5, 10, 42)

      result = SpotAttributor.attribute_posterior(obs, spots, draws, 0.0)

      [attr] = result.attributions
      assert attr.lift > 0
      assert attr.p_positive > 0.5
    end
  end

  # ---------------------------------------------------------------
  # 3. Overlapping spots use Shapley per draw
  # ---------------------------------------------------------------

  describe "overlapping spots" do
    test "uses Shapley allocation and lifts sum to total" do
      obs = [100.0, 160.0, 170.0, 180.0, 100.0]
      # a covers [1,3), b covers [2,4) — overlap at index 2
      spots = [make_spot("a", 1, 3), make_spot("b", 2, 4)]

      # Counterfactual draws centered around 100
      draws = generate_draws(100.0, 2.0, 5, 10, 42)

      result = SpotAttributor.attribute_posterior(obs, spots, draws, 0.0)

      assert length(result.attributions) == 2
      assert length(result.overlap_groups) == 1

      # Per-spot lifts should sum approximately to total_lift
      sum_lifts = Enum.reduce(result.attributions, 0.0, fn a, acc -> acc + a.lift end)
      assert_in_delta result.total_lift, sum_lifts, 1.0e-10
    end
  end

  # ---------------------------------------------------------------
  # 4. CI contains posterior mean
  # ---------------------------------------------------------------

  describe "credible interval" do
    test "CI contains the posterior mean lift" do
      obs = [115.0, 112.0, 118.0, 114.0, 116.0]
      spots = [make_spot("s1", 0, 5)]
      draws = generate_draws(100.0, 3.0, 5, 15, 42)

      result = SpotAttributor.attribute_posterior(obs, spots, draws, 0.0)

      [attr] = result.attributions
      assert attr.lift_lower <= attr.lift
      assert attr.lift_upper >= attr.lift
    end
  end

  # ---------------------------------------------------------------
  # 5. No lift scenario — lift near zero
  # ---------------------------------------------------------------

  describe "no lift scenario" do
    test "lift near zero when observations match counterfactual" do
      # Observations right at the counterfactual baseline
      obs = [100.0, 100.0, 100.0, 100.0, 100.0]
      spots = [make_spot("s1", 0, 5)]

      # Counterfactual draws centered exactly at 100 with small noise
      draws = generate_draws(100.0, 0.5, 5, 10, 42)

      result = SpotAttributor.attribute_posterior(obs, spots, draws, 0.0)

      [attr] = result.attributions
      # Lift should be near zero (within a few standard deviations of noise)
      assert abs(attr.lift) < 20.0
    end
  end

  # ---------------------------------------------------------------
  # 6. Empty spots — returns empty
  # ---------------------------------------------------------------

  describe "empty spots" do
    test "returns empty attributions when no spots provided" do
      obs = [110.0, 115.0, 112.0]
      draws = generate_draws(100.0, 2.0, 3, 5, 42)

      result = SpotAttributor.attribute_posterior(obs, [], draws, 0.0)

      assert result.attributions == []
      assert result.total_lift == 0.0
      assert result.total_lift_sd == 0.0
      assert result.overlap_groups == []
    end
  end

  # ---------------------------------------------------------------
  # 7. Empty draws — returns zero-effect result
  # ---------------------------------------------------------------

  describe "empty draws" do
    test "returns empty result when no draws provided" do
      obs = [110.0, 115.0, 112.0]
      spots = [make_spot("s1", 0, 3)]

      result = SpotAttributor.attribute_posterior(obs, spots, [], 0.0)

      assert result.attributions == []
      assert result.total_lift == 0.0
      assert result.total_lift_sd == 0.0
      assert result.overlap_groups == []
    end
  end

  # ---------------------------------------------------------------
  # 8. Deterministic with same seed
  # ---------------------------------------------------------------

  describe "determinism" do
    test "same draws produce identical results" do
      obs = [115.0, 112.0, 118.0]
      spots = [make_spot("s1", 0, 3)]
      draws = generate_draws(100.0, 2.0, 3, 10, 42)

      r1 = SpotAttributor.attribute_posterior(obs, spots, draws, 0.0)
      r2 = SpotAttributor.attribute_posterior(obs, spots, draws, 0.0)

      [a1] = r1.attributions
      [a2] = r2.attributions
      assert a1.lift == a2.lift
      assert a1.lift_sd == a2.lift_sd
      assert a1.p_positive == a2.p_positive
    end
  end

  describe "aggregation parity" do
    test "matches manual per-draw attribute aggregation for overlaps" do
      obs = [100.0, 160.0, 170.0, 180.0, 100.0]
      spots = [make_spot("a", 1, 3), make_spot("b", 2, 4)]
      draws = generate_draws(100.0, 2.0, 5, 8, 7)
      opts = [n_samples: 200, key: Nx.Random.key(123), alpha: 0.05]

      result = SpotAttributor.attribute_posterior(obs, spots, draws, 0.0, opts)

      zeros = List.duplicate(0.0, length(obs))

      manual_per_draw =
        Enum.map(draws, fn draw ->
          SpotAttributor.attribute(
            obs,
            spots,
            %{mean: draw, variance: zeros, obs_variance: 0.0},
            opts
          )
        end)

      manual_total_lifts = Enum.map(manual_per_draw, & &1.total_lift)

      manual_spot_lifts =
        manual_per_draw
        |> Enum.map(fn r -> Map.new(r.attributions, fn a -> {a.spot_id, a.lift} end) end)
        |> Enum.reduce(%{"a" => [], "b" => []}, fn m, acc ->
          %{
            "a" => [Map.get(m, "a", 0.0) | acc["a"]],
            "b" => [Map.get(m, "b", 0.0) | acc["b"]]
          }
        end)
        |> Map.new(fn {k, v} -> {k, Enum.reverse(v)} end)

      expected_mean_a = Enum.sum(manual_spot_lifts["a"]) / length(draws)
      expected_mean_b = Enum.sum(manual_spot_lifts["b"]) / length(draws)
      expected_total_mean = Enum.sum(manual_total_lifts) / length(draws)

      attr_a = Enum.find(result.attributions, &(&1.spot_id == "a"))
      attr_b = Enum.find(result.attributions, &(&1.spot_id == "b"))

      assert_in_delta attr_a.lift, expected_mean_a, @aggregation_delta_tol
      assert_in_delta attr_b.lift, expected_mean_b, @aggregation_delta_tol
      assert_in_delta result.total_lift, expected_total_mean, @aggregation_delta_tol
    end
  end

  describe "single-draw and validation branches" do
    test "single draw produces zero posterior standard deviations" do
      obs = [110.0, 112.0, 111.0]
      spots = [make_spot("s1", 0, 3)]
      draws = [[100.0, 100.0, 100.0]]

      result = SpotAttributor.attribute_posterior(obs, spots, draws, 0.0)
      [attr] = result.attributions

      assert attr.lift_sd == 0.0
      assert result.total_lift_sd == 0.0
    end

    test "rejects draw length mismatch" do
      obs = [110.0, 112.0, 111.0]
      spots = [make_spot("s1", 0, 3)]
      draws = [[100.0, 100.0, 100.0], [101.0, 101.0]]

      assert_raise ArgumentError, ~r/each counterfactual draw length/, fn ->
        SpotAttributor.attribute_posterior(obs, spots, draws, 0.0)
      end
    end
  end
end
