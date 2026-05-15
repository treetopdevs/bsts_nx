defmodule BstsNx.Applications.AnomalyDetectorTest do
  use ExUnit.Case, async: true

  alias BstsNx.Applications.AnomalyDetector

  describe "fit/2 and score/2 with MCMC" do
    test "fits baseline and scores normal data" do
      :rand.seed(:exsss, {100, 101, 102})
      normal_data = Enum.map(1..30, fn _ -> 100.0 + :rand.normal() * 2 end)

      detector = AnomalyDetector.fit(normal_data, num_samples: 10, burn_in: 5, seed: 42)

      assert detector.method == :mcmc
      assert is_list(detector.posterior_mean)
      assert length(detector.posterior_mean) == 30

      # Score the training data — most should be normal
      scores = AnomalyDetector.score(detector, normal_data)

      assert length(scores) == 30

      Enum.each(scores, fn s ->
        assert is_float(s.value)
        assert is_float(s.predicted)
        assert is_float(s.z_score)
        assert is_float(s.p_value)
        assert is_boolean(s.anomaly?)
        assert s.severity in [:normal, :warning, :critical]
      end)
    end

    test "detects anomalous observations" do
      :rand.seed(:exsss, {200, 201, 202})
      normal_data = Enum.map(1..30, fn _ -> 100.0 + :rand.normal() * 2 end)

      detector =
        AnomalyDetector.fit(normal_data, num_samples: 10, burn_in: 5, seed: 42, alpha: 0.05)

      # Score data with a clear anomaly
      test_data = Enum.map(1..5, fn _ -> 100.0 + :rand.normal() * 2 end) ++ [200.0]

      scores = AnomalyDetector.score(detector, test_data)
      last_score = List.last(scores)

      # The 200.0 value should be flagged as anomalous
      assert last_score.value == 200.0
      assert abs(last_score.z_score) > 2.0
    end
  end

  describe "fit/2 and score_one/2 with filter" do
    test "streaming anomaly detection" do
      :rand.seed(:exsss, {300, 301, 302})
      normal_data = Enum.map(1..30, fn _ -> 100.0 + :rand.normal() * 2 end)

      detector = AnomalyDetector.fit(normal_data, method: :filter, alpha: 0.01)

      assert detector.method == :filter
      assert detector.filter_state != nil

      # Score one observation at a time
      {score1, det1} = AnomalyDetector.score_one(detector, 101.0)
      assert is_float(score1.z_score)
      assert score1.severity == :normal

      # Score an anomalous value
      {score2, _det2} = AnomalyDetector.score_one(det1, 200.0)
      assert abs(score2.z_score) > 2.0
    end

    test "filter training preserves f64 precision for list observations" do
      high_precision = 16_777_217.0

      detector =
        AnomalyDetector.fit(List.duplicate(high_precision, 8),
          method: :filter,
          x0: 0.0,
          p0: 1.0e12,
          q: 1.0e-9,
          r: 1.0e-6
        )

      assert_in_delta detector.filter_state, high_precision, 0.1
    end
  end

  describe "detect/3" do
    test "returns only anomalous observations" do
      :rand.seed(:exsss, {400, 401, 402})
      normal_data = Enum.map(1..30, fn _ -> 100.0 + :rand.normal() * 2 end)

      detector =
        AnomalyDetector.fit(normal_data, num_samples: 10, burn_in: 5, seed: 42, alpha: 0.05)

      # Mix of normal and anomalous values
      test_data = [100.0, 101.0, 99.0, 200.0, 102.0]
      anomalies = AnomalyDetector.detect(detector, test_data)

      # Only the 200.0 spike should be detected
      assert Enum.any?(anomalies, fn a -> a.value == 200.0 end)
    end
  end

  describe "summary/1" do
    test "summarizes anomaly detection results" do
      scores = [
        %{
          value: 100.0,
          predicted: 100.0,
          predicted_sd: 2.0,
          residual: 0.0,
          z_score: 0.0,
          p_value: 1.0,
          anomaly?: false,
          severity: :normal
        },
        %{
          value: 200.0,
          predicted: 100.0,
          predicted_sd: 2.0,
          residual: 100.0,
          z_score: 50.0,
          p_value: 0.0,
          anomaly?: true,
          severity: :critical
        },
        %{
          value: 110.0,
          predicted: 100.0,
          predicted_sd: 2.0,
          residual: 10.0,
          z_score: 5.0,
          p_value: 0.001,
          anomaly?: true,
          severity: :warning
        }
      ]

      summary = AnomalyDetector.summary(scores)

      assert summary.total_observations == 3
      assert summary.anomalies == 2
      assert summary.warnings == 1
      assert summary.criticals == 1
      assert summary.max_abs_z_score == 50.0
      assert_in_delta summary.anomaly_rate, 2 / 3, 0.01
    end
  end

  describe "validation" do
    test "rejects empty observations" do
      assert_raise ArgumentError, ~r/non-empty/, fn ->
        AnomalyDetector.fit([])
      end
    end

    test "rejects invalid num_samples" do
      assert_raise ArgumentError, ~r/num_samples/, fn ->
        AnomalyDetector.fit([1.0, 2.0, 3.0], num_samples: -5)
      end
    end

    test "rejects non-integer num_samples" do
      assert_raise ArgumentError, ~r/num_samples/, fn ->
        AnomalyDetector.fit([1.0, 2.0, 3.0], num_samples: 10.5)
      end
    end

    test "rejects invalid alpha" do
      assert_raise ArgumentError, ~r/alpha/, fn ->
        AnomalyDetector.fit([1.0, 2.0, 3.0], alpha: 1.0)
      end
    end

    test "rejects unsupported method" do
      assert_raise ArgumentError, ~r/method/, fn ->
        AnomalyDetector.fit([1.0, 2.0, 3.0], method: :bad)
      end
    end

    test "rejects non-tensor regressors" do
      assert_raise ArgumentError, ~r/regressors must be an Nx tensor/, fn ->
        AnomalyDetector.fit([1.0, 2.0, 3.0], regressors: "not a tensor")
      end
    end

    test "rejects mismatched regressors length" do
      regressors = Nx.tensor([[1.0], [2.0]])

      assert_raise ArgumentError, ~r/regressors rows/, fn ->
        AnomalyDetector.fit([1.0, 2.0, 3.0], regressors: regressors)
      end
    end
  end

  describe "fit/2 with regressors" do
    test "produces detector with regression baseline" do
      :rand.seed(:exsss, {500, 501, 502})
      # Known promotions cause spikes — modeled in regressors
      promo = Enum.map(1..30, fn i -> if rem(i, 10) == 0, do: 1.0, else: 0.0 end)

      obs =
        Enum.zip(1..30, promo)
        |> Enum.map(fn {_, p} -> 50.0 + p * 20.0 + :rand.normal() * 2 end)

      regressors = Nx.tensor(Enum.map(promo, fn p -> [p] end))

      detector =
        AnomalyDetector.fit(obs,
          regressors: regressors,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      assert detector.method == :mcmc
      assert is_list(detector.posterior_mean)
      assert length(detector.posterior_mean) == 30
      assert detector.spec != nil
    end

    test "score/2 with regression detector handles known effects" do
      :rand.seed(:exsss, {510, 511, 512})
      promo = Enum.map(1..30, fn i -> if rem(i, 10) == 0, do: 1.0, else: 0.0 end)

      obs =
        Enum.zip(1..30, promo)
        |> Enum.map(fn {_, p} -> 50.0 + p * 20.0 + :rand.normal() * 2 end)

      regressors = Nx.tensor(Enum.map(promo, fn p -> [p] end))

      detector =
        AnomalyDetector.fit(obs,
          regressors: regressors,
          num_samples: 10,
          burn_in: 5,
          seed: 42
        )

      # Score the same data with matching known future effects.
      scores = AnomalyDetector.score(detector, obs, future_regressors: regressors)
      assert length(scores) == 30

      Enum.each(scores, fn s ->
        assert is_float(s.predicted)
        assert is_float(s.z_score)
      end)
    end

    test "score/2 with regression detector requires future regressors" do
      promo = Enum.map(1..12, fn i -> if rem(i, 4) == 0, do: 1.0, else: 0.0 end)
      obs = Enum.map(promo, fn p -> 50.0 + p * 20.0 end)
      regressors = Nx.tensor(Enum.map(promo, fn p -> [p] end))

      detector =
        AnomalyDetector.fit(obs,
          regressors: regressors,
          num_samples: 6,
          burn_in: 3,
          seed: 42
        )

      assert_raise ArgumentError, ~r/future_regressors are required/, fn ->
        AnomalyDetector.score(detector, obs)
      end
    end
  end

  describe "structured forward uncertainty" do
    test "fallback extrapolation reuses trailing posterior moments" do
      detector = %{
        method: :mcmc,
        z_threshold: 3.0,
        posterior_mean: [10.0, 12.0],
        posterior_sd: [2.0, 3.0],
        posterior_samples: nil,
        mcmc_model: nil,
        spec: nil
      }

      scores = AnomalyDetector.score(detector, [10.0, 12.0, 12.0, 12.0])

      assert Enum.map(scores, & &1.predicted) == [10.0, 12.0, 12.0, 12.0]

      assert Enum.map(scores, & &1.predicted_sd) == [
               2.0,
               3.0,
               3.0 * :math.sqrt(1.1),
               3.0 * :math.sqrt(1.2)
             ]
    end

    test "score/2 accumulates state covariance over the horizon" do
      spec =
        BstsNx.Components.local_level_spec(
          initial_state: 0.0,
          initial_cov: 1.0,
          process_var: 1.0,
          obs_var: 0.25
        )

      detector = %{
        method: :mcmc,
        z_threshold: 3.0,
        posterior_mean: nil,
        posterior_sd: nil,
        posterior_samples: [
          %{
            states: [Nx.tensor([0.0])],
            q_matrix: Nx.tensor([[1.0]]),
            obs_var: Nx.tensor(0.25)
          }
        ],
        mcmc_model: :structured,
        spec: spec
      }

      scores = AnomalyDetector.score(detector, [0.0, 0.0, 0.0])
      sds = Enum.map(scores, & &1.predicted_sd)

      assert Enum.at(sds, 0) < Enum.at(sds, 1)
      assert Enum.at(sds, 1) < Enum.at(sds, 2)
      assert_in_delta Enum.at(sds, 0), :math.sqrt(1.25), 1.0e-10
      assert_in_delta Enum.at(sds, 1), :math.sqrt(2.25), 1.0e-10
      assert_in_delta Enum.at(sds, 2), :math.sqrt(3.25), 1.0e-10
    end
  end
end
