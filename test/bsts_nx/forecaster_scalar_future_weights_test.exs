defmodule BstsNx.ForecasterScalarFutureWeightsTest do
  use ExUnit.Case, async: true

  alias BstsNx.Forecaster

  test "future observation weights widen an unstructured scalar forecast" do
    samples =
      Enum.map(1..32, fn _index ->
        %{
          states: [Nx.tensor(10.0)],
          state_covs: [Nx.tensor(0.0)],
          process_var: Nx.tensor(0.0),
          obs_var: Nx.tensor(1.0)
        }
      end)

    fit = %{
      posterior_samples: samples,
      spec: nil,
      training_length: 1,
      method: :scalar,
      n_regression_dims: 0,
      observation_variance_mode: :scalar
    }

    reference =
      Forecaster.predict(fit,
        horizon: 4,
        return: :both,
        future_observation_variance_weights: List.duplicate(1.0, 4),
        seed: 2026
      )

    less_reliable =
      Forecaster.predict(fit,
        horizon: 4,
        return: :both,
        future_observation_variance_weights: List.duplicate(25.0, 4),
        seed: 2026
      )

    assert Enum.sum(less_reliable.sd) > 3.0 * Enum.sum(reference.sd)
  end
end
