defmodule BstsNx.ForecasterObservationWeightsMissingTest do
  use ExUnit.Case, async: true

  alias BstsNx.Forecaster

  test "preserves missing observations through weighted fitting" do
    fit =
      Forecaster.fit([10.0, nil, 11.0, 10.5],
        observation_variance_weights: [1.0, 4.0, 1.0, 1.0],
        num_samples: 3,
        burn_in: 2,
        seed: 1_515
      )

    assert fit.method == :structured
    assert fit.observation_variance_mode == :scaled_known
    assert length(fit.posterior_samples) == 3

    forecast = Forecaster.predict(fit, horizon: 2, return: :both, seed: 1_616)

    assert forecast.num_draws == 3
    assert length(forecast.mean) == 2
    assert Enum.all?(forecast.draws, &(length(&1) == 2))
  end
end
