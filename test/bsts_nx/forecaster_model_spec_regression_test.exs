defmodule BstsNx.ForecasterModelSpecRegressionTest do
  use ExUnit.Case, async: true

  alias BstsNx.Components
  alias BstsNx.Forecaster

  test "predict accepts future_regressors with regression custom model_spec" do
    observations = Enum.map(1..12, &(&1 * 1.0))
    regressors = Nx.tensor(Enum.map(1..12, fn i -> [i * 0.1] end))

    spec =
      Components.compose_specs(
        Components.local_linear_trend_spec(initial_level: 1.0),
        Components.regression_spec(regressors)
      )

    fit_result =
      Forecaster.fit(observations, model_spec: spec, num_samples: 6, burn_in: 3, seed: 42)

    future_regressors = Nx.tensor([[1.3], [1.4], [1.5]])

    forecast =
      Forecaster.predict(
        fit_result,
        horizon: 3,
        future_regressors: future_regressors,
        seed: 42
      )

    assert forecast.horizon == 3
    assert length(forecast.mean) == 3
    assert length(forecast.lower) == 3
    assert length(forecast.upper) == 3
  end
end
