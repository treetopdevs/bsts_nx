defmodule BstsNxGibbsSamplerMissingObservationsTest do
  use ExUnit.Case
  import ExUnit.CaptureLog
  alias BstsNx.GibbsSampler

  test "logs a warning and skips nil observations in variance update" do
    observations = [1.0, nil, 2.0]

    log =
      capture_log(fn ->
        samples = GibbsSampler.sample(observations, 1, 0.0, 1.0, 1.0, 1.0, seed: 123)
        assert length(samples) == 1
      end)

    assert log =~ "missing values are skipped"
  end
end
