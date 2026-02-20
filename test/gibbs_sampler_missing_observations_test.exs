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

    assert log =~ "missing observations"
  end

  test "logs a warning and skips NaN observations in variance update" do
    nan = Nx.Constants.nan()
    observations = [1.0, nan, 2.0]

    log =
      capture_log(fn ->
        samples = GibbsSampler.sample(observations, 1, 0.0, 1.0, 1.0, 1.0, seed: 456)
        assert length(samples) == 1

        # Verify no NaN propagation into sampled variances
        sample = hd(samples)
        assert Nx.to_number(sample.process_var) > 0
        assert Nx.to_number(sample.obs_var) > 0
        refute BstsNx.Utils.has_non_finite?(sample.process_var)
        refute BstsNx.Utils.has_non_finite?(sample.obs_var)
      end)

    assert log =~ "missing observations"
  end
end
