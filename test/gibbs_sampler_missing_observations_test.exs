defmodule BstsNxGibbsSamplerMissingObservationsTest do
  use ExUnit.Case
  import ExUnit.CaptureLog
  alias BstsNx.Components
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

  test "sample_general skips NaN observations in variance update" do
    observations = [1.0, Nx.Constants.nan(), 2.0]

    samples = GibbsSampler.sample_general(observations, 1.0, 1.0, 4, burn_in: 0, seed: 123)

    assert length(samples) == 4

    Enum.each(samples, fn sample ->
      assert finite_number?(Nx.to_number(sample.process_var))
      assert finite_number?(Nx.to_number(sample.obs_var))
    end)
  end

  test "sample_structured skips NaN observations in variance update" do
    observations = [1.0, Nx.Constants.nan(), 2.0]
    spec = Components.local_level_spec(process_var: 0.5, obs_var: 1.0)

    samples = GibbsSampler.sample_structured(observations, spec, 4, burn_in: 0, seed: 123)

    assert length(samples) == 4

    Enum.each(samples, fn sample ->
      q_diag = sample.q_matrix |> Nx.take_diagonal() |> Nx.to_flat_list()

      Enum.each(q_diag, fn q -> assert finite_number?(q) end)
      assert finite_number?(Nx.to_number(sample.obs_var))
    end)
  end

  defp finite_number?(v) when is_number(v), do: v == v and v not in [:infinity, :neg_infinity]
  defp finite_number?(_), do: false
end
