defmodule BstsNx.RSidecarTest do
  use ExUnit.Case, async: false

  alias BstsNx.RSidecar

  @moduletag :external
  @moduletag timeout: 180_000

  @run_sidecar System.get_env("BSTS_NX_ENABLE_R_SIDECAR") in ["1", "true", "TRUE"]

  unless @run_sidecar do
    @moduletag skip: "set BSTS_NX_ENABLE_R_SIDECAR=1 to run R sidecar tests"
  end

  if @run_sidecar and not RSidecar.available?() do
    @moduletag skip: "Rscript with CRAN packages CausalImpact and bsts is not available"
  end

  test "runs an offline CausalImpact report" do
    pre = Enum.map(1..30, fn i -> 100.0 + 0.1 * i end)
    post = Enum.map(31..40, fn i -> 110.0 + 0.1 * i end)

    assert {:ok, result} =
             RSidecar.run_report(pre ++ post, {1, 30}, {31, 40},
               niter: 100,
               timeout_ms: 120_000
             )

    assert result.execution.engine == :r_sidecar
    assert result.execution.method_used == :r_causal_impact
    assert result.report != ""
    assert length(result.summary) > 0
    assert length(result.series) == 40
  end
end
