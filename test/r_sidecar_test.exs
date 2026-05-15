defmodule BstsNx.RSidecarTest do
  use ExUnit.Case, async: false

  alias BstsNx.RSidecar
  import ExUnit.CaptureLog

  @moduletag timeout: 180_000

  setup do
    old_rscript = System.get_env("BSTS_NX_RSCRIPT")
    old_timeout = System.get_env("BSTS_NX_R_TIMEOUT_MS")

    on_exit(fn ->
      restore_env("BSTS_NX_RSCRIPT", old_rscript)
      restore_env("BSTS_NX_R_TIMEOUT_MS", old_timeout)
    end)

    :ok
  end

  test "availability rejects a command that exits successfully but is not Rscript" do
    System.put_env("BSTS_NX_RSCRIPT", "/usr/bin/true")

    assert {:error, :invalid_rscript_path} = RSidecar.availability()
    refute RSidecar.available?()
  end

  test "availability accepts a trimmed executable Rscript override" do
    rscript =
      fake_rscript!("""
      echo "BSTS_NX_R_SIDECAR_READY"
      exit 0
      """)

    System.put_env("BSTS_NX_RSCRIPT", "  #{rscript}  ")

    assert :ok = RSidecar.availability()
    assert RSidecar.available?()
  end

  test "run_report fails when R output is successful but missing parse markers" do
    rscript =
      fake_rscript!("""
      if echo "$*" | grep -q "requireNamespace"; then
        echo "BSTS_NX_R_SIDECAR_READY"
        exit 0
      fi

      echo "report completed without sidecar markers"
      exit 0
      """)

    System.put_env("BSTS_NX_RSCRIPT", rscript)

    assert {:error, %{reason: :missing_r_output_marker, marker: "SUMMARY"}} =
             RSidecar.run_report(Enum.map(1..10, &(&1 * 1.0)), {1, 6}, {7, 10}, timeout_ms: 1_000)
  end

  test "invalid BSTS_NX_R_TIMEOUT_MS returns a typed configuration error" do
    rscript =
      fake_rscript!("""
      if echo "$*" | grep -q "requireNamespace"; then
        echo "BSTS_NX_R_SIDECAR_READY"
        exit 0
      fi

      cat <<'EOF'
      BSTS_NX_BEGIN_SUMMARY
      metric\tActual
      Average\t1.0
      BSTS_NX_END_SUMMARY
      BSTS_NX_BEGIN_REPORT
      ok
      BSTS_NX_END_REPORT
      BSTS_NX_BEGIN_SERIES
      index\tresponse
      1\t1.0
      2\t2.0
      3\t3.0
      4\t4.0
      5\t5.0
      6\t6.0
      7\t7.0
      8\t8.0
      9\t9.0
      10\t10.0
      BSTS_NX_END_SERIES
      EOF
      exit 0
      """)

    System.put_env("BSTS_NX_RSCRIPT", rscript)
    System.put_env("BSTS_NX_R_TIMEOUT_MS", "not-an-integer")

    assert {:error, %{reason: :invalid_timeout_env, value: "not-an-integer"}} =
             RSidecar.run_report(Enum.map(1..10, &(&1 * 1.0)), {1, 6}, {7, 10}, [])
  end

  test "timeout option strings are parsed as positive millisecond values" do
    rscript =
      fake_rscript!("""
      if echo "$*" | grep -q "requireNamespace"; then
        echo "BSTS_NX_R_SIDECAR_READY"
        exit 0
      fi

      sleep 1
      exit 0
      """)

    System.put_env("BSTS_NX_RSCRIPT", rscript)

    assert {:error, %{reason: :r_timeout, timeout_ms: 1}} =
             RSidecar.run_report(Enum.map(1..10, &(&1 * 1.0)), {1, 6}, {7, 10}, timeout_ms: "1")
  end

  test "run_report clears unrelated parent environment variables from Rscript" do
    old_secret = System.get_env("BSTS_NX_SIDE_SECRET")

    on_exit(fn -> restore_env("BSTS_NX_SIDE_SECRET", old_secret) end)

    rscript =
      fake_rscript!("""
      if echo "$*" | grep -q "requireNamespace"; then
        echo "BSTS_NX_R_SIDECAR_READY"
        exit 0
      fi

      if [ -n "$BSTS_NX_SIDE_SECRET" ]; then
        echo "secret leaked into sidecar"
        exit 7
      fi

      cat <<'EOF'
      BSTS_NX_BEGIN_SUMMARY
      metric\tActual
      Average\t1.0
      BSTS_NX_END_SUMMARY
      BSTS_NX_BEGIN_REPORT
      ok
      BSTS_NX_END_REPORT
      BSTS_NX_BEGIN_SERIES
      index\tresponse
      1\t1.0
      2\t2.0
      3\t3.0
      4\t4.0
      5\t5.0
      6\t6.0
      7\t7.0
      8\t8.0
      9\t9.0
      10\t10.0
      BSTS_NX_END_SERIES
      EOF
      exit 0
      """)

    System.put_env("BSTS_NX_RSCRIPT", rscript)
    System.put_env("BSTS_NX_SIDE_SECRET", "do-not-forward")

    assert {:ok, result} =
             RSidecar.run_report(Enum.map(1..10, &(&1 * 1.0)), {1, 6}, {7, 10}, [])

    assert result.report == "ok"
  end

  test "run_report returns bounded error output instead of full stdout and stderr" do
    rscript =
      fake_rscript!("""
      if echo "$*" | grep -q "requireNamespace"; then
        echo "BSTS_NX_R_SIDECAR_READY"
        exit 0
      fi

      yes "very long failure line" | head -n 400
      exit 9
      """)

    System.put_env("BSTS_NX_RSCRIPT", rscript)

    log =
      capture_log([level: :debug], fn ->
        assert {:error, error} =
                 RSidecar.run_report(Enum.map(1..10, &(&1 * 1.0)), {1, 6}, {7, 10}, [])

        send(self(), {:sidecar_error, error})
      end)

    assert_receive {:sidecar_error, error}

    assert error.reason == :r_failed
    assert error.status == 9
    assert is_binary(error.output_excerpt)
    assert byte_size(error.output_excerpt) <= 4_200
    refute Map.has_key?(error, :output)
    assert log =~ "R sidecar failed with status 9"
    assert log =~ "very long failure line"
  end

  test "timeout terminates the direct Rscript process" do
    pid_file =
      Path.join(System.tmp_dir!(), "bsts_nx_r_sidecar_pid_#{System.unique_integer([:positive])}")

    on_exit(fn -> File.rm(pid_file) end)

    rscript =
      fake_rscript!("""
      if echo "$*" | grep -q "requireNamespace"; then
        echo "BSTS_NX_R_SIDECAR_READY"
        exit 0
      fi

      echo $$ > "#{pid_file}"
      sleep 3
      exit 0
      """)

    System.put_env("BSTS_NX_RSCRIPT", rscript)

    assert {:error, %{reason: :r_timeout, timeout_ms: 10}} =
             RSidecar.run_report(Enum.map(1..10, &(&1 * 1.0)), {1, 6}, {7, 10}, timeout_ms: 10)

    assert {:ok, pid} = read_pid(pid_file)
    refute process_alive?(pid)
  end

  test "availability rejects a symlinked Rscript override" do
    target =
      fake_rscript!("""
      echo "BSTS_NX_R_SIDECAR_READY"
      exit 0
      """)

    dir =
      Path.join(
        System.tmp_dir!(),
        "bsts_nx_r_sidecar_symlink_#{System.unique_integer([:positive])}"
      )

    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf(dir) end)

    link = Path.join(dir, "Rscript")
    File.ln_s!(target, link)
    System.put_env("BSTS_NX_RSCRIPT", link)

    assert {:error, :invalid_rscript_path} = RSidecar.availability()
  end

  @run_sidecar System.get_env("BSTS_NX_ENABLE_R_SIDECAR") in ["1", "true", "TRUE"]
  @sidecar_skip_reason (if @run_sidecar do
                          if RSidecar.available?() do
                            false
                          else
                            "Rscript with CRAN packages CausalImpact and bsts is not available"
                          end
                        else
                          "set BSTS_NX_ENABLE_R_SIDECAR=1 to run R sidecar tests"
                        end)

  @tag :external
  @tag skip: @sidecar_skip_reason
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

  defp fake_rscript!(body) do
    dir =
      Path.join(System.tmp_dir!(), "bsts_nx_r_sidecar_test_#{System.unique_integer([:positive])}")

    File.mkdir_p!(dir)
    on_exit(fn -> File.rm_rf(dir) end)

    path = Path.join(dir, "Rscript")

    File.write!(path, """
    #!/bin/sh
    #{body}
    """)

    File.chmod!(path, 0o700)
    path
  end

  defp restore_env(key, nil), do: System.delete_env(key)
  defp restore_env(key, value), do: System.put_env(key, value)

  defp read_pid(path) do
    deadline = System.monotonic_time(:millisecond) + 500
    read_pid_until(path, deadline)
  end

  defp read_pid_until(path, deadline) do
    case File.read(path) do
      {:ok, contents} ->
        {:ok, String.trim(contents)}

      {:error, :enoent} ->
        if System.monotonic_time(:millisecond) < deadline do
          Process.sleep(10)
          read_pid_until(path, deadline)
        else
          {:error, :enoent}
        end
    end
  end

  defp process_alive?(pid) do
    case System.cmd("kill", ["-0", pid], stderr_to_stdout: true) do
      {_out, 0} -> true
      {_out, _} -> false
    end
  end
end
