defmodule BstsNx.RSidecar do
  @moduledoc """
  Optional bridge to R's CRAN CausalImpact/bsts stack for offline reports.

  This module is intentionally opt-in. It is useful for parity checks and
  batch reports, but it is not used by operational Elixir/Nx hot paths.
  """

  alias BstsNx.Execution
  import Bitwise, only: [band: 2]
  require Logger

  @default_timeout_ms 120_000
  @max_payload_cells 500_000
  @max_error_output_bytes 4_096
  @availability_sentinel "BSTS_NX_R_SIDECAR_READY"
  @missing_packages_sentinel "BSTS_NX_R_PACKAGES_MISSING"
  @sidecar_env_allowlist [
    "PATH",
    "HOME",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "R_HOME",
    "R_LIBS",
    "R_LIBS_USER",
    "R_LIBS_SITE"
  ]

  @type run_result :: %{
          summary: [map()],
          report: String.t(),
          series: [map()],
          execution: Execution.t()
        }

  @doc """
  Returns true when `Rscript`, `CausalImpact`, and `bsts` are available.
  """
  @spec available?() :: boolean()
  def available? do
    match?(:ok, availability())
  end

  @doc """
  Returns `:ok` or `{:error, reason}` with the missing sidecar dependency.
  """
  @spec availability() :: :ok | {:error, atom()}
  def availability do
    with {:ok, rscript} <- rscript_path(),
         :ok <- require_r_packages(rscript) do
      :ok
    end
  end

  @doc """
  Runs an offline CausalImpact report through R.

  `observations` is the response vector. `pre_period` and `post_period` use the
  same 1-based inclusive period convention as the Elixir APIs.

  Options:

    * `:regressors` - optional `{T, p}` tensor/list of regressor rows
    * `:niter` - CausalImpact model iterations, default `1000`
    * `:alpha` - interval tail probability, default `0.05`
    * `:timeout_ms` - process timeout, default `BSTS_NX_R_TIMEOUT_MS` or 120s
    * `:nseasons` and `:season_duration` - optional CausalImpact model args
  """
  @spec run_report(
          [number()],
          {pos_integer(), pos_integer()},
          {pos_integer(), pos_integer()},
          keyword()
        ) ::
          {:ok, run_result()} | {:error, map()}
  def run_report(observations, pre_period, post_period, opts) do
    with {:ok, timeout_ms} <- timeout_ms(opts),
         {:ok, rscript} <- rscript_path(),
         :ok <- require_r_packages(rscript),
         :ok <- validate_periods(observations, pre_period, post_period),
         {:ok, data_path, temp_dir} <-
           write_payload_file(observations, Keyword.get(opts, :regressors)) do
      env =
        [
          {"BSTS_NX_DATA_PATH", data_path},
          {"BSTS_NX_PRE_START", elem(pre_period, 0) |> to_string()},
          {"BSTS_NX_PRE_END", elem(pre_period, 1) |> to_string()},
          {"BSTS_NX_POST_START", elem(post_period, 0) |> to_string()},
          {"BSTS_NX_POST_END", elem(post_period, 1) |> to_string()},
          {"BSTS_NX_NITER", Keyword.get(opts, :niter, 1000) |> to_string()},
          {"BSTS_NX_ALPHA", Keyword.get(opts, :alpha, 0.05) |> to_string()},
          {"BSTS_NX_NSEASONS", Keyword.get(opts, :nseasons, 1) |> to_string()},
          {"BSTS_NX_SEASON_DURATION", Keyword.get(opts, :season_duration, 1) |> to_string()}
        ]

      try do
        {elapsed_us, result} =
          Execution.measure(fn ->
            system_cmd(rscript, ["-e", r_code()], [env: env, stderr_to_stdout: true], timeout_ms)
          end)

        case result do
          {out, 0} ->
            case parse_output(out) do
              {:ok, parsed} ->
                {:ok,
                 Map.put(
                   parsed,
                   :execution,
                   Execution.metadata(:bayesian, :r_sidecar, :r_causal_impact, elapsed_us)
                 )}

              {:error, reason} ->
                {:error, reason}
            end

          {out, status} ->
            Logger.debug(fn ->
              "R sidecar failed with status #{status}:\n#{String.trim(out)}"
            end)

            {:error, %{reason: :r_failed, status: status, output_excerpt: output_excerpt(out)}}

          :timeout ->
            {:error, %{reason: :r_timeout, timeout_ms: timeout_ms}}
        end
      after
        cleanup_payload(data_path, temp_dir)
      end
    else
      {:error, reason} when is_atom(reason) -> {:error, %{reason: reason}}
      {:error, reason} -> {:error, reason}
    end
  rescue
    e ->
      Logger.error(Exception.format(:error, e, __STACKTRACE__))
      {:error, %{reason: :r_system_error, message: Exception.message(e)}}
  end

  defp rscript_path do
    env_path =
      case System.get_env("BSTS_NX_RSCRIPT") do
        nil -> nil
        value -> String.trim(value)
      end

    cond do
      is_binary(env_path) and env_path != "" ->
        validate_rscript_override(env_path)

      path = System.find_executable("Rscript") ->
        {:ok, path}

      true ->
        {:error, :rscript_not_found}
    end
  end

  defp validate_rscript_override(path) do
    expanded = Path.expand(path)

    with true <- Path.basename(expanded) in ["Rscript", "Rscript.exe"],
         {:ok, %{type: :regular}} <- File.lstat(expanded),
         {:ok, %{type: :regular, mode: mode}} <- File.stat(expanded),
         true <- band(mode, 0o111) != 0 do
      {:ok, expanded}
    else
      _ -> {:error, :invalid_rscript_path}
    end
  end

  defp require_r_packages(rscript) do
    code = """
    ok <- requireNamespace("CausalImpact", quietly = TRUE) &&
      requireNamespace("bsts", quietly = TRUE)
    if (ok) {
      cat("#{@availability_sentinel}\\n")
      quit(status = 0)
    } else {
      cat("#{@missing_packages_sentinel}\\n")
      quit(status = 10)
    }
    """

    case system_cmd(rscript, ["-e", code], [stderr_to_stdout: true], 30_000) do
      {out, 0} ->
        if sentinel_line?(out, @availability_sentinel) do
          :ok
        else
          {:error, :rscript_invalid}
        end

      {out, _status} ->
        if sentinel_line?(out, @missing_packages_sentinel) do
          {:error, :r_packages_not_installed}
        else
          {:error, :r_unavailable}
        end

      :timeout ->
        {:error, :r_timeout}
    end
  rescue
    _ -> {:error, :r_unavailable}
  end

  defp sentinel_line?(output, sentinel) do
    output
    |> String.split(["\r\n", "\n", "\r"], trim: true)
    |> Enum.any?(&(String.trim(&1) == sentinel))
  end

  defp system_cmd(command, args, opts, timeout_ms) do
    port_opts =
      [
        :binary,
        :exit_status,
        args: args,
        env: sidecar_env(Keyword.get(opts, :env, []))
      ]
      |> maybe_stderr_to_stdout(Keyword.get(opts, :stderr_to_stdout, false))

    port = Port.open({:spawn_executable, String.to_charlist(command)}, port_opts)
    deadline = System.monotonic_time(:millisecond) + timeout_ms

    collect_port(port, [], deadline)
  end

  defp maybe_stderr_to_stdout(port_opts, true), do: [:stderr_to_stdout | port_opts]
  defp maybe_stderr_to_stdout(port_opts, _false), do: port_opts

  defp collect_port(port, chunks, deadline) do
    timeout = max(deadline - System.monotonic_time(:millisecond), 0)

    receive do
      {^port, {:data, data}} ->
        collect_port(port, [data | chunks], deadline)

      {^port, {:exit_status, status}} ->
        {chunks |> Enum.reverse() |> IO.iodata_to_binary(), status}
    after
      timeout ->
        terminate_port(port)
        :timeout
    end
  end

  defp terminate_port(port) do
    os_pid =
      case Port.info(port, :os_pid) do
        {:os_pid, pid} -> pid
        _ -> nil
      end

    if is_integer(os_pid) do
      kill_os_pid(os_pid, "-TERM")
    end

    try do
      Port.close(port)
    rescue
      _ -> :ok
    catch
      :exit, _ -> :ok
    end

    if is_integer(os_pid) do
      Process.sleep(20)
      kill_os_pid(os_pid, "-KILL")
    end

    :ok
  end

  defp kill_os_pid(pid, signal) do
    System.cmd("kill", [signal, Integer.to_string(pid)], stderr_to_stdout: true)
    :ok
  rescue
    _ -> :ok
  end

  defp sidecar_env(extra_env) do
    cleared_parent =
      System.get_env()
      |> Map.keys()
      |> Enum.map(fn key -> {String.to_charlist(key), false} end)

    allowed_parent =
      @sidecar_env_allowlist
      |> Enum.flat_map(fn key ->
        case System.get_env(key) do
          nil -> []
          value -> [{String.to_charlist(key), String.to_charlist(value)}]
        end
      end)

    explicit =
      Enum.map(extra_env, fn {key, value} ->
        {to_env_charlist(key), to_env_charlist(value)}
      end)

    cleared_parent ++ allowed_parent ++ explicit
  end

  defp to_env_charlist(value) when is_binary(value), do: String.to_charlist(value)
  defp to_env_charlist(value), do: value |> to_string() |> String.to_charlist()

  defp validate_periods(observations, {pre_start, pre_end}, {post_start, post_end}) do
    n = length(observations)

    cond do
      n == 0 ->
        {:error, :empty_observations}

      pre_start < 1 or pre_end < pre_start or pre_end > n ->
        {:error, %{reason: :invalid_pre_period, n: n}}

      post_start <= pre_end or post_end < post_start or post_end > n ->
        {:error, %{reason: :invalid_post_period, n: n}}

      true ->
        :ok
    end
  end

  defp write_payload_file(observations, regressors) do
    with {:ok, response_values} <- normalize_numeric_list(observations, :response),
         {:ok, regressor_rows, regressor_cols} <- normalize_regressors(observations, regressors),
         :ok <- validate_payload_size(length(response_values), regressor_cols) do
      data =
        response_values
        |> payload_lines(regressor_rows, regressor_cols)
        |> Enum.join("\n")
        |> Kernel.<>("\n")

      temp_dir = private_temp_dir()
      path = Path.join(temp_dir, "payload.tsv")

      case File.mkdir(temp_dir) do
        :ok ->
          with :ok <- File.chmod(temp_dir, 0o700),
               :ok <- File.write(path, data, [:write, :exclusive]),
               :ok <- File.chmod(path, 0o600) do
            {:ok, path, temp_dir}
          else
            {:error, reason} ->
              cleanup_payload(path, temp_dir)
              payload_write_error(reason)
          end

        {:error, reason} ->
          payload_write_error(reason)
      end
    end
  end

  defp payload_write_error(reason) do
    {:error, %{reason: :payload_write_failed, message: :file.format_error(reason)}}
  end

  defp private_temp_dir do
    Path.join(
      System.tmp_dir!(),
      "bsts_nx_r_sidecar_#{System.unique_integer([:positive, :monotonic])}"
    )
  end

  defp cleanup_payload(data_path, temp_dir) do
    expanded_temp = Path.expand(temp_dir)
    expected_prefix = Path.join(System.tmp_dir!(), "bsts_nx_r_sidecar_") |> Path.expand()

    if String.starts_with?(expanded_temp, expected_prefix) do
      File.rm_rf(temp_dir)
    else
      File.rm(data_path)
    end
  end

  defp normalize_regressors(observations, nil) do
    {:ok, List.duplicate([], length(observations)), 0}
  end

  defp normalize_regressors(observations, regressors) do
    tensor =
      case regressors do
        %Nx.Tensor{} = t ->
          t

        rows when is_list(rows) ->
          Nx.tensor(rows)

        other ->
          raise ArgumentError, "regressors must be an Nx tensor or list, got: #{inspect(other)}"
      end

    if Nx.rank(tensor) != 2 do
      {:error, %{reason: :invalid_regressors, shape: Nx.shape(tensor)}}
    else
      {rows, cols} = Nx.shape(tensor)

      if rows != length(observations) do
        {:error,
         %{reason: :invalid_regressor_rows, rows: rows, observations: length(observations)}}
      else
        with {:ok, values} <- tensor |> Nx.to_flat_list() |> normalize_numeric_list(:regressors) do
          {:ok, Enum.chunk_every(values, cols), cols}
        end
      end
    end
  rescue
    e in ArgumentError -> {:error, %{reason: :invalid_regressors, message: Exception.message(e)}}
  end

  defp normalize_numeric_list(values, field) do
    values
    |> Enum.with_index()
    |> Enum.reduce_while({:ok, []}, fn {value, idx}, {:ok, acc} ->
      cond do
        is_integer(value) ->
          {:cont, {:ok, [value * 1.0 | acc]}}

        is_float(value) and value == value ->
          {:cont, {:ok, [value | acc]}}

        true ->
          {:halt, {:error, %{reason: :invalid_numeric_value, field: field, index: idx}}}
      end
    end)
    |> case do
      {:ok, normalized} -> {:ok, Enum.reverse(normalized)}
      error -> error
    end
  end

  defp validate_payload_size(rows, regressor_cols) do
    cells = rows * (regressor_cols + 1)

    if cells > @max_payload_cells do
      {:error, %{reason: :payload_too_large, cells: cells, max_cells: @max_payload_cells}}
    else
      :ok
    end
  end

  defp payload_lines(response_values, regressor_rows, regressor_cols) do
    header = ["response" | Enum.map(1..regressor_cols//1, &"x#{&1}")]

    rows =
      Enum.zip(response_values, regressor_rows)
      |> Enum.map(fn {response, regressors} ->
        [response | regressors]
        |> Enum.map_join("\t", &to_string/1)
      end)

    [Enum.join(header, "\t") | rows]
  end

  defp timeout_ms(opts) do
    if Keyword.has_key?(opts, :timeout_ms) do
      case parse_timeout_ms(Keyword.fetch!(opts, :timeout_ms)) do
        {:ok, timeout} ->
          {:ok, timeout}

        :error ->
          {:error, %{reason: :invalid_timeout_option, value: Keyword.fetch!(opts, :timeout_ms)}}
      end
    else
      env_timeout_ms()
    end
  end

  defp env_timeout_ms do
    case System.get_env("BSTS_NX_R_TIMEOUT_MS") do
      nil ->
        {:ok, @default_timeout_ms}

      value ->
        case parse_timeout_ms(value) do
          {:ok, timeout} -> {:ok, timeout}
          :error -> {:error, %{reason: :invalid_timeout_env, value: value}}
        end
    end
  end

  defp parse_timeout_ms(timeout_ms) when is_integer(timeout_ms) and timeout_ms > 0,
    do: {:ok, timeout_ms}

  defp parse_timeout_ms(value) when is_binary(value) do
    case Integer.parse(String.trim(value)) do
      {timeout_ms, ""} when timeout_ms > 0 -> {:ok, timeout_ms}
      _ -> :error
    end
  end

  defp parse_timeout_ms(_value), do: :error

  defp output_excerpt(output) do
    trimmed = String.trim(output)

    if byte_size(trimmed) <= @max_error_output_bytes do
      trimmed
    else
      binary_part(trimmed, 0, @max_error_output_bytes) <> "\n...[truncated]"
    end
  end

  defp r_code do
    """
    suppressPackageStartupMessages(library(CausalImpact))

    data_path <- Sys.getenv("BSTS_NX_DATA_PATH")
    if (!nzchar(data_path)) {
      stop("BSTS_NX_DATA_PATH is required")
    }

    data_frame <- read.delim(data_path, check.names = FALSE)
    response <- as.numeric(data_frame$response)
    regressor_names <- setdiff(names(data_frame), "response")

    if (length(regressor_names) > 0) {
      regressors <- as.matrix(data_frame[, regressor_names, drop = FALSE])
      data <- cbind(response, regressors)
    } else {
      data <- response
    }

    pre.period <- c(as.integer(Sys.getenv("BSTS_NX_PRE_START")), as.integer(Sys.getenv("BSTS_NX_PRE_END")))
    post.period <- c(as.integer(Sys.getenv("BSTS_NX_POST_START")), as.integer(Sys.getenv("BSTS_NX_POST_END")))
    alpha <- as.numeric(Sys.getenv("BSTS_NX_ALPHA"))
    niter <- as.integer(Sys.getenv("BSTS_NX_NITER"))
    nseasons <- as.integer(Sys.getenv("BSTS_NX_NSEASONS"))
    season.duration <- as.integer(Sys.getenv("BSTS_NX_SEASON_DURATION"))

    model.args <- list(niter = niter)
    if (!is.na(nseasons) && nseasons > 1) {
      model.args$nseasons <- nseasons
      model.args$season.duration <- season.duration
    }

    impact <- CausalImpact(data, pre.period, post.period, model.args = model.args, alpha = alpha)

    cat("BSTS_NX_BEGIN_SUMMARY\\n")
    write.table(cbind(metric = rownames(impact$summary), impact$summary),
      sep = "\\t", row.names = FALSE, quote = FALSE, na = "")
    cat("BSTS_NX_END_SUMMARY\\n")

    cat("BSTS_NX_BEGIN_REPORT\\n")
    cat(summary(impact, "report"))
    cat("\\nBSTS_NX_END_REPORT\\n")

    cat("BSTS_NX_BEGIN_SERIES\\n")
    write.table(cbind(index = seq_len(nrow(impact$series)), as.data.frame(impact$series)),
      sep = "\\t", row.names = FALSE, quote = FALSE, na = "")
    cat("BSTS_NX_END_SERIES\\n")
    """
  end

  defp parse_output(output) do
    with {:ok, summary} <- extract_block(output, "SUMMARY"),
         {:ok, report} <- extract_block(output, "REPORT"),
         {:ok, series} <- extract_block(output, "SERIES") do
      {:ok,
       %{
         summary: parse_table(summary),
         report: String.trim(report),
         series: parse_table(series)
       }}
    end
  end

  defp extract_block(output, name) do
    begin_marker = "BSTS_NX_BEGIN_#{name}"
    end_marker = "BSTS_NX_END_#{name}"

    case String.split(output, begin_marker, parts: 2) do
      [_before, rest] ->
        case String.split(rest, end_marker, parts: 2) do
          [block, _after] -> {:ok, block}
          _ -> {:error, %{reason: :missing_r_output_marker, marker: name}}
        end

      _ ->
        {:error, %{reason: :missing_r_output_marker, marker: name}}
    end
  end

  defp parse_table(""), do: []

  defp parse_table(table) do
    table
    |> String.trim()
    |> String.split("\n", trim: true)
    |> case do
      [] ->
        []

      [header | rows] ->
        keys = String.split(header, "\t")

        Enum.map(rows, fn row ->
          values = String.split(row, "\t")
          keys |> Enum.zip(values) |> Map.new(fn {k, v} -> {k, parse_value(v)} end)
        end)
    end
  end

  defp parse_value(""), do: nil

  defp parse_value(value) do
    case Float.parse(value) do
      {number, ""} -> number
      _ -> value
    end
  end
end
