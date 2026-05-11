defmodule BstsNx.RSidecar do
  @moduledoc """
  Optional bridge to R's CRAN CausalImpact/bsts stack for offline reports.

  This module is intentionally opt-in. It is useful for parity checks and
  batch reports, but it is not used by operational Elixir/Nx hot paths.
  """

  alias BstsNx.Execution

  @default_timeout_ms 120_000
  @max_payload_cells 500_000

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
    with {:ok, rscript} <- rscript_path(),
         :ok <- require_r_packages(rscript),
         :ok <- validate_periods(observations, pre_period, post_period),
         {:ok, data_path} <- write_payload_file(observations, Keyword.get(opts, :regressors)) do
      timeout_ms = timeout_ms(opts)

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
            parsed = parse_output(out)

            {:ok,
             Map.put(
               parsed,
               :execution,
               Execution.metadata(:bayesian, :r_sidecar, :r_causal_impact, elapsed_us)
             )}

          {out, status} ->
            {:error, %{reason: :r_failed, status: status, output: String.trim(out)}}

          :timeout ->
            {:error, %{reason: :r_timeout, timeout_ms: timeout_ms}}
        end
      after
        File.rm(data_path)
      end
    else
      {:error, reason} when is_atom(reason) -> {:error, %{reason: reason}}
      {:error, reason} -> {:error, reason}
    end
  rescue
    e ->
      {:error, %{reason: :r_system_error, message: Exception.message(e)}}
  end

  defp rscript_path do
    cond do
      path = System.get_env("BSTS_NX_RSCRIPT") ->
        {:ok, path}

      path = System.find_executable("Rscript") ->
        {:ok, path}

      true ->
        {:error, :rscript_not_found}
    end
  end

  defp require_r_packages(rscript) do
    code = """
    ok <- requireNamespace("CausalImpact", quietly = TRUE) &&
      requireNamespace("bsts", quietly = TRUE)
    quit(status = if (ok) 0 else 10)
    """

    case system_cmd(rscript, ["-e", code], [stderr_to_stdout: true], 30_000) do
      {_out, 0} -> :ok
      {_out, _status} -> {:error, :r_packages_not_installed}
      :timeout -> {:error, :r_timeout}
    end
  rescue
    _ -> {:error, :r_unavailable}
  end

  defp system_cmd(command, args, opts, timeout_ms) do
    task = Task.async(fn -> System.cmd(command, args, opts) end)

    case Task.yield(task, timeout_ms) || Task.shutdown(task, :brutal_kill) do
      {:ok, result} -> result
      nil -> :timeout
    end
  end

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

      path =
        Path.join(
          System.tmp_dir!(),
          "bsts_nx_r_sidecar_#{System.unique_integer([:positive, :monotonic])}.tsv"
        )

      with :ok <- File.write(path, data, [:write, :exclusive]),
           :ok <- File.chmod(path, 0o600) do
        {:ok, path}
      else
        {:error, reason} ->
          File.rm(path)
          {:error, %{reason: :payload_write_failed, message: :file.format_error(reason)}}
      end
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
    env_timeout =
      case System.get_env("BSTS_NX_R_TIMEOUT_MS") do
        nil -> nil
        value -> String.to_integer(value)
      end

    Keyword.get(opts, :timeout_ms, env_timeout || @default_timeout_ms)
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
    %{
      summary: output |> extract_block("SUMMARY") |> parse_table(),
      report: output |> extract_block("REPORT") |> String.trim(),
      series: output |> extract_block("SERIES") |> parse_table()
    }
  end

  defp extract_block(output, name) do
    begin_marker = "BSTS_NX_BEGIN_#{name}"
    end_marker = "BSTS_NX_END_#{name}"

    case String.split(output, begin_marker, parts: 2) do
      [_before, rest] ->
        rest |> String.split(end_marker, parts: 2) |> hd()

      _ ->
        ""
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
