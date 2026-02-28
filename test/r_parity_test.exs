defmodule BstsNx.RParityTest do
  use ExUnit.Case

  alias BstsNx.GibbsSampler

  @moduletag :external
  @moduletag timeout: 360_000

  @r_parity_skip_reason (
                          enabled? =
                            System.get_env("BSTS_NX_ENABLE_R_PARITY") in ["1", "true", "TRUE"]

                          cond do
                            not enabled? ->
                              "set BSTS_NX_ENABLE_R_PARITY=1 to run R parity integration"

                            is_nil(System.find_executable("Rscript")) ->
                              "Rscript not found"

                            true ->
                              rscript = System.find_executable("Rscript")

                              {_out, status} =
                                System.cmd(rscript, [
                                  "-e",
                                  "quit(status = if (requireNamespace('bsts', quietly = TRUE)) 0 else 10)"
                                ])

                              if status == 0, do: nil, else: "R package 'bsts' is not installed"
                          end
                        )

  if @r_parity_skip_reason do
    @moduletag skip: @r_parity_skip_reason
  end

  test "posterior process and observation variances are close to R bsts" do
    rscript = System.find_executable("Rscript")

    :rand.seed(:exsss, {2_021, 2_022, 2_023})

    n = 160
    true_q = 0.4
    true_r = 0.9

    {_state, observations} =
      Enum.reduce(1..n, {0.0, []}, fn _i, {state, acc} ->
        state_new = state + :rand.normal() * :math.sqrt(true_q)
        y = state_new + :rand.normal() * :math.sqrt(true_r)
        {state_new, [y | acc]}
      end)

    observations = Enum.reverse(observations)

    samples =
      GibbsSampler.sample(
        observations,
        1_000,
        List.first(observations),
        10.0,
        1.0,
        1.0,
        burn_in: 1_000,
        seed: 2_024,
        prior_shape: 1.0,
        prior_scale: 1.0
      )

    elixir_q_mean = samples |> Enum.map(&Nx.to_number(&1.process_var)) |> mean()
    elixir_r_mean = samples |> Enum.map(&Nx.to_number(&1.obs_var)) |> mean()

    y_csv = observations |> Enum.map_join(",", &Float.to_string/1)

    r_code = """
    suppressPackageStartupMessages(library(bsts))

    y <- as.numeric(strsplit(Sys.getenv("Y_CSV"), ",")[[1]])
    obs.prior <- SdPrior(sigma.guess = 1.2, sample.size = 2.0, initial.value = 1.0)
    level.prior <- SdPrior(sigma.guess = 1.2, sample.size = 2.0, initial.value = 1.0)
    initial.state.prior <- NormalPrior(mu = y[1], sigma = sqrt(10.0), initial.value = y[1])

    state.spec <- AddLocalLevel(
      list(),
      y,
      sigma.prior = level.prior,
      initial.state.prior = initial.state.prior
    )

    model <- bsts(
      y,
      state.specification = state.spec,
      prior = obs.prior,
      niter = 2000,
      ping = 0,
      seed = 2024
    )
    burn <- 1001

    sigma_obs <- as.numeric(model$sigma.obs)
    sigma_obs <- sigma_obs[burn:length(sigma_obs)]

    as_draw_vector <- function(v) {
      if (is.null(v)) return(NULL)

      if (is.matrix(v)) {
        if (ncol(v) < 1) return(NULL)
        return(as.numeric(v[, 1]))
      }

      if (length(v) > 1) return(as.numeric(v))
      NULL
    }

    extract_level_sigma <- function(m) {
      for (nm in c("sigma.level", "sigma.trend", "sigma.state")) {
        v <- as_draw_vector(m[[nm]])
        if (!is.null(v)) return(v)
      }

      comp <- m$state.specification[[1]]

      for (nm in c("sigma.draws", "sigma", "sigma.level", "sigma_level")) {
        v <- as_draw_vector(comp[[nm]])
        if (!is.null(v)) return(v)
      }

      v <- as_draw_vector(m$state.sigma)
      if (!is.null(v)) return(v)

      stop("Unable to extract local-level sigma draws from bsts object")
    }

    sigma_level <- extract_level_sigma(model)
    if (length(sigma_level) >= burn) {
      sigma_level <- sigma_level[burn:length(sigma_level)]
    }

    cat(mean(sigma_level^2), ",", mean(sigma_obs^2), sep = "")
    """

    {out, status} = System.cmd(rscript, ["-e", r_code], env: [{"Y_CSV", y_csv}])

    if status != 0 do
      flunk("R parity script failed: #{String.trim(out)}")
    end

    [r_q_mean, r_r_mean] =
      out
      |> String.trim()
      |> String.split(",")
      |> Enum.map(&String.to_float/1)

    tol = read_tolerance()

    assert relative_diff(elixir_q_mean, r_q_mean) <= tol,
           "process variance mean mismatch: elixir=#{elixir_q_mean}, r=#{r_q_mean}"

    assert relative_diff(elixir_r_mean, r_r_mean) <= tol,
           "observation variance mean mismatch: elixir=#{elixir_r_mean}, r=#{r_r_mean}"
  end

  defp mean(xs), do: Enum.sum(xs) / max(length(xs), 1)

  defp relative_diff(a, b) do
    denom = max(abs(b), 1.0e-12)
    abs(a - b) / denom
  end

  defp read_tolerance do
    case System.get_env("BSTS_NX_R_PARITY_TOL") do
      nil -> 0.02
      s -> String.to_float(s)
    end
  end
end
