defmodule BstsSiteWeb.Trust.DiagnosticsLive do
  @moduledoc """
  Trust: break your own results. One causal-impact fit on planted data, then
  the `BstsNx.Validation` suite tries to knock it over five different ways.
  """
  use BstsSiteWeb, :live_view

  alias BstsSite.Demos.Diagnostics
  alias BstsSite.Demos.Limiter

  @impl true
  def mount(_params, _session, socket) do
    {:ok,
     socket
     |> assign(page_title: "Break your own results")
     |> assign(scenario: Diagnostics.scenario(), result: nil, running: false, busy: false)
     |> assign(revealed: false)}
  end

  @impl true
  def handle_event("run_checks", _params, %{assigns: %{running: true}} = socket) do
    {:noreply, socket}
  end

  def handle_event("run_checks", _params, socket) do
    scenario = socket.assigns.scenario
    socket = assign(socket, running: true, busy: false)

    {:noreply,
     start_async(socket, :checks, fn ->
       case Limiter.run(fn -> Diagnostics.run(scenario) end) do
         {:ok, result} -> result
         :busy -> :busy
       end
     end)}
  end

  def handle_event("reveal_truth", _params, socket) do
    {:noreply, assign(socket, revealed: true)}
  end

  @impl true
  def handle_async(:checks, {:ok, :busy}, socket) do
    {:noreply, assign(socket, running: false, busy: true)}
  end

  def handle_async(:checks, {:ok, result}, socket) do
    {:noreply, assign(socket, running: false, busy: false, result: result)}
  end

  def handle_async(:checks, {:exit, _reason}, socket) do
    {:noreply, assign(socket, running: false, busy: true)}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <Layouts.app flash={@flash} track={:trust}>
      <div class="mx-auto max-w-4xl">
        <.mantra active={:compare} class="mb-4" />
        <.section_heading eyebrow="Trust · one fit, five attacks" title="Break your own results">
          A causal estimate that merely <em>looks</em> plausible is not worth much. The
          library ships a validation suite that attacks a finished analysis from five
          directions: does the baseline even track the data, are the intervals honest,
          is there structure left in the residuals, does the method find effects where
          nothing happened, and does the answer survive a nudge to the analysis window?
          Below, one fit on planted data takes all five hits in public.
        </.section_heading>

        <.figure
          no="1"
          caption={fig1_caption(@result)}
          execution={@result && @result.execution_fit}
        >
          <:legend>
            <.swatch color={:ink} label="observed" />
            <.swatch :if={@result} color={:model} label="model baseline / counterfactual" dash />
          </:legend>
          <.line_figure
            id="diagnostics-fit"
            height={300}
            y_label="units/day"
            series={fit_series(@scenario, @result)}
            bands={fit_bands(@result)}
            vlines={[
              %{x: Diagnostics.post_start() - 1, label: "campaign on"},
              %{x: Diagnostics.post_end(), label: "window ends"}
            ]}
          />
        </.figure>

        <div class="flex flex-wrap items-center gap-4">
          <button
            type="button"
            phx-click="run_checks"
            disabled={@running}
            class="btn btn-primary font-data"
          >
            {if @result, do: "Run the fit and its checks again", else: "Run the fit and its checks"}
          </button>
          <p :if={@running} class="font-data text-sm text-(--color-ink-soft)" aria-live="polite">
            Fitting, then re-fitting three more times for the placebo and stability checks…
          </p>
        </div>

        <.verdict_card
          :if={@busy}
          class="mt-6"
          verdict="Another visitor is sampling — try again in a moment."
          tone={:warning}
        >
          This page runs four MCMC fits back to back, so concurrent runs are limited on
          this small shared machine.
        </.verdict_card>

        <div :if={@result} class="mt-2">
          <div class="grid gap-4 sm:grid-cols-[1fr_minmax(240px,0.9fr)]">
            <.verdict_card
              verdict={effect_verdict(@result)}
              tone={if @result.significant?, do: :lift, else: :neutral}
            >
              <:stat
                label="Cumulative effect"
                value={"#{signed(@result.cumulative.mean)} units"}
                hint={"95% CI [#{fmt(@result.cumulative.lower)}, #{fmt(@result.cumulative.upper)}]"}
              />
              <:stat label="Relative lift" value={pct(@result.relative.mean)} />
              <:stat label="Checks passed" value={"#{@result.pass_count} of 5"} />
            </.verdict_card>

            <.reveal_truth
              revealed={@revealed}
              truth={"Planted: +#{fmt(@scenario.truth.per_step)}/day from day #{Diagnostics.post_start()} — #{signed(@scenario.truth.cumulative)} cumulative over the #{@scenario.truth.window_days}-day window."}
              grade={grade_line(@result, @scenario)}
            />
          </div>

          <.figure
            no="2"
            caption={"Pre-period residuals (observed minus the model's baseline) against the 95% band the coverage check tests — #{cov_pct(@result)}% of the #{Diagnostics.n_pre()} points fall inside."}
            execution={@result.execution_checks}
          >
            <:legend>
              <.swatch color={:ink} label="residuals" />
              <.swatch color={:model} label="95% band" />
            </:legend>
            <.line_figure
              id="diagnostics-residuals"
              height={220}
              y_label="residual"
              series={[%{points: @result.residuals, color: :ink, width: 1.3}]}
              bands={[
                %{lower: @result.resid_lower, upper: @result.resid_upper, x_offset: 0}
              ]}
              hlines={[%{y: 0.0, label: nil, color: :ink}]}
            />
          </.figure>

          <h3 class="font-display mt-8 text-xl font-bold">The five checks</h3>
          <div class="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            <.verdict_card
              verdict={check_verdict(@result.verdicts.prediction_error, "Prediction error")}
              tone={check_tone(@result.verdicts.prediction_error)}
            >
              Would catch a baseline that cannot even track the pre-period — everything
              downstream inherits that failure.
              <:stat label="MAPE" value={pct_plain(@result.details.prediction_error.mape)} />
              <:stat label="RMSE" value={fmt(@result.details.prediction_error.rmse)} />
            </.verdict_card>

            <.verdict_card
              verdict={check_verdict(@result.verdicts.coverage, "CI coverage")}
              tone={check_tone(@result.verdicts.coverage)}
            >
              Would catch dishonest intervals — too narrow means overconfident, too wide
              means the uncertainty is theater. Target: 85–99% of points inside.
              <:stat
                label="Coverage"
                value={"#{cov_pct(@result)}%"}
                hint={"#{@result.details.coverage.n_covered}/#{@result.details.coverage.n_total} points"}
              />
            </.verdict_card>

            <.verdict_card
              verdict={check_verdict(@result.verdicts.durbin_watson, "Durbin-Watson")}
              tone={check_tone(@result.verdicts.durbin_watson)}
            >
              Would catch structure left in the residuals — a trend or cycle the model
              never absorbed. 2.0 means uncorrelated; 1.5–2.5 passes.
              <:stat label="Statistic" value={fmt2(@result.details.durbin_watson.statistic)} />
            </.verdict_card>

            <.verdict_card
              verdict={check_verdict(@result.verdicts.placebo, "Placebo test")}
              tone={check_tone(@result.verdicts.placebo)}
            >
              Would catch a method that finds effects where none exist: it plants a fake
              intervention at days {elem(@result.details.placebo_window, 0) + 1}–{elem(
                @result.details.placebo_window,
                1
              ) + 1}, where nothing happened, and re-fits.
              <:stat
                label="Fake-window effect"
                value={signed(@result.details.placebo.lift)}
                hint={"CI [#{fmt(@result.details.placebo.ci95.lower)}, #{fmt(@result.details.placebo.ci95.upper)}] — #{if @result.details.placebo.is_significant, do: "significant (bad)", else: "not significant"}"}
              />
            </.verdict_card>

            <.verdict_card
              verdict={check_verdict(@result.verdicts.effect_stability, "Effect stability")}
              tone={check_tone(@result.verdicts.effect_stability)}
            >
              Would catch an answer that hinges on exactly where the window was drawn:
              it re-fits at windows of {@result.details.effect_stability.window_low} and {@result.details.effect_stability.window_high} days and compares per-day rates.
              <:stat
                label="Max change"
                value={pct_plain(@result.details.effect_stability.max_pct_change)}
                hint={"rate #{fmt2(@result.details.effect_stability.lift_base)}/day at #{Diagnostics.n_post()} days"}
              />
            </.verdict_card>

            <.verdict_card verdict="What this run can't check" tone={:neutral}>
              Coverage here is judged on {Diagnostics.n_pre()} pre-period points — each
              point is worth about 1%, so the pass band is coarse. And all five checks
              validate the <em>baseline and method</em>, not the causal claim itself:
              only a truth you planted (or a randomized experiment) can grade that.
            </.verdict_card>
          </div>

          <div class="report-prose mt-6 text-sm text-(--color-ink-soft)">
            Two honest footnotes. The pre-period baseline for the first three checks
            comes from a second, identically configured run of the same sampler —
            <code>estimate/4</code>
            keeps its posterior states private and derives its
            own PRNG stream from the seed, so this is an independent chain from the
            same posterior. And the
            placebo and stability checks re-fit the model three more times through
            callbacks — the Validation API deliberately takes functions, not fitted
            models, so any estimator can be plugged in. All four fits are included in
            the badge's elapsed time.
          </div>
        </div>

        <.under_the_hood code={Diagnostics.code_snippet()} />

        <div class="mt-10 grid gap-4 md:grid-cols-3">
          <.cross_link
            navigate={~p"/trust/calibration"}
            eyebrow="The systematic version"
            title="The planted truth, systematically"
          >
            Five planted lifts, five fits, every interval graded against its truth.
          </.cross_link>
          <.cross_link
            navigate={~p"/engine/counterfactual"}
            eyebrow="What's being checked"
            title="The counterfactual"
          >
            The baseline these five checks are interrogating, built up step by step.
          </.cross_link>
          <.cross_link
            navigate={~p"/trust/honesty"}
            eyebrow="The fine print"
            title="What this is and isn't"
          >
            Where these methods come from and what the library doesn't do yet.
          </.cross_link>
        </div>
      </div>
    </Layouts.app>
    """
  end

  # ── figure helpers ──────────────────────────────────────────────────

  defp fig1_caption(nil) do
    "130 days of a generated series. The campaign turns on at day #{Diagnostics.post_start()} and stays on; the measured window is days #{Diagnostics.post_start()}–#{Diagnostics.post_end()}."
  end

  defp fig1_caption(_result) do
    "The fit: posterior-mean baseline over the pre-period, and the counterfactual the model projected for the measured window with its 95% posterior-predictive band."
  end

  defp fit_series(scenario, nil) do
    [%{points: scenario.observations, color: :ink, width: 1.5}]
  end

  defp fit_series(scenario, result) do
    [
      %{points: scenario.observations, color: :ink, width: 1.5},
      %{points: result.baseline_pre, color: :model, dash: true, width: 1.8},
      %{
        points: result.counterfactual_mean,
        x_offset: Diagnostics.post_start() - 1,
        color: :model,
        dash: true,
        width: 2.0
      }
    ]
  end

  defp fit_bands(nil), do: []

  defp fit_bands(result) do
    [
      %{
        lower: result.counterfactual_lower,
        upper: result.counterfactual_upper,
        x_offset: Diagnostics.post_start() - 1,
        fill: "var(--color-model-band)"
      }
    ]
  end

  # ── verdict helpers ─────────────────────────────────────────────────

  defp effect_verdict(%{significant?: true, cumulative: cum}) do
    "The fit says: real effect, #{signed(cum.mean)} units."
  end

  defp effect_verdict(%{significant?: false}) do
    "The fit says: can't tell — the interval straddles zero."
  end

  defp grade_line(result, scenario) do
    truth = scenario.truth.cumulative
    %{lower: lo, upper: hi} = result.cumulative

    if truth >= lo and truth <= hi do
      "The planted truth sits inside the fit's 95% interval — and the five checks below graded the same fit without knowing it."
    else
      "The planted truth fell outside the 95% interval this run — rare by design, and exactly what the calibration page measures across many plants."
    end
  end

  defp check_verdict(:pass, name), do: "Pass — #{name}"
  defp check_verdict(:warn, name), do: "Marginal — #{name}"
  defp check_verdict(:fail, name), do: "Fail — #{name}"
  defp check_verdict(:skip, name), do: "Skipped — #{name}"

  defp check_tone(:pass), do: :truth
  defp check_tone(:warn), do: :warning
  defp check_tone(:fail), do: :warning
  defp check_tone(:skip), do: :neutral

  defp cov_pct(result) do
    :erlang.float_to_binary(result.details.coverage.pct * 100.0, decimals: 1)
  end

  # ── formatting ──────────────────────────────────────────────────────

  defp fmt(v) when is_number(v), do: :erlang.float_to_binary(v * 1.0, decimals: 1)
  defp fmt2(v) when is_number(v), do: :erlang.float_to_binary(v * 1.0, decimals: 2)

  defp signed(v) when is_number(v) do
    prefix = if v >= 0, do: "+", else: ""
    prefix <> :erlang.float_to_binary(v * 1.0, decimals: 1)
  end

  defp pct(v) when is_number(v) do
    prefix = if v >= 0, do: "+", else: ""
    prefix <> :erlang.float_to_binary(v * 100.0, decimals: 1) <> "%"
  end

  defp pct_plain(v) when is_number(v) do
    :erlang.float_to_binary(v * 100.0, decimals: 1) <> "%"
  end
end
