defmodule BstsSiteWeb.Engine.GibbsLive do
  @moduledoc """
  Engine chapter: the Gibbs sampler, watched live.

  The MCMC runs for real in a `start_async` (guarded by the Limiter), then
  the finished chains are replayed a few iterations per tick so visitors see
  the trace grow and R-hat settle; exactly as the draws happened, at a pace
  a human can follow. The caption says so.
  """
  use BstsSiteWeb, :live_view

  alias BstsSite.Demos.Gibbs

  @reveal_per_tick 3
  @tick_ms 100

  @impl true
  def mount(_params, _session, socket) do
    {:ok,
     socket
     |> assign(page_title: "Honest about the unknown")
     |> assign(scenario: Gibbs.scenario())
     |> assign(run: nil, running: false, busy: false, replaying: false)
     |> assign(shown: 0, r_hat_shown: :nan, revealed: false)}
  end

  @impl true
  def handle_event("run", _params, %{assigns: %{running: true}} = socket),
    do: {:noreply, socket}

  def handle_event("run", _params, %{assigns: %{replaying: true}} = socket),
    do: {:noreply, socket}

  def handle_event("run", _params, socket) do
    observations = socket.assigns.scenario.observations

    socket =
      assign(socket,
        running: true,
        busy: false,
        run: nil,
        shown: 0,
        r_hat_shown: :nan,
        revealed: false
      )

    {:noreply,
     start_async(socket, :sample, fn ->
       case BstsSite.Demos.Limiter.run(fn -> Gibbs.sample(observations) end) do
         {:ok, result} -> result
         :busy -> :busy
       end
     end)}
  end

  def handle_event("reveal_truth", _params, socket) do
    {:noreply, assign(socket, revealed: true)}
  end

  @impl true
  def handle_async(:sample, {:ok, :busy}, socket) do
    {:noreply, assign(socket, running: false, busy: true)}
  end

  def handle_async(:sample, {:ok, run}, socket) do
    Process.send_after(self(), :tick, @tick_ms)
    {:noreply, assign(socket, running: false, run: run, replaying: true, shown: 0)}
  end

  def handle_async(:sample, {:exit, _reason}, socket) do
    {:noreply, assign(socket, running: false, busy: true)}
  end

  @impl true
  def handle_info(:tick, %{assigns: %{replaying: true, run: %{} = run}} = socket) do
    shown = min(socket.assigns.shown + @reveal_per_tick, run.total)
    r_hat = Gibbs.r_hat_prefix(run.obs_var_traces, shown)

    socket = assign(socket, shown: shown, r_hat_shown: r_hat)

    if shown < run.total do
      Process.send_after(self(), :tick, @tick_ms)
      {:noreply, socket}
    else
      {:noreply, assign(socket, replaying: false)}
    end
  end

  def handle_info(:tick, socket), do: {:noreply, socket}

  @impl true
  def render(assigns) do
    ~H"""
    <Layouts.app flash={@flash} track={:engine}>
      <div class="mx-auto max-w-4xl">
        <.section_heading level={1} eyebrow="Engine · chapter 4" title="Honest about the unknown">
          The filter and smoother chapters assumed we knew how noisy the world is.
          Real data doesn't come with its variances printed on the label; how much
          the level truly drifts, and how much is just measurement noise, are exactly
          the things you don't know. This chapter runs the machinery that learns them:
          a Gibbs sampler, alternating between two easy questions because it can't
          answer the hard one directly. And you get to watch it think.
        </.section_heading>
        <.mantra active={:compare} class="mb-6" />

        <.figure
          no="1"
          caption={"#{@scenario.n} simulated observations: a level that drifts as a random walk, seen through observation noise; the sampler is told neither variance."}
        >
          <:legend>
            <.swatch color={:ink} label="observed" />
          </:legend>
          <.line_figure
            id="gibbs-data"
            title="Gibbs sampler observations"
            desc="Simulated observations from a drifting level are shown before the sampler learns the hidden variances."
            height={260}
            y_label="observed value"
            series={[%{points: @scenario.observations, color: :ink, width: 1.5}]}
          />
        </.figure>

        <div class="report-prose text-sm text-(--color-ink-soft)">
          <p>
            The sampler can't estimate the level path and the variances at once, so it
            alternates. <strong>Draw states given variances:</strong>
            hold both variances
            fixed and draw one plausible level path through the data; the simulation
            smoother from the previous chapter, hindsight with luck thrown in.
            <strong>Draw variances given states:</strong>
            hold that path fixed, and each
            variance becomes a textbook update; draw those too. Repeat. The draws never
            settle on one answer; they wander around it, and after enough round trips
            that wandering <em>is</em>
            the posterior distribution.
          </p>
          <p class="mt-3">
            One chain could be fooling itself, so we run {Gibbs.num_chains()} from
            deliberately wrong starting guesses. R-hat compares the spread <em>between</em>
            chains to the spread <em>within</em>
            each chain: while the
            chains disagree about where they live it sits above 1, and when they become
            indistinguishable it settles to 1.0. Convention says distrust anything above
            1.05. A healthy trace looks like a hairy caterpillar; rapid wiggle around a
            stable center, no drift, no chain off on its own.
          </p>
        </div>

        <div class="mt-6 flex flex-wrap items-center gap-4">
          <button
            type="button"
            phx-click="run"
            disabled={@running or @replaying}
            class="btn btn-primary font-data"
          >
            {if @run || @running, do: "Run it again; same seed, same chains", else: "Watch it think"}
          </button>
          <span
            :if={@running}
            role="status"
            aria-live="polite"
            aria-busy="true"
            class="font-data text-xs text-(--color-ink-soft) motion-safe:animate-pulse"
          >
            sampling: {Gibbs.num_chains()} chains x {Gibbs.burn_in() + Gibbs.num_samples()} iterations each…
          </span>
        </div>

        <.verdict_card
          :if={@busy}
          class="mt-4"
          role="status"
          aria-live="polite"
          tone={:warning}
          verdict="Another visitor is sampling, try again in a moment."
        >
          This demo runs real MCMC on a small shared server, so concurrent runs are
          capped. The button will work again in a few seconds.
        </.verdict_card>

        <section :if={@run}>
          <div class="mt-6 flex flex-wrap items-center gap-3 font-data text-xs text-(--color-ink-soft)">
            <span class="tabular-nums">draw {@shown} of {@run.total} per chain</span>
            <span class={[
              "rounded-full border px-2.5 py-0.5 tabular-nums",
              rhat_chip_class(@r_hat_shown)
            ]}>
              R-hat {fmt_rhat(@r_hat_shown)}
            </span>
            <span>{rhat_note(@r_hat_shown, @shown, @run.total)}</span>
          </div>

          <.figure
            no="2"
            caption={"The observation-variance draw per iteration, one line per chain; replaying the #{Gibbs.num_chains()} chains exactly as they were drawn; total compute: #{@run.elapsed_ms} ms on this server."}
            execution={@run.execution}
          >
            <:legend>
              <.swatch
                color={:model}
                label={"σ²_obs draws; one line per chain (#{Gibbs.num_chains()} shades)"}
              />
            </:legend>
            <.line_figure
              id="gibbs-trace"
              title="Gibbs observation-variance trace"
              desc="Retained observation-variance draws are plotted by iteration for each chain."
              height={280}
              y_label="sampled observation variance σ²_obs"
              x_domain={{0, @run.total - 1}}
              y_domain={@run.trace_domain}
              series={trace_series(@run, @shown)}
            />
          </.figure>

          <div :if={!@replaying} class="report-prose text-sm text-(--color-ink-soft)">
            The {Gibbs.burn_in()} burn-in iterations per chain; the walk in from the
            wrong starting guesses; were discarded before you saw anything, as is
            standard. What remains is the sampler's actual answer: not two numbers,
            but two piles of draws.
          </div>

          <div :if={!@replaying}>
            <.figure
              no="3"
              caption={"All #{@run.obs_var_summary.n} retained draws of the observation variance σ²_obs, pooled across the #{Gibbs.num_chains()} chains."}
              execution={@run.execution}
            >
              <:legend>
                <.swatch color={:model} label="posterior draws" />
                <.swatch :if={@revealed} color={:truth} label="planted truth" dash />
              </:legend>
              <.histogram
                id="gibbs-hist-obs"
                title="Observation-variance posterior"
                desc="Pooled posterior draws of observation variance are shown with planted-truth markers when revealed."
                values={List.flatten(@run.obs_var_traces)}
                bins={24}
                vlines={truth_vlines(@revealed, @scenario.truth.obs_var)}
              />
            </.figure>

            <.figure
              no="4"
              caption={"All #{@run.process_var_summary.n} retained draws of the level-drift variance σ²_level, pooled across the #{Gibbs.num_chains()} chains."}
              execution={@run.execution}
            >
              <:legend>
                <.swatch color={:model} label="posterior draws" />
                <.swatch :if={@revealed} color={:truth} label="planted truth" dash />
              </:legend>
              <.histogram
                id="gibbs-hist-process"
                title="Level-drift variance posterior"
                desc="Pooled posterior draws of level-drift variance are shown with planted-truth markers when revealed."
                values={List.flatten(@run.process_var_traces)}
                bins={24}
                vlines={truth_vlines(@revealed, @scenario.truth.process_var)}
              />
            </.figure>

            <div class="grid gap-4 sm:grid-cols-[1fr_minmax(240px,0.8fr)]">
              <.verdict_card
                verdict={verdict_line(@r_hat_shown)}
                tone={if converged?(@r_hat_shown), do: :truth, else: :warning}
              >
                The width of those intervals is the honesty: {@scenario.n} points
                genuinely cannot pin the level drift much tighter, and the sampler
                says so instead of pretending.
                <:stat
                  label="σ²_obs posterior"
                  value={fmt2(@run.obs_var_summary.mean)}
                  hint={"95% [#{fmt2(@run.obs_var_summary.lower)}, #{fmt2(@run.obs_var_summary.upper)}]"}
                />
                <:stat
                  label="σ²_level posterior"
                  value={fmt2(@run.process_var_summary.mean)}
                  hint={"95% [#{fmt2(@run.process_var_summary.lower)}, #{fmt2(@run.process_var_summary.upper)}]"}
                />
                <:stat label="R-hat (σ²_obs)" value={fmt_rhat(@r_hat_shown)} />
                <:stat
                  label="Effective samples"
                  value={fmt_ess(@run.ess_obs_var)}
                  hint={"of #{@run.obs_var_summary.n} draws"}
                />
              </.verdict_card>

              <.reveal_truth
                revealed={@revealed}
                truth={"Planted at generation: observation variance #{fmt1(@scenario.truth.obs_var)}, level-drift variance #{fmt1(@scenario.truth.process_var)}. The sampler never saw either."}
                grade={grade_line(@run, @scenario.truth)}
              />
            </div>
          </div>
        </section>

        <.under_the_hood code={Gibbs.code_snippet()} />

        <div class="mt-10 grid gap-4 md:grid-cols-3">
          <.cross_link
            navigate={~p"/engine/smoother"}
            eyebrow="The state-draw step"
            title="The power of hindsight"
          >
            Each iteration begins by drawing a plausible level path through the data; that's the simulation smoother from the previous chapter.
          </.cross_link>
          <.cross_link
            navigate={~p"/trust/diagnostics"}
            eyebrow="When to distrust a result"
            title="Break your own results"
          >
            R-hat and ESS caught this chain converging; the diagnostics page attacks the
            finished fit five more ways; prediction-error, coverage, residual, placebo,
            and stability checks.
          </.cross_link>
          <.cross_link navigate={~p"/speed"} eyebrow="What sampling costs" title="The speed ledger">
            The run you just watched reports its own wall-clock time; the speed page
            measures every path in the library the same way.
          </.cross_link>
        </div>
      </div>
    </Layouts.app>
    """
  end

  # ── view helpers ────────────────────────────────────────────────────

  defp trace_series(run, shown) do
    n = length(run.obs_var_traces)

    run.obs_var_traces
    |> Enum.with_index()
    |> Enum.map(fn {trace, i} ->
      # Shades from 0.4 up to 1.0, evenly spaced, however many chains there are.
      opacity = 0.4 + 0.6 * i / max(n - 1, 1)
      %{points: Enum.take(trace, shown), color: :model, width: 1.6, opacity: opacity}
    end)
  end

  defp truth_vlines(false, _truth), do: []
  defp truth_vlines(true, truth), do: [%{x: truth, label: "planted", color: :truth}]

  defp converged?(r_hat), do: is_number(r_hat) and r_hat < 1.05

  defp rhat_chip_class(:nan), do: "border-(--color-ink-faint) text-(--color-ink-faint)"

  defp rhat_chip_class(r_hat) do
    if converged?(r_hat),
      do: "border-(--color-truth) text-(--color-truth)",
      else: "border-(--color-warning) text-(--color-warning)"
  end

  defp rhat_note(:nan, _shown, _total), do: "warming up; too few draws to compare chains"

  defp rhat_note(r_hat, shown, total) do
    cond do
      converged?(r_hat) and shown >= total -> "below 1.05; the chains agree"
      converged?(r_hat) -> "below 1.05; converging"
      true -> "above 1.05; the chains still disagree"
    end
  end

  defp verdict_line(r_hat) do
    if converged?(r_hat) do
      "The chains agree; and the answer is a distribution, not a number."
    else
      "The chains haven't fully agreed; treat these intervals with suspicion."
    end
  end

  defp grade_line(run, truth) do
    obs_in = Gibbs.recovered?(run.obs_var_summary, truth.obs_var)
    process_in = Gibbs.recovered?(run.process_var_summary, truth.process_var)

    case {obs_in, process_in} do
      {true, true} ->
        "Both planted variances sit inside the sampler's 95% intervals; recovered, " <>
          "with the level-drift variance the harder of the two."

      {true, false} ->
        "The observation variance was recovered; the planted level-drift variance " <>
          "fell outside its 95% interval this run; separating drift from noise in " <>
          "#{length(hd(run.obs_var_traces))} draws per chain is genuinely hard."

      {false, true} ->
        "The level-drift variance was recovered; the planted observation variance " <>
          "fell outside its 95% interval this run. Intervals are promises about the " <>
          "long run, not every draw."

      {false, false} ->
        "Neither planted variance landed inside its 95% interval this run; that " <>
          "should be rare. See the calibration page for how often it happens."
    end
  end

  # ── formatting ──────────────────────────────────────────────────────

  defp fmt_rhat(:nan), do: "n/a"
  defp fmt_rhat(v) when is_number(v), do: :erlang.float_to_binary(v * 1.0, decimals: 3)

  defp fmt_ess(:nan), do: "n/a"
  defp fmt_ess(v), do: Integer.to_string(v)

  defp fmt1(v) when is_number(v), do: :erlang.float_to_binary(v * 1.0, decimals: 1)
  defp fmt2(v) when is_number(v), do: :erlang.float_to_binary(v * 1.0, decimals: 2)
end
