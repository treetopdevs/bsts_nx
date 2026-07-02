defmodule BstsSiteWeb.Trust.CalibrationLive do
  @moduledoc """
  Trust: the planted truth, systematically. Five copies of one series, five
  sealed planted lifts (one of them zero), five MCMC fits — then the answer
  key comes out and every interval gets graded in public.
  """
  use BstsSiteWeb, :live_view

  alias BstsSite.Demos.Calibration
  alias BstsSite.Demos.Limiter

  @impl true
  def mount(_params, _session, socket) do
    {:ok,
     socket
     |> assign(page_title: "The planted truth, systematically")
     |> assign(series: Calibration.series(), sweep: nil, running: false, busy: false)
     |> assign(revealed: false)}
  end

  @impl true
  def handle_event("run_sweep", _params, %{assigns: %{running: true}} = socket) do
    {:noreply, socket}
  end

  def handle_event("run_sweep", _params, socket) do
    series = socket.assigns.series
    socket = assign(socket, running: true, busy: false)

    {:noreply,
     start_async(socket, :sweep, fn ->
       case Limiter.run(fn -> Calibration.run_sweep(series) end) do
         {:ok, result} -> result
         :busy -> :busy
       end
     end)}
  end

  def handle_event("reveal_truth", _params, socket) do
    {:noreply, assign(socket, revealed: true)}
  end

  @impl true
  def handle_async(:sweep, {:ok, :busy}, socket) do
    {:noreply, assign(socket, running: false, busy: true)}
  end

  def handle_async(:sweep, {:ok, sweep}, socket) do
    {:noreply, assign(socket, running: false, busy: false, sweep: sweep)}
  end

  def handle_async(:sweep, {:exit, _reason}, socket) do
    {:noreply, assign(socket, running: false, busy: true)}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <Layouts.app flash={@flash} track={:trust}>
      <div class="mx-auto max-w-4xl">
        <.mantra active={:compare} class="mb-4" />
        <.section_heading
          eyebrow="Trust · does it recover what we hid?"
          title="The planted truth, systematically"
        >
          One planted truth recovered could be luck. So here are five: the same 90-point
          series, five times over, each with a different lift sealed into the last 20
          points — and one of the five is zero, a placebo hiding among the real effects.
          The model fits each series knowing nothing about what we planted, and then we
          grade all five intervals at once.
        </.section_heading>

        <.figure
          no="1"
          caption={"The five case series overlaid — identical seeded noise (sd #{@series.noise_sd}), different planted per-day lifts from day 71 onward, ranging up to +#{@series.max_per_step}/day. Per day, even the largest plant barely clears the noise."}
        >
          <:legend>
            <.swatch color={:ink} label="observed (5 cases)" />
            <.swatch :if={@revealed} color={:truth} label="planted means" dash />
          </:legend>
          <.line_figure
            id="calibration-data"
            height={280}
            y_label="level"
            series={data_series(@series, @revealed)}
            vlines={[%{x: Calibration.pre_end(), label: "intervention"}]}
          />
        </.figure>

        <div class="flex flex-wrap items-center gap-4">
          <button
            type="button"
            phx-click="run_sweep"
            disabled={@running}
            class="btn btn-primary font-data"
          >
            {if @sweep, do: "Run the sweep again", else: "Run the sweep"}
          </button>
          <p :if={@running} class="font-data text-sm text-(--color-ink-soft)" aria-live="polite">
            Fitting five models, one after another, on this server…
          </p>
        </div>

        <.verdict_card
          :if={@busy}
          class="mt-6"
          verdict="Another visitor is sampling — try again in a moment."
          tone={:warning}
        >
          This page runs real MCMC on a small shared machine, so concurrent sweeps are
          limited. Nothing is cached; every run is live.
        </.verdict_card>

        <div :if={@sweep} class="mt-2">
          <.figure
            no="2"
            caption="Estimated cumulative effect per case with its 95% credible interval. Each fit saw only its own series — never the planted value."
            execution={@sweep.execution}
          >
            <:legend>
              <.swatch color={:lift} label="estimated effect" />
              <.swatch :if={@revealed} color={:truth} label="planted truth" />
            </:legend>
            <.bar_figure
              id="calibration-sweep"
              items={sweep_items(@sweep, @revealed)}
              height={if @revealed, do: 400, else: 230}
            />
          </.figure>

          <div class="grid gap-4 sm:grid-cols-[1fr_minmax(240px,0.9fr)]">
            <.verdict_card
              verdict={verdict_line(@sweep, @revealed)}
              tone={verdict_tone(@sweep, @revealed)}
            >
              <p :if={!@revealed}>
                Five intervals, one sealed answer key. Before you look: the estimates
                below are all the model gets to say.
              </p>
              <:stat
                :for={{c, i} <- Enum.with_index(@sweep.cases)}
                label={"case #{case_label(i)}"}
                value={signed(c.mean)}
                hint={"CI [#{fmt(c.lower)}, #{fmt(c.upper)}]"}
              />
            </.verdict_card>

            <.reveal_truth
              revealed={@revealed}
              truth="Planted, in order: +0 (placebo), +5, +10, +20, +40 cumulative — never shown to the model."
              grade={grade_line(@sweep)}
            />
          </div>

          <div class="report-prose mt-6 text-sm text-(--color-ink-soft)">
            A 95% interval is a promise about the long run: over many studies, it should
            contain the truth about 19 times in 20 — which also means that roughly one
            miss in twenty is the promise <em>working</em>, not breaking. Five fits are a
            spot-check of that promise, not its proof. The library ships the systematic
            version as <code>BstsNx.Validation.known_lift_injection/3</code> — the same
            plant-and-grade loop, run offline with structured models and hundreds of
            samples per fit.
          </div>
        </div>

        <.under_the_hood code={Calibration.code_snippet()} />

        <div class="mt-10 grid gap-4 md:grid-cols-3">
          <.cross_link
            navigate={~p"/engine/gibbs"}
            eyebrow="The engine behind it"
            title="The Gibbs sampler"
          >
            The MCMC machinery this sweep stresses five times over — watch a chain converge.
          </.cross_link>
          <.cross_link
            navigate={~p"/trust/diagnostics"}
            eyebrow="More ways to check"
            title="Break your own results"
          >
            Coverage, placebo, and stability checks on a single fit.
          </.cross_link>
          <.cross_link navigate={~p"/speed"} eyebrow="Why it takes seconds" title="The two lanes">
            This page uses the honest slow lane. There is a millisecond lane too.
          </.cross_link>
        </div>
      </div>
    </Layouts.app>
    """
  end

  # ── figure helpers ──────────────────────────────────────────────────

  defp data_series(series, revealed) do
    pre_end = Calibration.pre_end()
    pre = %{points: Enum.take(series.base, pre_end), color: :ink, width: 1.5}

    posts =
      series.cases
      |> Enum.with_index()
      |> Enum.map(fn {c, i} ->
        %{
          points: Enum.slice(c.observations, (pre_end - 1)..-1//1),
          x_offset: pre_end - 1,
          color: :ink,
          width: 1.2,
          opacity: 0.3 + 0.175 * i
        }
      end)

    truths =
      if revealed do
        Enum.map(series.cases, fn c ->
          %{
            points: List.duplicate(c.truth_mean, Calibration.n_post()),
            x_offset: pre_end,
            color: :truth,
            width: 1.3,
            dash: true
          }
        end)
      else
        []
      end

    [pre | posts] ++ truths
  end

  defp sweep_items(sweep, false) do
    sweep.cases
    |> Enum.with_index()
    |> Enum.map(fn {c, i} ->
      %{
        label: "case #{case_label(i)}",
        value: c.mean,
        lower: c.lower,
        upper: c.upper,
        color: :lift
      }
    end)
  end

  defp sweep_items(sweep, true) do
    sweep.cases
    |> Enum.with_index()
    |> Enum.flat_map(fn {c, i} ->
      [
        %{
          label: "#{case_label(i)} · estimate",
          value: c.mean,
          lower: c.lower,
          upper: c.upper,
          color: :lift
        },
        %{label: "#{case_label(i)} · truth +#{c.lift}", value: c.truth, color: :truth}
      ]
    end)
  end

  defp case_label(i), do: Enum.at(~w(A B C D E), i)

  # ── verdict helpers ─────────────────────────────────────────────────

  defp verdict_line(_sweep, false), do: "Five estimates, sealed answer key."

  defp verdict_line(sweep, true) do
    "#{sweep.covered_count} of 5 planted truths inside their 95% intervals."
  end

  defp verdict_tone(_sweep, false), do: :neutral
  defp verdict_tone(%{covered_count: 5}, true), do: :truth
  defp verdict_tone(_sweep, true), do: :warning

  defp grade_line(sweep) do
    placebo =
      if sweep.placebo_straddles? do
        "The placebo case straddles zero — the model declined to invent an effect where none was planted."
      else
        "The placebo case did not straddle zero this run — a false alarm worth noticing."
      end

    "#{sweep.covered_count} of 5 landed inside their intervals. #{placebo}"
  end

  # ── formatting ──────────────────────────────────────────────────────

  defp fmt(v) when is_number(v), do: :erlang.float_to_binary(v * 1.0, decimals: 1)

  defp signed(v) when is_number(v) do
    prefix = if v >= 0, do: "+", else: ""
    prefix <> :erlang.float_to_binary(v * 1.0, decimals: 1)
  end
end
