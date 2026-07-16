defmodule BstsSiteWeb.SpeedLive do
  @moduledoc """
  The performance argument, made honestly: race the millisecond lane against
  the MCMC lane on identical planted-lift data, timed live on this server.
  """
  use BstsSiteWeb, :live_view

  alias BstsSite.Demos.Speed

  @impl true
  def mount(_params, _session, socket) do
    demo = Speed.prepare()

    {:ok,
     socket
     |> assign(page_title: "The two lanes")
     |> assign(demo: demo, race: nil, racing: false, busy: false, revealed: false)}
  end

  @impl true
  def handle_event("race", _params, %{assigns: %{racing: true}} = socket) do
    {:noreply, socket}
  end

  def handle_event("race", _params, socket) do
    socket = assign(socket, racing: true, busy: false)

    {:noreply,
     start_async(socket, :race, fn ->
       case BstsSite.Demos.Limiter.run(fn -> Speed.race() end) do
         {:ok, result} -> result
         :busy -> :busy
       end
     end)}
  end

  def handle_event("reveal_truth", _params, socket) do
    {:noreply, assign(socket, revealed: true)}
  end

  @impl true
  def handle_async(:race, {:ok, :busy}, socket) do
    {:noreply, assign(socket, racing: false, busy: true)}
  end

  def handle_async(:race, {:ok, result}, socket) do
    {:noreply, assign(socket, racing: false, busy: false, race: result)}
  end

  def handle_async(:race, {:exit, _reason}, socket) do
    {:noreply, assign(socket, racing: false, busy: true)}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <Layouts.app flash={@flash} track={:speed}>
      <div class="mx-auto max-w-4xl">
        <.section_heading level={1} eyebrow="The performance argument" title="The two lanes">
          Everything on this site runs in one of two lanes. The <strong>milliseconds lane</strong>
          is Kalman filtering with fixed variances; fast enough to run per event, per request, per slider drag. The
          <strong>MCMC lane</strong>
          is Gibbs sampling; it learns the variances instead
          of assuming them and hands you a full posterior, at a price measured in seconds.
          This page makes the trade concrete: same data, three estimators, one wall clock.
        </.section_heading>

        <div class="mt-6 grid gap-4 sm:grid-cols-2">
          <div class="rounded-md border border-(--color-grid) bg-(--color-paper-raised) p-5">
            <div class="font-data text-[0.65rem] uppercase tracking-widest text-(--color-ink-faint)">
              Lane one · milliseconds
            </div>
            <div class="font-display mt-1 text-lg font-bold">Per-event safe</div>
            <ul class="report-prose mt-2 list-disc space-y-1 pl-4 text-sm text-(--color-ink-soft)">
              <li>
                <code>Operational.prepare</code> + <code>run</code>; the forecast-first
                counterfactual behind the hub, the TV demo, and the marketing fast pass
              </li>
              <li>
                <code>CausalImpact.estimate_from_filter</code>; one compiled filter +
                smoother pass over the whole series
              </li>
              <li>
                <code>AnomalyDetector</code> fit and score; a filter pass per window
              </li>
            </ul>
            <p class="report-prose mt-2 text-sm text-(--color-ink-soft)">
              Variances are fixed up front, so a run is a handful of matrix recursions.
              Safe inside a request path.
            </p>
          </div>
          <div class="rounded-md border border-(--color-grid) bg-(--color-paper-raised) p-5">
            <div class="font-data text-[0.65rem] uppercase tracking-widest text-(--color-ink-faint)">
              Lane two · seconds and up
            </div>
            <div class="font-display mt-1 text-lg font-bold">The full posterior</div>
            <ul class="report-prose mt-2 list-disc space-y-1 pl-4 text-sm text-(--color-ink-soft)">
              <li>
                Scalar <code>CausalImpact.estimate</code>
                and <code>GibbsSampler.sample</code>; under a second at the {Speed.mcmc_samples()} samples raced here, seconds at
                the hundreds you'd use for real work
              </li>
              <li>
                Structured or seasonal <code>ModelSpec</code>s through MCMC; <em>minutes</em>
                on CPU. Bluntly: precompute or background-job territory, never a request path
              </li>
            </ul>
            <p class="report-prose mt-2 text-sm text-(--color-ink-soft)">
              What the time buys: the sampler learns the variances from the data and reports
              parameter uncertainty the fast lane simply assumes away.
            </p>
          </div>
        </div>

        <p class="report-prose mt-6 text-sm text-(--color-ink-soft)">
          The racetrack: {Speed.total()} days of orders around a level of 100, with a lift
          planted at day {Speed.pre_end() + 1} and left on. The figure below is the
          milliseconds lane's opening move, computed on page load; a counterfactual
          projected from the pre-period alone.
        </p>

        <.figure
          no="1"
          caption={"The #{Speed.total()}-day series and the Operational forecast-first counterfactual; projected from days 1–#{Speed.pre_end()} without seeing a single post-period observation."}
          execution={@demo.execution}
        >
          <:legend>
            <.swatch color={:ink} label="observed orders" />
            <.swatch color={:model} label="counterfactual" dash />
          </:legend>
          <.line_figure
            id="speed-scenario"
            title="Speed benchmark scenario"
            desc="Observed order volume is compared with the Operational forecast-first counterfactual over the post-period."
            height={300}
            y_label="orders/day"
            series={[
              %{points: @demo.scenario.observations, color: :ink, width: 1.5},
              %{
                points: @demo.counterfactual_mean,
                x_offset: Speed.pre_end(),
                color: :model,
                dash: true,
                width: 2.0
              }
            ]}
            bands={[
              %{
                lower: @demo.band_lower,
                upper: @demo.band_upper,
                x_offset: Speed.pre_end(),
                fill: "var(--color-model-band)"
              }
            ]}
            vlines={[%{x: Speed.pre_end(), label: "lift switches on"}]}
          />
        </.figure>

        <div class="mt-6 flex flex-wrap items-center gap-3">
          <button
            id="race-button"
            type="button"
            phx-click="race"
            disabled={@racing}
            class="btn btn-outline font-data text-sm"
          >
            {race_button_label(@racing, @race)}
          </button>
          <span class="font-data text-xs text-(--color-ink-faint)">
            times all three estimators on this server, right now
          </span>
        </div>

        <.verdict_card
          :if={@busy}
          role="status"
          aria-live="polite"
          verdict="Another visitor is sampling, try again in a moment."
          tone={:warning}
          class="mt-4"
        >
          The MCMC lane is real work, so we cap how many races run at once.
        </.verdict_card>

        <%= if @race do %>
          <.figure
            no="2"
            caption={"Wall-clock time for each estimator on the identical #{Speed.total()}-point series, measured with :timer.tc during this race. The fast lanes are the barely-visible bars; that is the argument."}
          >
            <:legend>
              <.swatch color={:ink} label="measured on this server" />
            </:legend>
            <.bar_figure
              id="race-latency"
              title="Estimator latency race"
              desc="Wall-clock runtime for each estimator lane on the same series, measured on this server in milliseconds."
              unit=" ms"
              height={180}
              items={
                for lane <- @race.lanes do
                  %{label: lane.label, value: lane.ms, color: :ink}
                end
              }
            />
          </.figure>

          <.figure
            no="3"
            caption={"The answers, side by side: each lane's cumulative-effect estimate over the #{@demo.scenario.truth.post_days}-day post period, with 95% intervals#{if @revealed, do: ", and the planted truth in green"}."}
          >
            <:legend>
              <.swatch color={:lift} label="estimated cumulative effect" />
              <.swatch :if={@revealed} color={:truth} label="planted truth" />
            </:legend>
            <.bar_figure
              id="race-answers"
              title="Estimator answer comparison"
              desc="Each estimator lane's cumulative-effect estimate is shown with interval bounds and the planted answer when revealed."
              unit=" orders"
              height={200}
              items={answer_items(@race, @demo, @revealed)}
            />
          </.figure>

          <p class="report-prose text-sm text-(--color-ink-soft)">
            The honest footnotes. <code>estimate_from_filter</code> masks the effect window
            and smooths; its baseline may use observations from <em>after</em> the window,
            which makes it an interpolation, not a forecast. In this race the window runs to
            the end of the series, so there was nothing later to peek at; that's why its
            mean lands within {fmt1(fast_lane_gap(@race))} orders of the forecast-first lane.
            Point the window at the middle of a series and the two part ways. <code>Operational</code>'s forecast-first baseline never sees post-period data,
            by construction. And the MCMC lane isn't just a slower way to the same number:
            its interval carries parameter uncertainty; the sampler learned the variances
            the fast lanes were handed for free.
          </p>

          <div class="mt-4 grid gap-4 sm:grid-cols-[1fr_minmax(240px,0.8fr)]">
            <.verdict_card verdict={verdict_line(@race)} tone={:lift}>
              <:stat
                label="Fastest lane"
                value={"#{fmt_ms(fastest(@race).ms)} ms"}
                hint={"#{fastest(@race).label}"}
              />
              <:stat label="MCMC lane" value={"#{fmt_ms(mcmc_lane(@race).ms)} ms"} />
              <:stat
                label="Speedup"
                value={"#{fmt(@race.speedup)}×"}
                hint="operational vs MCMC"
              />
              <:stat
                label="Spread of the three means"
                value={"#{fmt(@race.mean_spread)} orders"}
              />
            </.verdict_card>

            <.reveal_truth
              revealed={@revealed}
              truth={"Planted: +#{fmt(@demo.scenario.truth.lift_per_day)} orders/day for #{@demo.scenario.truth.post_days} days; +#{fmt(@demo.scenario.truth.cumulative_lift)} cumulative."}
              grade={grade_line(@race, @demo.scenario.truth)}
            />
          </div>
        <% end %>

        <p class="report-prose mt-6 text-sm text-(--color-ink-soft)">
          A note on the machine: this server runs Nx's <code>{backend_name(@demo.execution)}</code>
          ; pure Elixir, no native acceleration.
          In your own deployment, EXLA can JIT-compile the <code>defn</code>
          paths the fast
          lanes are built on. Every number on this page is measured live when you click, so
          it is machine-dependent by nature; that's a feature, not a caveat.
        </p>

        <.under_the_hood code={Speed.code_snippet()} />

        <div class="mt-8 grid gap-4 md:grid-cols-3">
          <.cross_link
            navigate={~p"/engine/kalman"}
            eyebrow="The fast lane's engine"
            title="The Kalman filter chapter"
          >
            Both millisecond lanes are this one algorithm: a handful of predict-update
            recursions per observation.
          </.cross_link>
          <.cross_link
            navigate={~p"/engine/counterfactual"}
            eyebrow="What the slow lane buys"
            title="The world that didn't happen"
          >
            Watch what {Speed.mcmc_samples()} posterior draws actually look like; the
            spaghetti the fast lanes summarize away.
          </.cross_link>
          <.cross_link
            navigate={~p"/demos/anomaly"}
            eyebrow="The fast lane at work"
            title="Anomaly detection, per event"
          >
            A production-shaped use of the milliseconds lane: fit and score on every
            interaction, no queue, no spinner.
          </.cross_link>
        </div>
      </div>
    </Layouts.app>
    """
  end

  # -- Race presentation -------------------------------------------------------

  defp answer_items(race, demo, revealed) do
    lanes =
      for lane <- race.lanes do
        %{
          label: lane.label,
          value: lane.cumulative.mean,
          lower: lane.cumulative.lower,
          upper: lane.cumulative.upper,
          color: :lift
        }
      end

    if revealed do
      lanes ++
        [%{label: "planted truth", value: demo.scenario.truth.cumulative_lift, color: :truth}]
    else
      lanes
    end
  end

  defp fastest(race), do: Enum.min_by(race.lanes, & &1.ms)

  defp mcmc_lane(race), do: Enum.find(race.lanes, &(&1.id == :mcmc))

  defp fast_lane_gap(race) do
    op = Enum.find(race.lanes, &(&1.id == :operational))
    filt = Enum.find(race.lanes, &(&1.id == :filter))
    abs(op.cumulative.mean - filt.cumulative.mean) |> max(0.1)
  end

  defp verdict_line(race) do
    "Three lanes, roughly one answer; means within #{fmt(race.mean_spread)} orders of each other."
  end

  defp grade_line(race, truth) do
    %{n: n, of: of_n} = BstsSite.Demos.Speed.coverage(race, truth.cumulative_lift)

    case n do
      ^of_n ->
        "The planted truth sits inside all #{of_n} intervals. The lanes disagree on cost, not on the answer."

      0 ->
        "The planted truth fell outside every interval this race; rare, and worth a visit to the calibration page."

      _ ->
        "The planted truth sits inside #{n} of #{of_n} intervals. Interval widths differ by lane; the MCMC interval also carries parameter uncertainty."
    end
  end

  defp race_button_label(true, _race), do: "Racing…"
  defp race_button_label(false, nil), do: "Race the lanes"
  defp race_button_label(false, _race), do: "Race again"

  # "{Nx.BinaryBackend, []}" -> "Nx.BinaryBackend"
  defp backend_name(%{backend: backend}) when is_binary(backend) do
    case Regex.run(~r/[A-Za-z0-9_.]*Backend/, backend) do
      [name] -> name
      _ -> backend
    end
  end

  defp backend_name(_execution), do: "Nx.BinaryBackend"

  # -- Formatting --------------------------------------------------------------

  defp fmt(v) when is_number(v), do: :erlang.float_to_binary(v * 1.0, decimals: 0)

  defp fmt1(v) when is_number(v), do: :erlang.float_to_binary(v * 1.0, decimals: 1)

  defp fmt_ms(v) when is_number(v) do
    if v < 100 do
      :erlang.float_to_binary(v * 1.0, decimals: 1)
    else
      :erlang.float_to_binary(v * 1.0, decimals: 0)
    end
  end
end
