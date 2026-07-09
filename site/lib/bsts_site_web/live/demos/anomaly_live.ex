defmodule BstsSiteWeb.Demos.AnomalyLive do
  @moduledoc """
  The ops-monitoring demo: plant latency spikes in a 36-hour monitoring
  window and watch `BstsNx.Applications.AnomalyDetector` grade them
  against a streaming Kalman baseline; refit and rescored inline on
  every interaction.
  """
  use BstsSiteWeb, :live_view

  alias BstsSite.Demos.Anomaly
  alias BstsSite.Demos.DefaultCache

  @impl true
  def mount(_params, _session, socket) do
    %{anomalies: anomalies, alpha: alpha, demo: demo} =
      DefaultCache.get(:anomaly, fn ->
        anomalies = Anomaly.default_anomalies()
        alpha = Anomaly.default_alpha()
        %{anomalies: anomalies, alpha: alpha, demo: Anomaly.run(anomalies, alpha)}
      end)

    {:ok,
     socket
     |> assign(page_title: "Is this spike an anomaly?")
     |> assign(anomalies: anomalies, alpha: alpha, hour: 15, magnitude: 20, revealed: false)
     |> assign(demo: demo)}
  end

  @impl true
  def handle_event("adjust", params, socket) do
    socket =
      assign(socket,
        hour: Anomaly.clamp_hour(params["hour"] || socket.assigns.hour),
        magnitude: Anomaly.clamp_magnitude(params["magnitude"] || socket.assigns.magnitude),
        alpha: Anomaly.clamp_alpha(params["alpha"] || socket.assigns.alpha)
      )

    {:noreply, rerun(socket)}
  end

  def handle_event("add_anomaly", _params, socket) do
    %{anomalies: anomalies, hour: hour, magnitude: magnitude} = socket.assigns
    {:noreply, rerun(assign(socket, anomalies: Anomaly.add_anomaly(anomalies, hour, magnitude)))}
  end

  def handle_event("remove_anomaly", %{"hour" => hour}, socket) do
    {:noreply,
     rerun(assign(socket, anomalies: Anomaly.remove_anomaly(socket.assigns.anomalies, hour)))}
  end

  def handle_event("clear_anomalies", _params, socket) do
    {:noreply, rerun(assign(socket, anomalies: []))}
  end

  def handle_event("reveal_truth", _params, socket) do
    {:noreply, assign(socket, revealed: true)}
  end

  defp rerun(socket) do
    assign(socket, demo: Anomaly.run(socket.assigns.anomalies, socket.assigns.alpha))
  end

  @impl true
  def render(assigns) do
    ~H"""
    <Layouts.app flash={@flash} track={:questions}>
      <div class="mx-auto max-w-4xl">
        <.section_heading level={1} eyebrow="Ops monitoring" title="Is this spike an anomaly?">
          A service hums along at roughly {fmt(hd(@demo.predicted))} ms.
          Lines like this one never hold still; the question is never "did it move?"
          but "did it move more than its own noise says it should?". Below are {Anomaly.history_hours()} hours of normal latency the detector trained on, then a {Anomaly.window_hours()}-hour monitoring window it scores one hour at a time.
          You plant the spikes; the model decides which ones deserve a page.
        </.section_heading>
        <.mantra active={:compare} />

        <p class="report-prose mt-4 text-sm text-(--color-ink-soft)">
          Two spikes are planted to start: an obvious +{obvious_default().magnitude}&nbsp;ms
          and a subtle +{subtle_default().magnitude}&nbsp;ms.
          Every slider drag and click refits the baseline and rescores the window; the whole round trip is a single Kalman filter pass, measured live in the
          figure captions. Alpha is the false-alarm budget: the fraction of quiet
          hours you're willing to page someone for.
        </p>

        <form phx-change="adjust" class="mt-5 grid gap-x-8 gap-y-3 sm:grid-cols-3">
          <.param_slider
            name="hour"
            label="Plant at hour"
            min={Anomaly.hour_range().first}
            max={Anomaly.hour_range().last}
            value={@hour}
            display={"hour #{@hour} of the window"}
          />
          <.param_slider
            name="magnitude"
            label="Spike magnitude"
            min={Anomaly.magnitude_range().first}
            max={Anomaly.magnitude_range().last}
            value={@magnitude}
            display={"+#{@magnitude} ms"}
          />
          <label class="block">
            <div class="flex items-baseline justify-between gap-2">
              <span class="font-data text-[0.7rem] uppercase tracking-wider text-(--color-ink-soft)">
                Alpha; false-alarm budget
              </span>
              <span class="font-data text-xs tabular-nums text-(--color-ink)">
                |z| &gt; {fmt2(@demo.threshold)}
              </span>
            </div>
            <select name="alpha" class="select select-sm mt-1 w-full font-data text-sm">
              <option value="0.01" selected={@alpha == 0.01}>0.01; strict (99% band)</option>
              <option value="0.05" selected={@alpha == 0.05}>0.05; standard (95% band)</option>
              <option value="0.10" selected={@alpha == 0.10}>0.10; loose (90% band)</option>
            </select>
          </label>
        </form>

        <div class="mt-4 flex flex-wrap items-center gap-2">
          <button
            type="button"
            phx-click="add_anomaly"
            class="btn btn-sm btn-outline font-data text-xs"
          >
            Plant +{@magnitude} ms at hour {@hour}
          </button>
          <span
            :for={a <- @demo.anomalies}
            class="inline-flex items-center gap-1.5 rounded-full border border-(--color-grid) bg-(--color-paper-raised) px-3 py-1 font-data text-xs tabular-nums"
          >
            +{a.magnitude} ms @ hour {a.hour}
            <button
              type="button"
              phx-click="remove_anomaly"
              phx-value-hour={a.hour}
              class="touch-target rounded-full text-(--color-ink-faint) hover:text-(--color-lift)"
              aria-label={"remove the spike at hour #{a.hour}"}
            >
              ×
            </button>
          </span>
          <button
            :if={@demo.anomalies != []}
            type="button"
            phx-click="clear_anomalies"
            class="btn btn-sm btn-ghost font-data text-xs text-(--color-ink-faint)"
          >
            clear all
          </button>
        </div>

        <.figure
          no="1"
          caption={"#{Anomaly.history_hours() + Anomaly.window_hours()} hours of server latency; the detector trains on the first #{Anomaly.history_hours()}, then checks each monitoring hour against a one-step-ahead prediction; the shaded band is its ±#{fmt1(@demo.tolerance_ms)} ms tolerance at alpha #{fmt_alpha(@alpha)}."}
          execution={@demo.execution}
        >
          <:legend>
            <.swatch color={:ink} label="observed latency" />
            <.swatch color={:model} label="one-step prediction" dash />
            <.swatch color={:lift} label="flagged" />
            <.swatch :if={@revealed} color={:truth} label="planted spike" />
          </:legend>
          <.line_figure
            id="anomaly-stream"
            title="Latency stream with anomaly flags"
            desc="Observed latency is compared with the detector's one-step predictions; flagged and planted spike markers appear in the monitoring window."
            height={320}
            y_label="latency (ms)"
            series={[
              %{points: @demo.observations, color: :ink, width: 1.5},
              %{
                points: @demo.predicted,
                x_offset: @demo.window_start,
                color: :model,
                dash: true,
                width: 2.0
              }
            ]}
            bands={[
              %{
                lower: @demo.band_lower,
                upper: @demo.band_upper,
                x_offset: @demo.window_start,
                fill: "var(--color-model-band)"
              }
            ]}
            vlines={[%{x: @demo.window_start, label: "monitoring starts"}]}
            markers={stream_markers(@demo, @revealed)}
          />
        </.figure>

        <.figure
          no="2"
          caption={"The anomaly score for each monitoring hour; the one-step prediction error divided by the sd the filter reports for it; hours beyond the dashed ±#{fmt2(@demo.threshold)} lines are flagged."}
          execution={@demo.execution}
        >
          <:legend>
            <.swatch color={:model} label="z-score" />
            <.swatch color={:ink} label={"±#{fmt2(@demo.threshold)} threshold"} dash />
            <.swatch color={:lift} label="flagged" />
            <.swatch :if={@revealed} color={:truth} label="planted spike" />
          </:legend>
          <.line_figure
            id="anomaly-zscores"
            title="Anomaly z-scores"
            desc="Monitoring-hour z-scores are plotted against positive and negative threshold lines; flagged and planted spike markers appear when present."
            height={240}
            y_label="z-score"
            series={[%{points: @demo.z_scores, color: :model, width: 1.75}]}
            hlines={[%{y: @demo.threshold}, %{y: -@demo.threshold}]}
            markers={z_markers(@demo, @revealed)}
          />
        </.figure>

        <p class="report-prose text-sm text-(--color-ink-soft)">
          This is the sensitivity dial, laid bare. At alpha {fmt_alpha(@alpha)} the tolerance
          band is ±{fmt1(@demo.tolerance_ms)}&nbsp;ms, and pure noise is expected to cross it about {fmt1(
            @demo.expected_false_alarms
          )} times per clean {Anomaly.window_hours()}-hour window.
          Tighten alpha and small spikes slip inside the band; loosen it and quiet hours start
          paging you. One more honest detail: the detector is streaming, so each observation it
          scores gets absorbed into its level estimate; plant a big spike and watch the
          prediction get dragged upward for the next few hours.
        </p>

        <div class="mt-4 grid gap-4 sm:grid-cols-[1fr_minmax(240px,0.8fr)]">
          <.verdict_card verdict={verdict_line(@demo)} tone={verdict_tone(@demo)}>
            <:stat
              label="Flagged"
              value={"#{length(@demo.flagged_hours)} of #{Anomaly.window_hours()} hours"}
            />
            <:stat
              label="Tolerance band"
              value={"±#{fmt1(@demo.tolerance_ms)} ms"}
              hint={"|z| > #{fmt2(@demo.threshold)}"}
            />
            <:stat
              label="Expected false alarms"
              value={fmt1(@demo.expected_false_alarms)}
              hint="per clean window"
            />
            <:stat label="Max |z| seen" value={fmt1(@demo.summary.max_abs_z_score)} />
          </.verdict_card>

          <.reveal_truth
            revealed={@revealed}
            label="Reveal & grade the detector"
            truth={truth_line(@demo)}
            grade={grade_line(@demo)}
          />
        </div>

        <.under_the_hood code={Anomaly.code_snippet()} />

        <div class="mt-8 grid gap-4 sm:grid-cols-2">
          <.cross_link
            navigate={~p"/engine/kalman"}
            eyebrow="The engine underneath"
            title="The Kalman filter chapter"
          >
            The anomaly score is nothing exotic; it's the filter's one-step prediction
            error, divided by the sd the filter itself reports. That chapter shows the
            same machinery in slow motion.
          </.cross_link>
          <.cross_link
            navigate={~p"/speed"}
            eyebrow="Why this page never spins"
            title="The speed lane"
          >
            Fit and score is a single filter pass; {@demo.execution.elapsed_ms} ms on this
            server just now, no MCMC anywhere. That's why it reruns on every slider drag.
          </.cross_link>
        </div>
      </div>
    </Layouts.app>
    """
  end

  # -- Figure markers --------------------------------------------------------

  defp stream_markers(demo, revealed) do
    truth =
      if revealed do
        Enum.map(demo.anomalies, fn a ->
          %{x: demo.window_start + a.hour, y: obs_at(demo, a.hour), color: :truth, shape: :dot}
        end)
      else
        []
      end

    flags =
      Enum.map(demo.flagged_hours, fn h ->
        %{x: demo.window_start + h, y: obs_at(demo, h), color: :lift, shape: :x}
      end)

    truth ++ flags
  end

  defp z_markers(demo, revealed) do
    truth =
      if revealed do
        Enum.map(demo.anomalies, fn a ->
          %{x: a.hour, y: Enum.at(demo.z_scores, a.hour), color: :truth, shape: :dot}
        end)
      else
        []
      end

    flags =
      Enum.map(demo.flagged_hours, fn h ->
        %{x: h, y: Enum.at(demo.z_scores, h), color: :lift, shape: :x}
      end)

    truth ++ flags
  end

  defp obs_at(demo, hour), do: Enum.at(demo.observations, demo.window_start + hour)

  # -- Verdict ---------------------------------------------------------------

  defp verdict_line(%{anomalies: [], flagged_hours: []}) do
    "All quiet; nothing flagged in the monitoring window."
  end

  defp verdict_line(%{anomalies: [], flagged_hours: flagged}) do
    "#{length(flagged)} #{plural(length(flagged), "false alarm", "false alarms")}; noise alone crossed the threshold."
  end

  defp verdict_line(%{anomalies: planted, hits: hits, misses: [], false_alarms: []}) do
    "Caught #{length(hits)} of #{length(planted)} planted spikes; zero false alarms."
  end

  defp verdict_line(%{anomalies: planted, hits: hits, misses: misses}) when misses != [] do
    "Caught #{length(hits)} of #{length(planted)}; #{length(misses)} slipped inside the tolerance band."
  end

  defp verdict_line(%{anomalies: planted, hits: hits, false_alarms: fa}) do
    "Caught all #{length(hits)} of #{length(planted)}; but cried wolf on #{length(fa)} quiet #{plural(length(fa), "hour", "hours")}."
  end

  defp verdict_tone(%{anomalies: [], flagged_hours: []}), do: :neutral

  defp verdict_tone(%{misses: [], false_alarms: [], anomalies: planted}) when planted != [],
    do: :truth

  defp verdict_tone(_demo), do: :warning

  # -- Reveal ----------------------------------------------------------------

  defp truth_line(%{anomalies: []}) do
    "Nothing planted; every flag in this window is a false alarm."
  end

  defp truth_line(%{anomalies: anomalies}) do
    "Planted: " <>
      Enum.map_join(anomalies, ", ", fn a -> "+#{a.magnitude} ms at hour #{a.hour}" end) <>
      " (hours into the monitoring window)."
  end

  defp grade_line(%{anomalies: [], flagged_hours: []}) do
    "The detector agreed: a quiet window, zero flags."
  end

  defp grade_line(demo) do
    [misses_sentence(demo), false_alarm_sentence(demo), clean_sentence(demo)]
    |> Enum.reject(&is_nil/1)
    |> Enum.join(" ")
  end

  defp misses_sentence(%{misses: []}), do: nil

  defp misses_sentence(demo) do
    detail =
      Enum.map_join(demo.misses, "; ", fn h ->
        a = Enum.find(demo.anomalies, &(&1.hour == h))
        z = Enum.at(demo.z_scores, h)

        "+#{a.magnitude} ms at hour #{h} scored |z| = #{fmt2(abs(z))}, under the #{fmt2(demo.threshold)} bar"
      end)

    "Missed: #{detail}. A spike smaller than the tolerance band is, to this model, indistinguishable from noise; loosen alpha and the bar drops."
  end

  defp false_alarm_sentence(%{false_alarms: []}), do: nil

  defp false_alarm_sentence(%{anomalies: []} = demo) do
    hours = Enum.map_join(demo.false_alarms, ", ", &"#{&1}")

    "False #{plural(length(demo.false_alarms), "alarm", "alarms")} at #{plural(length(demo.false_alarms), "hour", "hours")} #{hours}; quiet #{plural(length(demo.false_alarms), "hour", "hours")} whose noise crossed the bar; expect about #{fmt1(demo.expected_false_alarms)} per clean window at alpha #{fmt_alpha(demo.alpha)}."
  end

  defp false_alarm_sentence(demo) do
    hours = Enum.map_join(demo.false_alarms, ", ", &"#{&1}")

    "Also flagged: #{plural(length(demo.false_alarms), "hour", "hours")} #{hours}; #{plural(length(demo.false_alarms), "an hour", "hours")} we didn't plant; noise, or the aftershock of a real spike being absorbed into the level estimate."
  end

  defp clean_sentence(%{misses: [], false_alarms: [], hits: hits}) when hits != [] do
    "Every planted spike crossed the threshold and nothing else did; a clean scorecard at this alpha."
  end

  defp clean_sentence(_demo), do: nil

  # -- Defaults for copy -------------------------------------------------------

  defp obvious_default, do: Enum.max_by(Anomaly.default_anomalies(), & &1.magnitude)
  defp subtle_default, do: Enum.min_by(Anomaly.default_anomalies(), & &1.magnitude)

  # -- Formatting ------------------------------------------------------------

  defp fmt(v) when is_number(v), do: :erlang.float_to_binary(v * 1.0, decimals: 0)
  defp fmt1(v) when is_number(v), do: :erlang.float_to_binary(v * 1.0, decimals: 1)
  defp fmt2(v) when is_number(v), do: :erlang.float_to_binary(v * 1.0, decimals: 2)

  defp fmt_alpha(alpha), do: :erlang.float_to_binary(alpha * 1.0, decimals: 2)

  defp plural(1, singular, _plural), do: singular
  defp plural(_n, _singular, plural), do: plural
end
