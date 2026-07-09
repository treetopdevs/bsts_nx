defmodule BstsSiteWeb.Demos.TvLive do
  @moduledoc """
  Which TV spots earned their airtime? Overlapping-credit attribution on
  the Operational fast lane, re-run on every slider drag.
  """
  use BstsSiteWeb, :live_view

  alias BstsSite.Demos.DefaultCache
  alias BstsSite.Demos.Tv

  @impl true
  def mount(_params, _session, socket) do
    %{prepared: prepared, demo: demo} =
      DefaultCache.get(:tv, fn ->
        prepared = Tv.prepare()
        d = Tv.defaults()
        %{prepared: prepared, demo: Tv.run(prepared, d.start_a, d.start_b, d.start_c, d.strength)}
      end)

    {:ok,
     socket
     |> assign(page_title: "Which TV spots earned their airtime?")
     |> assign(prepared: prepared, demo: demo, revealed: false)}
  end

  @impl true
  def handle_event(
        "adjust",
        %{"start_a" => a, "start_b" => b, "start_c" => c, "strength" => s},
        socket
      ) do
    demo = Tv.run(socket.assigns.prepared, a, b, c, s)
    {:noreply, assign(socket, demo: demo)}
  end

  def handle_event("reveal_truth", _params, socket) do
    {:noreply, assign(socket, revealed: true)}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <Layouts.app flash={@flash} track={:questions}>
      <div class="mx-auto max-w-4xl">
        <.section_heading
          level={1}
          eyebrow="I have a question; attribution"
          title="Which TV spots earned their airtime?"
        >
          Three TV spots air over the last two days of a six-day traffic series, and each
          leaves a carry-over bump that decays across the hours after it runs. We planted
          those bumps ourselves, so the fair answer is known; the model just isn't told.
          It forecasts what traffic would have done with no TV at all, measures the gap,
          and splits credit between the spots. Drag two windows into each other and watch
          the split stay fair where the spots share hours.
        </.section_heading>

        <form phx-change="adjust" class="mt-5 grid gap-x-10 gap-y-3 sm:grid-cols-2">
          <.param_slider
            :for={{spot, name} <- Enum.zip(@demo.spots, ~w(start_a start_b start_c))}
            name={name}
            label={"#{spot.id} airs at"}
            min={Tv.start_range().first}
            max={Tv.start_range().last}
            value={spot.window_start}
            display={"post hour #{spot.window_start} → #{spot.window_end}"}
          />
          <.param_slider
            name="strength"
            label="Spot strength (A's peak; B and C air at 70% and 40%)"
            min={Tv.strength_range().first}
            max={Tv.strength_range().last}
            value={@demo.strength}
            display={"peak +#{@demo.strength} visits/hour"}
          />
        </form>

        <.figure
          no="1"
          caption={"The three #{Tv.window_hours()}-hour spot windows inside the #{Tv.post_hours()}-hour post period; 0-based and half-open, so [18, 26) covers hours 18 through 25, and windows are free to overlap."}
          execution={@demo.execution}
        >
          <:legend>
            <.swatch color={:model} label="spot window" />
          </:legend>
          <.gantt
            id="tv-windows"
            title="TV spot windows"
            desc="Each horizontal bar shows a half-open TV spot attribution window inside the post-period; overlapping bars share credit."
            domain={{0, Tv.post_hours()}}
            height={130}
            rows={
              for spot <- @demo.spots do
                %{id: spot.id, start: spot.window_start, stop: spot.window_end, color: :model}
              end
            }
          />
        </.figure>

        <p class="report-prose text-sm text-(--color-ink-soft)">
          {overlap_note(@demo.overlap_groups)}
        </p>

        <.figure
          no="2"
          caption="Hourly traffic against the counterfactual forecast from pre-period data alone; the model never sees post-period outcomes when projecting. Shaded magenta is the measured gap; dashed vertical lines mark where each spot's window opens."
          execution={@demo.execution}
        >
          <:legend>
            <.swatch color={:ink} label="observed" />
            <.swatch color={:model} label="counterfactual" dash />
            <.swatch color={:lift} label="measured lift" />
            <.swatch :if={@revealed} color={:truth} label="planted path" />
          </:legend>
          <.line_figure
            id="tv-traffic"
            title="TV traffic counterfactual"
            desc="Hourly traffic is compared with a counterfactual forecast; shaded lift and spot-window markers show where TV could have contributed."
            height={320}
            y_label="visits/hour"
            series={
              [
                %{points: @demo.observations, color: :ink, width: 1.5},
                %{
                  points: @demo.counterfactual_mean,
                  x_offset: Tv.pre_end(),
                  color: :model,
                  dash: true,
                  width: 2.0
                }
              ] ++
                if @revealed do
                  [%{points: @demo.truth_path, x_offset: Tv.pre_end(), color: :truth, width: 1.75}]
                else
                  []
                end
            }
            bands={[
              %{
                lower: @demo.band_lower,
                upper: @demo.band_upper,
                x_offset: Tv.pre_end(),
                fill: "var(--color-model-band)"
              },
              %{
                lower: @demo.counterfactual_mean,
                upper: Enum.drop(@demo.observations, Tv.pre_end()),
                x_offset: Tv.pre_end(),
                fill: "var(--color-lift-band)"
              }
            ]}
            vlines={
              [%{x: Tv.pre_end()}] ++
                for spot <- @demo.spots do
                  %{x: Tv.pre_end() + spot.window_start, label: spot.short}
                end
            }
          />
        </.figure>

        <.figure
          no="3"
          caption={"Per-spot attributed lift with 95% intervals. The three attributions sum to #{signed1(@demo.attributed_sum)} visits; reconciling exactly with the #{signed1(@demo.union_lift)} visits of lift measured across the union of the windows, no hour counted twice."}
          execution={@demo.execution}
        >
          <:legend>
            <.swatch color={:lift} label="attributed lift" />
            <.swatch :if={@revealed} color={:truth} label="planted contribution" />
          </:legend>
          <.bar_figure
            id="tv-attribution"
            title="Per-spot attributed lift"
            desc="Each TV spot's attributed lift is shown with uncertainty, and planted contributions appear when the answer key is revealed."
            height={if @revealed, do: 300, else: 200}
            items={
              Enum.flat_map(@demo.spots, fn spot ->
                [
                  %{
                    label: spot.id,
                    value: spot.lift,
                    lower: spot.lift_lower,
                    upper: spot.lift_upper,
                    color: :lift
                  }
                ] ++
                  if @revealed do
                    [%{label: "#{spot.short} · planted", value: spot.truth, color: :truth}]
                  else
                    []
                  end
              end)
            }
          />
        </.figure>

        <div class="grid gap-4 sm:grid-cols-[1fr_minmax(240px,0.8fr)]">
          <.verdict_card
            verdict={verdict_line(@demo.spots)}
            tone={if Enum.any?(@demo.spots, & &1.earned?), do: :lift, else: :neutral}
          >
            <p>
              A spot "earned its airtime" when the lower edge of its 95% interval clears
              zero. Values are visits attributed over each spot's own window.
            </p>
            <:stat
              :for={spot <- @demo.spots}
              label={spot.id}
              value={"#{signed(spot.lift)} visits"}
              hint={"[#{fmt(spot.lift_lower)}, #{fmt(spot.lift_upper)}] · P(>0) #{prob(spot.p_positive)}"}
            />
            <:stat
              label="All spots"
              value={"#{signed(@demo.attributed_sum)} visits"}
              hint={"± #{fmt(@demo.attributed_total_sd)} sd"}
            />
          </.verdict_card>

          <.reveal_truth
            revealed={@revealed}
            truth={truth_line(@demo.spots)}
            grade={grade_line(@demo.spots)}
          />
        </div>

        <.under_the_hood code={Tv.code_snippet()} />

        <div class="mt-10 grid gap-4 md:grid-cols-3">
          <.cross_link
            navigate={~p"/demos/marketing"}
            eyebrow="Same engine, one lever"
            title="Did the campaign work?"
          >
            One campaign instead of three spots: the counterfactual gap without the
            credit split.
          </.cross_link>
          <.cross_link
            navigate={~p"/engine/counterfactual"}
            eyebrow="The engine underneath"
            title="Where the no-TV world comes from"
          >
            The counterfactual here is a projection of what traffic would have done with
            no spots at all; this chapter shows how it's built.
          </.cross_link>
          <.cross_link
            navigate={~p"/speed"}
            eyebrow="Why this can run per drag"
            title="Prepare once, run many"
          >
            Every drag here re-runs the Operational fast lane. The speed page measures
            what that costs on this server.
          </.cross_link>
        </div>
      </div>
    </Layouts.app>
    """
  end

  defp verdict_line(spots) do
    earned = Enum.filter(spots, & &1.earned?)
    missed = Enum.reject(spots, & &1.earned?)

    case {earned, missed} do
      {_, []} ->
        "All three spots earned their airtime."

      {[], _} ->
        "Honestly? Can't call a single spot; every interval straddles zero."

      _ ->
        "#{join_names(names(earned))} earned it; #{join_names(names(missed))} #{verb(missed)} inside the noise."
    end
  end

  defp truth_line(spots) do
    planted =
      spots
      |> Enum.map(fn spot -> "#{spot.id} #{signed1(spot.truth)}" end)
      |> Enum.join(", ")

    "Planted: #{planted} visits; each an 8-hour decaying pulse starting with its window."
  end

  defp grade_line(spots) do
    recovered = Enum.count(spots, fn s -> s.truth >= s.lift_lower and s.truth <= s.lift_upper end)

    case recovered do
      3 ->
        "All three planted contributions sit inside their spots' 95% intervals. Recovered."

      n ->
        "#{n} of 3 planted contributions sit inside their 95% intervals. Where windows " <>
          "overlap the per-spot interval is a variance heuristic (Shapley splits means, " <>
          "not variances), so an occasional miss there is honest, not hidden."
    end
  end

  defp overlap_note([]) do
    "No windows overlap right now; each spot keeps its own hours and takes full credit " <>
      "for them. Drag two starts within 8 hours of each other to force a collision."
  end

  defp overlap_note(groups) do
    listed = groups |> Enum.map(&join_names/1) |> Enum.join("; ")

    "#{listed} share hours right now. Inside shared hours the model sees one combined " <>
      "bump, so the group's lift is split by exact Shapley values; each spot's average " <>
      "marginal contribution over every order it could have arrived in. The split is " <>
      "exact for the means; per-spot intervals inside shared hours are approximate."
  end

  defp names(spots), do: Enum.map(spots, & &1.id)

  defp join_names([a]), do: a
  defp join_names([a, b]), do: "#{a} and #{b}"

  defp join_names(list) do
    {rest, [last]} = Enum.split(list, -1)
    Enum.join(rest, ", ") <> " and " <> last
  end

  defp verb([_]), do: "is"
  defp verb(_), do: "are"

  defp fmt(v) when is_number(v), do: :erlang.float_to_binary(v * 1.0, decimals: 0)

  defp signed(v) when is_number(v) do
    prefix = if v >= 0, do: "+", else: ""
    prefix <> :erlang.float_to_binary(v * 1.0, decimals: 0)
  end

  defp signed1(v) when is_number(v) do
    prefix = if v >= 0, do: "+", else: ""
    prefix <> :erlang.float_to_binary(v * 1.0, decimals: 1)
  end

  defp prob(v) when is_number(v), do: :erlang.float_to_binary(v * 1.0, decimals: 2)
end
