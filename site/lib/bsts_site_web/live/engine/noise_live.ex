defmodule BstsSiteWeb.Engine.NoiseLive do
  @moduledoc """
  Engine chapter 1: the four-walks quiz. Three series are pure accumulated
  noise; one carries a planted shift. Eyes guess, then the filter grades
  all four with cumulative-effect intervals.
  """
  use BstsSiteWeb, :live_view

  alias BstsSite.Demos.NoiseQuiz

  @impl true
  def mount(_params, _session, socket) do
    {:ok,
     socket
     |> assign(page_title: "Why raw data lies")
     |> assign(batch: NoiseQuiz.batch(0), counter: 0, guess: nil, revealed: false)}
  end

  @impl true
  def handle_event("guess", %{"walk" => walk}, socket) do
    if socket.assigns.guess do
      {:noreply, socket}
    else
      {:noreply, assign(socket, guess: parse_walk(walk))}
    end
  end

  def handle_event("regenerate", _params, socket) do
    counter = socket.assigns.counter + 1

    {:noreply,
     assign(socket,
       counter: counter,
       batch: NoiseQuiz.batch(counter),
       guess: nil,
       revealed: false
     )}
  end

  def handle_event("reveal_truth", _params, socket) do
    {:noreply, assign(socket, revealed: true)}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <Layouts.app flash={@flash} track={:engine}>
      <div class="mx-auto max-w-4xl">
        <.mantra active={:decompose} />
        <.section_heading
          level={1}
          class="mt-4"
          eyebrow="Engine · chapter 1"
          title="Why raw data lies"
        >
          Each series below is {NoiseQuiz.n()} steps of a random walk; accumulated coin-flip
          noise with no trend, no seasonality, no cause. Except one: from step {NoiseQuiz.half()},
          it carries a planted shift ramping to +{fmt(NoiseQuiz.shift())} per step.
          Your eyes evolved to find patterns, so all four will look like they're going somewhere.
          Click the one you think is hiding the real effect.
        </.section_heading>

        <div class="flex flex-wrap items-center gap-3">
          <button
            type="button"
            phx-click="regenerate"
            class="btn btn-sm btn-outline font-data text-xs"
          >
            Deal four fresh walks
          </button>
          <span class="font-data text-xs text-(--color-ink-faint)">
            set #{@counter + 1}; the counter seeds the generator, so every visitor's set
            #{@counter + 1} is identical
          </span>
        </div>

        <div class="grid gap-x-4 sm:grid-cols-2">
          <.figure
            :for={{walk, i} <- Enum.with_index(@batch.walks)}
            no={"#{i + 1}"}
            caption={walk_caption(i, @guess, @batch)}
          >
            <:legend :if={@guess}>
              <.swatch color={:model} label="filter's baseline" dash />
            </:legend>
            <button
              type="button"
              id={"walk-#{i}"}
              phx-click="guess"
              phx-value-walk={i}
              disabled={@guess != nil}
              aria-label={"Guess walk #{Enum.at(@batch.labels, i)}"}
              class={[
                "block w-full rounded-sm transition-shadow disabled:opacity-100",
                @guess == nil && "cursor-pointer hover:ring-2 hover:ring-(--color-model)",
                @guess != nil && i == @batch.planted && "ring-2 ring-(--color-truth)",
                @guess != nil && i == @guess && i != @batch.planted &&
                  "ring-2 ring-(--color-warning)"
              ]}
            >
              <.line_figure
                id={"walk-chart-#{i}"}
                title={"Random walk #{Enum.at(@batch.labels, i)}"}
                desc="One random walk candidate in the noise quiz, with the filter baseline and uncertainty band shown after a guess."
                height={180}
                series={walk_series(walk, Enum.at(@batch.verdicts, i), @guess)}
                bands={walk_bands(Enum.at(@batch.verdicts, i), @guess)}
                vlines={[%{x: NoiseQuiz.half()}]}
              />
            </button>
          </.figure>
        </div>

        <p :if={@guess == nil} class="report-prose mt-1 text-sm text-(--color-ink-soft)">
          One honest warning before you click: each chart is scaled to fill its own frame; exactly how dashboards do it. The axes tell the truth; the shapes exaggerate.
        </p>

        <section :if={@guess != nil}>
          <.figure
            no="5"
            caption="Cumulative second-half effect estimated for each walk, with 95% intervals. A walk is flagged only if its interval excludes zero."
            execution={@batch.execution}
          >
            <:legend>
              <.swatch color={:lift} label="estimated cumulative effect" />
            </:legend>
            <.bar_figure
              id="quiz-verdict-bars"
              title="Noise quiz effect intervals"
              desc="Cumulative second-half effects for the four walks are shown with 95% intervals."
              height={220}
              items={verdict_items(@batch)}
            />
          </.figure>

          <p class="report-prose -mt-3 mb-4 px-1 text-xs text-(--color-ink-faint)">
            Footnote: <code>estimate_from_filter</code>
            is the library's fast lane; a
            filter-and-smooth interpolation that gets to see every point it isn't told to mask.
            Here we masked the whole second half, so it behaves like a forecast; in general its
            baseline is not a causal counterfactual. It also gets handed the true step size; learning that from data is the Gibbs chapter. And because a 95% interval clears the
            bar on pure noise about one time in twenty, we deal only vetted sets; the
            calibration page shows what happens when nobody vets.
            <.link navigate={~p"/speed"} class="underline underline-offset-2">
              More on the fast lane →
            </.link>
          </p>

          <div class="grid gap-4 sm:grid-cols-[1fr_minmax(240px,0.8fr)]">
            <.verdict_card verdict={verdict_line(@guess, @batch)} tone={verdict_tone(@guess, @batch)}>
              Your eyes had one look. The filter had a model: it projected each walk's
              second half from its first, then asked whether the gap was bigger than
              the walk's own step noise could explain.
              <:stat
                label={"Walk #{Enum.at(@batch.labels, @batch.planted)} (planted)"}
                value={signed(planted_verdict(@batch).mean)}
                hint={ci_hint(planted_verdict(@batch))}
              />
              <:stat
                :if={@guess != @batch.planted}
                label={"Walk #{Enum.at(@batch.labels, @guess)} (your pick)"}
                value={signed(guess_verdict(@guess, @batch).mean)}
                hint={ci_hint(guess_verdict(@guess, @batch))}
              />
              <:stat label="Second half" value={"#{NoiseQuiz.n() - NoiseQuiz.half()} steps"} />
            </.verdict_card>

            <.reveal_truth
              revealed={@revealed}
              truth={"Planted: a shift ramping to +#{fmt(@batch.truth.shift)} per step from step #{@batch.truth.start} of walk #{Enum.at(@batch.labels, @batch.planted)}; #{signed(@batch.truth.cumulative)} cumulative."}
              grade={grade_line(@batch)}
            />
          </div>
        </section>

        <p class="report-prose mt-6 text-sm text-(--color-ink-soft)">
          This is the whole reason the site exists. A random walk grows "trends" out of
          nothing; it has to end up somewhere, and wherever it ends up looks like a story.
          Your eyes find the pattern; the filter quantifies the doubt. Three of those
          intervals straddle zero, which is the model saying, plainly, "a walk like this
          wanders that much on its own."
        </p>

        <.under_the_hood code={NoiseQuiz.code_snippet()} />

        <div class="mt-8 grid gap-4 md:grid-cols-3">
          <.cross_link
            navigate={~p"/engine/kalman"}
            eyebrow="Engine · chapter 2"
            title="Tracking a signal through noise"
          >
            The machine that drew those baselines: a Kalman filter you can detune yourself.
          </.cross_link>
          <.cross_link
            navigate={~p"/engine/gibbs"}
            eyebrow="Engine · chapter 4"
            title="Learning the noise"
          >
            We told the filter the true step size. The Gibbs sampler has to learn it from data.
          </.cross_link>
          <.cross_link
            navigate={~p"/trust/calibration"}
            eyebrow="Trust"
            title="Are the intervals honest?"
          >
            Five planted truths, every interval graded in public; a spot-check of the
            one-in-twenty promise.
          </.cross_link>
        </div>
      </div>
    </Layouts.app>
    """
  end

  # Before the guess the charts show only raw walks; afterwards each gets the
  # filter's projected baseline and 95% band overlaid.
  defp walk_series(walk, _verdict, nil), do: [%{points: walk, color: :ink, width: 1.3}]

  defp walk_series(walk, verdict, _guess) do
    [
      %{points: walk, color: :ink, width: 1.3},
      %{
        points: verdict.baseline,
        x_offset: NoiseQuiz.half(),
        color: :model,
        dash: true,
        width: 1.6
      }
    ]
  end

  defp walk_bands(_verdict, nil), do: []

  defp walk_bands(verdict, _guess) do
    [
      %{
        lower: verdict.band_lower,
        upper: verdict.band_upper,
        x_offset: NoiseQuiz.half(),
        fill: "var(--color-model-band)"
      }
    ]
  end

  defp parse_walk(walk) when is_binary(walk) do
    case Integer.parse(walk) do
      {n, _} -> n |> max(0) |> min(3)
      :error -> nil
    end
  end

  defp parse_walk(walk) when is_integer(walk), do: walk |> max(0) |> min(3)
  defp parse_walk(_), do: nil

  defp walk_caption(i, nil, batch) do
    "Walk #{Enum.at(batch.labels, i)}; #{NoiseQuiz.n()} steps, axes scaled to fill the frame."
  end

  defp walk_caption(i, guess, batch) do
    label = Enum.at(batch.labels, i)

    cond do
      i == batch.planted -> "Walk #{label}; the planted shift lived here."
      i == guess -> "Walk #{label}; your pick. Pure noise."
      true -> "Walk #{label}; pure noise."
    end
  end

  defp verdict_items(batch) do
    Enum.with_index(batch.verdicts, fn v, i ->
      suffix = if i == batch.planted, do: "; planted", else: ""

      %{
        label: "#{Enum.at(batch.labels, i)}#{suffix}",
        value: v.mean,
        lower: v.lower,
        upper: v.upper,
        color: :lift
      }
    end)
  end

  defp planted_verdict(batch), do: Enum.at(batch.verdicts, batch.planted)
  defp guess_verdict(guess, batch), do: Enum.at(batch.verdicts, guess)

  defp verdict_line(guess, batch) do
    planted_label = Enum.at(batch.labels, batch.planted)

    if guess == batch.planted do
      "Your eyes got it; and the filter agrees: walk #{planted_label}."
    else
      "Your eyes said #{Enum.at(batch.labels, guess)}. The filter says #{planted_label}; the only interval that excludes zero."
    end
  end

  defp verdict_tone(guess, batch), do: if(guess == batch.planted, do: :truth, else: :neutral)

  defp grade_line(batch) do
    truth = batch.truth.cumulative
    v = planted_verdict(batch)

    if truth >= v.lower and truth <= v.upper do
      "The planted #{signed(truth)} sits inside the filter's 95% interval [#{fmt(v.lower)}, #{fmt(v.upper)}]. Recovered."
    else
      "The planted #{signed(truth)} fell outside the filter's interval [#{fmt(v.lower)}, #{fmt(v.upper)}]; rare, and exactly what the calibration page keeps score on."
    end
  end

  defp ci_hint(v), do: "95% CI [#{fmt(v.lower)}, #{fmt(v.upper)}]"

  defp fmt(v) when is_number(v), do: :erlang.float_to_binary(v * 1.0, decimals: 0)

  defp signed(v) when is_number(v) do
    prefix = if v >= 0, do: "+", else: ""
    prefix <> :erlang.float_to_binary(v * 1.0, decimals: 0)
  end
end
