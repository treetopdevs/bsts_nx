defmodule BstsSiteWeb.HubLive do
  @moduledoc """
  The hub. Poses the site's one question and answers it live, then opens
  three doors: Questions, Engine, Trust.
  """
  use BstsSiteWeb, :live_view

  alias BstsSite.Demos.Hero

  @impl true
  def mount(_params, _session, socket) do
    prepared = Hero.prepare()
    demo = Hero.run(prepared, 12, 4)

    {:ok,
     socket
     |> assign(page_title: "Was it you, or was it noise?")
     |> assign(prepared: prepared, demo: demo, revealed: false)}
  end

  @impl true
  def handle_event("adjust", %{"lift" => lift, "noise" => noise}, socket) do
    demo = Hero.run(socket.assigns.prepared, lift, noise)
    {:noreply, assign(socket, demo: demo)}
  end

  def handle_event("reveal_truth", _params, socket) do
    {:noreply, assign(socket, revealed: true)}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <Layouts.app flash={@flash}>
      <div class="mx-auto max-w-3xl text-center">
        <p class="font-data text-xs uppercase tracking-[0.2em] text-(--color-lift)">
          Bayesian structural time series · pure Elixir · built on Nx
        </p>
        <h1 class="font-display mt-3 text-4xl font-bold leading-tight sm:text-5xl">
          The numbers changed.<br />
          <em>Was it you — or was it noise?</em>
        </h1>
        <p class="report-prose mx-auto mt-5 text-(--color-ink-soft)">
          A campaign launches, a policy lands, a server gets upgraded — and the line moves.
          BstsNx tells you whether it moved <em>because of you</em>, with honest uncertainty.
          You're watching it work right now: the figure below is computed by this server
          on every drag of a slider.
        </p>
      </div>

      <section class="mx-auto mt-10 max-w-4xl">
        <div class="report-prose text-sm text-(--color-ink-soft)">
          We generated six days of hourly website traffic and <strong>planted</strong>
          a campaign effect after 96 untreated hours, ramping in the way real carry-over
          effects do.
          The model is never shown the truth: it learns the pre-campaign baseline and its
          noise, projects what <em>should</em> have happened, and measures the gap. Make
          the effect smaller and the world noisier — and watch the model honestly refuse
          to be sure when the signal drowns.
        </div>

        <form phx-change="adjust" class="mt-5 grid gap-x-10 gap-y-3 sm:grid-cols-2">
          <.param_slider
            name="lift"
            label="Planted campaign effect"
            min={Hero.lift_range().first}
            max={Hero.lift_range().last}
            value={@demo.lift}
            display={"+#{@demo.lift} visits/hour"}
          />
          <.param_slider
            name="noise"
            label="Noise in the world"
            min={Hero.noise_range().first}
            max={Hero.noise_range().last}
            value={@demo.noise_sd}
            display={"sd #{@demo.noise_sd}"}
          />
        </form>

        <.figure
          no="1"
          caption="Hourly traffic vs. the counterfactual the model projected without seeing the post-campaign data. The shaded gap is the estimated effect."
          execution={@demo.execution}
        >
          <:legend>
            <.swatch color={:ink} label="observed" />
            <.swatch color={:model} label="counterfactual" dash />
            <.swatch color={:lift} label="estimated lift" />
          </:legend>
          <.line_figure
            id="hero-chart"
            height={320}
            y_label="visits/hour"
            series={[
              %{points: @demo.observations, color: :ink, width: 1.5},
              %{
                points: @demo.counterfactual_mean,
                x_offset: 96,
                color: :model,
                dash: true,
                width: 2.0
              }
            ]}
            bands={[
              %{
                lower: @demo.band_lower,
                upper: @demo.band_upper,
                x_offset: 96,
                fill: "var(--color-model-band)"
              },
              %{
                lower: @demo.counterfactual_mean,
                upper: Enum.drop(@demo.observations, 96),
                x_offset: 96,
                fill: "var(--color-lift-band)"
              }
            ]}
            vlines={[%{x: 96, label: "campaign starts"}]}
          />
        </.figure>

        <div class="grid gap-4 sm:grid-cols-[1fr_minmax(240px,0.8fr)]">
          <.verdict_card
            verdict={verdict_line(@demo)}
            tone={if @demo.significant?, do: :lift, else: :neutral}
          >
            <:stat
              label="Cumulative effect"
              value={"#{signed(@demo.cumulative.mean)} visits"}
              hint={"95% CI [#{fmt(@demo.cumulative.lower)}, #{fmt(@demo.cumulative.upper)}]"}
            />
            <:stat label="Relative lift" value={pct(@demo.relative.mean)} />
            <:stat label="Post-period" value="48 hours" />
          </.verdict_card>

          <.reveal_truth
            revealed={@revealed}
            truth={"Planted: ramps to +#{trunc(@demo.scenario.truth.lift_per_hour)} visits/hour — #{signed(@demo.scenario.truth.cumulative_lift)} cumulative."}
            grade={grade_line(@demo)}
          />
        </div>

        <.under_the_hood code={Hero.code_snippet()} />
      </section>

      <section class="mt-16">
        <div class="text-center">
          <.mantra class="justify-center" />
          <p class="report-prose mx-auto mt-3 text-sm text-(--color-ink-soft)">
            One method underneath everything here: <strong>decompose</strong> the series into
            structure and noise, <strong>project</strong> the structure forward,
            <strong>compare</strong> the projection with what actually happened.
            Every page on this site is that sentence, asked a different way.
          </p>
        </div>

        <div class="mt-8 grid gap-4 md:grid-cols-3">
          <.door
            navigate={~p"/demos"}
            eyebrow="I have a question"
            title="Five questions, answered live"
          >
            Did the campaign work? Which TV spot earns credit? How much stock do we need?
            Is this spike real? Did the policy do anything?
          </.door>
          <.door navigate={~p"/engine"} eyebrow="How does it work?" title="Open the engine">
            Six short chapters from "noise lies to you" to the world that didn't happen —
            with a Gibbs sampler you can watch converge along the way. Every figure is alive.
          </.door>
          <.door navigate={~p"/trust"} eyebrow="Should I trust it?" title="Grade the homework">
            We plant truths and grade the model against them, run placebo tests, and say
            plainly what this library doesn't do yet.
          </.door>
        </div>
      </section>

      <section class="mx-auto mt-16 max-w-3xl">
        <.section_heading eyebrow="For Elixir engineers" title="It's a library, not a service.">
          Everything on this site is <code>BstsNx</code> running inside a Phoenix app —
          the same calls you'd make from your own code.
        </.section_heading>
        <.under_the_hood
          code={quick_start_code()}
          title="The sixty-second version"
        />
        <div class="mt-2">
          <.link navigate={~p"/start"} class="font-data text-sm text-(--color-model) hover:text-(--color-ink) underline underline-offset-4">
            Install &amp; full quick start →
          </.link>
        </div>
      </section>
    </Layouts.app>
    """
  end

  attr :navigate, :string, required: true
  attr :eyebrow, :string, required: true
  attr :title, :string, required: true
  slot :inner_block, required: true

  defp door(assigns) do
    ~H"""
    <.link
      navigate={@navigate}
      class="group block rounded-md border border-(--color-grid) bg-(--color-paper-raised) p-5 transition-all hover:border-(--color-ink) hover:shadow-sm"
    >
      <div class="font-data text-[0.65rem] uppercase tracking-widest text-(--color-lift)">
        {@eyebrow}
      </div>
      <div class="font-display mt-2 text-xl font-bold">
        {@title} <span class="opacity-0 transition-opacity group-hover:opacity-100">→</span>
      </div>
      <p class="mt-2 text-sm leading-relaxed text-(--color-ink-soft)">
        {render_slot(@inner_block)}
      </p>
    </.link>
    """
  end

  defp verdict_line(%{significant?: true, cumulative: cum}) do
    "Yes — the campaign shows: #{signed(cum.mean)} visits."
  end

  defp verdict_line(%{significant?: false}) do
    "Honestly? Can't tell — the interval straddles zero."
  end

  defp grade_line(demo) do
    truth = demo.scenario.truth.cumulative_lift
    %{lower: lo, upper: hi} = demo.cumulative

    if truth >= lo and truth <= hi do
      "The planted truth sits inside the model's 95% interval. Recovered."
    else
      "The planted truth fell outside the 95% interval this time — that should be rare. Intervals are promises about the long run, not every draw; see the calibration page."
    end
  end

  defp quick_start_code do
    ~S"""
    result =
      BstsNx.CausalImpact.estimate(observations, {1, 96}, {97, 144},
        num_samples: 200,
        burn_in: 100,
        seed: 42
      )

    BstsNx.CausalImpact.summary(result).cumulative_effect
    #=> %{mean: ..., lower: ..., upper: ..., sd: ...}
    """
  end

  defp fmt(v) when is_number(v), do: :erlang.float_to_binary(v * 1.0, decimals: 0)

  defp signed(v) when is_number(v) do
    prefix = if v >= 0, do: "+", else: ""
    prefix <> :erlang.float_to_binary(v * 1.0, decimals: 0)
  end

  defp pct(v) when is_number(v) do
    prefix = if v >= 0, do: "+", else: ""
    prefix <> :erlang.float_to_binary(v * 100.0, decimals: 1) <> "%"
  end
end
