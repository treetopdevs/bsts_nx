defmodule BstsSiteWeb.Engine.ComposeLive do
  @moduledoc """
  Engine chapter: models as Lego bricks. Toggle components on and off and
  watch `Components.compose_specs/2` assemble the block-diagonal transition
  matrix live. No sampling runs on this page — composition is instant.
  """
  use BstsSiteWeb, :live_view

  alias BstsSite.Demos.Compose

  @impl true
  def mount(_params, _session, socket) do
    demo = Compose.build(:local_level, :none, :none)

    {:ok,
     socket
     |> assign(page_title: "Models as Lego bricks")
     |> assign(demo: demo)}
  end

  @impl true
  def handle_event("configure", %{"group" => group, "choice" => choice}, socket) do
    demo = socket.assigns.demo

    {trend, seasonal, regression} =
      case group do
        "trend" -> {Compose.parse_choice("trend", choice), demo.seasonal, demo.regression}
        "seasonal" -> {demo.trend, Compose.parse_choice("seasonal", choice), demo.regression}
        "regression" -> {demo.trend, demo.seasonal, Compose.parse_choice("regression", choice)}
        _other -> {demo.trend, demo.seasonal, demo.regression}
      end

    {:noreply, assign(socket, demo: Compose.build(trend, seasonal, regression))}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <Layouts.app flash={@flash} track={:engine}>
      <div class="mx-auto max-w-4xl">
        <.mantra active={:decompose} class="mb-4" />
        <.section_heading eyebrow="Engine · chapter 5" title="Snap a model together">
          Real series are rarely just a level: they trend, they cycle weekly, they lean on
          control series. BstsNx builds all of that from small bricks — each component is a
          tiny state-space model, and <code>compose_specs/2</code> snaps them together by
          placing their transition matrices on a block diagonal. Toggle the bricks below and
          watch the composed matrix take shape. Nothing is sampled on this page; composition
          is pure matrix bookkeeping, re-run live on every click.
        </.section_heading>

        <div class="mt-6 grid gap-4 sm:grid-cols-3">
          <.chip_group label="Trend">
            <.chip
              group="trend"
              choice="local_level"
              active={@demo.trend == :local_level}
              label="local level"
              hint="1 state"
            />
            <.chip
              group="trend"
              choice="local_linear_trend"
              active={@demo.trend == :local_linear_trend}
              label="local linear trend"
              hint="2 states"
            />
          </.chip_group>

          <.chip_group label="Seasonality">
            <.chip group="seasonal" choice="none" active={@demo.seasonal == :none} label="none" />
            <.chip
              group="seasonal"
              choice="4"
              active={@demo.seasonal == 4}
              label="period 4"
              hint="3 states"
            />
            <.chip
              group="seasonal"
              choice="7"
              active={@demo.seasonal == 7}
              label="period 7"
              hint="6 states"
            />
          </.chip_group>

          <.chip_group label="Regression">
            <.chip group="regression" choice="none" active={@demo.regression == :none} label="none" />
            <.chip
              group="regression"
              choice="two_controls"
              active={@demo.regression == :two_controls}
              label="2 controls"
              hint="2 states"
            />
          </.chip_group>
        </div>

        <.figure no="1" caption={figure_caption(@demo)} execution={@demo.execution}>
          <:legend>
            <.swatch color={:model} label="nonzero entry" />
            <.swatch color={:lift} label="component block" />
          </:legend>
          <.matrix_grid id="compose-f" matrix={@demo.f_matrix} blocks={@demo.blocks} />
        </.figure>

        <div class="grid gap-4 sm:grid-cols-[1fr_minmax(240px,0.9fr)]">
          <.verdict_card verdict={verdict_line(@demo)} tone={:neutral}>
            The sampler would learn {@demo.q_count + 1} variances for this spec: {Enum.join(
              @demo.variance_labels,
              ", "
            )} — plus one observation noise
            variance shared by the whole model. Everything else in the matrix is fixed
            structure. <:stat label="State dimension" value={"#{@demo.state_dim}"} />
            <:stat label="Process variances" value={"#{@demo.q_count}"} hint="one per q_spec" />
            <:stat label="Observation variance" value="1" />
          </.verdict_card>

          <div class="rounded-md border border-(--color-grid) bg-(--color-paper-raised) p-4 text-sm text-(--color-ink-soft)">
            <div class="font-data text-[0.65rem] uppercase tracking-wider text-(--color-ink-faint)">
              Observation matrix
            </div>
            <p class="report-prose mt-2">
              <span :if={!@demo.time_varying_h?}>
                H stays a single static row: it reads the level{if @demo.seasonal != :none,
                  do: " and the current seasonal effect"}, and ignores the rest of the state.
              </span>
              <span :if={@demo.time_varying_h?}>
                With regression on, H becomes time-varying: {@demo.h_steps} rows, one per
                timestep, each carrying that step's regressor values. The coefficients live
                in the state; the data lives in H.
              </span>
            </p>
          </div>
        </div>

        <.under_the_hood code={Compose.code_snippet(@demo)} />

        <section class="mt-10">
          <.section_heading eyebrow="How complexity flows" title="One spec, the whole stack">
            The struct you just composed is the contract for everything above it: each layer
            hands its output to the next, and every high-level answer on this site bottoms
            out in a spec like this one.
          </.section_heading>
          <div class="rounded-md border border-(--color-grid) bg-(--color-paper-raised) p-4 text-center">
            <div class="font-data text-sm">
              Components <span aria-hidden="true">→</span>
              ModelSpec <span aria-hidden="true">→</span>
              GibbsSampler <span aria-hidden="true">→</span>
              CausalImpact <span aria-hidden="true">→</span>
              Pipeline
            </div>
            <div class="font-data mt-1 text-[0.65rem] uppercase tracking-wider text-(--color-ink-faint)">
              compose · specify · sample · compare · attribute
            </div>
          </div>
          <p class="report-prose mt-4 text-sm text-(--color-ink-soft)">
            One honest caveat before wiring this spec into MCMC: the structured Gibbs sampler
            resamples every one of those {@demo.q_count + 1} variances each iteration, and on
            CPU that is minutes-per-fit territory for seasonal specs — which is why this page
            composes but does not sample. The speed page maps which paths are milliseconds
            and which are minutes.
          </p>
        </section>

        <div class="mt-8 grid gap-4 md:grid-cols-3">
          <.cross_link
            navigate={~p"/engine/gibbs"}
            eyebrow="Engine · chapter 4"
            title="Feed it to the sampler"
          >
            The Gibbs chapter shows what "learning the variances" actually looks like — a
            chain you can watch converge.
          </.cross_link>
          <.cross_link
            navigate={~p"/engine/counterfactual"}
            eyebrow="Engine · chapter 6"
            title="The world that didn't happen"
          >
            Where a spec like this earns its keep: fit on the pre-period, then simulate
            the world where nothing happened.
          </.cross_link>
          <.cross_link
            navigate={~p"/speed"}
            eyebrow="Latency honesty"
            title="What runs fast, what doesn't"
          >
            Composition is instant; structured MCMC is not. The speed page measures both on
            this server.
          </.cross_link>
        </div>
      </div>
    </Layouts.app>
    """
  end

  attr :label, :string, required: true
  slot :inner_block, required: true

  defp chip_group(assigns) do
    ~H"""
    <div>
      <div class="font-data text-[0.7rem] uppercase tracking-wider text-(--color-ink-soft)">
        {@label}
      </div>
      <div class="mt-2 flex flex-wrap gap-2">
        {render_slot(@inner_block)}
      </div>
    </div>
    """
  end

  attr :group, :string, required: true
  attr :choice, :string, required: true
  attr :active, :boolean, required: true
  attr :label, :string, required: true
  attr :hint, :string, default: nil

  defp chip(assigns) do
    ~H"""
    <button
      type="button"
      phx-click="configure"
      phx-value-group={@group}
      phx-value-choice={@choice}
      aria-pressed={to_string(@active)}
      class={[
        "rounded-full border px-3 py-1 font-data text-xs transition-colors",
        @active && "border-(--color-ink) bg-(--color-ink) text-(--color-paper)",
        !@active &&
          "border-(--color-grid) bg-(--color-paper-raised) text-(--color-ink-soft) hover:border-(--color-ink)"
      ]}
    >
      {@label}<span :if={@hint} class="opacity-60"> · {@hint}</span>
    </button>
    """
  end

  defp figure_caption(demo) do
    blocks = Enum.map_join(demo.blocks, ", ", fn b -> "#{b.label} (#{b.size}×#{b.size})" end)

    "The composed #{demo.state_dim}×#{demo.state_dim} transition matrix F. " <>
      "Outlined diagonal blocks, top-left to bottom-right: #{blocks}. " <>
      "Off-block zeros mean the components evolve independently."
  end

  defp verdict_line(demo) do
    case length(demo.blocks) do
      1 ->
        "One brick, #{demo.state_dim} latent state#{plural(demo.state_dim)} — the simplest honest model."

      parts ->
        "#{parts} bricks snap into one spec with #{demo.state_dim} latent states."
    end
  end

  defp plural(1), do: ""
  defp plural(_), do: "s"
end
