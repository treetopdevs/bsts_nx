defmodule BstsSiteWeb.Engine.CounterfactualLive do
  @moduledoc """
  Engine chapter: the counterfactual. One MCMC fit, all posterior draws kept,
  drawn as spaghetti — the spread IS the uncertainty. An alpha slider
  recomputes intervals from the stored draws without refitting.
  """
  use BstsSiteWeb, :live_view

  alias BstsSite.Demos.Counterfactual, as: Demo
  alias BstsSite.Demos.Limiter

  @impl true
  def mount(_params, _session, socket) do
    effect = 10

    {:ok,
     socket
     |> assign(page_title: "The counterfactual")
     |> assign(
       effect: effect,
       alpha: 0.05,
       scenario: Demo.scenario(effect),
       fit: nil,
       stats: nil,
       running: false,
       busy: false,
       revealed: false
     )}
  end

  @impl true
  def handle_event("adjust", %{"effect" => effect}, socket) do
    effect = Demo.clamp_effect(effect)

    {:noreply,
     assign(socket,
       effect: effect,
       scenario: Demo.scenario(effect),
       fit: nil,
       stats: nil,
       busy: false,
       revealed: false
     )}
  end

  def handle_event("run", _params, %{assigns: %{running: true}} = socket),
    do: {:noreply, socket}

  def handle_event("run", _params, socket) do
    effect = socket.assigns.effect
    socket = assign(socket, running: true, busy: false)

    {:noreply,
     start_async(socket, :fit, fn ->
       case Limiter.run(fn -> Demo.fit(effect) end) do
         {:ok, fit} -> fit
         :busy -> :busy
       end
     end)}
  end

  def handle_event("adjust_alpha", %{"alpha" => alpha}, socket) do
    alpha = Demo.clamp_alpha(alpha)

    stats =
      case socket.assigns.fit do
        nil -> nil
        fit -> Demo.stats(fit.result, alpha)
      end

    {:noreply, assign(socket, alpha: alpha, stats: stats)}
  end

  def handle_event("reveal_truth", _params, socket) do
    {:noreply, assign(socket, revealed: true)}
  end

  @impl true
  def handle_async(:fit, {:ok, :busy}, socket) do
    {:noreply, assign(socket, running: false, busy: true)}
  end

  def handle_async(:fit, {:ok, fit}, socket) do
    if fit.effect == socket.assigns.effect do
      stats = Demo.stats(fit.result, socket.assigns.alpha)
      {:noreply, assign(socket, running: false, busy: false, fit: fit, stats: stats)}
    else
      # The effect slider moved while the sampler ran — this result answers
      # a question the page is no longer asking. Drop it.
      {:noreply, assign(socket, running: false)}
    end
  end

  def handle_async(:fit, {:exit, _reason}, socket) do
    {:noreply, assign(socket, running: false, busy: true)}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <Layouts.app flash={@flash} track={:engine}>
      <div class="mx-auto max-w-4xl">
        <.mantra active={:compare} class="mb-4" />
        <.section_heading
          eyebrow="Engine · chapter 6"
          title="The world that didn't happen"
        >
          A causal effect is a comparison against a world we never observed. The model can't
          show us that world, but it can show us its honest guesses: fit on the pre-period
          only, then forward-simulate what should have happened — once per posterior draw.
          Below, {Demo.n_post()} days follow a planted step. Fit the model and every thin
          blue line is one complete answer to "what if we had done nothing?" The spread of
          the spaghetti is the uncertainty; no band is drawn because none is needed.
        </.section_heading>

        <form phx-change="adjust" class="mt-5 grid gap-x-10 gap-y-3 sm:grid-cols-2">
          <.param_slider
            name="effect"
            label="Planted step at day 91"
            min={Demo.effect_range().first}
            max={Demo.effect_range().last}
            value={@effect}
            display={"+#{@effect} units/day"}
          />
          <div class="flex items-end">
            <button
              type="button"
              phx-click="run"
              disabled={@running}
              class="btn btn-sm btn-outline font-data text-xs"
            >
              {run_label(@running, @fit)}
            </button>
          </div>
        </form>

        <.figure no="1" caption={figure_caption(@fit)} execution={@fit && @fit.execution}>
          <:legend>
            <.swatch color={:ink} label="observed" />
            <.swatch :if={@fit} color={:model} label="counterfactual draws" />
            <.swatch :if={@fit} color={:model} label="posterior mean" dash />
          </:legend>
          <.line_figure
            id="cf-spaghetti"
            height={320}
            y_label="units/day"
            series={
              spaghetti_series(@fit) ++ [%{points: @scenario.observations, color: :ink, width: 1.5}]
            }
            vlines={[%{x: Demo.pre_end(), label: "step planted"}]}
          />
        </.figure>

        <div :if={@busy} class="my-4">
          <.verdict_card
            verdict="Another visitor is sampling — try again in a moment."
            tone={:warning}
          >
            MCMC slots on this server are limited so everyone's demo stays responsive.
          </.verdict_card>
        </div>

        <div :if={@fit} class="mt-8">
          <form phx-change="adjust_alpha" class="max-w-sm">
            <.param_slider
              name="alpha"
              label="Interval width (no refit)"
              min={Demo.alpha_min()}
              max={Demo.alpha_max()}
              step={0.05}
              value={@alpha}
              display={"#{@stats.coverage_pct}% interval (α = #{@alpha})"}
            />
          </form>
          <p class="report-prose mt-2 text-sm text-(--color-ink-soft)">
            Drag it: nothing refits. The {@fit.num_draws} draws are already stored, so a new
            interval is just new percentiles — recomputed in about a millisecond.
          </p>

          <.figure
            no="2"
            caption={"The #{@fit.num_draws} cumulative-effect draws, one per counterfactual trajectory. The dashed lines mark the #{@stats.coverage_pct}% interval; they move with the slider above."}
          >
            <:legend>
              <.swatch color={:model} label="posterior draws" />
              <.swatch :if={@revealed} color={:truth} label="planted truth" dash />
            </:legend>
            <.histogram
              id="cf-cumulative-hist"
              values={@fit.cumulative_draws}
              bins={18}
              vlines={hist_vlines(@stats, @scenario, @revealed)}
            />
          </.figure>

          <div class="grid gap-4 sm:grid-cols-[1fr_minmax(240px,0.8fr)]">
            <.verdict_card
              verdict={verdict_line(@stats)}
              tone={if @stats.significant?, do: :lift, else: :neutral}
            >
              <:stat
                label="Cumulative effect"
                value={"#{signed(@stats.cumulative.mean)} units"}
                hint={"#{@stats.coverage_pct}% CI [#{fmt(@stats.cumulative.lower)}, #{fmt(@stats.cumulative.upper)}]"}
              />
              <:stat label="Relative lift" value={pct(@stats.relative.mean)} />
              <:stat label="Posterior draws" value={"#{@fit.num_draws}"} />
            </.verdict_card>

            <.reveal_truth
              revealed={@revealed}
              truth={"Planted: a +#{@scenario.effect}/day step from day 91 — #{signed(@scenario.truth.cumulative)} cumulative over #{@scenario.truth.post_days} days."}
              grade={grade_line(@stats, @scenario)}
            />
          </div>
        </div>

        <div :if={is_nil(@fit) and not @busy} class="my-4">
          <.verdict_card verdict="No verdict yet — the model hasn't looked." tone={:neutral}>
            Press "Fit the model" above. The sampler sees only days 1–{Demo.pre_end()}; the
            planted step stays hidden from it.
          </.verdict_card>
        </div>

        <.under_the_hood code={Demo.code_snippet()} />

        <div class="mt-8 grid gap-4 sm:grid-cols-3">
          <.cross_link
            navigate={~p"/demos/marketing"}
            eyebrow="Same engine, real question"
            title="Did the campaign work?"
          >
            The marketing demo asks this page's question with money on the line.
          </.cross_link>
          <.cross_link
            navigate={~p"/trust/calibration"}
            eyebrow="Grade the intervals"
            title="Are 95% intervals honest?"
          >
            Intervals are promises about the long run. The calibration page checks whether
            they're kept.
          </.cross_link>
          <.cross_link
            navigate={~p"/engine/gibbs"}
            eyebrow="Engine · chapter 4"
            title="Where the draws come from"
          >
            Each spaghetti strand starts as one Gibbs draw of states and variances.
          </.cross_link>
        </div>
      </div>
    </Layouts.app>
    """
  end

  defp run_label(true, _fit), do: "Sampling…"
  defp run_label(false, nil), do: "Fit the model (~1–2 s)"
  defp run_label(false, _fit), do: "Refit"

  defp spaghetti_series(nil), do: []

  defp spaghetti_series(fit) do
    strands =
      Enum.with_index(fit.spaghetti)
      |> Enum.map(fn {draw, _i} ->
        %{
          points: draw,
          x_offset: Demo.pre_end(),
          color: :model,
          width: 1.0,
          opacity: 0.12
        }
      end)

    strands ++
      [
        %{
          points: fit.mean_counterfactual,
          x_offset: Demo.pre_end(),
          color: :model,
          dash: true,
          width: 2.0
        }
      ]
  end

  defp figure_caption(nil) do
    "120 days of observations; a step is planted at day 91. Fit the model to overlay its counterfactual draws."
  end

  defp figure_caption(fit) do
    "#{length(fit.spaghetti)} of the #{fit.num_draws} posterior counterfactual trajectories over days 91–120 — " <>
      "each thin line is one possible what-would-have-happened. The dashed line is their mean."
  end

  defp hist_vlines(stats, scenario, revealed) do
    interval = [
      %{x: stats.cumulative.lower, label: "lower", color: :model},
      %{x: stats.cumulative.upper, label: "upper", color: :model}
    ]

    if revealed do
      interval ++ [%{x: scenario.truth.cumulative, label: "truth", color: :truth}]
    else
      interval
    end
  end

  defp verdict_line(%{significant?: true} = stats) do
    "An effect shows: #{signed(stats.cumulative.mean)} units, and the #{stats.coverage_pct}% interval clears zero."
  end

  defp verdict_line(%{significant?: false} = stats) do
    "Honestly? Can't tell — the #{stats.coverage_pct}% interval straddles zero."
  end

  defp grade_line(stats, scenario) do
    truth = scenario.truth.cumulative
    %{lower: lo, upper: hi} = stats.cumulative

    cond do
      not is_number(lo) or not is_number(hi) ->
        nil

      truth >= lo and truth <= hi ->
        "The planted truth sits inside the #{stats.coverage_pct}% interval. Recovered."

      true ->
        "The planted truth fell outside the #{stats.coverage_pct}% interval. Narrow intervals miss more often — that's the trade the slider is making; see the calibration page."
    end
  end

  defp fmt(v) when is_number(v), do: :erlang.float_to_binary(v * 1.0, decimals: 0)
  defp fmt(v), do: to_string(v)

  defp signed(v) when is_number(v) do
    prefix = if v >= 0, do: "+", else: ""
    prefix <> :erlang.float_to_binary(v * 1.0, decimals: 0)
  end

  defp pct(v) when is_number(v) do
    prefix = if v >= 0, do: "+", else: ""
    prefix <> :erlang.float_to_binary(v * 100.0, decimals: 1) <> "%"
  end

  defp pct(v), do: to_string(v)
end
