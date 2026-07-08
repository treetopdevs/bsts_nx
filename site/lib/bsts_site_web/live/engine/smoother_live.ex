defmodule BstsSiteWeb.Engine.SmootherLive do
  @moduledoc """
  Engine chapter 3: the RTS smoother revisits the Kalman chapter's series
  with hindsight, and the Carter-Kohn simulation smoother draws whole
  alternative histories from the posterior.
  """
  use BstsSiteWeb, :live_view

  alias BstsSite.Demos.Hindsight

  @impl true
  def mount(_params, _session, socket) do
    prepared = Hindsight.prepare()

    {:ok,
     socket
     |> assign(page_title: "The power of hindsight")
     |> assign(
       prepared: prepared,
       show_smoothed: false,
       draws: [],
       draw_count: 0,
       draw_exec: nil,
       revealed: false
     )}
  end

  @impl true
  def handle_event("toggle", params, socket) do
    {:noreply, assign(socket, show_smoothed: params["smoothed"] == "true")}
  end

  def handle_event("draw", _params, socket) do
    %{prepared: prepared, draws: draws, draw_count: count} = socket.assigns
    draw = Hindsight.draw(prepared, count)

    draws =
      (draws ++ [draw.points])
      |> Enum.take(-Hindsight.max_draws())

    {:noreply, assign(socket, draws: draws, draw_count: count + 1, draw_exec: draw.execution)}
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
          eyebrow="Engine · chapter 3"
          title="The power of hindsight"
        >
          This is the same wandering level you tuned in chapter 2, with the true Q and R.
          The filter's estimate at step 40 used only steps 1–40; it couldn't know the
          future. But the series is over now; we do know it. The Rauch–Tung–Striebel
          smoother walks backwards through the filter's output, revising every estimate
          in light of everything that came after. Same data, same model; strictly less
          uncertainty.
        </.section_heading>

        <div class="mt-5 flex flex-wrap items-center gap-x-6 gap-y-3">
          <form phx-change="toggle" class="flex items-center gap-2">
            <input type="hidden" name="smoothed" value="false" />
            <input
              type="checkbox"
              id="smoothed-toggle"
              name="smoothed"
              value="true"
              checked={@show_smoothed}
              class="toggle toggle-sm"
            />
            <label
              for="smoothed-toggle"
              class="font-data text-xs uppercase tracking-wider text-(--color-ink-soft)"
            >
              Turn on hindsight; overlay the RTS smoother
            </label>
          </form>

          <div class="flex items-center gap-3">
            <button type="button" phx-click="draw" class="btn btn-sm btn-outline font-data text-xs">
              Draw another possible history
            </button>
            <span :if={@draw_count > 0} class="font-data text-xs text-(--color-ink-faint)">
              {min(@draw_count, Hindsight.max_draws())} of {Hindsight.max_draws()} kept
            </span>
            <.exec_badge :if={@draw_exec} execution={@draw_exec} />
          </div>
        </div>

        <.figure
          no="1"
          caption={figure_caption(@show_smoothed, @draws, @revealed)}
          execution={@prepared.execution}
        >
          <:legend>
            <.swatch color={:ink} label="observed" />
            <.swatch color={:model} label="filtered" dash />
            <.swatch :if={@show_smoothed} color={:model} label="smoothed" />
            <.swatch :if={@draws != []} color={:model} label="posterior draws" />
            <.swatch :if={@revealed} color={:truth} label="latent level" />
          </:legend>
          <.line_figure
            id="smoother-chart"
            title="Filtered and smoothed latent level"
            desc="Observed values are compared with filtered estimates, smoothed hindsight estimates, optional posterior histories, and the latent truth when revealed."
            height={340}
            y_label="level"
            series={chart_series(@prepared, @draws, @show_smoothed, @revealed)}
          />
        </.figure>

        <div class="grid gap-4 sm:grid-cols-[1fr_minmax(240px,0.8fr)]">
          <.verdict_card
            verdict={"Hindsight shrinks the model's own uncertainty by #{fmt1(sd_cut_pct(@prepared))}%."}
            tone={:neutral}
          >
            No new data, no new model; the smoother just refuses to pretend it doesn't
            know the future. Each thin blue line is one complete history drawn from the
            posterior; the smoothed line is their mean, not "the" answer.
            <:stat label="Filtered avg ±sd" value={"±#{fmt2(@prepared.avg_sd_filt)}"} />
            <:stat label="Smoothed avg ±sd" value={"±#{fmt2(@prepared.avg_sd_smooth)}"} />
            <:stat
              :if={@revealed}
              label="Filtered RMSE vs truth"
              value={fmt2(@prepared.rmse_filt)}
            />
            <:stat
              :if={@revealed}
              label="Smoothed RMSE vs truth"
              value={fmt2(@prepared.rmse_smooth)}
            />
          </.verdict_card>

          <.reveal_truth
            revealed={@revealed}
            truth="The green line is the latent level we generated; the exact path hidden under the noise."
            grade={grade_line(@prepared)}
          />
        </div>

        <p class="report-prose mt-6 text-sm text-(--color-ink-soft)">
          Keep pressing the draw button. Every trajectory is a history that is fully
          consistent with the observations and the model; the posterior is a
          distribution over pasts, and the spread of those lines is honest uncertainty
          you'd never see from a single smoothed curve. This is the machinery MCMC runs
          thousands of times: the next chapter's Gibbs sampler draws a history like
          these, then uses it to re-learn Q and R.
        </p>

        <.under_the_hood code={Hindsight.code_snippet()} />

        <div class="mt-8 grid gap-4 md:grid-cols-3">
          <.cross_link
            navigate={~p"/engine/kalman"}
            eyebrow="Engine · chapter 2"
            title="Tracking a signal through noise"
          >
            The forward pass this chapter revises; tune Q and R yourself.
          </.cross_link>
          <.cross_link
            navigate={~p"/engine/noise"}
            eyebrow="Engine · chapter 1"
            title="Why raw data lies"
          >
            The quiz that motivates all of this: eyes vs filter on pure noise.
          </.cross_link>
          <.cross_link
            navigate={~p"/engine/gibbs"}
            eyebrow="Engine · chapter 4"
            title="Learning the noise"
          >
            We handed the smoother the true Q and R. The Gibbs sampler earns them; watch it converge.
          </.cross_link>
        </div>
      </div>
    </Layouts.app>
    """
  end

  defp chart_series(prepared, draws, show_smoothed, revealed) do
    observed = [%{points: prepared.observations, color: :ink, width: 1.2, opacity: 0.5}]

    spaghetti =
      Enum.map(draws, fn points ->
        %{points: points, color: :model, width: 1.0, opacity: 0.2}
      end)

    filtered = [%{points: prepared.filt_means, color: :model, dash: true, width: 1.6}]

    smoothed =
      if show_smoothed do
        [%{points: prepared.smooth_means, color: :model, width: 2.2}]
      else
        []
      end

    truth =
      if revealed do
        [%{points: prepared.latent, color: :truth, width: 2.0}]
      else
        []
      end

    observed ++ spaghetti ++ filtered ++ smoothed ++ truth
  end

  defp figure_caption(show_smoothed, draws, revealed) do
    smoothed_part =
      if show_smoothed do
        "; solid: the RTS smoother's backward revision."
      else
        "; toggle above to overlay the smoother's backward revision."
      end

    draws_part =
      if draws == [] do
        ""
      else
        " Thin lines: #{length(draws)} complete histories drawn from the posterior."
      end

    truth_part = if revealed, do: " Green: the planted latent level.", else: ""

    "Filtered estimate of the latent level (dashed), one forward pass" <>
      smoothed_part <> draws_part <> truth_part
  end

  defp sd_cut_pct(prepared) do
    (1.0 - prepared.avg_sd_smooth / prepared.avg_sd_filt) * 100.0
  end

  defp grade_line(prepared) do
    cut = (1.0 - prepared.rmse_smooth / prepared.rmse_filt) * 100.0

    "Graded against it: filtered RMSE #{fmt2(prepared.rmse_filt)}, smoothed RMSE #{fmt2(prepared.rmse_smooth)}; hindsight cut the error by #{fmt1(cut)}%."
  end

  defp fmt1(v) when is_number(v), do: :erlang.float_to_binary(v * 1.0, decimals: 1)
  defp fmt2(v) when is_number(v), do: :erlang.float_to_binary(v * 1.0, decimals: 2)
end
