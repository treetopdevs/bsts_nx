defmodule BstsSiteWeb.EngineIndexLive do
  @moduledoc """
  The Engine track index: six chapters, one method; decompose, project,
  compare; each chapter a live instrument, not a diagram.
  """
  use BstsSiteWeb, :live_view

  @impl true
  def mount(_params, _session, socket) do
    {:ok, assign(socket, page_title: "The engine")}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <Layouts.app flash={@flash} track={:engine}>
      <div class="mx-auto max-w-4xl">
        <.section_heading level={1} eyebrow="Track two · The engine" title="Open the engine">
          One method underneath everything on this site: <strong>decompose</strong>
          the
          series into structure and noise, <strong>project</strong>
          the structure forward, <strong>compare</strong>
          the projection with what actually happened. Six chapters
          teach it one moving part at a time; every figure is computed live, and most of
          them let you turn the knobs yourself.
        </.section_heading>

        <div class="my-8 flex justify-center">
          <.mantra class="scale-110" />
        </div>

        <div class="grid gap-4 sm:grid-cols-2">
          <.cross_link
            navigate={~p"/engine/noise"}
            eyebrow="Chapter 1 · decompose"
            title="Why raw data lies"
          >
            Random walks with and without a planted shift. Guess by eye, then watch the
            filter find what you couldn't; the case for decomposing at all.
          </.cross_link>
          <.cross_link
            navigate={~p"/engine/kalman"}
            eyebrow="Chapter 2 · decompose"
            title="Tracking a signal through noise"
          >
            A Kalman filter with its two variance knobs exposed. Turn them and watch it
            trade trust between the model and the data, one observation at a time.
          </.cross_link>
          <.cross_link
            navigate={~p"/engine/smoother"}
            eyebrow="Chapter 3 · decompose"
            title="The power of hindsight"
          >
            The RTS smoother re-reads the past with the whole series in hand; the backward
            pass that quietly sharpens every estimate the filter made.
          </.cross_link>
          <.cross_link
            navigate={~p"/engine/gibbs"}
            eyebrow="Chapter 4 · compare"
            title="Honest about the unknown"
          >
            Real data doesn't print its variances on the label. Watch MCMC chains learn
            them; and get graded against each other with R-hat.
          </.cross_link>
          <.cross_link
            navigate={~p"/engine/compose"}
            eyebrow="Chapter 5 · decompose"
            title="Models as Lego bricks"
          >
            Snap level, trend, seasonality and regression specs into one block-diagonal
            model. Pure matrix assembly; you can see the blocks.
          </.cross_link>
          <.cross_link
            navigate={~p"/engine/counterfactual"}
            eyebrow="Chapter 6 · project → compare"
            title="The world that didn't happen"
          >
            Fit on the pre-period, then forward-simulate once per posterior draw: the
            spaghetti of worlds where you did nothing, and the gap that is your effect.
          </.cross_link>
        </div>

        <div class="mt-14 text-center">
          <p class="font-data text-xs uppercase tracking-widest text-(--color-ink-faint)">
            Non-linear by design; start anywhere
          </p>
          <div class="mt-4 grid gap-4 md:grid-cols-2">
            <.cross_link
              navigate={~p"/demos"}
              eyebrow="I have a question"
              title="Five questions, answered live"
            >
              The same engine wearing business clothes: campaigns, TV spots, stock levels,
              anomalies, policy breaks.
            </.cross_link>
            <.cross_link
              navigate={~p"/trust"}
              eyebrow="Should I trust it?"
              title="Grade the homework"
            >
              Where the engine gets stress-tested: calibration sweeps, placebo tests, and
              plain talk about limits.
            </.cross_link>
          </div>
        </div>
      </div>
    </Layouts.app>
    """
  end
end
