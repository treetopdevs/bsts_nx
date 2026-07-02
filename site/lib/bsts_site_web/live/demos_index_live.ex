defmodule BstsSiteWeb.DemosIndexLive do
  @moduledoc """
  The Questions track index: five business questions, each answered live by
  the same engine on planted-truth data.
  """
  use BstsSiteWeb, :live_view

  @impl true
  def mount(_params, _session, socket) do
    {:ok, assign(socket, page_title: "Five questions")}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <Layouts.app flash={@flash} track={:questions}>
      <div class="mx-auto max-w-4xl">
        <.section_heading eyebrow="Track one · Questions" title="Five questions, answered live">
          Every page here is a business question that is secretly the same statistical
          question: the numbers changed — was it you, or was it noise? Each demo plants a
          truth in synthetic data, estimates without peeking, and lets you grade the answer.
          Pick the question that sounds like yours.
        </.section_heading>

        <div class="mt-8 grid gap-4 sm:grid-cols-2">
          <.cross_link
            navigate={~p"/demos/marketing"}
            eyebrow="Marketing"
            title="Did the campaign actually work?"
          >
            A lift planted in daily conversions, the fast lane's instant read, then the full
            Bayesian run — you watch what the extra second buys.
          </.cross_link>
          <.cross_link
            navigate={~p"/demos/tv"}
            eyebrow="TV attribution"
            title="Which TV spots earned their airtime?"
          >
            Overlapping spot windows on a timeline, a forecast-first counterfactual, and
            Shapley values splitting the credit — recomputed as you move the spots.
          </.cross_link>
          <.cross_link
            navigate={~p"/demos/demand"}
            eyebrow="Demand planning"
            title="How much should we stock?"
          >
            A forecast fan drawn from posterior samples, and safety stock read off the
            interval instead of the point estimate.
          </.cross_link>
          <.cross_link
            navigate={~p"/demos/anomaly"}
            eyebrow="Ops monitoring"
            title="Is this spike an anomaly?"
          >
            Plant latency spikes yourself and watch a streaming filter decide which ones
            deserve a page — with a false-alarm budget you control.
          </.cross_link>
          <.cross_link
            navigate={~p"/demos/policy"}
            eyebrow="Policy evaluation"
            title="Did the policy change anything?"
          >
            An interrupted time series verdict on a planted policy break — including the
            honest "can't tell" when the signal drowns.
          </.cross_link>
        </div>

        <div class="mt-14 text-center">
          <p class="font-data text-xs uppercase tracking-widest text-(--color-ink-faint)">
            Non-linear by design — start anywhere
          </p>
          <div class="mt-4 grid gap-4 md:grid-cols-2">
            <.cross_link navigate={~p"/engine"} eyebrow="How does it work?" title="Open the engine">
              Six short chapters from "noise lies to you" to the world that didn't
              happen — the machinery behind every demo above.
            </.cross_link>
            <.cross_link navigate={~p"/trust"} eyebrow="Should I trust it?" title="Grade the homework">
              Planted truths graded systematically, placebo tests, and a plain list of what
              this library doesn't do yet.
            </.cross_link>
          </div>
        </div>
      </div>
    </Layouts.app>
    """
  end
end
