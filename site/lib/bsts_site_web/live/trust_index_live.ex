defmodule BstsSiteWeb.TrustIndexLive do
  @moduledoc """
  The Trust track index: the site's honesty pitch — planted truths graded
  systematically, tools for breaking your own results, and plain talk about
  what the library doesn't do yet.
  """
  use BstsSiteWeb, :live_view

  @impl true
  def mount(_params, _session, socket) do
    {:ok, assign(socket, page_title: "Trust, but verify")}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <Layouts.app flash={@flash} track={:trust}>
      <div class="mx-auto max-w-4xl">
        <.section_heading eyebrow="Track three · Trust" title="Grade the homework">
          This site never asks you to take a model's word for it. Every demo runs on data we
          generated, with the answer sealed at generation time — so the model can be graded
          in public, and so can its uncertainty. These three pages are where the grading
          gets systematic: coverage spot-checked, in public, results deliberately broken,
          and the library's limits stated plainly.
        </.section_heading>

        <div class="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <.cross_link
            navigate={~p"/trust/calibration"}
            eyebrow="Coverage, counted"
            title="The planted truth, systematically"
          >
            Five planted lifts on one seeded series — one of them zero, a placebo in plain
            sight — every 95% interval graded against the sealed answer key. A live
            spot-check of the coverage promise; the systematic audit ships as <code>BstsNx.Validation.known_lift_injection/3</code>.
          </.cross_link>
          <.cross_link
            navigate={~p"/trust/diagnostics"}
            eyebrow="Sabotage, invited"
            title="Break your own results"
          >
            Placebo tests, coverage, residual and stability checks — five ways to knock
            over a finished fit, pointed at our own planted data first.
          </.cross_link>
          <.cross_link
            navigate={~p"/trust/honesty"}
            eyebrow="Limits, stated"
            title="What this is and isn't"
          >
            v0.1.0, plainly: validated against the R bsts / CausalImpact reference stack,
            not claiming to beat it — plus what's slow, and what's missing entirely.
          </.cross_link>
        </div>

        <div class="mt-14 text-center">
          <p class="font-data text-xs uppercase tracking-widest text-(--color-ink-faint)">
            Non-linear by design — start anywhere
          </p>
          <div class="mt-4 grid gap-4 md:grid-cols-2">
            <.cross_link
              navigate={~p"/demos"}
              eyebrow="I have a question"
              title="Five questions, answered live"
            >
              The demos these pages keep honest: campaigns, TV spots, stock levels,
              anomalies, policy breaks.
            </.cross_link>
            <.cross_link
              navigate={~p"/engine"}
              eyebrow="How does it work?"
              title="Open the engine"
            >
              The machinery being graded — six chapters, every figure alive.
            </.cross_link>
          </div>
        </div>
      </div>
    </Layouts.app>
    """
  end
end
