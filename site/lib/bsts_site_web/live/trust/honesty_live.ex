defmodule BstsSiteWeb.Trust.HonestyLive do
  @moduledoc """
  Trust: what this is and isn't. The page with no demo — just the facts about
  the library's maturity, lineage, license, conventions, and gaps.
  """
  use BstsSiteWeb, :live_view

  @impl true
  def mount(_params, _session, socket) do
    {:ok, assign(socket, page_title: "What this is and isn't")}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <Layouts.app flash={@flash} track={:trust}>
      <div class="mx-auto max-w-3xl">
        <.section_heading eyebrow="Trust · the fine print" title="What this is and isn't">
          Every other page on this site tries to convince you with live computation.
          This one just states facts. If you are deciding whether to build on <code>BstsNx</code>, read this page last — it is the honest edge of everything
          you saw before it.
        </.section_heading>

        <h3 class="font-display mt-8 text-xl font-bold">Where it comes from</h3>
        <div class="report-prose mt-2 text-(--color-ink-soft)">
          <p>
            BstsNx is an early-stage library, version 0.1.0. It reimplements the ideas of
            Bayesian structural time series and counterfactual causal impact — the
            approach of Brodersen et al. — in pure Elixir on Nx. The R packages <code>bsts</code>
            and <code>CausalImpact</code>
            are used as reference
            implementations: the library's test suite includes an optional, offline
            R-parity lane that runs the same analyses through CRAN's stack via
            <code>BstsNx.RSidecar</code>
            and compares results. That sidecar is validation
            tooling — it requires a local R installation, is opt-in, and is not part of
            anything running on this site. No claim is made of parity with or superiority
            to the R stack; R is the yardstick, not the competition.
          </p>
          <p>
            The license is LGPL-2.1-only. The library is not yet published on Hex — you
            install it as a git dependency; the
            <.link
              navigate={~p"/start"}
              class="underline underline-offset-4 text-(--color-model) hover:text-(--color-ink)"
            >start page</.link>
            has the exact lines.
          </p>
        </div>

        <h3 class="font-display mt-8 text-xl font-bold">The conventions, stated once</h3>
        <div class="report-prose mt-2 text-(--color-ink-soft)">
          <p>
            Index semantics cause the quietest bugs, so the library's top-level moduledoc
            states them explicitly. Quoting it verbatim:
          </p>
          <blockquote class="border-l-2 border-(--color-grid) pl-4">
            <p>Be explicit with index semantics:</p>
            <ul>
              <li>
                <code>pre_period</code>
                and <code>post_period</code>
                are <strong>1-based inclusive</strong>
              </li>
              <li>
                spot windows are <strong>0-based half-open</strong>
                (<code>[window_start, window_end)</code>)
              </li>
            </ul>
          </blockquote>
          <p>
            One consequence worth spelling out: spot windows are indexed <em>within the
            post-period</em>, not within the whole series. Every demo on this site
            follows these conventions; if a figure caption says "days 91–120", that is
            the 1-based inclusive reading.
          </p>
        </div>

        <h3 class="font-display mt-8 text-xl font-bold">What's not here yet</h3>
        <div class="report-prose mt-2 text-(--color-ink-soft)">
          <p>From the project's own README and docs, the current limitations:</p>
          <ul>
            <li>
              The compiled <code>defn</code> filter and smoother paths are
              scalar-oriented; multi-dimensional state runs through the eager
              implementations.
            </li>
            <li>
              The structured Gibbs samplers currently learn a diagonal <code>Q</code> and
              a scalar observation variance <code>R</code>, and structured MCMC targets
              scalar observations per time step — full multivariate-observation support
              is roadmap, not release.
            </li>
            <li>
              Structured MCMC on CPU is slow — minutes per run for composed seasonal
              models. That is why this site never runs it live: pages that sample use the
              scalar path, and pages that need structure use the filter lane. No figure
              here fakes a structured posterior.
            </li>
            <li>
              EMLX GPU execution for structured workflows is limited by missing linear
              algebra primitives; EMLX CPU, EXLA, and the pure BinaryBackend are the
              practical paths today.
            </li>
            <li>Not on Hex yet, as above.</li>
          </ul>
        </div>

        <h3 class="font-display mt-8 text-xl font-bold">The fast lane, honestly</h3>
        <div class="report-prose mt-2 text-(--color-ink-soft)">
          <p>
            The library has two millisecond-scale paths, and they are not the same thing.
            <code>BstsNx.Operational</code>
            is forecast-first: it fits the pre-period,
            then projects the post-period counterfactual <em>without ever observing
            post-period outcomes</em>. That is an honest counterfactual.
            <code>BstsNx.CausalImpact.estimate_from_filter/3</code>
            is different, and its
            own docstring says so plainly:
          </p>
          <blockquote class="border-l-2 border-(--color-grid) pl-4">
            "RTS smoothing uses all non-masked observations, including those after the
            intervention period. The baseline is therefore an interpolation rather than a
            pure counterfactual forecast. This is acceptable when post-intervention
            observations outside the masked window are unaffected by treatment, but can
            introduce bias otherwise."
          </blockquote>
          <p>
            On this site, the fast figures label which one they use. When a page says
            "counterfactual", it is the forecast-first lane; anything built on
            <code>estimate_from_filter</code>
            is called a fast interpolation, because
            that is what it is.
          </p>
        </div>

        <.under_the_hood
          code={two_lanes_code()}
          title="The two fast paths — one never peeks, one does"
        />

        <h3 class="font-display mt-8 text-xl font-bold">About the figures on this site</h3>
        <div class="report-prose mt-2 text-(--color-ink-soft)">
          <p>
            Every figure is computed server-side by the library at the moment you load or
            interact with the page — there are no screenshots, no cached results, no
            JavaScript chart library. The small badge under each figure
            ("<span class="font-data">method · elapsed · computed live on this server</span>")
            is not site copy: it renders the <code>execution</code>
            metadata map that the
            library itself returns — <code>method_used</code>
            and <code>elapsed_ms</code>
            as reported by <code>BstsNx.Execution</code>. Where a page runs MCMC, the
            elapsed time is the measured wall time of the sampling you just triggered.
          </p>
        </div>

        <div class="mt-10 grid gap-4 md:grid-cols-3">
          <.cross_link
            navigate={~p"/speed"}
            eyebrow="The lanes, measured"
            title="The two lanes"
          >
            Milliseconds vs seconds, with the trade both lanes are making.
          </.cross_link>
          <.cross_link
            navigate={~p"/trust/calibration"}
            eyebrow="Don't take our word"
            title="The planted truth, systematically"
          >
            Five planted lifts, five fits, every interval graded in public.
          </.cross_link>
          <.cross_link navigate={~p"/start"} eyebrow="If you're in" title="Install &amp; quick start">
            The git dependency, the first model, and the sixty-second version.
          </.cross_link>
        </div>
      </div>
    </Layouts.app>
    """
  end

  defp two_lanes_code do
    ~S"""
    # Forecast-first: fits the pre-period, projects days 97-144 without
    # ever seeing them. An honest counterfactual, in milliseconds.
    prepared = BstsNx.Operational.prepare(spec, {1, 96}, {97, 144})
    result = BstsNx.Operational.run(prepared, observations, [], return: :lists)
    result.execution.method_used
    #=> :scalar_forecast_filter

    # Mask-and-smooth: masks the window, then RTS-smooths across it using
    # data from BOTH sides. Fast, useful — and an interpolation, not a
    # counterfactual forecast. The docstring says so; so do we.
    BstsNx.CausalImpact.estimate_from_filter(
      observations,
      Enum.to_list(96..143),   # 0-based intervention indices, not period tuples
      q: 0.8, r: 16.0, x0: 84.0, p0: 10.0
    )
    """
  end
end
