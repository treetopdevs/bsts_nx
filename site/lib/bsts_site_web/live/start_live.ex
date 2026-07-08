defmodule BstsSiteWeb.StartLive do
  @moduledoc """
  Get started: install, a sixty-second quick start, the which-question →
  which-module decision table, and where to go deeper. Static and
  terminal-quiet; no demos run here.
  """
  use BstsSiteWeb, :live_view

  # Every row verified against lib/bsts_nx/; modules and arities exist.
  @decision_table [
    {"Did this intervention have a causal effect?", "CausalImpact",
     "estimate/4 or estimate_structured/5"},
    {"Just tell me if it was significant", "InterventionAnalysis", "analyze/3"},
    {"Which marketing channel drove the most lift?", "Applications.MarketingLift",
     "measure_lift/3"},
    {"Did this policy change work?", "Applications.PolicyEvaluator", "evaluate/3"},
    {"Which TV spots deserve credit?", "Applications.TVAttribution", "attribute/3"},
    {"How much should we order next week?", "Applications.DemandForecaster",
     "forecast/2 + safety_stock/2"},
    {"What will happen in the next N periods?", "Forecaster", "fit_predict/3"},
    {"Is this data point abnormal?", "Applications.AnomalyDetector", "fit/2 + score/2"},
    {"Causal impact + attribution in one call", "Pipeline", "run/6"},
    {"Can I trust these results?", "Validation", "assess/1"}
  ]

  @impl true
  def mount(_params, _session, socket) do
    {:ok, assign(socket, page_title: "Get started")}
  end

  @impl true
  def render(assigns) do
    assigns =
      assign(assigns,
        hex_published: Application.get_env(:bsts_site, :hex_published),
        github_url: Application.get_env(:bsts_site, :github_url),
        decision_table: @decision_table
      )

    ~H"""
    <Layouts.app flash={@flash} track={:start}>
      <div class="mx-auto max-w-3xl">
        <.section_heading level={1} eyebrow="Get started" title="From install to a causal claim">
          BstsNx is a library, not a service; everything you've watched on this site is a
          function call you can make from your own code. This page is the practical part:
          install it, run the sixty-second version, and find the right entry point for your
          question.
        </.section_heading>

        <section class="mt-8">
          <h3 class="font-display text-xl font-bold">1 · Install</h3>
          <%= if @hex_published do %>
            <p class="report-prose mt-2 text-sm text-(--color-ink-soft)">
              Add the dependency to your <code>mix.exs</code>:
            </p>
            <div class="code-block mt-3">
              <pre><code>{hex_snippet()}</code></pre>
            </div>
          <% else %>
            <p class="report-prose mt-2 text-sm text-(--color-ink-soft)">
              Hex publishing is in progress; until it lands, use the git dependency form in
              your <code>mix.exs</code>:
            </p>
            <div class="code-block mt-3">
              <pre><code>{git_snippet(@github_url)}</code></pre>
            </div>
            <p
              :if={is_nil(@github_url)}
              class="mt-2 font-data text-xs text-(--color-ink-faint)"
            >
              The public source link is coming; swap in the repository URL when it's
              announced.
            </p>
          <% end %>

          <h4 class="font-data mt-5 text-xs uppercase tracking-wider text-(--color-ink-soft)">
            Requirements
          </h4>
          <ul class="report-prose mt-2 list-disc space-y-1 pl-4 text-sm text-(--color-ink-soft)">
            <li>Elixir <code>~> 1.19</code></li>
            <li><code>nx ~> 0.12.0</code>; the core numerical dependency</li>
            <li>
              Optional accelerators: <code>exla ~> 0.12.0</code>
              (JIT compilation) and <code>emlx ~> 0.3.0</code>
              (MLX-backed execution)
            </li>
          </ul>
          <p class="report-prose mt-2 text-sm text-(--color-ink-soft)">
            One warning worth repeating from the README: if your app or <code>Mix.install/2</code>
            script pins <code>nx</code>
            below 0.12 (a lot of older
            examples pin 0.6.x), dependency resolution will fail. Upgrade the full graph to
            Nx 0.12.
          </p>
        </section>

        <section class="mt-10">
          <h3 class="font-display text-xl font-bold">2 · The sixty-second quick start</h3>
          <p class="report-prose mt-2 text-sm text-(--color-ink-soft)">
            The lowest floor of the library: a Kalman filter over a local-level model.
            Seven arguments; observations, then F, H, Q, R, and the initial state and
            covariance. It returns one <code>{"{state, covariance}"}</code> pair per
            observation.
          </p>
          <div class="code-block mt-3">
            <pre><code>{kalman_snippet()}</code></pre>
          </div>
          <p class="report-prose mt-4 text-sm text-(--color-ink-soft)">
            And the highest floor: hand <code>CausalImpact.estimate/4</code> a series and
            two 1-based inclusive periods; pre for fitting, post for judging; and it fits
            the pre-period by Gibbs sampling, projects the counterfactual, and measures the
            gap. Seed it and the answer is reproducible.
          </p>
          <div class="code-block mt-3">
            <pre><code>{causal_impact_snippet()}</code></pre>
          </div>
        </section>

        <section class="mt-10">
          <h3 class="font-display text-xl font-bold">3 · Which question → which module</h3>
          <p class="report-prose mt-2 text-sm text-(--color-ink-soft)">
            Every high-level API is the same engine; decompose, project, compare; wearing
            different domain clothes. Start from your question; all modules live under the
            <code>BstsNx.</code>
            namespace.
          </p>
          <div class="mt-4 overflow-x-auto rounded-md border border-(--color-grid)">
            <table class="w-full text-left text-sm">
              <thead>
                <tr class="border-b border-(--color-grid) bg-(--color-paper-raised)">
                  <th class="px-4 py-2.5 font-data text-[0.65rem] font-medium uppercase tracking-wider text-(--color-ink-faint)">
                    Your question
                  </th>
                  <th class="px-4 py-2.5 font-data text-[0.65rem] font-medium uppercase tracking-wider text-(--color-ink-faint)">
                    Module
                  </th>
                  <th class="px-4 py-2.5 font-data text-[0.65rem] font-medium uppercase tracking-wider text-(--color-ink-faint)">
                    Key function
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr
                  :for={{question, module, fun} <- @decision_table}
                  class="border-b border-(--color-grid) last:border-b-0"
                >
                  <td class="px-4 py-2.5 text-(--color-ink)">{question}</td>
                  <td class="px-4 py-2.5 font-data text-xs text-(--color-ink-soft)">{module}</td>
                  <td class="px-4 py-2.5 font-data text-xs text-(--color-ink-soft)">{fun}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p class="report-prose mt-3 text-sm text-(--color-ink-soft)">
            Two supporting names worth knowing: <code>SpotAttributor</code> splits measured
            lift across overlapping windows, and <code>ShapleyAllocator</code> is the
            Shapley engine underneath it (exact for small groups, Monte-Carlo for large
            ones).
          </p>
        </section>

        <section class="mt-10">
          <h3 class="font-display text-xl font-bold">4 · Going deeper</h3>
          <div class="report-prose mt-2 space-y-3 text-sm text-(--color-ink-soft)">
            <p>
              Two Livebooks ship with the repository as companion deep-dives, and every idea
              on this site appears in them with runnable cells:
            </p>
            <ul class="list-disc space-y-2 pl-4">
              <li>
                <strong>From noisy data to causal claims</strong>
                (<code>livebooks/bsts_nx_guide.livemd</code>); the full arc: synthetic data,
                Kalman filtering, Gibbs sampling, causal impact, attribution, validation.
              </li>
              <li>
                <strong>BSTS for web engineers</strong>
                (<code>livebooks/bsts_nx_statistical_journey.livemd</code>); the same
                machinery built up from intuition, for readers who think in requests and
                latencies rather than posteriors.
              </li>
            </ul>
            <p>
              Backend acceleration is optional and orthogonal: the library runs anywhere
              Elixir runs on the pure-Elixir <code>Nx.BinaryBackend</code> (that's what this
              site uses), and EXLA can JIT-compile the <code>defn</code> fast paths in your
              own deployment:
            </p>
          </div>
          <div class="code-block mt-3">
            <pre><code>config :nx, :default_backend, EXLA.Backend</code></pre>
          </div>
          <p class="report-prose mt-4 text-sm text-(--color-ink-soft)">
            BstsNx is early-stage (v0.1.0), validated against the R <code>bsts</code>
            / <code>CausalImpact</code>
            reference stack, and licensed under <strong>LGPL-2.1-only</strong>.
          </p>
        </section>

        <div class="mt-10 grid gap-4 md:grid-cols-3">
          <.cross_link navigate={~p"/"} eyebrow="Start at the question" title="The hub">
            The one question this library answers, computed live on a slider.
          </.cross_link>
          <.cross_link
            navigate={~p"/engine"}
            eyebrow="Understand the machinery"
            title="Open the engine"
          >
            Six chapters from "noise lies to you" to the world that didn't happen.
          </.cross_link>
          <.cross_link navigate={~p"/speed"} eyebrow="Plan your deployment" title="The two lanes">
            Which calls are request-path safe and which belong in a background job; timed
            live.
          </.cross_link>
        </div>
      </div>
    </Layouts.app>
    """
  end

  # -- Install snippets (config-dependent) ------------------------------------

  defp hex_snippet do
    ~S"""
    def deps do
      [
        {:bsts_nx, "~> 0.1"}
      ]
    end
    """
  end

  defp git_snippet(nil) do
    ~S"""
    def deps do
      [
        {:bsts_nx, git: "<repo-url>"}
      ]
    end
    """
  end

  defp git_snippet(github_url) do
    """
    def deps do
      [
        {:bsts_nx, git: "#{github_url}"}
      ]
    end
    """
  end

  # -- Quick-start snippets (mirrors the README; verified against lib) --------

  defp kalman_snippet do
    ~S"""
    obs = [1.0, 2.0, 3.0]

    {xs, _ps} =
      BstsNx.KalmanFilter.filter(obs, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0)
      |> Enum.unzip()

    Enum.map(xs, &Nx.to_number/1)
    #=> [0.666..., 1.5, 2.428...]
    """
  end

  defp causal_impact_snippet do
    ~S"""
    pre = Enum.map(1..60, fn _ -> 50.0 + :rand.normal() * 5 end)
    post = Enum.map(1..30, fn _ -> 60.0 + :rand.normal() * 5 end)
    obs = pre ++ post

    result =
      BstsNx.CausalImpact.estimate(obs, {1, 60}, {61, 90},
        num_samples: 100,
        burn_in: 50,
        seed: 42
      )

    summary = BstsNx.CausalImpact.summary(result)
    summary.cumulative_effect
    #=> %{mean: ..., sd: ..., lower: ..., upper: ...}
    """
  end
end
