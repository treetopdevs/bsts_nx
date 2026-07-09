defmodule BstsSiteWeb.Demos.MarketingLive do
  @moduledoc """
  /demos/marketing; "Did the campaign actually work?"

  Two lanes on the same planted data: an instant filter lane so the page is
  never empty, and the full Bayesian lane behind a button and the limiter.
  """
  use BstsSiteWeb, :live_view

  alias BstsSite.Demos.DefaultCache
  alias BstsSite.Demos.Limiter
  alias BstsSite.Demos.Marketing

  @impl true
  def mount(_params, _session, socket) do
    %{scenario: scenario, fast: fast} =
      DefaultCache.get(:marketing, fn ->
        defaults = Marketing.defaults()
        scenario = Marketing.scenario(defaults.lift, defaults.noise_sd)
        %{scenario: scenario, fast: Marketing.fast_lane(scenario)}
      end)

    {:ok,
     socket
     |> assign(page_title: "Did the campaign actually work?")
     |> assign(scenario: scenario, fast: fast, mcmc: nil)
     |> assign(running: false, busy: false, revealed: false)}
  end

  @impl true
  def handle_event("adjust", %{"lift" => lift, "noise" => noise}, socket) do
    scenario = Marketing.scenario(lift, noise)
    fast = Marketing.fast_lane(scenario)

    # New data invalidates any Bayesian result and reseals the answer key.
    {:noreply,
     assign(socket, scenario: scenario, fast: fast, mcmc: nil, revealed: false, busy: false)}
  end

  def handle_event("run_mcmc", _params, %{assigns: %{running: true}} = socket) do
    {:noreply, socket}
  end

  def handle_event("run_mcmc", _params, socket) do
    %{lift: lift, noise_sd: noise_sd} = socket.assigns.scenario
    socket = assign(socket, running: true, busy: false)

    {:noreply,
     start_async(socket, :mcmc, fn ->
       case Limiter.run(fn -> Marketing.fit(lift, noise_sd) end) do
         {:ok, result} -> result
         :busy -> :busy
       end
     end)}
  end

  def handle_event("reveal_truth", _params, socket) do
    {:noreply, assign(socket, revealed: true)}
  end

  @impl true
  def handle_async(:mcmc, {:ok, :busy}, socket) do
    {:noreply, assign(socket, running: false, busy: true)}
  end

  def handle_async(:mcmc, {:ok, result}, socket) do
    %{lift: lift, noise_sd: noise_sd} = socket.assigns.scenario

    if result.scenario.lift == lift and result.scenario.noise_sd == noise_sd do
      {:noreply, assign(socket, running: false, mcmc: result)}
    else
      # The sliders moved while this run was sampling; drop the stale result.
      {:noreply, assign(socket, running: false)}
    end
  end

  def handle_async(:mcmc, {:exit, _reason}, socket) do
    {:noreply, assign(socket, running: false, busy: true)}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <Layouts.app flash={@flash} track={:questions}>
      <section class="mx-auto max-w-4xl">
        <.section_heading
          level={1}
          eyebrow="Did the campaign actually work?"
          title="Ninety days of conversions, one campaign, two lanes."
        >
          We generated 90 days of daily conversions around a steady 200/day and planted a
          campaign that starts on day 61. You choose how big the true lift is and how noisy
          the world is. The model never sees the truth; it learns the baseline from the
          first 60 days, projects what should have happened, and measures the gap.
          Then you grade it.
        </.section_heading>
        <.mantra active={:compare} class="mb-4" />

        <form id="marketing-controls" phx-change="adjust" class="grid gap-x-10 gap-y-3 sm:grid-cols-2">
          <.param_slider
            name="lift"
            label="Planted campaign lift"
            min={Marketing.lift_range().first}
            max={Marketing.lift_range().last}
            value={@scenario.lift}
            display={"+#{@scenario.lift} conversions/day"}
          />
          <.param_slider
            name="noise"
            label="Noise in the world"
            min={Marketing.noise_range().first}
            max={Marketing.noise_range().last}
            value={@scenario.noise_sd}
            display={"sd #{@scenario.noise_sd}"}
          />
        </form>

        <.figure
          no="1"
          caption="Fast lane; the operational filter's forecast-first counterfactual, recomputed on every slider drag. The magenta shading is its estimated lift."
          execution={@fast.execution}
        >
          <:legend>
            <.swatch color={:ink} label="observed" />
            <.swatch color={:model} label="counterfactual" dash />
            <.swatch color={:lift} label="estimated lift" />
          </:legend>
          <.line_figure
            id="marketing-fast"
            title="Fast marketing lift estimate"
            desc="Daily conversions are compared with a forecast-first counterfactual; the post-campaign gap is the fast lift estimate."
            height={300}
            y_label="conversions/day"
            series={[
              %{points: @scenario.observations, color: :ink, width: 1.5},
              %{
                points: @fast.counterfactual_mean,
                x_offset: @scenario.pre_end,
                color: :model,
                dash: true,
                width: 2.0
              }
            ]}
            bands={[
              %{
                lower: @fast.band_lower,
                upper: @fast.band_upper,
                x_offset: @scenario.pre_end,
                fill: "var(--color-model-band)"
              },
              %{
                lower: @fast.counterfactual_mean,
                upper: Enum.drop(@scenario.observations, @scenario.pre_end),
                x_offset: @scenario.pre_end,
                fill: "var(--color-lift-band)"
              }
            ]}
            vlines={[%{x: @scenario.pre_end, label: "campaign starts"}]}
          />
        </.figure>

        <.verdict_card
          verdict={fast_verdict(@fast)}
          tone={if @fast.significant?, do: :lift, else: :neutral}
        >
          <p>
            This is the millisecond lane; a Kalman filter we hand the noise level you dialed
            in. A first impression, not a measurement. The Bayesian lane below has to learn
            the noise from the data itself, and reports a full posterior for its trouble.
          </p>
          <:stat
            label="Cumulative effect"
            value={"#{signed(@fast.cumulative.mean)} conversions"}
            hint={"95% CI [#{fmt(@fast.cumulative.lower)}, #{fmt(@fast.cumulative.upper)}]"}
          />
          <:stat label="Relative lift" value={pct(@fast.relative.mean)} />
        </.verdict_card>
      </section>

      <section class="mx-auto mt-12 max-w-4xl">
        <.section_heading eyebrow="The slow lane" title="Run the full Bayesian analysis">
          <code>MarketingLift.measure_lift</code> fits the pre-campaign baseline by Gibbs
          sampling; {Marketing.num_samples()} posterior draws of the level and both
          variances; then simulates {Marketing.num_samples()} counterfactual futures for
          the campaign window. It takes about a second on this server, and unlike the fast
          lane it earns its uncertainty instead of being told it.
        </.section_heading>

        <div class="flex flex-wrap items-center gap-3">
          <button
            :if={!@running}
            type="button"
            phx-click="run_mcmc"
            class="btn btn-primary font-data text-sm"
          >
            Run the full Bayesian analysis
          </button>
          <span
            :if={@running}
            role="status"
            aria-live="polite"
            aria-busy="true"
            class="font-data motion-safe:animate-pulse text-sm text-(--color-ink-soft)"
          >
            Sampling {Marketing.num_samples()} posterior draws on this server…
          </span>
          <span :if={@mcmc && !@running} class="font-data text-xs text-(--color-ink-faint)">
            Done in {@mcmc.execution.elapsed_ms} ms. Move a slider to start over.
          </span>
        </div>

        <.verdict_card
          :if={@busy}
          class="mt-4"
          role="status"
          aria-live="polite"
          tone={:warning}
          verdict="Another visitor is sampling, try again in a moment."
        >
          <p>MCMC runs are limited to a few at a time so the demo stays responsive.</p>
        </.verdict_card>

        <div :if={@mcmc}>
          <.figure
            no="2"
            caption={"Bayesian lane; the posterior counterfactual (mean and 95% band across #{@mcmc.num_samples} draws). The magenta shading between it and the observed line is the estimated lift."}
            execution={@mcmc.execution}
          >
            <:legend>
              <.swatch color={:ink} label="observed" />
              <.swatch color={:model} label="posterior counterfactual" dash />
              <.swatch color={:lift} label="estimated lift" />
            </:legend>
            <.line_figure
              id="marketing-mcmc"
              title="Bayesian marketing lift estimate"
              desc="Daily conversions are compared with the posterior counterfactual mean and effect band from the full Bayesian analysis."
              height={300}
              y_label="conversions/day"
              series={[
                %{points: @mcmc.scenario.observations, color: :ink, width: 1.5},
                %{
                  points: @mcmc.counterfactual_mean,
                  x_offset: @mcmc.scenario.pre_end,
                  color: :model,
                  dash: true,
                  width: 2.0
                }
              ]}
              bands={[
                %{
                  lower: @mcmc.band_lower,
                  upper: @mcmc.band_upper,
                  x_offset: @mcmc.scenario.pre_end,
                  fill: "var(--color-model-band)"
                },
                %{
                  lower: @mcmc.counterfactual_mean,
                  upper: Enum.drop(@mcmc.scenario.observations, @mcmc.scenario.pre_end),
                  x_offset: @mcmc.scenario.pre_end,
                  fill: "var(--color-lift-band)"
                }
              ]}
              vlines={[%{x: @mcmc.scenario.pre_end, label: "campaign starts"}]}
            />
          </.figure>

          <.figure
            no="3"
            caption={"The posterior distribution of cumulative lift; #{@mcmc.num_samples} draws. The spread is the answer, not a nuisance."}
            execution={@mcmc.execution}
          >
            <:legend>
              <.swatch color={:model} label="posterior draws" />
              <.swatch :if={@revealed} color={:truth} label="planted truth" dash />
            </:legend>
            <.histogram
              id="marketing-posterior"
              title="Marketing lift posterior"
              desc="Posterior draws of cumulative campaign lift, with interval and planted-truth markers when revealed."
              values={@mcmc.cumulative_draws}
              bins={24}
              vlines={histogram_vlines(@mcmc, @revealed)}
            />
          </.figure>

          <div class="grid gap-4 sm:grid-cols-[1fr_minmax(240px,0.8fr)]">
            <.verdict_card
              verdict={mcmc_verdict(@mcmc)}
              tone={if @mcmc.significant?, do: :lift, else: :neutral}
            >
              <:stat
                label="Cumulative lift"
                value={"#{signed(@mcmc.effect.cumulative)} conversions"}
                hint={"95% CI [#{fmt(@mcmc.effect.cumulative_lower)}, #{fmt(@mcmc.effect.cumulative_upper)}]"}
              />
              <:stat
                label="Relative lift"
                value={pct(@mcmc.effect.relative)}
                hint={"95% CI [#{pct(@mcmc.relative_lower)}, #{pct(@mcmc.relative_upper)}]"}
              />
              <:stat label="Daily average" value={"#{signed(@mcmc.effect.per_period)}/day"} />
            </.verdict_card>

            <.reveal_truth
              revealed={@revealed}
              truth={truth_line(@mcmc.scenario)}
              grade={grade_line(@mcmc)}
            />
          </div>
        </div>

        <p :if={!@mcmc && !@running} class="report-prose mt-4 text-sm text-(--color-ink-soft)">
          The answer key stays sealed until the Bayesian lane has committed to an estimate.
        </p>

        <.under_the_hood code={Marketing.code_snippet()} />
      </section>

      <section class="mx-auto mt-12 max-w-4xl">
        <div class="grid gap-4 md:grid-cols-3">
          <.cross_link
            navigate={~p"/demos/policy"}
            eyebrow="Same estimator, opposite sign"
            title="Did the policy change anything?"
          >
            Point the same machinery at a speed-limit change and it measures accidents
            <em>prevented</em>
            instead of conversions gained.
          </.cross_link>
          <.cross_link
            navigate={~p"/engine/counterfactual"}
            eyebrow="The engine underneath"
            title="Where the counterfactual comes from"
          >
            How a model that never saw the campaign projects what should have happened.
          </.cross_link>
          <.cross_link
            navigate={~p"/speed"}
            eyebrow="Why two lanes"
            title="Milliseconds vs. seconds"
          >
            The fast/slow trade you just watched, measured properly.
          </.cross_link>
        </div>
      </section>
    </Layouts.app>
    """
  end

  defp fast_verdict(%{significant?: true, cumulative: cum}) do
    "First impression: the campaign shows; #{signed(cum.mean)} conversions."
  end

  defp fast_verdict(%{significant?: false}) do
    "First impression: can't tell; the interval straddles zero."
  end

  defp mcmc_verdict(%{significant?: true, effect: %{cumulative: cum}}) when cum >= 0 do
    "Yes; the campaign added #{signed(cum)} conversions."
  end

  defp mcmc_verdict(%{significant?: true, effect: %{cumulative: cum}}) do
    "The model sees #{signed(cum)} conversions; a drop, not a lift."
  end

  defp mcmc_verdict(_mcmc) do
    "Honestly? Can't tell; the interval straddles zero."
  end

  defp histogram_vlines(mcmc, revealed) do
    zero = %{x: 0.0, label: "no effect", color: :ink}
    truth = %{x: mcmc.scenario.truth.cumulative_lift, label: "planted truth", color: :truth}

    if revealed, do: [zero, truth], else: [zero]
  end

  defp truth_line(%{truth: %{lift_per_day: lift}}) when lift == 0.0 do
    "Planted: nothing. The campaign has zero true effect; any lift the model reports is noise."
  end

  defp truth_line(%{truth: truth}) do
    "Planted: +#{trunc(truth.lift_per_day)}/day from day 61; #{signed(truth.cumulative_lift)} conversions over #{truth.post_days} days."
  end

  defp grade_line(mcmc) do
    truth = mcmc.scenario.truth.cumulative_lift
    %{cumulative_lower: lo, cumulative_upper: hi} = mcmc.effect

    if truth >= lo and truth <= hi do
      "The planted truth sits inside the model's 95% interval. Recovered."
    else
      "The planted truth fell outside the 95% interval this time; that should be rare. " <>
        "Intervals are promises about the long run, not every draw; see the calibration page."
    end
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
