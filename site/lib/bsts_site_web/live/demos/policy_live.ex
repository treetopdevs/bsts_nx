defmodule BstsSiteWeb.Demos.PolicyLive do
  @moduledoc """
  /demos/policy; "Did the policy change anything?"

  An interrupted time series: the same estimator the marketing page uses,
  pointed at a planted *reduction*. The MCMC run sits behind the limiter.
  """
  use BstsSiteWeb, :live_view

  alias BstsSite.Demos.Limiter
  alias BstsSite.Demos.Policy

  @impl true
  def mount(_params, _session, socket) do
    defaults = Policy.defaults()
    scenario = Policy.scenario(defaults.reduction, defaults.noise_sd)

    {:ok,
     socket
     |> assign(page_title: "Did the policy change anything?")
     |> assign(scenario: scenario, mcmc: nil)
     |> assign(running: false, busy: false, revealed: false)}
  end

  @impl true
  def handle_event("adjust", %{"reduction" => reduction, "noise" => noise}, socket) do
    scenario = Policy.scenario(reduction, noise)

    # New data invalidates any fitted result and reseals the answer key.
    {:noreply, assign(socket, scenario: scenario, mcmc: nil, revealed: false, busy: false)}
  end

  def handle_event("run_mcmc", _params, %{assigns: %{running: true}} = socket) do
    {:noreply, socket}
  end

  def handle_event("run_mcmc", _params, socket) do
    %{reduction: reduction, noise_sd: noise_sd} = socket.assigns.scenario
    socket = assign(socket, running: true, busy: false)

    {:noreply,
     start_async(socket, :mcmc, fn ->
       case Limiter.run(fn -> Policy.fit(reduction, noise_sd) end) do
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
    %{reduction: reduction, noise_sd: noise_sd} = socket.assigns.scenario

    if result.scenario.reduction == reduction and result.scenario.noise_sd == noise_sd do
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
          eyebrow="Did the policy change anything?"
          title="A speed limit changes. Do the accident counts?"
        >
          We generated 48 months of accident counts around a steady 120/month and planted a
          speed-limit change at month 37. You choose how many accidents the change truly
          prevents each month, and how noisy the counts are. This is an interrupted time
          series; the same estimator the marketing demo runs, except here the interesting
          effect is negative.
        </.section_heading>
        <.mantra active={:compare} class="mb-4" />

        <form id="policy-controls" phx-change="adjust" class="grid gap-x-10 gap-y-3 sm:grid-cols-2">
          <.param_slider
            name="reduction"
            label="Planted monthly change"
            min={Policy.reduction_range().first}
            max={Policy.reduction_range().last}
            value={@scenario.reduction}
            display={"#{@scenario.reduction} accidents/month"}
          />
          <.param_slider
            name="noise"
            label="Noise in the counts"
            min={Policy.noise_range().first}
            max={Policy.noise_range().last}
            value={@scenario.noise_sd}
            display={"sd #{@scenario.noise_sd}"}
          />
        </form>

        <.figure
          no="1"
          caption={series_caption(@mcmc)}
          execution={@mcmc && @mcmc.execution}
        >
          <:legend>
            <.swatch color={:ink} label="observed" />
            <.swatch :if={@mcmc} color={:model} label="posterior counterfactual" dash />
            <.swatch :if={@mcmc} color={:lift} label="estimated effect" />
          </:legend>
          <.line_figure
            id="policy-series"
            title="Policy intervention series"
            desc="Monthly accident counts are shown before and after the policy, with model counterfactual and effect overlays after fitting."
            height={300}
            y_label="accidents/month"
            series={series_lines(@scenario, @mcmc)}
            bands={series_bands(@scenario, @mcmc)}
            vlines={[%{x: @scenario.pre_end, label: "policy takes effect"}]}
          />
        </.figure>

        <p :if={!@mcmc} class="report-prose text-sm text-(--color-ink-soft)">
          Before any analysis, all you can honestly say is that the line looks different
          after month 37; or doesn't, depending on your sliders. Whether that difference
          is the policy or just noise is exactly the question the model answers.
        </p>
      </section>

      <section class="mx-auto mt-12 max-w-4xl">
        <.section_heading eyebrow="Interrupted time series" title="Run the evaluation">
          <code>PolicyEvaluator.evaluate</code> fits the 36 pre-change months by Gibbs
          sampling; {Policy.num_samples()} posterior draws; projects the counterfactual
          "no policy change" world across the last 12 months, and measures the gap. About
          a second on this server.
        </.section_heading>

        <div class="flex flex-wrap items-center gap-3">
          <button
            :if={!@running}
            type="button"
            phx-click="run_mcmc"
            class="btn btn-primary font-data text-sm"
          >
            Run the interrupted time series analysis
          </button>
          <span
            :if={@running}
            role="status"
            aria-live="polite"
            aria-busy="true"
            class="font-data motion-safe:animate-pulse text-sm text-(--color-ink-soft)"
          >
            Sampling {Policy.num_samples()} posterior draws on this server…
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
            caption={"The posterior distribution of the cumulative effect; #{@mcmc.num_samples} draws. Mass below zero is evidence the policy prevented accidents."}
            execution={@mcmc.execution}
          >
            <:legend>
              <.swatch color={:model} label="posterior draws" />
              <.swatch :if={@revealed} color={:truth} label="planted truth" dash />
            </:legend>
            <.histogram
              id="policy-posterior"
              title="Policy effect posterior"
              desc="Posterior draws of the cumulative policy effect, with interval and planted-truth markers when revealed."
              values={@mcmc.cumulative_draws}
              bins={24}
              vlines={histogram_vlines(@mcmc, @revealed)}
            />
          </.figure>

          <div class="grid gap-4 sm:grid-cols-[1fr_minmax(240px,0.8fr)]">
            <.verdict_card
              verdict={mcmc_verdict(@mcmc)}
              tone={verdict_tone(@mcmc)}
            >
              <:stat
                label="Cumulative effect"
                value={"#{signed(@mcmc.effect.cumulative)} accidents"}
                hint={"95% CI [#{fmt(@mcmc.effect.cumulative_lower)}, #{fmt(@mcmc.effect.cumulative_upper)}]"}
              />
              <:stat
                label="Monthly effect"
                value={"#{signed(@mcmc.effect.per_period)}/month"}
              />
              <:stat label="Relative change" value={pct(@mcmc.effect.relative)} />
            </.verdict_card>

            <.reveal_truth
              revealed={@revealed}
              truth={truth_line(@mcmc.scenario)}
              grade={grade_line(@mcmc)}
            />
          </div>
        </div>

        <p :if={!@mcmc && !@running} class="report-prose mt-4 text-sm text-(--color-ink-soft)">
          The answer key stays sealed until the model has committed to an estimate.
        </p>

        <.under_the_hood code={Policy.code_snippet()} />
      </section>

      <section class="mx-auto mt-12 max-w-4xl">
        <div class="grid gap-4 md:grid-cols-2">
          <.cross_link
            navigate={~p"/demos/marketing"}
            eyebrow="Same estimator, opposite sign"
            title="Did the campaign actually work?"
          >
            The identical machinery pointed at a positive effect; conversions gained
            instead of accidents prevented.
          </.cross_link>
          <.cross_link
            navigate={~p"/engine/counterfactual"}
            eyebrow="The engine underneath"
            title="Where the counterfactual comes from"
          >
            How a model that never saw the policy projects the world where nothing changed.
          </.cross_link>
        </div>
      </section>
    </Layouts.app>
    """
  end

  defp series_lines(scenario, nil) do
    [%{points: scenario.observations, color: :ink, width: 1.5}]
  end

  defp series_lines(scenario, mcmc) do
    [
      %{points: scenario.observations, color: :ink, width: 1.5},
      %{
        points: mcmc.counterfactual_mean,
        x_offset: scenario.pre_end,
        color: :model,
        dash: true,
        width: 2.0
      }
    ]
  end

  defp series_bands(_scenario, nil), do: []

  defp series_bands(scenario, mcmc) do
    [
      %{
        lower: mcmc.band_lower,
        upper: mcmc.band_upper,
        x_offset: scenario.pre_end,
        fill: "var(--color-model-band)"
      },
      %{
        lower: Enum.drop(scenario.observations, scenario.pre_end),
        upper: mcmc.counterfactual_mean,
        x_offset: scenario.pre_end,
        fill: "var(--color-lift-band)"
      }
    ]
  end

  defp series_caption(nil) do
    "Monthly accident counts. The dashed rule marks month 37, when the speed-limit change takes effect."
  end

  defp series_caption(mcmc) do
    "Monthly accident counts vs. the posterior counterfactual (mean and 95% band across " <>
      "#{mcmc.num_samples} draws); the world where the speed limit never changed. " <>
      "The magenta shading is the estimated effect."
  end

  defp mcmc_verdict(%{significant?: true, effect: %{cumulative: cum} = effect})
       when cum < 0 do
    "Yes; the change prevented ~#{fmt(-cum)} accidents " <>
      "(95% CI #{fmt(-effect.cumulative_upper)} to #{fmt(-effect.cumulative_lower)} fewer)."
  end

  defp mcmc_verdict(%{significant?: true, effect: %{cumulative: cum}}) do
    "The model sees ~#{fmt(cum)} more accidents; an increase, not a reduction."
  end

  defp mcmc_verdict(_mcmc) do
    "Honestly? Can't tell; the interval straddles zero."
  end

  # Prevented accidents are the good outcome here, so a significant
  # reduction gets the effect color; anything else stays neutral.
  defp verdict_tone(%{significant?: true, effect: %{cumulative: cum}}) when cum < 0, do: :lift
  defp verdict_tone(%{significant?: true}), do: :warning
  defp verdict_tone(_), do: :neutral

  defp histogram_vlines(mcmc, revealed) do
    zero = %{x: 0.0, label: "no effect", color: :ink}
    truth = %{x: mcmc.scenario.truth.cumulative_change, label: "planted truth", color: :truth}

    if revealed, do: [zero, truth], else: [zero]
  end

  defp truth_line(%{truth: %{change_per_month: change}}) when change == 0.0 do
    "Planted: nothing. The speed-limit change has zero true effect; any change the model reports is noise."
  end

  defp truth_line(%{truth: truth}) do
    "Planted: #{trunc(truth.change_per_month)} accidents/month from month 37; " <>
      "#{fmt(truth.cumulative_change)} over #{truth.post_months} months " <>
      "(#{fmt(-truth.cumulative_change)} accidents prevented)."
  end

  defp grade_line(mcmc) do
    truth = mcmc.scenario.truth.cumulative_change
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
