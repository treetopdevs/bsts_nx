defmodule BstsSiteWeb.Demos.DemandLive do
  @moduledoc """
  How much should we stock? Forecasting under uncertainty, then turning the
  posterior into an order quantity at a chosen service level.
  """
  use BstsSiteWeb, :live_view

  alias BstsSite.Demos.Demand

  @impl true
  def mount(_params, _session, socket) do
    scenario = Demand.scenario()

    {:ok,
     socket
     |> assign(page_title: "How much should we stock?")
     |> assign(
       scenario: scenario,
       forecast: nil,
       decision: nil,
       coverage: nil,
       service_level: 95,
       running: false,
       busy: false,
       error: nil,
       revealed: false
     )}
  end

  @impl true
  def handle_event("fit", _params, %{assigns: %{running: true}} = socket) do
    {:noreply, socket}
  end

  def handle_event("fit", _params, socket) do
    scenario = socket.assigns.scenario
    socket = assign(socket, running: true, busy: false, error: nil)

    {:noreply,
     start_async(socket, :fit, fn ->
       case BstsSite.Demos.Limiter.run(fn -> Demand.fit_and_forecast(scenario) end) do
         {:ok, result} -> result
         :busy -> :busy
       end
     end)}
  end

  def handle_event("service", %{"service_level" => level}, socket) do
    service_level = Demand.clamp(level, Demand.service_range())

    decision =
      case socket.assigns.forecast do
        nil -> nil
        forecast -> Demand.decision(forecast, service_level)
      end

    {:noreply, assign(socket, service_level: service_level, decision: decision)}
  end

  def handle_event("reveal_truth", _params, socket) do
    {:noreply, assign(socket, revealed: true)}
  end

  @impl true
  def handle_async(:fit, {:ok, :busy}, socket) do
    {:noreply, assign(socket, running: false, busy: true, error: nil)}
  end

  def handle_async(:fit, {:ok, forecast}, socket) do
    {:noreply,
     assign(socket,
       running: false,
       busy: false,
       error: nil,
       forecast: forecast,
       decision: Demand.decision(forecast, socket.assigns.service_level),
       coverage: Demand.coverage(forecast, socket.assigns.scenario.future)
     )}
  end

  def handle_async(:fit, {:exit, reason}, socket) do
    require Logger
    Logger.error("DemandLive async :fit crashed: #{inspect(reason)}")

    {:noreply,
     assign(socket,
       running: false,
       busy: false,
       error: "The forecast failed. Please try again.",
       forecast: nil,
       decision: nil,
       coverage: nil
     )}
  end

  @impl true
  def render(assigns) do
    ~H"""
    <Layouts.app flash={@flash} track={:questions}>
      <div class="mx-auto max-w-4xl">
        <.section_heading
          level={1}
          eyebrow="How much should we stock?"
          title="A forecast is not a number. It's a distribution."
        >
          We generated {Demand.history_days()} days of daily demand for one product; a level
          that meanders gently around {fmt(@scenario.history_mean)} units a day; and generated {Demand.horizon()} more days the same way, then sealed them. One purchase order has to
          cover those three weeks. The model fits the history, simulates {Demand.num_samples()} possible three-week futures; {Demand.num_samples() *
            Demand.horizon()} simulated
          unit-days; and the order quantity falls out of a quantile.
        </.section_heading>

        <.mantra active={:project} />

        <div class="mt-6 flex flex-wrap items-center gap-4">
          <button
            type="button"
            phx-click="fit"
            disabled={@running}
            class="btn btn-primary font-data text-sm"
          >
            {if @forecast, do: "Refit and forecast", else: "Fit and forecast"}
          </button>
          <span
            :if={@running}
            role="status"
            aria-live="polite"
            aria-busy="true"
            class="motion-safe:animate-pulse font-data text-xs text-(--color-ink-soft)"
          >
            sampling {Demand.num_samples()} posterior draws on this server…
          </span>
          <span :if={@forecast && !@running} class="font-data text-xs text-(--color-ink-faint)">
            fitted in {@forecast.execution.elapsed_ms} ms; the slider below never refits
          </span>
        </div>

        <form :if={@forecast} id="service-form" phx-change="service" class="mt-5 max-w-md">
          <.param_slider
            name="service_level"
            label="Target service level"
            min={Demand.service_range().first}
            max={Demand.service_range().last}
            value={@decision.service_level}
            display={"#{@decision.service_level}% of futures covered"}
          />
        </form>

        <.figure
          no="1"
          caption={
            if @forecast,
              do:
                "Daily demand history and the #{Demand.horizon()}-day forecast fan: nested 50% and 95% bands around the posterior mean, drawn from #{@forecast.n_draws} simulated futures.",
              else:
                "#{Demand.history_days()} days of daily demand history; the blank space right of the line is the question."
          }
          execution={@forecast && @forecast.execution}
        >
          <:legend>
            <.swatch color={:ink} label="observed demand" />
            <.swatch :if={@forecast} color={:model} label="point forecast" dash />
            <.swatch :if={@revealed} color={:truth} label="actual (held back)" />
          </:legend>
          <.line_figure
            id="demand-fan"
            title="Demand forecast distribution"
            desc="Historical demand appears before the order point; forecast and held-back actual demand appear after it when available."
            height={320}
            y_label="units/day"
            x_domain={{0, Demand.history_days() + Demand.horizon() - 1}}
            series={fan_series(@scenario, @forecast, @revealed)}
            bands={fan_bands(@forecast)}
            vlines={[%{x: Demand.history_days(), label: "order placed"}]}
          />
        </.figure>

        <div :if={@busy} class="my-6">
          <.verdict_card
            role="status"
            aria-live="polite"
            verdict="Another visitor is sampling, try again in a moment."
            tone={:warning}
          >
            The MCMC demos share a small worker pool so this machine stays honest about its
            latencies. Your click wasn't lost; the button works again right now.
          </.verdict_card>
        </div>

        <.verdict_card
          :if={@error}
          class="my-6"
          role="alert"
          tone={:warning}
          verdict={@error}
        >
          <p>The failure was logged so it can be investigated.</p>
        </.verdict_card>

        <section :if={@forecast}>
          <.figure
            no="2"
            caption={"Total demand over the next #{Demand.horizon()} days, once per posterior draw; the distribution the order decision is actually made on."}
            execution={@forecast.execution}
          >
            <:legend>
              <.swatch color={:model} label="order quantity" dash />
              <.swatch :if={@revealed} color={:truth} label="actual total" dash />
            </:legend>
            <.histogram
              id="demand-totals"
              title="Posterior total demand"
              desc="Posterior draws of total demand over the planning horizon, with order quantity and actual total markers when available."
              values={@forecast.totals}
              bins={16}
              height={190}
              vlines={total_vlines(@decision, @scenario, @revealed)}
            />
          </.figure>

          <div class="grid gap-4 sm:grid-cols-[1fr_minmax(240px,0.8fr)]">
            <.verdict_card
              verdict={"Order #{fmt(@decision.order_units)} units for a #{@decision.service_level}% service level."}
              tone={verdict_tone(@decision, @scenario, @revealed)}
            >
              That's expected demand plus {fmt(@decision.safety_stock)} units of safety stock.
              A textbook formula that treats each day's error as independent would hold only {fmt(
                @decision.naive_safety_stock
              )}; but when the level itself wanders, a high day
              tends to be followed by more high days, so {Demand.horizon()}-day totals spread
              wider than independent errors suggest. The trajectories carry that correlation;
              the formula throws it away.
              <:stat
                label="Expected demand"
                value={"#{fmt(@decision.expected_total)} units"}
                hint={"over #{Demand.horizon()} days"}
              />
              <:stat
                label="Safety stock"
                value={"+#{fmt(@decision.safety_stock)} units"}
                hint="above expected"
              />
              <:stat
                label="Stockout risk"
                value={"#{@decision.exceed_draws} of #{@decision.n_draws}"}
                hint="posterior futures exceed the order"
              />
            </.verdict_card>

            <.reveal_truth
              revealed={@revealed}
              truth={"Held back: the actual next #{Demand.horizon()} days totalled #{fmt(@scenario.truth.future_total)} units (#{fmt(@scenario.truth.future_daily_mean)}/day)."}
              grade={grade_line(@decision, @coverage, @scenario)}
            />
          </div>

          <.under_the_hood code={Demand.code_snippet()} />
        </section>

        <div class="mt-12 grid gap-4 sm:grid-cols-2">
          <.cross_link
            navigate={~p"/engine/gibbs"}
            eyebrow="Where the fan comes from"
            title="The Gibbs sampler"
          >
            Every trajectory in the fan is one posterior draw of the model's variances,
            projected forward. This is the chapter where those draws are made.
          </.cross_link>
          <.cross_link
            navigate={~p"/trust/calibration"}
            eyebrow="Do the bands keep their promise?"
            title="Calibration, graded"
          >
            A 95% band should cover about 95% of held-back futures; over many worlds, not
            just this one. That page grades five planted lifts on one seeded series; a live
            spot-check; the library ships the many-runs version offline
            (<code>BstsNx.Validation.known_lift_injection/3</code>).
          </.cross_link>
        </div>
      </div>
    </Layouts.app>
    """
  end

  # ── figure data ─────────────────────────────────────────────────────

  defp fan_series(scenario, nil, _revealed) do
    [%{points: scenario.history, color: :ink, width: 1.5}]
  end

  defp fan_series(scenario, forecast, revealed) do
    offset = Demand.history_days()

    base = [
      %{points: scenario.history, color: :ink, width: 1.5},
      %{points: forecast.mean, x_offset: offset, color: :model, dash: true, width: 2.0}
    ]

    if revealed do
      base ++ [%{points: scenario.future, x_offset: offset, color: :truth, width: 1.75}]
    else
      base
    end
  end

  defp fan_bands(nil), do: []

  defp fan_bands(forecast) do
    offset = Demand.history_days()

    [
      %{
        lower: forecast.fan95.lower,
        upper: forecast.fan95.upper,
        x_offset: offset,
        fill: "var(--color-model-band)"
      },
      %{
        lower: forecast.band50.lower,
        upper: forecast.band50.upper,
        x_offset: offset,
        fill: "var(--color-model-band)"
      }
    ]
  end

  defp total_vlines(decision, scenario, revealed) do
    order = [
      %{x: decision.order_units, label: "order #{fmt(decision.order_units)}", color: :model}
    ]

    if revealed do
      order ++
        [
          %{
            x: scenario.truth.future_total,
            label: "actual #{fmt(scenario.truth.future_total)}",
            color: :truth
          }
        ]
    else
      order
    end
  end

  # ── verdict & grade ─────────────────────────────────────────────────

  defp verdict_tone(_decision, _scenario, false), do: :neutral

  defp verdict_tone(decision, scenario, true) do
    if decision.order_units >= scenario.truth.future_total, do: :truth, else: :warning
  end

  defp grade_line(decision, coverage, scenario) do
    actual = scenario.truth.future_total
    order = decision.order_units

    outcome =
      if order >= actual do
        "ordering #{fmt(order)} units would have covered demand with #{fmt(order - actual)} left on the shelf; that spare is the price of a #{decision.service_level}% promise."
      else
        "ordering #{fmt(order)} units would have run out #{fmt(actual - order)} units short; inside the #{100 - decision.service_level}% of futures the promise deliberately leaves uncovered."
      end

    "The 95% band covered #{coverage} of #{Demand.horizon()} actual days. At your #{decision.service_level}% service level, " <>
      outcome
  end

  # ── formatting ──────────────────────────────────────────────────────

  defp fmt(v) when is_number(v), do: :erlang.float_to_binary(v * 1.0, decimals: 0)
end
