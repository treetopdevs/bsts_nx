defmodule BstsSiteWeb.Engine.KalmanLive do
  @moduledoc """
  Engine chapter 2: one noisy series, two trust knobs (Q and R), and the
  eager Kalman filter re-run on every slider drag.
  """
  use BstsSiteWeb, :live_view

  alias BstsSite.Demos.KalmanTuner

  @impl true
  def mount(_params, _session, socket) do
    demo = KalmanTuner.run(KalmanTuner.default_q_idx(), KalmanTuner.default_r_idx())

    {:ok,
     socket
     |> assign(page_title: "Tracking a signal through noise")
     |> assign(demo: demo, revealed: false)}
  end

  @impl true
  def handle_event("tune", %{"q_idx" => q_idx, "r_idx" => r_idx}, socket) do
    {:noreply, assign(socket, demo: KalmanTuner.run(q_idx, r_idx))}
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
          eyebrow="Engine · chapter 2"
          title="Tracking a signal through noise"
        >
          Your phone's GPS solves this problem every second. It has a model; you probably
          kept moving the way you were moving; and a measurement: a satellite fix, each one
          a little wrong. The Kalman filter blends the two, weighted by how much it trusts
          each. Below, a level wanders under heavy observation noise; the two sliders are
          the two trusts. Q says "the level itself can move this much per step";
          R says "each reading is this noisy". Drag them to the extremes and watch
          the filter change character.
        </.section_heading>

        <form phx-change="tune" class="mt-5 grid gap-x-10 gap-y-3 sm:grid-cols-2">
          <.param_slider
            name="q_idx"
            label="Process variance Q; how much the level may drift"
            min={0}
            max={KalmanTuner.max_idx()}
            value={@demo.q_idx}
            display={"Q = #{fmt_g(@demo.q)}"}
          />
          <.param_slider
            name="r_idx"
            label="Observation variance R; how noisy each reading is"
            min={0}
            max={KalmanTuner.max_idx()}
            value={@demo.r_idx}
            display={"R = #{fmt_g(@demo.r)}"}
          />
        </form>

        <.figure
          no="1"
          caption="96 noisy observations of a wandering level. The dashed line is each step's one-step-ahead prediction, made before seeing that observation; the solid line is the filtered estimate after seeing it, with its 95% band shaded."
          execution={@demo.execution}
        >
          <:legend>
            <.swatch color={:ink} label="observed" />
            <.swatch color={:model} label="one-step prediction" dash />
            <.swatch color={:model} label="filtered estimate" />
          </:legend>
          <.line_figure
            id="kalman-chart"
            title="Kalman filter tuning"
            desc="Noisy observations are shown with one-step predictions, filtered estimates, and a 95% model band."
            height={330}
            y_label="level"
            series={[
              %{points: @demo.observations, color: :ink, width: 1.3},
              %{points: @demo.pred_means, color: :model, dash: true, width: 1.5},
              %{points: @demo.filt_means, color: :model, width: 2.1}
            ]}
            bands={[
              %{
                lower: @demo.band_lower,
                upper: @demo.band_upper,
                fill: "var(--color-model-band)"
              }
            ]}
          />
        </.figure>

        <div class="grid gap-4 sm:grid-cols-[1fr_minmax(240px,0.8fr)]">
          <.verdict_card verdict={verdict_line(@demo)} tone={verdict_tone(@demo)}>
            The one-step prediction is the honest score: it never saw the point it's guessing.
            Filtered error you can cheat to zero; crank Q and the filter simply re-draws the
            data, noise and all. Prediction error you can't cheat: it's governed by the trust
            ratio Q/R, and on this grid it bottoms out at the true ratio.
            <:stat
              label="One-step prediction MAE"
              value={fmt2(@demo.mae_pred)}
              hint="honest; computed before each observation"
            />
            <:stat
              label="Filtered MAE"
              value={fmt2(@demo.mae_filt)}
              hint="after seeing each observation"
            />
            <:stat label="Q / R" value={fmt_ratio(@demo.q / @demo.r)} />
          </.verdict_card>

          <.reveal_truth
            revealed={@revealed}
            truth={"This series was generated with Q = #{fmt_g(KalmanTuner.q_true())} and R = #{fmt_g(KalmanTuner.r_true())}; exactly where the sliders started."}
            grade={grade_line(@demo)}
          />
        </div>

        <.under_the_hood code={KalmanTuner.code_snippet()} />

        <div class="mt-8 grid gap-4 md:grid-cols-3">
          <.cross_link
            navigate={~p"/engine/noise"}
            eyebrow="Engine · chapter 1"
            title="Why raw data lies"
          >
            The quiz this filter graded: four random walks, one planted shift.
          </.cross_link>
          <.cross_link
            navigate={~p"/engine/smoother"}
            eyebrow="Engine · chapter 3"
            title="The power of hindsight"
          >
            This filter only looks backwards-to-now. The smoother revisits every estimate
            knowing the whole series.
          </.cross_link>
          <.cross_link navigate={~p"/speed"} eyebrow="Speed" title="The compiled twin">
            The same filter, JIT-compiled, is what makes the site's fast lane fast.
          </.cross_link>
        </div>
      </div>
    </Layouts.app>
    """
  end

  defp verdict_line(%{regime: :trusts_data}) do
    "Q ≫ R: the filter believes every wiggle; it's re-drawing the noise."
  end

  defp verdict_line(%{regime: :trusts_model}) do
    "R ≫ Q: the filter barely believes its eyes; smooth, confident, late to every turn."
  end

  defp verdict_line(_demo) do
    "Balanced: the filter splits the difference between model and measurement."
  end

  defp verdict_tone(%{regime: :balanced}), do: :neutral
  defp verdict_tone(_demo), do: :warning

  defp grade_line(demo) do
    if demo.q == KalmanTuner.q_true() and demo.r == KalmanTuner.r_true() do
      "You're parked on the truth right now (prediction MAE #{fmt2(demo.mae_pred)}). One honest wrinkle: prediction error identifies the ratio Q/R, not the pair; settings that scale both together predict almost identically. Separating them is the Gibbs chapter's job."
    else
      "You're at Q = #{fmt_g(demo.q)}, R = #{fmt_g(demo.r)}, so Q/R = #{fmt_ratio(demo.q / demo.r)}; the data were generated at Q/R = #{fmt_ratio(KalmanTuner.q_true() / KalmanTuner.r_true())}. Head back there and watch the prediction MAE bottom out; the filtered MAE will happily mislead you along the way."
    end
  end

  defp fmt_g(v) when is_number(v) do
    if v >= 1.0 do
      :erlang.float_to_binary(v * 1.0, decimals: 0)
    else
      :erlang.float_to_binary(v * 1.0, decimals: 2)
    end
  end

  defp fmt2(v) when is_number(v), do: :erlang.float_to_binary(v * 1.0, decimals: 2)

  defp fmt_ratio(v) when is_number(v) do
    # Guard the formatting boundary: 0.3 / 3.0 = 0.09999999999999999, which
    # would fall through to the 4-decimal branch and render "0.1000".
    v = Float.round(v * 1.0, 6)

    cond do
      v >= 10.0 -> :erlang.float_to_binary(v * 1.0, decimals: 0)
      v >= 0.1 -> :erlang.float_to_binary(v * 1.0, decimals: 1)
      true -> :erlang.float_to_binary(v * 1.0, decimals: 4)
    end
  end
end
