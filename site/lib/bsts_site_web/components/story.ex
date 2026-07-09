defmodule BstsSiteWeb.Story do
  @moduledoc """
  The site's storytelling components.

  The recurring devices:

    * `figure/1`: every chart is a numbered Figure with a journal-style
      caption; the caption carries the live execution readout.
    * `exec_badge/1`: the library's own `execution` metadata as an
      instrument readout: "method: :scalar_forecast_filter · 6 ms · live".
    * `reveal_truth/1`: planted ground truth behind a redaction bar.
    * `under_the_hood/1`: the actual Elixir the demo just ran, highlighted
      server-side with Makeup.
    * `verdict_card/1`: the plain-language answer with its interval.
    * `mantra/1`: decompose → project → compare, with the current stage lit.
  """

  use Phoenix.Component

  @doc """
  A numbered figure. The body slot holds the chart; `caption` is the
  journal-style caption; optional `execution` renders the live readout.
  """
  attr(:no, :string, required: true, doc: ~s(figure number, e.g. "1" or "3.2"))
  attr(:caption, :string, required: true)
  attr(:execution, :map, default: nil, doc: "the library's execution metadata map")
  attr(:class, :string, default: nil)
  slot(:inner_block, required: true)
  slot(:legend)

  def figure(assigns) do
    ~H"""
    <figure class={["my-6", @class]}>
      <div class="rounded-md border border-(--color-grid) bg-(--color-paper-raised) p-3 sm:p-4">
        {render_slot(@inner_block)}
      </div>
      <figcaption class="figure-caption mt-2 flex flex-wrap items-baseline gap-x-3 gap-y-1 px-1">
        <span>
          <span class="fig-no">Fig. {@no}</span>: {@caption}
        </span>
        <span :if={@legend != []} class="inline-flex flex-wrap items-center gap-x-3">
          {render_slot(@legend)}
        </span>
        <.exec_badge :if={@execution} execution={@execution} class="ml-auto" />
      </figcaption>
    </figure>
    """
  end

  @doc "A legend swatch for figure captions."
  attr(:color, :atom, required: true, values: [:ink, :model, :lift, :truth])
  attr(:label, :string, required: true)
  attr(:dash, :boolean, default: false)

  def swatch(assigns) do
    ~H"""
    <span class="inline-flex items-center gap-1.5 whitespace-nowrap">
      <svg viewBox="0 0 20 6" class="w-5 h-1.5" aria-hidden="true">
        <line
          x1="0"
          y1="3"
          x2="20"
          y2="3"
          stroke={BstsSiteWeb.Charts.color(@color)}
          stroke-width="2.5"
          stroke-dasharray={if @dash, do: "4 3"}
        />
      </svg>
      {@label}
    </span>
    """
  end

  @doc """
  Instrument readout of the library's self-reported execution metadata.
  Nothing on this site is a screenshot: the badge proves the demo ran here.
  """
  attr(:execution, :map, required: true)
  attr(:class, :string, default: nil)

  def exec_badge(assigns) do
    ~H"""
    <span class={["exec-badge", @class]} title="Self-reported by the library's execution metadata">
      <span class="exec-dot" aria-hidden="true"></span>
      <span>{@execution.method_used}</span>
      <span aria-hidden="true">·</span>
      <span>{@execution.elapsed_ms} ms</span>
      <span class="hidden sm:inline" aria-hidden="true">·</span>
      <span class="hidden sm:inline">computed live on this server</span>
    </span>
    """
  end

  @doc """
  The plain-language answer. `tone` colors the leading verdict word:
  :lift for effects, :truth for recovered truths, :neutral otherwise.
  """
  attr(:verdict, :string, required: true, doc: ~s(e.g. "Yes, the campaign worked."))
  attr(:tone, :atom, default: :lift, values: [:lift, :truth, :neutral, :warning])
  attr(:class, :string, default: nil)
  attr(:rest, :global, include: ~w(role aria-live aria-busy))
  slot(:inner_block)

  slot :stat, doc: "key numbers" do
    attr(:label, :string, required: true)
    attr(:value, :string, required: true)
    attr(:hint, :string)
  end

  def verdict_card(assigns) do
    ~H"""
    <div
      {@rest}
      class={[
        "verdict-card min-w-0 rounded-md p-4 sm:p-5",
        verdict_class(@tone),
        @class
      ]}
    >
      <p class="font-display text-lg sm:text-xl font-semibold break-words">{@verdict}</p>
      <div :if={@inner_block != []} class="report-prose mt-1 text-sm text-(--color-ink-soft)">
        {render_slot(@inner_block)}
      </div>
      <dl :if={@stat != []} class="mt-3 flex flex-wrap gap-x-8 gap-y-2">
        <div :for={stat <- @stat} class="min-w-0">
          <dt class="font-data text-[0.65rem] uppercase tracking-wider text-(--color-ink-faint)">
            {stat.label}
          </dt>
          <dd class="font-data text-base font-medium tabular-nums break-words">
            {stat.value}
            <span :if={stat[:hint]} class="text-xs font-normal text-(--color-ink-faint)">
              {stat.hint}
            </span>
          </dd>
        </div>
      </dl>
    </div>
    """
  end

  defp verdict_class(:lift), do: "verdict-card-lift"
  defp verdict_class(:truth), do: "verdict-card-truth"
  defp verdict_class(:warning), do: "verdict-card-warning"
  defp verdict_class(:neutral), do: "verdict-card-neutral"

  @doc """
  The planted-truth reveal. We generated the data, so we know the answer;
  the estimate stands on its own before the reveal, and is graded after.

  The parent LiveView owns the `revealed` flag and handles the event
  (default: "reveal_truth").
  """
  attr(:revealed, :boolean, required: true)
  attr(:event, :string, default: "reveal_truth")
  attr(:label, :string, default: "Reveal the planted truth")
  attr(:truth, :string, required: true, doc: "the ground-truth statement")
  attr(:grade, :string, default: nil, doc: "how the estimate did against it")
  attr(:class, :string, default: nil)

  def reveal_truth(assigns) do
    ~H"""
    <div class={[
      "rounded-md border border-dashed p-4",
      @revealed && "border-(--color-truth)",
      !@revealed && "border-(--color-ink-faint)",
      @class
    ]}>
      <div class="flex flex-wrap items-center justify-between gap-2">
        <span class="font-data text-[0.65rem] uppercase tracking-wider text-(--color-ink-faint)">
          Answer key: sealed at data generation
        </span>
        <button
          :if={!@revealed}
          type="button"
          phx-click={@event}
          class="btn btn-sm btn-outline font-data text-xs"
        >
          {@label}
        </button>
      </div>
      <p class="mt-2 font-prose text-base">
        <span class={["redacted px-2 py-0.5", @revealed && "revealed"]} aria-live="polite">
          {if @revealed, do: @truth, else: placeholder_for(@truth)}
        </span>
      </p>
      <p :if={@revealed && @grade} class="report-prose mt-2 text-sm text-(--color-ink-soft)">
        {@grade}
      </p>
    </div>
    """
  end

  # Keep the redaction bar roughly the same width as the hidden text.
  defp placeholder_for(text), do: String.duplicate("█", min(String.length(text), 36))

  @doc """
  Collapsible panel showing the real, copy-pasteable Elixir the demo runs.
  Highlighted server-side with Makeup, no client JS.
  """
  attr(:code, :string, required: true)
  attr(:title, :string, default: "Under the hood: the code this demo just ran")
  attr(:class, :string, default: nil)

  def under_the_hood(assigns) do
    assigns = assign(assigns, :highlighted, highlight(assigns.code))

    ~H"""
    <details class={["group my-4 rounded-md border border-(--color-grid)", @class]}>
      <summary class="flex min-h-11 cursor-pointer select-none items-center gap-2 px-4 py-2.5 font-data text-xs text-(--color-ink-soft) hover:text-(--color-ink)">
        <span class="transition-transform group-open:rotate-90" aria-hidden="true">▸</span>
        {@title}
      </summary>
      <div class="code-block !rounded-t-none border-t border-(--color-grid)">
        <pre><code>{Phoenix.HTML.raw(@highlighted)}</code></pre>
      </div>
    </details>
    """
  end

  defp highlight(code) do
    Makeup.highlight_inner_html(code, lexer: Makeup.Lexers.ElixirLexer)
  rescue
    _ -> Phoenix.HTML.html_escape(code) |> Phoenix.HTML.safe_to_string()
  end

  @doc """
  The site's conceptual compass: decompose → project → compare.
  `active` lights the stage the current page teaches.
  """
  attr(:active, :atom, default: nil, values: [nil, :decompose, :project, :compare])
  attr(:class, :string, default: nil)

  def mantra(assigns) do
    ~H"""
    <div class={["flex items-center gap-2", @class]} aria-label="Method: decompose, project, compare">
      <span class={["mantra-step", @active == :decompose && "active"]}>decompose</span>
      <span class="text-(--color-ink-faint)" aria-hidden="true">→</span>
      <span class={["mantra-step", @active == :project && "active"]}>project</span>
      <span class="text-(--color-ink-faint)" aria-hidden="true">→</span>
      <span class={["mantra-step", @active == :compare && "active"]}>compare</span>
    </div>
    """
  end

  @doc "Cross-link card: 'this demo ran the same engine as → the Kalman chapter'."
  attr(:navigate, :string, required: true)
  attr(:eyebrow, :string, required: true)
  attr(:title, :string, required: true)
  attr(:class, :string, default: nil)
  slot(:inner_block)

  def cross_link(assigns) do
    ~H"""
    <.link
      navigate={@navigate}
      class={[
        "block rounded-md border border-(--color-grid) bg-(--color-paper-raised) p-4",
        "transition-colors hover:border-(--color-model)",
        @class
      ]}
    >
      <div class="font-data text-[0.65rem] uppercase tracking-wider text-(--color-ink-faint)">
        {@eyebrow}
      </div>
      <div class="font-display mt-1 font-semibold">{@title} <span aria-hidden="true">→</span></div>
      <div :if={@inner_block != []} class="mt-1 text-sm text-(--color-ink-soft)">
        {render_slot(@inner_block)}
      </div>
    </.link>
    """
  end

  @doc "Section eyebrow + display heading for report sections."
  attr(:eyebrow, :string, default: nil)
  attr(:title, :string, required: true)
  attr(:level, :integer, default: 2, values: [1, 2])
  attr(:class, :string, default: nil)
  slot(:inner_block)

  def section_heading(assigns) do
    ~H"""
    <header class={["mb-4", @class]}>
      <div :if={@eyebrow} class="font-data text-xs uppercase tracking-[0.15em] text-(--color-lift)">
        {@eyebrow}
      </div>
      <h1 :if={@level == 1} class="font-display mt-1 text-3xl font-bold sm:text-4xl">
        {@title}
      </h1>
      <h2 :if={@level == 2} class="font-display mt-1 text-2xl font-bold sm:text-3xl">
        {@title}
      </h2>
      <div :if={@inner_block != []} class="report-prose mt-2 text-(--color-ink-soft)">
        {render_slot(@inner_block)}
      </div>
    </header>
    """
  end

  @doc "A labeled range slider that pushes a phx-change event from its form."
  attr(:name, :string, required: true)
  attr(:label, :string, required: true)
  attr(:min, :any, required: true)
  attr(:max, :any, required: true)
  attr(:step, :any, default: 1)
  attr(:value, :any, required: true)
  attr(:display, :string, default: nil, doc: "formatted current value")
  attr(:class, :string, default: nil)

  def param_slider(assigns) do
    ~H"""
    <label class={["block", @class]}>
      <div class="flex items-baseline justify-between gap-2">
        <span class="font-data text-[0.7rem] uppercase tracking-wider text-(--color-ink-soft)">
          {@label}
        </span>
        <span class="font-data text-xs tabular-nums text-(--color-ink)">
          {@display || @value}
        </span>
      </div>
      <input
        type="range"
        name={@name}
        min={@min}
        max={@max}
        step={@step}
        value={@value}
        phx-debounce="150"
        class="range range-xs range-primary mt-1 w-full"
      />
    </label>
    """
  end
end
