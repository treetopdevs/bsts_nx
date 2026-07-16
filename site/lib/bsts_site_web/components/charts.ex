defmodule BstsSiteWeb.Charts do
  @moduledoc """
  Server-rendered SVG chart components.

  Every chart on the site is drawn here, in pure Elixir, from plain lists of
  numbers; no JS chart library. Semantic colors are fixed site-wide:

    * ink       ; observed data
    * model blue; model estimates / counterfactuals (bands at low alpha)
    * lift      ; effects, and nothing else
    * truth     ; planted ground truth, and nothing else

  Series and bands accept an `:x_offset` so post-period overlays (e.g. a
  counterfactual that only exists after the intervention) line up with the
  full series.
  """

  use Phoenix.Component

  @ink "var(--color-ink)"
  @model "var(--color-model)"
  @lift "var(--color-lift)"
  @truth "var(--color-truth)"

  @doc "Semantic stroke color for a series role."
  def color(:ink), do: @ink
  def color(:model), do: @model
  def color(:lift), do: @lift
  def color(:truth), do: @truth
  def color(other) when is_binary(other), do: other

  attr(:id, :string, required: true)

  attr(:series, :list,
    default: [],
    doc: "[%{points: [num], color:, dash:, width:, x_offset:, draw:, opacity:}]"
  )

  attr(:bands, :list, default: [], doc: "[%{lower: [num], upper: [num], x_offset:, fill:}]")
  attr(:vlines, :list, default: [], doc: "[%{x: num, label: str}]")
  attr(:hlines, :list, default: [], doc: "[%{y: num, label: str, color:}]")
  attr(:y_label, :string, default: nil)
  attr(:height, :integer, default: 300)
  attr(:y_domain, :any, default: :auto)
  attr(:x_domain, :any, default: :auto)
  attr(:class, :string, default: nil)
  attr(:markers, :list, default: [], doc: "[%{x:, y:, color:, label:, shape: :dot|:x}]")
  attr(:title, :string, default: nil, doc: "accessible SVG title")
  attr(:desc, :string, default: nil, doc: "accessible SVG description")

  def line_figure(assigns) do
    w = 720
    h = assigns.height
    pad = %{l: 46, r: 12, t: if(assigns.y_label, do: 28, else: 12), b: 24}

    xs_max =
      case assigns.x_domain do
        {_, xmax} ->
          xmax

        :auto ->
          assigns.series
          |> Enum.map(fn s -> Map.get(s, :x_offset, 0) + length(s.points) - 1 end)
          |> Enum.concat(
            Enum.map(assigns.bands, fn b -> Map.get(b, :x_offset, 0) + length(b.lower) - 1 end)
          )
          |> Enum.max(fn -> 1 end)
      end

    xs_min =
      case assigns.x_domain do
        {xmin, _} -> xmin
        :auto -> 0
      end

    all_values =
      Enum.flat_map(assigns.series, & &1.points) ++
        Enum.flat_map(assigns.bands, &(&1.lower ++ &1.upper)) ++
        Enum.map(assigns.hlines, & &1.y) ++
        Enum.map(assigns.markers, & &1.y)

    {y_min, y_max} =
      case assigns.y_domain do
        {a, b} -> {a, b}
        :auto -> pad_domain(finite_min_max(all_values))
      end

    scale = build_scale(xs_min, xs_max, y_min, y_max, w, h, pad)
    y_ticks = nice_ticks(y_min, y_max, 5)
    x_ticks = nice_ticks(xs_min, xs_max, 6)

    assigns =
      assign(assigns,
        w: w,
        h: h,
        pad: pad,
        scale: scale,
        y_ticks: y_ticks,
        x_ticks: x_ticks
      )

    ~H"""
    <svg
      id={@id}
      viewBox={"0 0 #{@w} #{@h}"}
      class={["w-full", @class]}
      role="img"
      aria-labelledby={chart_labelledby(@id)}
      focusable="false"
      preserveAspectRatio="xMidYMid meet"
    >
      <title id={chart_title_id(@id)}>{chart_title(@id, @title, "Line chart")}</title>
      <desc id={chart_desc_id(@id)}>{chart_desc(@desc)}</desc>
      <defs>
        <pattern id={"#{@id}-grid"} width="18" height="18" patternUnits="userSpaceOnUse">
          <path d="M 18 0 L 0 0 0 18" fill="none" stroke="var(--color-grid)" stroke-width="0.5" />
        </pattern>
        <clipPath id={"#{@id}-clip"}>
          <rect x={@pad.l} y={@pad.t} width={@w - @pad.l - @pad.r} height={@h - @pad.t - @pad.b} />
        </clipPath>
      </defs>

      <rect
        x={@pad.l}
        y={@pad.t}
        width={@w - @pad.l - @pad.r}
        height={@h - @pad.t - @pad.b}
        fill="var(--color-paper-raised)"
        stroke="none"
      />
      <rect
        x={@pad.l}
        y={@pad.t}
        width={@w - @pad.l - @pad.r}
        height={@h - @pad.t - @pad.b}
        fill={"url(##{@id}-grid)"}
        stroke="var(--color-grid)"
        stroke-width="1"
      />

      <g :for={t <- @y_ticks} class="font-data">
        <line
          x1={@pad.l - 4}
          x2={@pad.l}
          y1={sy(@scale, t)}
          y2={sy(@scale, t)}
          stroke="var(--color-ink-faint)"
          stroke-width="1"
        />
        <text
          x={@pad.l - 8}
          y={sy(@scale, t) + 3}
          text-anchor="end"
          font-size="10"
          fill="var(--color-ink-faint)"
        >
          {fmt_tick(t)}
        </text>
      </g>
      <g :for={t <- @x_ticks} class="font-data">
        <text
          x={sx(@scale, t)}
          y={@h - @pad.b + 16}
          text-anchor="middle"
          font-size="10"
          fill="var(--color-ink-faint)"
        >
          {fmt_tick(t)}
        </text>
      </g>

      <g clip-path={"url(##{@id}-clip)"}>
        <path
          :for={band <- @bands}
          d={band_path(@scale, band)}
          fill={Map.get(band, :fill, "var(--color-model-band)")}
          stroke="none"
        />
        <g :for={hl <- @hlines}>
          <line
            x1={@pad.l}
            x2={@w - @pad.r}
            y1={sy(@scale, hl.y)}
            y2={sy(@scale, hl.y)}
            stroke={color(Map.get(hl, :color, :ink))}
            stroke-width="1"
            stroke-dasharray="2 4"
            opacity="0.7"
          />
        </g>
        <path
          :for={s <- @series}
          d={line_path(@scale, s)}
          fill="none"
          stroke={color(Map.get(s, :color, :ink))}
          stroke-width={Map.get(s, :width, 1.75)}
          stroke-dasharray={if Map.get(s, :dash, false), do: "5 4"}
          stroke-linejoin="round"
          stroke-linecap="round"
          opacity={Map.get(s, :opacity, 1.0)}
          class={if Map.get(s, :draw, false), do: "draw-line"}
        />
        <g :for={m <- @markers}>
          <circle
            :if={Map.get(m, :shape, :dot) == :dot}
            cx={sx(@scale, m.x)}
            cy={sy(@scale, m.y)}
            r="3.5"
            fill={color(Map.get(m, :color, :lift))}
          />
          <g :if={Map.get(m, :shape) == :x}>
            <line
              x1={sx(@scale, m.x) - 4}
              x2={sx(@scale, m.x) + 4}
              y1={sy(@scale, m.y) - 4}
              y2={sy(@scale, m.y) + 4}
              stroke={color(Map.get(m, :color, :lift))}
              stroke-width="2"
            />
            <line
              x1={sx(@scale, m.x) - 4}
              x2={sx(@scale, m.x) + 4}
              y1={sy(@scale, m.y) + 4}
              y2={sy(@scale, m.y) - 4}
              stroke={color(Map.get(m, :color, :lift))}
              stroke-width="2"
            />
          </g>
        </g>
      </g>

      <g :for={vl <- @vlines}>
        <line
          x1={sx(@scale, vl.x)}
          x2={sx(@scale, vl.x)}
          y1={@pad.t}
          y2={@h - @pad.b}
          stroke="var(--color-ink-soft)"
          stroke-width="1"
          stroke-dasharray="4 4"
        />
        <text
          :if={vl[:label]}
          x={sx(@scale, vl.x) + 5}
          y={@pad.t + 12}
          font-size="10"
          fill="var(--color-ink-soft)"
          class="font-data"
        >
          {vl.label}
        </text>
      </g>

      <text
        :if={@y_label}
        x={@pad.l}
        y={16}
        font-size="10"
        fill="var(--color-ink-faint)"
        class="font-data"
      >
        {@y_label}
      </text>
    </svg>
    """
  end

  attr(:id, :string, required: true)
  attr(:items, :list, required: true, doc: "[%{label:, value:, lower:, upper:, color:}]")
  attr(:height, :integer, default: 220)
  attr(:unit, :string, default: "")
  attr(:class, :string, default: nil)
  attr(:title, :string, default: nil, doc: "accessible SVG title")
  attr(:desc, :string, default: nil, doc: "accessible SVG description")

  def bar_figure(assigns) do
    w = 720
    h = assigns.height
    pad = %{l: 110, r: 60, t: 14, b: 14}

    values =
      Enum.flat_map(assigns.items, fn it ->
        [it.value, Map.get(it, :lower, it.value), Map.get(it, :upper, it.value)]
      end)

    {vmin, vmax} = pad_domain(finite_min_max([0.0 | values]))
    n = max(length(assigns.items), 1)
    row_h = (h - pad.t - pad.b) / n
    xscale = fn v -> pad.l + (v - vmin) / max(vmax - vmin, 1.0e-9) * (w - pad.l - pad.r) end

    assigns =
      assign(assigns, w: w, h: h, pad: pad, row_h: row_h, xscale: xscale, zero: xscale.(0.0))

    ~H"""
    <svg
      id={@id}
      viewBox={"0 0 #{@w} #{@h}"}
      class={["w-full", @class]}
      role="img"
      aria-labelledby={chart_labelledby(@id)}
      focusable="false"
    >
      <title id={chart_title_id(@id)}>{chart_title(@id, @title, "Bar chart")}</title>
      <desc id={chart_desc_id(@id)}>{chart_desc(@desc)}</desc>
      <line
        x1={@zero}
        x2={@zero}
        y1={@pad.t}
        y2={@h - @pad.b}
        stroke="var(--color-ink-faint)"
        stroke-width="1"
      />
      <g :for={{it, i} <- Enum.with_index(@items)}>
        <% cy = @pad.t + @row_h * i + @row_h / 2 %>
        <% x0 = min(@zero, @xscale.(it.value)) %>
        <% x1 = max(@zero, @xscale.(it.value)) %>
        <text
          x={@pad.l - 10}
          y={cy + 4}
          text-anchor="end"
          font-size="12"
          fill="var(--color-ink)"
          class="font-data"
        >
          {it.label}
        </text>
        <rect
          x={x0}
          y={cy - min(@row_h * 0.32, 13)}
          width={max(x1 - x0, 1)}
          height={min(@row_h * 0.64, 26)}
          rx="2"
          fill={color(Map.get(it, :color, :lift))}
          opacity="0.85"
        />
        <g :if={it[:lower] && it[:upper]}>
          <line
            x1={@xscale.(it.lower)}
            x2={@xscale.(it.upper)}
            y1={cy}
            y2={cy}
            stroke="var(--color-ink)"
            stroke-width="1.5"
          />
          <line
            x1={@xscale.(it.lower)}
            x2={@xscale.(it.lower)}
            y1={cy - 5}
            y2={cy + 5}
            stroke="var(--color-ink)"
            stroke-width="1.5"
          />
          <line
            x1={@xscale.(it.upper)}
            x2={@xscale.(it.upper)}
            y1={cy - 5}
            y2={cy + 5}
            stroke="var(--color-ink)"
            stroke-width="1.5"
          />
        </g>
        <text
          x={max(x1, @xscale.(Map.get(it, :upper, it.value))) + 8}
          y={cy + 4}
          font-size="11"
          fill="var(--color-ink-soft)"
          class="font-data"
        >
          {fmt_tick(it.value)}{@unit}
        </text>
      </g>
    </svg>
    """
  end

  attr(:id, :string, required: true)
  attr(:rows, :list, required: true, doc: "[%{id:, start:, stop:, color:}] half-open windows")
  attr(:domain, :any, required: true, doc: "{min, max}")
  attr(:height, :integer, default: 140)
  attr(:class, :string, default: nil)
  attr(:title, :string, default: nil, doc: "accessible SVG title")
  attr(:desc, :string, default: nil, doc: "accessible SVG description")

  def gantt(assigns) do
    w = 720
    h = assigns.height
    pad = %{l: 90, r: 16, t: 10, b: 22}
    {dmin, dmax} = assigns.domain
    n = max(length(assigns.rows), 1)
    row_h = (h - pad.t - pad.b) / n
    xscale = fn v -> pad.l + (v - dmin) / max(dmax - dmin, 1.0e-9) * (w - pad.l - pad.r) end
    ticks = nice_ticks(dmin, dmax, 6)

    assigns = assign(assigns, w: w, h: h, pad: pad, row_h: row_h, xscale: xscale, ticks: ticks)

    ~H"""
    <svg
      id={@id}
      viewBox={"0 0 #{@w} #{@h}"}
      class={["w-full", @class]}
      role="img"
      aria-labelledby={chart_labelledby(@id)}
      focusable="false"
    >
      <title id={chart_title_id(@id)}>{chart_title(@id, @title, "Timeline chart")}</title>
      <desc id={chart_desc_id(@id)}>{chart_desc(@desc)}</desc>
      <g :for={t <- @ticks} class="font-data">
        <line
          x1={@xscale.(t)}
          x2={@xscale.(t)}
          y1={@pad.t}
          y2={@h - @pad.b}
          stroke="var(--color-grid)"
          stroke-width="1"
        />
        <text
          x={@xscale.(t)}
          y={@h - @pad.b + 14}
          text-anchor="middle"
          font-size="10"
          fill="var(--color-ink-faint)"
        >
          {fmt_tick(t)}
        </text>
      </g>
      <g :for={{row, i} <- Enum.with_index(@rows)}>
        <% cy = @pad.t + @row_h * i + @row_h / 2 %>
        <text
          x={@pad.l - 8}
          y={cy + 4}
          text-anchor="end"
          font-size="12"
          fill="var(--color-ink)"
          class="font-data"
        >
          {row.id}
        </text>
        <rect
          x={@xscale.(row.start)}
          y={cy - min(@row_h * 0.3, 11)}
          width={max(@xscale.(row.stop) - @xscale.(row.start), 2)}
          height={min(@row_h * 0.6, 22)}
          rx="3"
          fill={color(Map.get(row, :color, :model))}
          opacity="0.8"
        />
      </g>
    </svg>
    """
  end

  attr(:id, :string, required: true)
  attr(:matrix, :list, required: true, doc: "list of rows of numbers")
  attr(:blocks, :list, default: [], doc: "[%{from:, size:, label:}] diagonal block annotations")
  attr(:class, :string, default: nil)
  attr(:title, :string, default: nil, doc: "accessible SVG title")
  attr(:desc, :string, default: nil, doc: "accessible SVG description")

  def matrix_grid(assigns) do
    n = length(assigns.matrix)
    cell = min(div(360, max(n, 1)), 44)
    size = cell * n
    pad = 4

    assigns = assign(assigns, n: n, cell: cell, size: size, pad: pad)

    ~H"""
    <svg
      id={@id}
      viewBox={"0 0 #{@size + @pad * 2} #{@size + @pad * 2}"}
      class={["mx-auto", @class]}
      style={"max-width: #{@size + 8}px"}
      role="img"
      aria-labelledby={chart_labelledby(@id)}
      focusable="false"
    >
      <title id={chart_title_id(@id)}>{chart_title(@id, @title, "Matrix chart")}</title>
      <desc id={chart_desc_id(@id)}>{chart_desc(@desc)}</desc>
      <g :for={{row, i} <- Enum.with_index(@matrix)}>
        <g :for={{val, j} <- Enum.with_index(row)}>
          <rect
            x={@pad + j * @cell}
            y={@pad + i * @cell}
            width={@cell - 1}
            height={@cell - 1}
            rx="2"
            fill={if val == 0, do: "var(--color-base-200)", else: "var(--color-model)"}
            opacity={if val == 0, do: "0.6", else: cell_opacity(val)}
          />
          <text
            :if={val != 0 and @cell >= 26}
            x={@pad + j * @cell + @cell / 2}
            y={@pad + i * @cell + @cell / 2 + 4}
            text-anchor="middle"
            font-size="11"
            fill="var(--color-paper-raised)"
            class="font-data"
          >
            {fmt_cell(val)}
          </text>
        </g>
      </g>
      <rect
        :for={b <- @blocks}
        x={@pad + b.from * @cell - 1.5}
        y={@pad + b.from * @cell - 1.5}
        width={b.size * @cell + 2}
        height={b.size * @cell + 2}
        fill="none"
        stroke="var(--color-lift)"
        stroke-width="2"
        rx="4"
      />
    </svg>
    """
  end

  attr(:id, :string, required: true)
  attr(:values, :list, required: true)
  attr(:bins, :integer, default: 24)
  attr(:height, :integer, default: 170)
  attr(:vlines, :list, default: [], doc: "[%{x:, label:, color:}]")
  attr(:class, :string, default: nil)
  attr(:title, :string, default: nil, doc: "accessible SVG title")
  attr(:desc, :string, default: nil, doc: "accessible SVG description")

  def histogram(assigns) do
    w = 720
    h = assigns.height
    pad = %{l: 16, r: 16, t: 10, b: 24}

    {vmin, vmax} = pad_domain(finite_min_max(assigns.values))
    nbins = assigns.bins
    bw = (vmax - vmin) / max(nbins, 1)

    counts =
      assigns.values
      |> Enum.filter(&finite?/1)
      |> Enum.reduce(%{}, fn v, acc ->
        idx = min(trunc((v - vmin) / max(bw, 1.0e-9)), nbins - 1)
        Map.update(acc, idx, 1, &(&1 + 1))
      end)

    max_count = counts |> Map.values() |> Enum.max(fn -> 1 end)
    xscale = fn v -> pad.l + (v - vmin) / max(vmax - vmin, 1.0e-9) * (w - pad.l - pad.r) end
    ticks = nice_ticks(vmin, vmax, 6)

    assigns =
      assign(assigns,
        w: w,
        h: h,
        pad: pad,
        counts: counts,
        max_count: max_count,
        nbins: nbins,
        bw: bw,
        vmin: vmin,
        xscale: xscale,
        ticks: ticks
      )

    ~H"""
    <svg
      id={@id}
      viewBox={"0 0 #{@w} #{@h}"}
      class={["w-full", @class]}
      role="img"
      aria-labelledby={chart_labelledby(@id)}
      focusable="false"
    >
      <title id={chart_title_id(@id)}>{chart_title(@id, @title, "Histogram")}</title>
      <desc id={chart_desc_id(@id)}>{chart_desc(@desc)}</desc>
      <g :for={i <- 0..(@nbins - 1)}>
        <% count = Map.get(@counts, i, 0) %>
        <% bar_h = count / @max_count * (@h - @pad.t - @pad.b) %>
        <rect
          :if={count > 0}
          x={@xscale.(@vmin + i * @bw) + 1}
          y={@h - @pad.b - bar_h}
          width={max(@xscale.(@vmin + (i + 1) * @bw) - @xscale.(@vmin + i * @bw) - 2, 1)}
          height={bar_h}
          fill="var(--color-model)"
          opacity="0.55"
          rx="1.5"
        />
      </g>
      <line
        x1={@pad.l}
        x2={@w - @pad.r}
        y1={@h - @pad.b}
        y2={@h - @pad.b}
        stroke="var(--color-ink-faint)"
        stroke-width="1"
      />
      <g :for={t <- @ticks} class="font-data">
        <text
          x={@xscale.(t)}
          y={@h - @pad.b + 15}
          text-anchor="middle"
          font-size="10"
          fill="var(--color-ink-faint)"
        >
          {fmt_tick(t)}
        </text>
      </g>
      <g :for={vl <- @vlines}>
        <line
          x1={@xscale.(vl.x)}
          x2={@xscale.(vl.x)}
          y1={@pad.t}
          y2={@h - @pad.b}
          stroke={color(Map.get(vl, :color, :lift))}
          stroke-width="1.75"
          stroke-dasharray="4 3"
        />
        <text
          :if={vl[:label]}
          x={@xscale.(vl.x) + 5}
          y={@pad.t + 10}
          font-size="10"
          fill={color(Map.get(vl, :color, :lift))}
          class="font-data"
        >
          {vl.label}
        </text>
      </g>
    </svg>
    """
  end

  # ── scale helpers ─────────────────────────────────────────────────

  defp build_scale(x_min, x_max, y_min, y_max, w, h, pad) do
    %{
      x_min: x_min,
      x_max: max(x_max, x_min + 1),
      y_min: y_min,
      y_max: if(y_max == y_min, do: y_min + 1, else: y_max),
      plot_w: w - pad.l - pad.r,
      plot_h: h - pad.t - pad.b,
      l: pad.l,
      t: pad.t
    }
  end

  defp sx(s, x), do: Float.round(s.l + (x - s.x_min) / (s.x_max - s.x_min) * s.plot_w, 2)
  defp sy(s, y), do: Float.round(s.t + (s.y_max - y) / (s.y_max - s.y_min) * s.plot_h, 2)

  defp line_path(scale, series) do
    offset = Map.get(series, :x_offset, 0)

    series.points
    |> Enum.with_index(offset)
    |> Enum.filter(fn {y, _} -> finite?(y) end)
    |> Enum.map_join(" ", fn {y, i} ->
      prefix = if i == offset, do: "M", else: "L"
      "#{prefix}#{sx(scale, i)},#{sy(scale, y)}"
    end)
  end

  defp band_path(scale, band) do
    offset = Map.get(band, :x_offset, 0)
    upper = Enum.with_index(band.upper, offset)
    lower = band.lower |> Enum.with_index(offset) |> Enum.reverse()

    up =
      Enum.map_join(upper, " ", fn {y, i} ->
        prefix = if i == offset, do: "M", else: "L"
        "#{prefix}#{sx(scale, i)},#{sy(scale, clamp_finite(y))}"
      end)

    down =
      Enum.map_join(lower, " ", fn {y, i} ->
        "L#{sx(scale, i)},#{sy(scale, clamp_finite(y))}"
      end)

    up <> " " <> down <> " Z"
  end

  defp finite?(v) when is_number(v), do: v == v and v not in [:infinity, :neg_infinity]
  defp finite?(_), do: false

  defp clamp_finite(v), do: if(finite?(v), do: v, else: 0.0)

  defp finite_min_max(values) do
    finite = Enum.filter(values, &finite?/1)

    case finite do
      [] -> {0.0, 1.0}
      _ -> {Enum.min(finite) / 1, Enum.max(finite) / 1}
    end
  end

  defp pad_domain({min, max}) do
    span = max - min
    pad = if span == 0, do: max(abs(min) * 0.1, 1.0), else: span * 0.08
    {min - pad, max + pad}
  end

  @doc false
  def nice_ticks(min, max, target) when max > min do
    raw_step = (max - min) / target
    mag = :math.pow(10, :math.floor(:math.log10(raw_step)))
    residual = raw_step / mag

    step =
      cond do
        residual > 5 -> 10 * mag
        residual > 2 -> 5 * mag
        residual > 1 -> 2 * mag
        true -> mag
      end

    first = :math.ceil(min / step) * step
    Stream.iterate(first, &(&1 + step)) |> Enum.take_while(&(&1 <= max + step * 0.001))
  end

  def nice_ticks(min, _max, _target), do: [min]

  defp fmt_tick(v) when is_integer(v), do: Integer.to_string(v)

  defp fmt_tick(v) when is_float(v) do
    cond do
      abs(v) >= 1000 -> :erlang.float_to_binary(v / 1000, decimals: 1) <> "k"
      abs(v - trunc(v)) < 1.0e-9 -> Integer.to_string(trunc(v))
      abs(v) >= 10 -> :erlang.float_to_binary(v, decimals: 1)
      true -> :erlang.float_to_binary(v, decimals: 2)
    end
  end

  defp fmt_cell(v) when is_float(v) do
    if abs(v - trunc(v)) < 1.0e-9,
      do: Integer.to_string(trunc(v)),
      else: :erlang.float_to_binary(v, decimals: 1)
  end

  defp fmt_cell(v), do: to_string(v)

  defp cell_opacity(val) do
    v = min(abs(val * 1.0), 1.0)
    Float.round(0.45 + 0.55 * v, 2)
  end

  defp chart_labelledby(id), do: "#{chart_title_id(id)} #{chart_desc_id(id)}"
  defp chart_title_id(id), do: "#{id}-title"
  defp chart_desc_id(id), do: "#{id}-desc"

  defp chart_title(_id, title, _kind) when is_binary(title) and title != "", do: title

  defp chart_title(id, _title, kind), do: "#{kind}: #{humanize_id(id)}"

  defp chart_desc(desc) when is_binary(desc) and desc != "", do: desc

  defp chart_desc(_desc), do: "The surrounding figure caption summarizes this chart."

  defp humanize_id(id) do
    id
    |> String.replace(~r/[-_]+/, " ")
    |> String.trim()
  end
end
