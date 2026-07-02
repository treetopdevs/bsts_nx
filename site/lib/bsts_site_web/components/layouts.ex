defmodule BstsSiteWeb.Layouts do
  @moduledoc """
  Site chrome: header with the three tracks, quiet footer, flash handling.
  Single "paper" theme — a technical report is printed on paper.
  """
  use BstsSiteWeb, :html

  embed_templates "layouts/*"

  @doc """
  The app layout. `track` highlights the active track in the nav.
  """
  attr :flash, :map, required: true, doc: "the map of flash messages"
  attr :track, :atom, default: nil, values: [nil, :questions, :engine, :trust, :speed, :start]

  attr :current_scope, :map,
    default: nil,
    doc: "the current [scope](https://phoenix.hexdocs.pm/scopes.html)"

  slot :inner_block, required: true

  def app(assigns) do
    ~H"""
    <header class="border-b border-(--color-grid) bg-(--color-paper-raised)">
      <div class="mx-auto flex max-w-5xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3 sm:px-6">
        <.link navigate={~p"/"} class="flex items-baseline gap-2">
          <span class="font-display text-xl font-bold tracking-tight">BstsNx</span>
          <span class="hidden font-data text-[0.65rem] uppercase tracking-widest text-(--color-ink-faint) sm:inline">
            Bayesian structural time series · pure Elixir
          </span>
        </.link>
        <nav class="ml-auto flex flex-wrap items-center gap-1 font-data text-xs">
          <.nav_link navigate={~p"/demos"} active={@track == :questions}>Questions</.nav_link>
          <.nav_link navigate={~p"/engine"} active={@track == :engine}>Engine</.nav_link>
          <.nav_link navigate={~p"/trust"} active={@track == :trust}>Trust</.nav_link>
          <.nav_link navigate={~p"/speed"} active={@track == :speed}>Speed</.nav_link>
          <.link
            navigate={~p"/start"}
            class={[
              "ml-1 rounded-full border px-3 py-1.5 transition-colors",
              @track == :start && "border-(--color-ink) bg-(--color-ink) text-(--color-paper)",
              @track != :start &&
                "border-(--color-ink) hover:bg-(--color-ink) hover:text-(--color-paper)"
            ]}
          >
            Get started
          </.link>
        </nav>
      </div>
    </header>

    <main class="mx-auto max-w-5xl px-4 py-10 sm:px-6">
      {render_slot(@inner_block)}
    </main>

    <footer class="mt-16 border-t border-(--color-grid)">
      <div class="mx-auto flex max-w-5xl flex-wrap items-start justify-between gap-6 px-4 py-8 sm:px-6">
        <div class="max-w-md">
          <div class="font-display font-bold">BstsNx</div>
          <p class="mt-1 text-sm text-(--color-ink-soft)">
            An early-stage, pure-Elixir library for Bayesian structural time series,
            validated against the R <span class="font-data text-xs">bsts</span> /
            <span class="font-data text-xs">CausalImpact</span> reference stack.
            LGPL-2.1 licensed.
          </p>
        </div>
        <div class="font-data text-xs leading-6 text-(--color-ink-soft)">
          <div class="uppercase tracking-wider text-(--color-ink-faint)">Go deeper</div>
          <.link navigate={~p"/start"} class="block hover:text-(--color-ink)">
            Install &amp; quick start
          </.link>
          <a
            :if={github_url()}
            href={github_url()}
            class="block hover:text-(--color-ink)"
          >
            Source on GitHub
          </a>
          <.link navigate={~p"/trust/honesty"} class="block hover:text-(--color-ink)">
            What this is and isn't
          </.link>
        </div>
        <div class="font-data text-[0.65rem] text-(--color-ink-faint)">
          Every chart on this site is computed on this server, live,<br />
          then drawn as SVG by Phoenix LiveView. No screenshots.
        </div>
      </div>
    </footer>

    <.flash_group flash={@flash} />
    """
  end

  attr :navigate, :string, required: true
  attr :active, :boolean, default: false
  slot :inner_block, required: true

  defp nav_link(assigns) do
    ~H"""
    <.link
      navigate={@navigate}
      class={[
        "rounded-full px-3 py-1.5 transition-colors",
        @active && "bg-(--color-base-200) text-(--color-ink) font-medium",
        !@active && "text-(--color-ink-soft) hover:text-(--color-ink)"
      ]}
    >
      {render_slot(@inner_block)}
    </.link>
    """
  end

  defp github_url, do: Application.get_env(:bsts_site, :github_url)

  @doc """
  Shows the flash group with standard titles and content.
  """
  attr :flash, :map, required: true, doc: "the map of flash messages"
  attr :id, :string, default: "flash-group", doc: "the optional id of flash container"

  def flash_group(assigns) do
    ~H"""
    <div id={@id} aria-live="polite">
      <.flash kind={:info} flash={@flash} />
      <.flash kind={:error} flash={@flash} />

      <.flash
        id="client-error"
        kind={:error}
        title="Connection lost"
        phx-disconnected={
          show(".phx-client-error #client-error")
          |> JS.remove_attribute("hidden", to: ".phx-client-error #client-error")
        }
        phx-connected={hide("#client-error") |> JS.set_attribute({"hidden", ""})}
        hidden
      >
        Live demos need the socket — reconnecting
        <.icon name="hero-arrow-path" class="ml-1 size-3 motion-safe:animate-spin" />
      </.flash>

      <.flash
        id="server-error"
        kind={:error}
        title="Something went wrong"
        phx-disconnected={
          show(".phx-server-error #server-error")
          |> JS.remove_attribute("hidden", to: ".phx-server-error #server-error")
        }
        phx-connected={hide("#server-error") |> JS.set_attribute({"hidden", ""})}
        hidden
      >
        Reconnecting
        <.icon name="hero-arrow-path" class="ml-1 size-3 motion-safe:animate-spin" />
      </.flash>
    </div>
    """
  end
end
