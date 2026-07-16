defmodule BstsSiteWeb.Router do
  use BstsSiteWeb, :router

  pipeline :browser do
    plug :accepts, ["html"]
    plug :fetch_session
    plug :fetch_live_flash
    plug :put_root_layout, html: {BstsSiteWeb.Layouts, :root}
    plug :protect_from_forgery
    plug :put_secure_browser_headers
  end

  pipeline :api do
    plug :accepts, ["json"]
  end

  scope "/", BstsSiteWeb do
    pipe_through :browser

    live "/", HubLive

    live "/demos", DemosIndexLive
    live "/demos/marketing", Demos.MarketingLive
    live "/demos/tv", Demos.TvLive
    live "/demos/demand", Demos.DemandLive
    live "/demos/anomaly", Demos.AnomalyLive
    live "/demos/policy", Demos.PolicyLive

    live "/engine", EngineIndexLive
    live "/engine/noise", Engine.NoiseLive
    live "/engine/kalman", Engine.KalmanLive
    live "/engine/smoother", Engine.SmootherLive
    live "/engine/gibbs", Engine.GibbsLive
    live "/engine/compose", Engine.ComposeLive
    live "/engine/counterfactual", Engine.CounterfactualLive

    live "/trust", TrustIndexLive
    live "/trust/calibration", Trust.CalibrationLive
    live "/trust/diagnostics", Trust.DiagnosticsLive
    live "/trust/honesty", Trust.HonestyLive

    live "/speed", SpeedLive
    live "/start", StartLive
  end

  # Other scopes may use custom stacks.
  # scope "/api", BstsSiteWeb do
  #   pipe_through :api
  # end

  # Enable LiveDashboard in development
  if Application.compile_env(:bsts_site, :dev_routes) do
    # If you want to use the LiveDashboard in production, you should put
    # it behind authentication and allow only admins to access it.
    # If your application does not have an admins-only section yet,
    # you can use Plug.BasicAuth to set up some basic authentication
    # as long as you are also using SSL (which you should anyway).
    import Phoenix.LiveDashboard.Router

    scope "/dev" do
      pipe_through :browser

      live_dashboard "/dashboard", metrics: BstsSiteWeb.Telemetry
    end
  end
end
