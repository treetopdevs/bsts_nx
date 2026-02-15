defmodule BstsNx.MixProject do
  use Mix.Project

  @moduledoc """
  Mix project for the **BstsNx** library.  This project provides
  foundational components for building Bayesian Structural Time
  Series (BSTS) models in pure Elixir using the Nx numerical
  library.  It includes a Kalman filter implementation and a
  Gibbs sampler skeleton for state-space models.
  """

  def project do
    [
      app: :bsts_nx,
      version: "0.1.0",
      elixir: "~> 1.14",
      name: "BstsNx",
      description: description(),
      source_url: "https://github.com/Cleveland-Software-LLC/bsts_elixir",
      homepage_url: "https://github.com/Cleveland-Software-LLC/bsts_elixir",
      docs: docs(),
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [
      # Nx provides numerical computing primitives used throughout the library.
      {:nx, "~> 0.6"},
      # EXLA optionally enables just-in-time compilation and hardware acceleration.
      # To use EXLA, add it to your dependencies and configure Nx to use the
      # EXLA backend in your application (see README for details).  This
      # dependency is optional and will be ignored unless explicitly
      # installed.
      {:exla, "~> 0.6", optional: true},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false},
      {:stream_data, "~> 1.0", only: [:test, :dev]}
    ]
  end

  defp description do
    "Bayesian Structural Time Series (BSTS) utilities for Elixir/Nx."
  end

  defp docs do
    [
      main: "readme",
      extras: [
        "README.md",
        "docs/getting-started.md",
        "docs/components.md",
        "docs/causal-impact.md"
      ],
      groups_for_extras: [
        Guides: [
          "docs/getting-started.md",
          "docs/components.md",
          "docs/causal-impact.md"
        ]
      ]
    ]
  end
end
