defmodule BstsNx.Applications.TVAttribution do
  @moduledoc """
  TV-to-web attribution using Bayesian Structural Time Series.

  Measures the causal lift of TV ad spots on digital response metrics
  (website visits, app opens, conversions) by fitting a BSTS model to
  pre-airing baseline traffic and comparing post-airing actual response
  to the counterfactual baseline.

  This module wraps `BstsNx.Pipeline`, `BstsNx.SpotAttributor`, and
  `BstsNx.RollingBaseline` with a domain-specific API for TV attribution.

  ## Key features

  - **Per-spot attribution**: Measures incremental lift for each TV spot
  - **Overlap handling**: Shapley allocation for temporally overlapping spots
  - **Rolling baselines**: Warm-start support for nightly batch processing
  - **Posterior propagation**: Preserves correlation structure across time steps
  - **Uncertainty quantification**: Full credible intervals for each spot's lift

  ## Typical workflow

      # 1. Define spots that aired
      spots = [
        %{id: "spot_abc", window_start: 0, window_end: 5},
        %{id: "spot_def", window_start: 3, window_end: 8}
      ]

      # 2. Run attribution pipeline
      result = TVAttribution.attribute(
        minute_level_traffic,
        spots,
        pre_period: {1, 60},
        post_period: {61, 75},
        seasonality: 96    # 15-min intervals in a day
      )

      result.spots          # per-spot lift with CIs
      result.total_lift     # aggregate incremental traffic
      result.overlaps       # which spots overlapped

      # 3. Rolling baseline for batch processing
      {baseline, fit} = TVAttribution.rolling_baseline(
        day_traffic,
        horizon: 15,
        warm_start: previous_fit
      )
  """

  alias BstsNx.ModelBuilder
  alias BstsNx.Pipeline
  alias BstsNx.RollingBaseline
  alias BstsNx.SpotAttributor

  @type spot :: SpotAttributor.spot()

  @type attribution_result :: %{
          spots: [SpotAttributor.spot_attribution()],
          total_lift: float(),
          total_lift_sd: float(),
          overlaps: [[String.t()]],
          causal_impact_summary: map(),
          raw_pipeline: Pipeline.pipeline_result() | nil
        }

  @doc """
  Runs the full TV attribution pipeline.

  ## Parameters

    * `observations` - minute-level (or sub-minute) traffic time series
    * `spots` - list of spot definitions with 0-based half-open windows
      `[window_start, window_end)` relative to post-period start
    * `opts` - keyword options (see below)

  ## Options

    * `:pre_period` - `{start, end}` (1-based inclusive) for baseline fitting
    * `:post_period` - `{start, end}` (1-based inclusive) for impact measurement
    * `:seasonality` - seasonal periods (e.g., 96 for 15-min data with daily cycle)
    * `:model_spec` - custom `%BstsNx.ModelSpec{}` (overrides `:seasonality`)
    * `:regressors` - a `{T, p}` Nx tensor of regressor values (e.g., weather,
      competitor ad activity) to include as covariates in the baseline model
    * `:num_samples` - posterior MCMC draws (default: 200)
    * `:alpha` - significance level for CIs (default: 0.05)
    * `:seed` - PRNG seed
    * `:decay` - Shapley diminishing returns decay (default: 0.7)
  """
  @spec attribute([number()], [spot()], keyword()) :: attribution_result()
  def attribute(observations, spots, opts \\ []) do
    pre_period = Keyword.fetch!(opts, :pre_period)
    post_period = Keyword.fetch!(opts, :post_period)

    spec = resolve_spec(observations, pre_period, opts)

    pipeline_opts =
      opts
      |> Keyword.drop([:pre_period, :post_period, :seasonality, :model_spec, :regressors])

    result = Pipeline.run(observations, pre_period, post_period, spots, spec, pipeline_opts)

    %{
      spots: result.attributions.attributions,
      total_lift: result.attributions.total_lift,
      total_lift_sd: result.attributions.total_lift_sd,
      overlaps: result.attributions.overlap_groups,
      causal_impact_summary: result.summary,
      raw_pipeline: result
    }
  end

  @doc """
  Computes a rolling baseline for batch processing.

  Delegates to `BstsNx.RollingBaseline.fit_and_predict/3` with TV-specific
  defaults.

  ## Options

    * `:horizon` - number of post-period steps to predict (required)
    * `:num_seasons` - seasonal periods (default: 96)
    * `:warm_start` - previous fit result for warm-starting
    * `:num_samples` - posterior draws (default: 200)
    * `:seed` - PRNG seed
    * `:regressors` - a `{T, p}` Nx tensor of regressors for the training period.
      Passed through to `RollingBaseline.fit_and_predict/3`.
  """
  @spec rolling_baseline([number()], keyword()) ::
          {%{mean: [float()], variance: [float()], obs_variance: float()},
           RollingBaseline.fit_result()}
  def rolling_baseline(observations, opts \\ []) do
    horizon = Keyword.fetch!(opts, :horizon)

    baseline_opts =
      opts
      |> Keyword.delete(:horizon)
      |> Keyword.put_new(:num_seasons, 96)

    RollingBaseline.fit_and_predict(observations, horizon, baseline_opts)
  end

  @doc """
  Attributes pre-computed counterfactual baselines to spots.

  Use this when you've already computed the counterfactual (e.g., from
  `rolling_baseline/2`) and just need spot-level attribution.

  ## Examples

      iex> obs = [100.0, 110.0, 120.0, 105.0, 115.0]
      iex> spots = [%{id: "spot1", window_start: 0, window_end: 3}]
      iex> cf = %{mean: [100.0, 100.0, 100.0, 100.0, 100.0], variance: [1.0, 1.0, 1.0, 1.0, 1.0], obs_variance: 1.0}
      iex> result = BstsNx.Applications.TVAttribution.attribute_from_baseline(obs, spots, cf)
      iex> is_list(result.attributions)
      true
  """
  @spec attribute_from_baseline(
          [number()],
          [spot()],
          SpotAttributor.counterfactual(),
          keyword()
        ) :: SpotAttributor.attribution_result()
  def attribute_from_baseline(observations, spots, counterfactual, opts \\ []) do
    SpotAttributor.attribute(observations, spots, counterfactual, opts)
  end

  # -- Private --

  defp resolve_spec(observations, {pre_start, _pre_end}, opts) do
    # TV attribution defaults to 96 seasonal periods (15-min intervals in a day)
    tv_opts =
      opts
      |> Keyword.take([:model_spec, :seasonality, :regressors])
      |> Keyword.put_new(:seasonality, 96)

    obs_at_start = [Enum.at(observations, pre_start - 1) || 0.0]
    {spec, _method} = ModelBuilder.build_spec(obs_at_start, tv_opts)
    spec
  end
end
