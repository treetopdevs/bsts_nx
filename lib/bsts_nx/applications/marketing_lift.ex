defmodule BstsNx.Applications.MarketingLift do
  @moduledoc """
  Incrementality measurement for digital marketing campaigns.

  Measures the causal lift (incremental conversions, revenue, etc.) of
  marketing campaigns using Bayesian Structural Time Series.  Works with
  any marketing channel that has discrete on/off periods:

  - **Podcast ads**: episode drop → website visits, promo code usage
  - **Influencer posts**: post timestamp → social engagement, site traffic
  - **Radio spots**: air time → store visits, calls, web traffic
  - **Email campaigns**: send time → opens, clicks, conversions
  - **Push notifications**: send time → app opens, purchases
  - **Paid search**: campaign on/off → organic search lift
  - **Out-of-home**: campaign start → location-based traffic

  ## How it works

  For each campaign, the module:
  1. Fits a state-space model to pre-campaign baseline behavior
  2. Generates a Bayesian counterfactual ("what if we hadn't run the campaign?")
  3. Computes the incremental lift as the difference between actual and counterfactual
  4. Provides posterior credible intervals for uncertainty quantification
  5. For overlapping campaigns, uses Shapley allocation to attribute lift fairly

  ## Key concepts

  - **Incrementality**: Did this campaign cause *additional* conversions, or would
    they have happened anyway? This is the causal question BSTS answers.
  - **Counterfactual**: The predicted baseline behavior if the campaign had not run.
  - **Lift**: The difference between actual and counterfactual outcomes.
  - **Control series**: Optional correlated time series unaffected by the campaign
    (e.g., organic traffic from a region without the campaign) that strengthen
    the counterfactual prediction.

  ## Examples

      # Single campaign lift measurement
      result = MarketingLift.measure_lift(
        daily_conversions,
        %{
          name: "summer_email_blast",
          channel: :email,
          start_index: 31,
          end_index: 45,
          baseline_start: 1,
          baseline_end: 30
        }
      )

      result.effect.cumulative   # incremental conversions
      result.effect.relative     # % lift over baseline
      result.significant?        # statistically significant?

      # Multi-campaign with overlaps
      campaigns = [
        %{name: "podcast_a", start_index: 10, end_index: 15, baseline_start: 1, baseline_end: 9},
        %{name: "podcast_b", start_index: 12, end_index: 18, baseline_start: 1, baseline_end: 11}
      ]

      results = MarketingLift.measure_multi(daily_visits, campaigns)
  """

  alias BstsNx.InterventionAnalysis
  alias BstsNx.ModelBuilder

  @type campaign :: %{
          name: String.t(),
          start_index: pos_integer(),
          end_index: pos_integer(),
          baseline_start: pos_integer(),
          baseline_end: pos_integer(),
          channel: atom() | nil
        }

  @type campaign_result :: %{
          campaign: campaign(),
          effect: ModelBuilder.effect(),
          significant?: boolean(),
          baseline_mean: float(),
          actual_mean: float(),
          analysis: BstsNx.InterventionAnalysis.analysis_result()
        }

  @type multi_result :: %{
          campaigns: [campaign_result()],
          total_lift: float(),
          total_lift_sd: float(),
          overlap_groups: [[String.t()]]
        }

  @doc """
  Measures the incremental lift of a single marketing campaign.

  ## Parameters

    * `observations` - time series of the response metric (conversions,
      revenue, visits, etc.) covering both baseline and campaign periods
    * `campaign` - map defining the campaign (see type `campaign()`)

  ## Options

    * `:seasonality` - seasonal periods (e.g., 7 for daily data with
      weekly pattern)
    * `:control_series` - list of control time series (same length as
      `observations`) that are correlated with the response but unaffected
      by the campaign. These become regression covariates.
    * `:control_selection` - optional control-series selection before
      composing the regression component. Set to `true` for default Pearson
      screening, or pass options for `BstsNx.CovariateSelection.select/3`.
    * `:control_regression_mode` - `:dynamic` (default) or `:spike_and_slab`
      for in-loop Bayesian variable selection with a g-prior.
    * `:control_regression_opts` - keyword options forwarded to
      `Components.regression_spec/2` when controls are present.
    * `:alpha` - significance level (default: 0.05)
    * `:num_samples` - posterior MCMC draws (default: 200)
    * `:seed` - PRNG seed for reproducibility
  """
  @spec measure_lift([number()], campaign(), keyword()) :: campaign_result()
  def measure_lift(observations, campaign, opts \\ []) do
    config = %{
      pre_period: {campaign.baseline_start, campaign.baseline_end},
      post_period: {campaign.start_index, campaign.end_index}
    }

    analysis_opts = build_analysis_opts(observations, campaign, opts)
    analysis = InterventionAnalysis.analyze(observations, config, analysis_opts)

    build_campaign_result(campaign, analysis, observations)
  end

  @doc """
  Measures incremental lift for multiple campaigns, handling overlaps.

  When campaigns overlap in time, uses Shapley allocation via
  `SpotAttributor` to fairly distribute the joint lift.  When campaigns
  don't overlap, each is measured independently.

  ## Parameters

    * `observations` - time series of the response metric
    * `campaigns` - list of campaign definitions

  ## Options

  Same as `measure_lift/3`, plus:
    * `:decay` - diminishing returns decay for Shapley (default: 0.7)

  ## Examples

      iex> result = BstsNx.Applications.MarketingLift.measure_multi([1.0, 2.0, 3.0], [])
      iex> result.total_lift
      0.0
      iex> result.campaigns
      []
  """
  @spec measure_multi([number()], [campaign()], keyword()) :: multi_result()
  def measure_multi(observations, campaigns, opts \\ [])

  def measure_multi(_observations, [], _opts) do
    %{campaigns: [], total_lift: 0.0, total_lift_sd: 0.0, overlap_groups: []}
  end

  def measure_multi(observations, campaigns, opts) do
    # Detect overlapping campaigns
    groups = detect_campaign_overlaps(campaigns)

    overlap_groups =
      groups
      |> Enum.filter(fn g -> length(g) > 1 end)
      |> Enum.map(fn g -> Enum.map(g, & &1.name) end)

    # Process each overlap-connected group.
    group_results =
      Enum.map(groups, fn group ->
        if length(group) == 1 do
          campaign = hd(group)
          result = measure_lift(observations, campaign, opts)
          %{campaigns: [result], group_sd: result.effect.cumulative_sd}
        else
          measure_overlapping_group(observations, group, opts)
        end
      end)

    campaign_results = Enum.flat_map(group_results, & &1.campaigns)

    total_lift = Enum.reduce(campaign_results, 0.0, fn r, acc -> acc + r.effect.cumulative end)

    total_lift_sd =
      group_results
      |> Enum.reduce(0.0, fn g, acc -> acc + g.group_sd * g.group_sd end)
      |> :math.sqrt()

    %{
      campaigns: campaign_results,
      total_lift: total_lift,
      total_lift_sd: total_lift_sd,
      overlap_groups: overlap_groups
    }
  end

  @doc """
  Convenience: measures lift using date-indexed data.

  Converts date-based campaign definitions to index-based ones using
  the provided date series, then delegates to `measure_lift/3`.

  ## Parameters

    * `date_series` - list of dates corresponding to each observation
    * `observations` - response metric values
    * `campaign` - map with `:start_date`, `:end_date`, `:baseline_start_date`,
      `:baseline_end_date` keys (Date structs)
  """
  @spec measure_lift_by_date([Date.t()], [number()], map(), keyword()) :: campaign_result()
  def measure_lift_by_date(date_series, observations, campaign, opts \\ []) do
    idx_campaign = dates_to_indices(date_series, campaign)
    measure_lift(observations, idx_campaign, opts)
  end

  # -- Private implementation --

  defp build_analysis_opts(observations, campaign, opts) do
    controls = Keyword.get(opts, :control_series)

    opts_with_selection_window =
      Keyword.put_new(
        opts,
        :control_selection_pre_period,
        {campaign.baseline_start, campaign.baseline_end}
      )

    ModelBuilder.build_opts_with_controls(observations, controls, opts_with_selection_window)
  end

  defp build_campaign_result(campaign, analysis, observations) do
    {bs, be} = {campaign.baseline_start, campaign.baseline_end}
    {cs, ce} = {campaign.start_index, campaign.end_index}
    n_post = ce - cs + 1

    baseline_vals = Enum.slice(observations, (bs - 1)..(be - 1))
    actual_vals = Enum.slice(observations, (cs - 1)..(ce - 1))

    baseline_mean =
      if baseline_vals == [], do: 0.0, else: Enum.sum(baseline_vals) / length(baseline_vals)

    actual_mean =
      if actual_vals == [], do: 0.0, else: Enum.sum(actual_vals) / length(actual_vals)

    %{
      campaign: campaign,
      effect: ModelBuilder.build_effect(analysis.summary, n_post),
      significant?: analysis.significant?,
      baseline_mean: baseline_mean,
      actual_mean: actual_mean,
      analysis: analysis
    }
  end

  defp measure_overlapping_group(observations, group, opts) do
    # For overlapping campaigns, find the earliest baseline and latest campaign end,
    # run a single CausalImpact analysis, then use SpotAttributor to allocate
    earliest_baseline = Enum.min_by(group, & &1.baseline_start).baseline_start
    latest_baseline_end = Enum.max_by(group, & &1.baseline_end).baseline_end
    earliest_campaign = Enum.min_by(group, & &1.start_index).start_index
    latest_campaign = Enum.max_by(group, & &1.end_index).end_index

    config = %{
      pre_period: {earliest_baseline, latest_baseline_end},
      post_period: {earliest_campaign, latest_campaign}
    }

    # Use the first campaign's channel for analysis opts
    first = hd(group)
    analysis_opts = build_analysis_opts(observations, first, opts)
    analysis = InterventionAnalysis.analyze(observations, config, analysis_opts)

    # Convert campaigns to spots for SpotAttributor (0-based windows)
    post_offset = earliest_campaign

    spots =
      Enum.map(group, fn c ->
        %{
          id: c.name,
          window_start: c.start_index - post_offset,
          window_end: c.end_index - post_offset + 1
        }
      end)

    # If we have posterior draws, use posterior propagation.
    # Otherwise fall back to separate per-campaign analyses to avoid assigning
    # the same group-level effect to every campaign.
    case analysis.impact do
      nil ->
        campaigns = Enum.map(group, &measure_lift(observations, &1, opts))
        %{campaigns: campaigns, group_sd: pooled_sd(campaigns)}

      impact ->
        attr_result =
          BstsNx.SpotAttributor.attribute_posterior(
            impact.actual,
            spots,
            impact.counterfactual,
            0.0,
            Keyword.take(opts, [:alpha, :decay])
          )

        campaigns =
          Enum.map(group, fn c ->
            attr =
              Enum.find(attr_result.attributions, fn a -> a.spot_id == c.name end) ||
                %{lift: 0.0, lift_sd: 0.0, lift_lower: 0.0, lift_upper: 0.0, p_positive: 0.5}

            n_post = c.end_index - c.start_index + 1
            cumulative = ModelBuilder.safe_number(attr.lift)
            cumulative_sd = ModelBuilder.safe_number(attr.lift_sd)
            cumulative_lower = ModelBuilder.safe_number(attr.lift_lower)
            cumulative_upper = ModelBuilder.safe_number(attr.lift_upper)

            # Compute baseline and actual from observations
            baseline_vals = Enum.slice(observations, (c.baseline_start - 1)..(c.baseline_end - 1))
            actual_vals = Enum.slice(observations, (c.start_index - 1)..(c.end_index - 1))
            baseline_sum = Enum.sum(baseline_vals)

            baseline_mean =
              if baseline_vals == [], do: 0.0, else: baseline_sum / length(baseline_vals)

            actual_mean =
              if actual_vals == [], do: 0.0, else: Enum.sum(actual_vals) / length(actual_vals)

            baseline_total_for_window = baseline_mean * n_post

            relative =
              if abs(baseline_total_for_window) > 1.0e-10 do
                cumulative / baseline_total_for_window
              else
                0.0
              end

            relative_sd =
              if abs(baseline_total_for_window) > 1.0e-10 do
                cumulative_sd / abs(baseline_total_for_window)
              else
                0.0
              end

            %{
              campaign: c,
              effect: %{
                cumulative: cumulative,
                cumulative_sd: cumulative_sd,
                cumulative_lower: cumulative_lower,
                cumulative_upper: cumulative_upper,
                per_period: if(n_post > 0, do: cumulative / n_post, else: 0.0),
                per_period_sd: if(n_post > 0, do: cumulative_sd / n_post, else: 0.0),
                relative: relative,
                relative_sd: relative_sd
              },
              significant?: cumulative_lower > 0.0 or cumulative_upper < 0.0,
              baseline_mean: baseline_mean,
              actual_mean: actual_mean,
              analysis: analysis
            }
          end)

        %{
          campaigns: campaigns,
          group_sd: ModelBuilder.safe_number(attr_result.total_lift_sd)
        }
    end
  end

  defp pooled_sd(campaigns) do
    campaigns
    |> Enum.reduce(0.0, fn c, acc -> acc + c.effect.cumulative_sd * c.effect.cumulative_sd end)
    |> :math.sqrt()
  end

  defp detect_campaign_overlaps(campaigns) do
    # Convert campaigns to spot-like structures for overlap detection
    spots =
      Enum.map(campaigns, fn c ->
        %{id: c.name, window_start: c.start_index, window_end: c.end_index + 1}
      end)

    groups = BstsNx.ShapleyAllocator.detect_overlaps(spots)

    # Map back to campaign structs
    campaign_map = Map.new(campaigns, fn c -> {c.name, c} end)

    Enum.map(groups, fn group ->
      Enum.map(group, fn spot -> Map.fetch!(campaign_map, spot.id) end)
    end)
  end

  defp dates_to_indices(date_series, campaign) do
    date_index = Map.new(Enum.with_index(date_series, 1), fn {date, idx} -> {date, idx} end)

    %{
      name: campaign[:name] || "campaign",
      channel: campaign[:channel],
      start_index: Map.fetch!(date_index, campaign.start_date),
      end_index: Map.fetch!(date_index, campaign.end_date),
      baseline_start: Map.fetch!(date_index, campaign.baseline_start_date),
      baseline_end: Map.fetch!(date_index, campaign.baseline_end_date)
    }
  end
end
