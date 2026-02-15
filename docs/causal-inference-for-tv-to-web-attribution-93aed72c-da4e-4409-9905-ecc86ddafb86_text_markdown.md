# Causal inference for TV-to-web attribution

**Bayesian Structural Time Series combined with spot-level regression discontinuity offers the strongest methodological foundation for production TV attribution systems.** BSTS excels at counterfactual baseline estimation—modeling trend, seasonality, and covariates—while RDD-like spike analysis around each ad airing provides the per-spot causal identification that BSTS alone cannot deliver. This hybrid approach, validated by the leading academic papers (Liaukonyte et al. 2015; Arslan et al. 2024) and mirrored by industry vendors like Tatari, Innovid, and Google's TV Search Lift product, resolves the fundamental tension between campaign-level inference and spot-level attribution. The upgrade from wavelet-based baselines to BSTS is well-supported: BSTS provides principled uncertainty quantification, automatic covariate selection via spike-and-slab priors, and a Bayesian framework that naturally extends to hierarchical and multi-level models.

---

## Part 1: How five causal methods compare for TV attribution

### BSTS/CausalImpact — strong baselines, weak on multiple interventions

Google's CausalImpact (Brodersen et al. 2015, *Annals of Applied Statistics*) implements a diffusion-regression state-space model with three components: a **local linear trend** (random walk), **seasonal components** (day-of-week S=7, weekly S=52), and a **regression component with spike-and-slab priors** for automatic variable selection among control time series. The spike-and-slab prior combines point mass at zero with a weakly informative Gaussian for nonzero coefficients, with default inclusion probability M/J where M=3 expected covariates and J is total predictors. Posterior inference runs via Gibbs sampling (MCMC), implemented in C++ with R and Python (TensorFlow Probability) interfaces.

**The critical limitation for TV attribution is that CausalImpact assumes a single intervention with a clean pre/post split.** A typical TV advertiser runs dozens of spots daily with overlapping response windows—the method was not designed for this. Three workarounds exist in practice:

The **TV-Impact framework** (Arslan et al. 2024, *Entropy*) runs CausalImpact independently for each spot, using the pre-spot period as baseline and a short post-window for measurement. When consecutive ads overlap in response windows, they merge into "Group Ads" and decompose the joint effect using a Random Forest trained on individual ad characteristics. A second approach treats BSTS as a continuous counterfactual model with ad timing as a **time-varying regressor** (binary indicator), avoiding pre/post splitting entirely. Third, the `bsts` R package can build custom state-space models that explicitly include ad response functions in the observation equation.

For TV attribution, CausalImpact's assumptions are **partially satisfied**: the stability of control-series relationships holds over short windows (minutes/hours), and the spike-and-slab prior handles high-dimensional covariate selection well. The assumption that control series are unaffected by the intervention requires careful covariate selection—traffic from non-exposed DMAs, unrelated category search trends, or competitor brand traffic can serve as controls, but national campaigns contaminate all domestic series. Google validated CausalImpact against randomized experiments and found near-identical estimates. Industry adoption includes Google's internal TV Search Lift (8-minute default post-period), Samba TV's causal attribution product, AdQuick for OOH, and the TV-Impact framework deployed across 11 companies.

**Computational feasibility** is moderate: a single CausalImpact run with 500 time points and 10 covariates completes in seconds, but running thousands of per-spot MCMC chains daily requires parallelization infrastructure.

### Synthetic control — ideal for geo-testing, not spot-level work

The Synthetic Control Method (Abadie, Diamond & Hainmueller 2010, *JASA*) constructs a counterfactual by weighting untreated "donor" units to match the treated unit's pre-intervention trajectory. For TV attribution, this maps cleanly onto **geo-level incrementality testing**: DMAs receiving TV ads are compared against a synthetic control built from non-exposed DMAs with similar demographic and economic profiles.

**For per-spot attribution, SCM is fundamentally unsuited.** The method requires distinct pre/post periods for a single treated unit, has no mechanism for dozens of rapid-fire treatments within a day, and the donor-pool concept does not map to minute-level within-day time series. However, for campaign validation, SCM is the **gold standard for geo-experiments**. Meta's open-source GeoLift library implements Augmented Synthetic Control (Ben-Michael, Feller & Rothstein 2021) with power analysis, market selection, and bias-corrected inference. Brodersen et al. (2015) explicitly position BSTS as a "Bayesian generalization of synthetic controls," providing the same counterfactual logic with full posterior inference—a key advantage over SCM's permutation-based placebo tests.

Industry adoption is strong for geo-testing: **Meta's GeoLift**, Eppo's Bayesian Synthetic Control product, and Samba TV's causal attribution all use SCM variants. For the production system described here—which needs per-spot attribution at minute-level granularity—SCM serves as a validation tool rather than the primary method.

### Difference-in-differences — academically validated but requires control groups

DiD compares outcome changes between treatment and control groups, with the **parallel trends assumption** as its identifying condition. The most important academic application to TV attribution is **Liaukonyte, Teixeira & Wilbur (2015, *Marketing Science*)**, which used DiD with **non-advertising competitors' online shopping as the control group** to measure TV effects in 2-hour windows across $3.4 billion in ad spending by 20 brands. Lambrecht, Tucker & Zhang (2024, *Journal of Marketing Research*) used **Synthetic Difference-in-Differences** with a field test where part of the country saw TV ads while another served as control, measuring minute-level effects with county fixed effects.

The parallel trends assumption is **most plausible for short windows** (2 minutes, as in Liaukonyte et al.)—web traffic for a brand and its competitors should follow similar intraday patterns absent an ad airing. For longer windows or cross-market comparisons, SDID and covariate conditioning improve robustness. Modern staggered DiD estimators (Callaway & Sant'Anna 2021; Sun & Abraham 2021) address heterogeneous treatment effects across ad spots but assume **once treated, always treated**—a poor fit for transient TV ad effects lasting minutes.

DiD's computational cost is minimal (OLS regression), making it highly scalable. Its primary limitation is the requirement for clean control groups, which national TV campaigns often preclude.

### Regression discontinuity — the natural fit for spot-level attribution

RDD exploits the **sharp temporal cutoff at the moment of ad airing** as a quasi-experimental design. Liaukonyte et al. (2015) explicitly describe their identification strategy as "similar to the regression discontinuity approach of Hartmann, Nair & Narayanan (2011, *Marketing Science*)," using 2-minute pre/post windows around each airing. Near the cutoff, treatment assignment is effectively random—viewers cannot precisely control whether they are watching at the exact minute an ad airs, and ad timing is determined by the broadcaster's schedule.

**RDD handles multiple treatments more naturally than any other method** because each ad airing creates its own independent local cutoff. Effects are estimated via local polynomial regression around each cutoff, then aggregated across airings. When ads air within the same commercial break, overlapping response windows require merging (the "Group Ads" approach). Bandwidth selection—how many minutes before/after the ad to include—typically ranges from **2-10 minutes**, with standard selectors (Calonico, Cattaneo & Titiunik 2014) adaptable for time-based settings.

RDD estimates a **Local Average Treatment Effect at the cutoff**—the immediate response to an ad—which is precisely what per-spot TV attribution systems measure. It is the **most computationally efficient** method: local polynomial regression per spot is near-instantaneous, and the `rdrobust` package provides robust bias-corrected confidence intervals. While not widely labeled as "RDD" in industry, the underlying logic is exactly what spike-analysis systems implement: comparing traffic just before and just after an ad airs.

### Alternative methods worth considering

**Interrupted Time Series (ITS)** treats each ad as an intervention in a segmented regression framework. The TV-Impact framework and Veverka (2021) both implement ITS-style approaches with kernel smoothing for diurnal patterns and Weibull-distributed response functions. ITS is conceptually simpler than BSTS but lacks its covariate-driven counterfactual.

**Instrumental Variables** face a weak-instrument problem in TV attribution. Moshary, Shapiro & Song (2021, *Marketing Science*) evaluated political advertising cycles as instruments for commercial TV exposure across 274 product categories and found first-stage F-statistics below 10 for at least 221 categories—indicating weak instruments for most applications.

**BART (Bayesian Additive Regression Trees)** has been consistently among the best-performing methods in the Atlantic Causal Inference Data Analysis Challenge (Hill 2011; Hahn, Murray & Carvalho 2020) and could model heterogeneous treatment effects across ad characteristics without pre-specifying functional form. However, **no published application to TV-to-web attribution exists**, and BART does not natively handle the time-series structure of the problem.

**Neural Granger Causality** methods (GC-xLSTM 2025; sparse attention transformers, CODS-COMAD 2024) are advancing rapidly but remain untested for advertising attribution. These could potentially capture nonlinear temporal dependencies but lack the causal identification guarantees of the methods above.

### Comparative assessment at a glance

| Criterion | BSTS/CausalImpact | Synthetic Control | DiD | RDD | ITS |
|---|---|---|---|---|---|
| Multiple spots per day | Requires workarounds | Not designed for this | Possible with staggered DiD | Handles naturally | Each spot is independent |
| Minute-level data | Yes | No (geo-level) | Yes | Yes | Yes |
| Bayesian uncertainty | Full posteriors | Placebo-based | Bootstrap CIs | Robust bias-corrected CIs | Frequentist |
| Computational cost | Moderate (MCMC) | Low (geo) | Low (OLS) | Very low | Low |
| Best use | Counterfactual baseline | Geo-experiment validation | Market-level with controls | Per-spot immediate lift | Per-spot with simple baseline |
| Academic validation for TV | Strong | Strong (geo) | Strongest (Liaukonyte et al.) | Implicit in Liaukonyte et al. | Moderate |

---

## Part 2: Attribution windows and the timing of TV response

### The immediate spike peaks at one minute and decays within eight

Research converges on a remarkably consistent picture of immediate TV response. **Most viewers need approximately one minute to switch to a second screen after seeing an ad** (Veverka 2021), producing a peak in web traffic in the minute immediately following airing. Du, Xu & Wilbur (2019, *Journal of Marketing*) documented a **4.4-fold spike in brand searches** in the minute after national ad insertion for automotive brands, with an average brand search elasticity of **0.09**. Google's internal TV Search Lift methodology uses an **8-minute default post-period**, finding that "TV's impact on search volume is transient in most cases, usually less than 10 minutes" (Liu, Schwarzkopf & Koehler 2017). Tatari reports that spikes return to baseline in under **15 minutes**, and Mercury Media Technology uses a standard window of **5-10 minutes**. Approximately **one-third of all TV ads produce no measurable immediate web traffic impact** (Veverka 2021), a finding with significant implications for attribution system design.

### Delayed response multiplies the immediate effect by 2-8x

The immediate spike captures only a fraction of total TV-driven response. **Tatari's DragFactor metric quantifies this**: the default multiplier is **2-3x**, meaning for every immediate responder (within 5 minutes), 1-2 additional people respond over the following 30 days. For one streaming TV client, the DragFactor reached **8x**—only 12.5% of total response was immediate, and delayed responders converted at **10x the rate** of immediate responders.

Effectv and TVSquared's joint study of hundreds of advertisers found **4.7% average lift** in website visitors within 30 minutes, expanding to a **3-6x lift throughout the following week** and **23% prolonged impact** through the following month. Amazon Science's VARMAX analysis found that within a standard 2-week attribution window, **upper/middle-funnel ad products materialize only 30-50% of their total effects**, while lower-funnel products capture 60-90%. Thinkbox's UK research concluded that roughly **50% of all media-driven response comes in the long term** (3-24 months post-campaign).

### Optimal windows vary dramatically by campaign objective and category

No single attribution window fits all use cases. The window should be calibrated to the campaign's objective and product category:

- **Direct response / app installs**: 5-15 minutes captures the majority of immediate effect. Appropriate for performance TV optimization.
- **Insurance and considered purchases**: ISBA/Aviva's research argues that a 5-10 minute window is fundamentally wrong for categories where buyers are "disinterested for 11 months of the year." Weeks to months are needed.
- **QSR/restaurants**: 24-48 hours captures the decision cycle (JamLoop, EDO data showing **11.5% above-baseline traffic** while on air).
- **Automotive**: 30-day windows reflect the research-intensive purchase journey.
- **Brand campaigns**: At least 1 month (Amazon Science); full brand-building effects require 3-24 months (Thinkbox/GroupM).

**Daypart matters significantly.** Google's Android campaign study found that primetime spots drove more incremental searches per spot, but when normalized per impression and per cost, **weekend non-prime spots were most cost-efficient**. Over 80% of TV-driven web activity comes from mobile devices (Zigmond & Stipp). Ad creative type also modulates timing: direct response TV with clear CTAs generates faster, more concentrated response, while brand response TV produces a "slower burn" responsible for **52% of long-term media impact** and **40% more efficient** at driving long-term response per pound than the next best channels (Thinkbox/GroupM).

### Response decay follows geometric or Weibull functions

The Broadbent adstock model (1979) remains foundational: **A_t = T_t + λ · A_{t-1}**, where λ is the decay rate. Industry practitioners report half-lives of **2-5 weeks** for TV (FMCG average: 2.5 weeks), while academic studies suggest **7-12 weeks**. For minute-level spike attribution, the response function is better modeled as a **log-normal distribution** (Kitts et al. 2014) or **Weibull distribution** (Meta's Robyn). The Weibull's additional shape parameter allows modeling both immediate peaks and delayed response curves—critical for capturing the DragFactor phenomenon.

---

## Part 3: Confounders that sophisticated systems must address

### The concurrent digital advertising problem

The most pernicious confounder in TV attribution is **simultaneous digital advertising**. When a TV spot airs at 8:47 PM and a programmatic display campaign is also running, both systems claim credit for the resulting website visit. Lambrecht et al. (2024) explicitly re-estimated TV effects after excluding sessions from affiliate, display, social, and email marketing (about **16.4% of total sessions**), finding nearly identical direct-impact results—but this filtering is essential.

**Paid search creates a specific double-counting problem**: TV drives branded search, which gets captured by SEM ads. Both TV attribution and SEM attribution claim the conversion. Sellforte documents this "spillover effect" where TV commercials increase brand search queries, boosting SEM click-through rates. Google's bias correction research (arXiv: 1807.03292) addresses this endogeneity—paid search spend responds to consumer demand that is itself driven by TV. The recommended approach is to either filter SEM-originated visits from TV attribution or model the TV→SEM pathway explicitly.

### Temporal patterns require precise modeling

Web traffic follows strong **circadian and weekly rhythms** that must be decomposed from TV effects. CausalImpact models S=7 day-of-week and S=52 weekly seasonal components. Mercury Media Technology's system computes baselines "for every market, every minute of every day." Holiday effects create large traffic shifts—PyMC-Marketing implements Fourier-mode contributions for yearly seasonality, and explicit holiday indicators are standard in all MMM frameworks.

**Weather** affects both TV viewership (more indoor time increases exposure) and web behavior. **Sports events** create unique challenges: massive simultaneous audiences generate large spikes, but the event itself drives web behavior (score checking, social discussion) that confounds spot-level attribution. Zigmond & Stipp's seminal study showed "Chevy Volt" searches spiking during the 2008 Olympics, but separating ad effects from event-driven engagement requires careful control-group selection.

### A recommended covariate framework

Based on the academic literature and industry practice, a production TV attribution system should incorporate covariates in three tiers:

**Tier 1 — Essential (include always):** Time-of-day and day-of-week seasonal components; holiday indicators; traffic from channels unaffected by TV (email, direct/bookmarked, CRM-driven); category-level search trends from Google Trends; ad schedule data (network, creative, spot length, estimated impressions).

**Tier 2 — Important (include when available):** Concurrent digital media spend by channel (SEM, display, social, email); competitor TV ad airings (via iSpot or Kantar); website change log (launches, outages, redesigns); traffic from non-exposed geographies.

**Tier 3 — Contextual (include for precision):** Weather data; economic indicators; major event calendar (sports, elections, award shows); PR/earned media mentions; app store activity; competitor digital campaign timing.

PyMC-Marketing's documentation advocates a **causal DAG approach** to covariate selection: "We need to do a causal analysis to define the causal connections (DAG) and fit the model accordingly so that we do not induce biased estimates." Simulation studies confirm that omitting a confounder that affects both media and outcomes leads to systematic overestimation of channel effects. CausalImpact's spike-and-slab prior provides automatic variable selection among these covariates, but the DAG should determine which covariates are candidates in the first place.

---

## Part 4: What leading vendors and researchers are doing now

### Industry vendors have converged on hybrid architectures

**Tatari** (400+ brands, 18 USPTO patents) operates three distinct attribution methodologies: a proprietary view-through model using a device graph for household-level accuracy; a digital view-through model aligned with platform standards for cross-channel CPA comparison; and an incremental model crediting only responses solely driven by streaming TV. Their DragFactor metric and dynamic baselining (validated against test/control experiments) represent the most sophisticated public documentation of a production TV attribution system.

**Innovid/TVSquared** runs a dual-model architecture: a **linear spike model** for broadcast (calculating dynamic baselines per market/minute/day, filtering non-TV traffic, assigning probability scores to sessions) and an **impression-based model** for CTV using household-level ACR data. Their identity infrastructure ("Innovid Key") maps 95M+ US households.

**iSpot.tv** uses deterministic attribution from a panel of **83M+ smart TVs** with ACR technology, linking household IP addresses to digital outcomes while filtering institutional IPs (bars, hotels, hotspots). Their March 2025 "Outcomes at Scale" product connects impressions to outcomes in near-real-time.

**Google** maintains two relevant products: CausalImpact for intervention-level causal inference, and **Meridian** (launched January 2025, open-source), a Bayesian hierarchical marketing mix model with geo-level modeling, adstock/saturation transformations, and integration with Google Query Volume as a control variable.

**Meta** offers **Robyn** (open-source MMM using Ridge regression with Prophet for trend/seasonality, Nevergrad for optimization) and **GeoLift** (augmented synthetic control for geo-experiments). Meta advocates calibrating MMM with experimental lift tests, noting that observational methods "produced estimates that diverged dramatically from experimental ground truth despite conditioning on hundreds of covariates" (Gordon et al. 2023).

### Recent academic work highlights intertemporal substitution

The most consequential recent finding comes from **Lambrecht, Tucker & Zhang (2024, *Journal of Marketing Research*)**: TV ads cause instantaneous increases in browsing and sales but also **intertemporal substitution**—consumers pull forward activity they would have done anyway. Short attribution windows (5-15 minutes) can systematically **overstate** the total incremental effect of TV because they capture accelerated behavior rather than truly new behavior. This has profound implications: a spot that appears to drive 100 incremental visits in 10 minutes may have only generated 30-40 truly incremental visits when accounting for visits displaced from later periods.

Other notable recent work includes the **TV-Impact framework** (Arslan et al. 2024, *Entropy*) extending CausalImpact with dynamic control selection and Group Ads decomposition; **Veverka (2021)** proposing a three-stage kernel smoothing + Weibull MLE + random forest pipeline for minute-level attribution; and **Du et al. (2019, *Journal of Marketing*)** providing the most granular academic measurement of immediate TV-to-search response curves.

---

## Part 5: Ranked recommendations for a production system

### The optimal architecture: BSTS baseline with RDD-like spot attribution

For a system upgrading from wavelet-based baselines to BSTS, the recommended architecture operates at two timescales:

**Campaign/daily level — BSTS counterfactual.** Use the full BSTS framework (local linear trend + multiple seasonal components + regression with spike-and-slab covariate selection) to estimate what web traffic *would have been* absent all TV advertising. This replaces the wavelet baseline with a probabilistically grounded counterfactual that provides credible intervals, handles missing data, and automatically selects informative control series. Control covariates should include traffic from non-TV channels, category search trends, and competitor activity. This layer answers: "Did TV drive incremental traffic overall, and how much?"

**Spot level — RDD-inspired spike detection.** For each ad airing (or merged "Group Ad" when spots overlap), compare observed minute-level traffic against the BSTS-predicted counterfactual in a narrow window (recommended: **5-10 minutes post-airing**, with sensitivity analysis at 2, 8, and 15 minutes). The excess above the BSTS baseline in this window is the spot-level attributed lift. This is computationally fast (local comparison, no additional MCMC), benefits from the BSTS baseline's covariate adjustment, and provides a natural decomposition of campaign-level lift into individual spot contributions.

**Calibration layer.** Aggregate spot-level attributed lifts should be calibrated against the BSTS campaign-level estimate. If the sum of spot-level lifts exceeds the campaign-level BSTS estimate (likely, given overlapping windows and intertemporal substitution), apply a scaling factor to ensure consistency. Periodically validate both levels against **geo-holdout experiments** (turn off TV in matched markets and compare) using Meta's GeoLift or a custom augmented synthetic control implementation.

### Ranked method recommendations

1. **BSTS + RDD hybrid** (recommended primary method): Combines the best counterfactual baseline (BSTS) with the most natural per-spot causal identification (RDD logic). Strongest theoretical foundation for the specific use case of minute-level, multi-spot TV attribution.

2. **DiD with competitor controls** (recommended validation): Following Liaukonyte et al. (2015), use non-advertising competitor web traffic as a control group within 2-minute windows around each airing. Provides an independent cross-check on BSTS+RDD estimates without requiring separate control markets.

3. **Augmented Synthetic Control for geo-validation** (recommended periodic validation): Run quarterly or semi-annual geo-holdout experiments using GeoLift to establish ground-truth incrementality estimates. Use these to calibrate spot-level model parameters and validate overall system accuracy.

4. **Hierarchical Bayesian extension** (recommended future enhancement): Extend BSTS to a hierarchical model pooling information across networks, dayparts, and creative types. This enables partial pooling (spots with limited data borrow strength from similar spots), models parameter heterogeneity, and naturally produces uncertainty estimates for every effect dimension. Uber's BTVC framework (arXiv: 2106.03322) and PyMC-Marketing's hierarchical MMM provide implementation templates.

### Practical validation strategy

Validation should proceed in four stages. First, **parameter recovery on synthetic data**: generate simulated web traffic with known TV effects of varying magnitudes, inject realistic confounders, and verify the system recovers the true effects within credible intervals. Second, **out-of-sample forecast accuracy**: hold back recent data, generate BSTS counterfactual predictions, and compare to actuals during known TV-dark periods. Third, **cross-method comparison**: run BSTS+RDD, DiD, and simple spike-detection on the same data and compare estimates—large discrepancies signal model misspecification. Fourth, **experimental ground truth**: run at least one geo-holdout experiment annually and compare the system's predicted lift for those markets against the experimentally measured lift. Gordon et al. (2023) showed observational methods can diverge dramatically from experimental results, making this step non-negotiable.

### Implementation path

For a system already processing minute-level TV spot logs and web analytics data, the migration from wavelets to BSTS can proceed incrementally. Start by fitting BSTS baselines in parallel with the existing wavelet system and comparing counterfactual estimates. The `bsts` R package or `tfp-causalimpact` Python package provides the core modeling layer. Add control covariates progressively—beginning with day-of-week and time-of-day seasonality, then adding non-TV channel traffic, category search trends, and competitor activity. Deploy spot-level attribution against the BSTS baseline once counterfactual accuracy exceeds the wavelet system on held-out TV-dark periods. Finally, implement the calibration layer and schedule the first geo-holdout validation experiment. The entire BSTS infrastructure can run on standard compute (each BSTS model with 500 time points and 10 covariates completes in ~30 seconds with 10,000 MCMC iterations), though processing thousands of daily spots requires parallelization across a compute cluster or cloud batch processing.

The key open-source tools for this implementation are: **`bsts`/CausalImpact** (R) or **`tfp-causalimpact`** (Python) for BSTS modeling; **PyMC or Stan** for custom hierarchical extensions; **Meta's GeoLift** for geo-experiment design and analysis; **`rdrobust`** for formal RDD inference when needed; and **PyMC-Marketing** or **Google's Meridian** for eventual integration with a full marketing mix model that places TV attribution in the context of all channels.