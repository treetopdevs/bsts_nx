# Synthetic Data and Validation

This guide covers modules used to calibrate and stress-test BSTS workflows.

## 1. Scenario Library (`BstsNx.Synthetic.Scenarios`)

Use built-in scenarios for repeatable benchmarks.

```elixir
{:scenario_b, _name, config} = Enum.at(BstsNx.Synthetic.Scenarios.all(), 1)
config.name
```

Each scenario config includes baseline, spot schedule, noise, effect transform, controls, and seed.

## 2. Synthetic Generator (`BstsNx.Synthetic.Generator`)

Generate observable data and exact ground truth in one call.

```elixir
config = BstsNx.Synthetic.Scenarios.scenario_a()
result = BstsNx.Synthetic.Generator.generate(config)

result.observations
result.ground_truth.total_lift
result.ground_truth.spot_attributions
```

This is useful for regression tests and model recovery experiments.

## 3. Adstock and Saturation (`BstsNx.Synthetic.Adstock`)

Standalone transforms used by generator and available for custom DGP design.

```elixir
adstocked = BstsNx.Synthetic.Adstock.geometric_adstock([1.0, 0.0, 0.0], 0.5)
saturated = BstsNx.Synthetic.Adstock.hill_saturation(adstocked, 1.0, 2.0)
```

## 4. Validation Checks (`BstsNx.Validation`)

Run statistical checks on model quality and causal stability.

```elixir
res = BstsNx.Validation.assess(%{
  prediction_error: %{rmse: 5.0, mape: 0.08, n: 200},
  coverage: %{pct: 0.93, n_covered: 186, n_total: 200},
  durbin_watson: %{statistic: 1.9, n: 200},
  placebo: %{is_significant: false, lift_pct: 0.01, ci95: %{lower: -5.0, upper: 4.0}},
  effect_stability: %{max_pct_change: 0.07}
})

res
```

`known_lift_injection/3` provides calibration by testing recovery of known injected effects.

## 5. MCMC Convergence (`BstsNx.Diagnostics`)

```elixir
chains = [
  Enum.map(1..500, fn _ -> :rand.normal() end),
  Enum.map(1..500, fn _ -> :rand.normal() end)
]

{BstsNx.Diagnostics.r_hat(chains), BstsNx.Diagnostics.effective_sample_size(chains)}
```

Use R-hat and ESS before trusting posterior intervals in production.

## 6. Utility Distribution Samplers (`BstsNx.Distributions`)

These are used internally and can support custom experimental workflows.

```elixir
q = BstsNx.Distributions.inv_gamma_sample(2.0, 1.0)
{eps, key2} = BstsNx.Distributions.normal_sample(Nx.Random.key(42))
```

## Suggested Validation Loop

1. Generate synthetic data from a known scenario.
2. Run your production pipeline.
3. Compare estimated lift against known ground truth.
4. Run `Validation` checks and `Diagnostics` metrics.
5. Tune priors/model structure until calibration is stable.
