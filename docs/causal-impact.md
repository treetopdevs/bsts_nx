# Causal Impact

`BstsNx.CausalImpact` estimates intervention impact by comparing observed
post-period data to a counterfactual baseline forecasted from the pre-period.

## Estimators

`BstsNx.CausalImpact` exposes three estimators:

- `estimate/4`: local-level convenience API.
- `estimate_structured/5`: structured API for `%BstsNx.ModelSpec{}` models
  (trend, seasonal, regression, and composed specs).
- `estimate_from_filter/3`: non-MCMC fast path based on the compiled scalar
  filter/smoother.

For richer component-based baselines, prefer `estimate_structured/5`.

## Workflow

1. Choose a pre-period to fit the model.
2. Choose a post-period to evaluate the effect.
3. Use posterior samples to compute point, cumulative, and relative effects.

## Example

```elixir
pre = Enum.map(1..60, fn _ -> 50.0 + :rand.normal() * 5 end)
post = Enum.map(1..30, fn _ -> 60.0 + :rand.normal() * 5 end)
obs = pre ++ post

result =
  BstsNx.CausalImpact.estimate(obs, {1, 60}, {61, 90},
    num_samples: 200,
    burn_in: 100,
    seed: 42
  )

summary = BstsNx.CausalImpact.summary(result)
summary.cumulative_effect
```

## Structured Example

```elixir
spec =
  BstsNx.Components.compose_specs(
    BstsNx.Components.local_linear_trend_spec(),
    BstsNx.Components.seasonal_spec(7)
  )

result =
  BstsNx.CausalImpact.estimate_structured(obs, {1, 60}, {61, 90}, spec,
    num_samples: 200,
    burn_in: 100,
    seed: 42
  )

summary = BstsNx.CausalImpact.summary(result)
summary.cumulative_effect
```

## Notes

- `pre_period` and `post_period` are 1-based index ranges.
- `post_period` must immediately follow `pre_period`.
- If you pass a single posterior sample, `summary/1` returns `:nan` for
  standard deviation and interval bounds.
- Structured estimation supports static and time-varying observation matrices
  (`H_t`) through `ModelSpec.h`.
- Current structured MCMC limitations:
  - observations are scalar per time step,
  - observation variance is scalar,
  - process covariance learning is diagonal (`Q` entries sampled independently).
