# Causal Impact

`BstsNx.CausalImpact` estimates intervention impact by comparing observed
post-period data to a counterfactual baseline forecasted from the pre-period.

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

## Notes

- `pre_period` and `post_period` are 1-based index ranges.
- If you pass a single posterior sample, `summary/1` returns `:nan` for
  standard deviation and interval bounds.
- The current implementation assumes a local-level model; future versions
  may support richer component specifications.
