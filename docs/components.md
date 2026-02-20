# Components and Composition

BSTS models are built from components such as local level, trend, and
regression. BstsNx represents local level/trend/seasonal components as maps
with keys `:f`, `:q`, and `:h`, which can be composed into a larger
state-space model. Regression is built as a `%BstsNx.ModelSpec{}`.

## Local level

```elixir
component = BstsNx.Components.local_level(0.5)
component.f  # [[1.0]]
component.q  # [[0.5]]
component.h  # [[1.0]]
```

## Local linear trend

```elixir
component = BstsNx.Components.local_linear_trend(0.1, 0.01)
```

## Regression model spec

```elixir
regressors = Nx.tensor([
  [1.0, 0.5],
  [0.8, 1.2],
  [1.1, 0.9]
])

spec = BstsNx.Components.regression(regressors,
  var_beta: 0.01,
  obs_var: 1.0
)
```

## Compose components

```elixir
level = BstsNx.Components.local_level(0.5)
trend = BstsNx.Components.local_linear_trend(0.1, 0.01)
model = BstsNx.StateSpace.compose(level, trend)
```

## Time-varying regression contribution

Regression specs model coefficient evolution with time-varying observation
rows. To compute a direct per-step contribution `βᵀ·X_t`, use
`BstsNx.Components.predict_with_regressors/2`.
