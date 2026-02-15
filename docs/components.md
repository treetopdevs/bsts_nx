# Components and Composition

BSTS models are built from components such as local level, trend, and
regression. BstsNx represents components as maps with keys `:f`, `:q`, and
`:h`, which can be composed into a larger state-space model.

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

## Regression component

```elixir
betas = Nx.tensor([0.5, -0.2])
sigma = Nx.eye(2) |> Nx.multiply(0.01)
component = BstsNx.Components.regression(betas, sigma)
```

## Compose components

```elixir
level = BstsNx.Components.local_level(0.5)
trend = BstsNx.Components.local_linear_trend(0.1, 0.01)
model = BstsNx.StateSpace.compose(level, trend)
```

## Time-varying regressors

The regression component models coefficient evolution. To use time‑varying
regressors `X_t`, compute `βᵀ·X_t` at each time step, e.g. with
`BstsNx.Components.predict_with_regressors/2`.
