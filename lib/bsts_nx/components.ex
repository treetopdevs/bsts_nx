defmodule BstsNx.Components do
  @moduledoc """
  Factory functions for common state‑space model components.

  Bayesian Structural Time Series models are typically constructed from
  modular building blocks such as local level, trend and regression
  components.  This module provides helper functions to instantiate
  these components as maps with keys `:f`, `:q` and `:h`, consistent
  with the expectations of `BstsNx.StateSpace.compose/2` and the
  Kalman filter.

  Additionally, this module provides `*_spec` factory functions that
  return `%BstsNx.ModelSpec{}` structs for use with the structured
  Gibbs sampler (`BstsNx.GibbsSampler.sample_structured/4`).  These
  specs fully describe a model's matrices, initial conditions and
  inverse-gamma priors for each diagonal Q entry.
  """

  alias BstsNx.ModelSpec

  @doc """
  Constructs a local level component for a univariate time series.

  The local level model assumes that the latent state follows a random
  walk: `x_t = x_{t-1} + w_t`, with `w_t ∼ N(0, σ²)`.  Observations are
  `y_t = x_t + v_t`, with observation noise `v_t ∼ N(0, τ²)` supplied
  by the caller to the Kalman filter or Gibbs sampler.  The returned
  component has:

    * `:f` – the 1×1 matrix `[1]`.
    * `:q` – the 1×1 process variance matrix `[σ²]`.
    * `:h` – the 1×1 observation matrix `[1]`.

  ## Examples

      iex> c = BstsNx.Components.local_level(0.5)
      iex> Nx.shape(c.f)
      {1, 1}
  """
  @spec local_level(number | list | Nx.t()) :: %{f: Nx.t(), q: Nx.t(), h: Nx.t()}
  def local_level(process_var) do
    q = Nx.tensor([[process_var]])
    %{f: Nx.tensor([[1.0]]), q: q, h: Nx.tensor([[1.0]])}
  end

  @doc """
  Constructs a local linear trend component for a univariate series.

  The local linear trend model has two state dimensions: level and
  slope.  The transition is:

      [level_t]   = [1 1] [level_{t-1}] + [w1_t]
      [slope_t]     [0 1] [slope_{t-1}]   [w2_t]

  where `w1_t` and `w2_t` are Gaussian noises with variances `σ_level`
  and `σ_slope` respectively.  Observations are `y_t = [1 0] x_t + v_t`.

  ## Examples

      iex> c = BstsNx.Components.local_linear_trend(0.1, 0.01)
      iex> Nx.shape(c.f)
      {2, 2}
  """
  @spec local_linear_trend(number | list | Nx.t(), number | list | Nx.t()) :: %{
          f: Nx.t(),
          q: Nx.t(),
          h: Nx.t()
        }
  def local_linear_trend(var_level, var_slope) do
    f = Nx.tensor([[1.0, 1.0], [0.0, 1.0]])
    q = BstsNx.StateSpace.block_diag([var_level, var_slope])
    h = Nx.tensor([[1.0, 0.0]])
    %{f: f, q: q, h: h}
  end

  @doc """
  Constructs a regression component for one or more regressors.

  Given a vector of regression coefficients `betas` and a matrix of
  design covariances `sigma_betas`, this component models the latent
  coefficients as a random walk with innovation variance `sigma_betas`.
  The transition matrix is an identity matrix of size equal to the
  number of regressors, the process covariance is `sigma_betas`, and
  the observation matrix is the row vector of betas.  This component
  can be composed with others via `BstsNx.StateSpace.compose/2`.

  The initial beta estimates and their covariance should be provided
  separately to the Kalman filter.

  ## Examples

      iex> betas = Nx.tensor([0.5, -0.2])
      iex> sigma = Nx.eye(2) |> Nx.multiply(0.01)
      iex> c = BstsNx.Components.regression(betas, sigma)
      iex> Nx.shape(c.f)
      {2, 2}
  """
  @spec regression(Nx.t(), Nx.t()) :: %{f: Nx.t(), q: Nx.t(), h: Nx.t()}
  def regression(betas, sigma_betas) do
    # ensure betas is a vector tensor
    beta_vec =
      case betas do
        %Nx.Tensor{} -> betas
        v when is_list(v) or is_number(v) -> Nx.tensor(v)
      end

    dim = Nx.axis_size(beta_vec, 0)
    f = Nx.eye(dim)
    q = sigma_betas
    # observation matrix is 1×dim row vector of betas
    h = Nx.reshape(beta_vec, {1, dim})
    %{f: f, q: q, h: h}
  end

  @doc """
  Constructs a seasonal component with `num_seasons` seasons.

  The dummy-variable seasonal model uses `num_seasons - 1` state dimensions
  to represent the current and lagged seasonal effects.  The constraint that
  seasonal effects sum to approximately zero over a full cycle is encoded in
  the transition matrix.

  The state vector is `[γ_t, γ_{t-1}, ..., γ_{t-S+2}]` where S is the
  number of seasons.  The transition is:

      γ_t = -(γ_{t-1} + γ_{t-2} + ... + γ_{t-S+1}) + w_t

  with `w_t ~ N(0, σ²)`.  The transition matrix has `-1` in the first row
  and an identity shift below.  Only the first state dimension receives
  innovation noise; the others are deterministic shifts.

  Observations are `y_t = γ_t + v_t`, so `H = [1, 0, ..., 0]`.

  ## Examples

      iex> c = BstsNx.Components.seasonal(7, 0.1)
      iex> Nx.shape(c.f)
      {6, 6}
      iex> Nx.shape(c.h)
      {1, 6}

      iex> c = BstsNx.Components.seasonal(4, 0.5)
      iex> Nx.shape(c.f)
      {3, 3}
  """
  @spec seasonal(pos_integer(), number()) :: %{f: Nx.t(), q: Nx.t(), h: Nx.t()}
  def seasonal(num_seasons, process_var) when is_integer(num_seasons) and num_seasons >= 2 do
    dim = num_seasons - 1
    f = build_seasonal_transition(dim)

    # Q: innovation variance only on the first state dimension
    q = Nx.broadcast(0.0, {dim, dim}) |> Nx.put_slice([0, 0], Nx.tensor([[process_var]]))

    # H: observe only the current seasonal effect
    h = Nx.broadcast(0.0, {1, dim}) |> Nx.put_slice([0, 0], Nx.tensor([[1.0]]))

    %{f: f, q: q, h: h}
  end

  def seasonal(num_seasons, _process_var) do
    raise ArgumentError, "num_seasons must be an integer >= 2, got: #{inspect(num_seasons)}"
  end

  @doc """
  Computes the regression contribution for a given latent coefficient state and
  a vector of regressor values.

  Note: The observation matrix `h` contains the regression coefficients
  directly. This means the regression contribution to the observation is
  βᵀ · x where x is the latent coefficient state. To incorporate time‑varying
  regressors `X_t`, multiply the design vector by the latent state outside of
  this component.

  Given a coefficient state vector `state` of shape `{n}` and a
  design vector `regressors` of the same length, this function returns
  the scalar contribution `βᵀ·x` as a 0‑dimensional tensor.  This is
  useful when working with time‑varying regressors, where the
  regression coefficients evolve according to a random walk (modeled
  by `BstsNx.Components.regression/2`) but the regressor values
  themselves change at each time step.  You can use this function
  within the observation model to compute the regression part of the
  predicted observation.

  ## Examples

      iex> state = Nx.tensor([0.5, -0.2])
      iex> x_t = Nx.tensor([10.0, 3.0])  # regressor values at time t
      iex> contrib = BstsNx.Components.predict_with_regressors(state, x_t)
      iex> Nx.to_number(contrib) |> Float.round(1)
      4.4

  """
  @spec predict_with_regressors(Nx.t(), Nx.t()) :: Nx.t()
  def predict_with_regressors(state, regressors) do
    # ensure both are vectors
    s =
      case state do
        %Nx.Tensor{} -> state
        v when is_list(v) or is_number(v) -> Nx.tensor(v)
      end

    x =
      case regressors do
        %Nx.Tensor{} -> regressors
        v when is_list(v) or is_number(v) -> Nx.tensor(v)
      end

    Nx.dot(s, x) |> Nx.squeeze()
  end

  # ---------------------------------------------------------------------------
  # ModelSpec factory functions for the structured Gibbs sampler
  # ---------------------------------------------------------------------------

  @doc """
  Creates a `%ModelSpec{}` for a local level (random walk) model.

  The local level model has a single state dimension:

      x_t = x_{t-1} + w_t,   w_t ~ N(0, σ²_process)
      y_t = x_t + v_t,        v_t ~ N(0, σ²_obs)

  ## Options

    * `:process_var` — initial process variance (default: 1.0)
    * `:obs_var` — initial observation variance (default: 1.0)
    * `:initial_state` — prior mean for the state at time 0 (default: 0.0)
    * `:initial_cov` — prior variance for the state at time 0 (default: 1.0)
    * `:prior_shape` — shape parameter for inverse-gamma priors on both
      process and observation variances (default: 1.0)
    * `:prior_scale` — scale parameter for inverse-gamma priors (default: 1.0)

  ## Examples

      iex> spec = BstsNx.Components.local_level_spec()
      iex> spec.__struct__
      BstsNx.ModelSpec
      iex> Nx.shape(spec.f)
      {1, 1}
  """
  @spec local_level_spec(keyword()) :: ModelSpec.t()
  def local_level_spec(opts \\ []) do
    process_var = Keyword.get(opts, :process_var, 1.0)
    obs_var = Keyword.get(opts, :obs_var, 1.0)
    initial_state = Keyword.get(opts, :initial_state, 0.0)
    initial_cov = Keyword.get(opts, :initial_cov, 1.0)
    prior_shape = Keyword.get(opts, :prior_shape, 1.0)
    prior_scale = Keyword.get(opts, :prior_scale, 1.0)

    %ModelSpec{
      f: Nx.tensor([[1.0]]),
      h: Nx.tensor([[1.0]]),
      x0: Nx.tensor([initial_state]),
      p0: Nx.tensor([[initial_cov]]),
      obs_var: obs_var,
      q_specs: [
        %{dim_index: 0, initial: process_var, prior_shape: prior_shape, prior_scale: prior_scale}
      ],
      obs_prior_shape: prior_shape,
      obs_prior_scale: prior_scale
    }
  end

  @doc """
  Creates a `%ModelSpec{}` for a local linear trend model.

  The local linear trend model has two state dimensions (level and slope):

      [level_t]   = [1 1] [level_{t-1}] + [w1_t]
      [slope_t]     [0 1] [slope_{t-1}]   [w2_t]

      y_t = [1 0] x_t + v_t

  Each process variance (level and slope) has its own inverse-gamma prior
  and is resampled independently.

  ## Options

    * `:var_level` — initial level process variance (default: 1.0)
    * `:var_slope` — initial slope process variance (default: 0.1)
    * `:obs_var` — initial observation variance (default: 1.0)
    * `:initial_level` — prior mean for the level (default: 0.0)
    * `:initial_slope` — prior mean for the slope (default: 0.0)
    * `:initial_cov_level` — prior variance for the level (default: 1.0)
    * `:initial_cov_slope` — prior variance for the slope (default: 1.0)
    * `:prior_shape` — shape parameter for all inverse-gamma priors (default: 1.0)
    * `:prior_scale` — scale parameter for all inverse-gamma priors (default: 1.0)

  ## Examples

      iex> spec = BstsNx.Components.local_linear_trend_spec(var_level: 0.5, var_slope: 0.05)
      iex> Nx.shape(spec.f)
      {2, 2}
      iex> length(spec.q_specs)
      2
  """
  @spec local_linear_trend_spec(keyword()) :: ModelSpec.t()
  def local_linear_trend_spec(opts \\ []) do
    var_level = Keyword.get(opts, :var_level, 1.0)
    var_slope = Keyword.get(opts, :var_slope, 0.1)
    obs_var = Keyword.get(opts, :obs_var, 1.0)
    initial_level = Keyword.get(opts, :initial_level, 0.0)
    initial_slope = Keyword.get(opts, :initial_slope, 0.0)
    initial_cov_level = Keyword.get(opts, :initial_cov_level, 1.0)
    initial_cov_slope = Keyword.get(opts, :initial_cov_slope, 1.0)
    prior_shape = Keyword.get(opts, :prior_shape, 1.0)
    prior_scale = Keyword.get(opts, :prior_scale, 1.0)

    %ModelSpec{
      f: Nx.tensor([[1.0, 1.0], [0.0, 1.0]]),
      h: Nx.tensor([[1.0, 0.0]]),
      x0: Nx.tensor([initial_level, initial_slope]),
      p0: BstsNx.StateSpace.block_diag([initial_cov_level, initial_cov_slope]),
      obs_var: obs_var,
      q_specs: [
        %{dim_index: 0, initial: var_level, prior_shape: prior_shape, prior_scale: prior_scale},
        %{dim_index: 1, initial: var_slope, prior_shape: prior_shape, prior_scale: prior_scale}
      ],
      obs_prior_shape: prior_shape,
      obs_prior_scale: prior_scale
    }
  end

  @doc """
  Creates a `%ModelSpec{}` for a regression component with time-varying regressors.

  The regression model treats each regression coefficient as a latent state
  evolving via a random walk:

      β_t = β_{t-1} + w_t,   w_t ~ N(0, diag(σ²_β))
      y_t = X_t' β_t + v_t,  v_t ~ N(0, σ²_obs)

  The observation matrix H is time-varying — each row of the `regressors`
  matrix becomes the observation vector at that timestep.

  ## Arguments

    * `regressors` — a `{T, p}` tensor where T is the number of time steps
      and p is the number of regressors.

  ## Options

    * `:var_beta` — initial process variance for each coefficient (default: 0.01).
      Can be a single number (applied to all) or a list of length p.
    * `:obs_var` — initial observation variance (default: 1.0)
    * `:initial_betas` — prior mean for coefficients (default: zeros)
    * `:initial_cov_beta` — prior variance for each coefficient (default: 1.0).
      Can be a single number or list of length p.
    * `:prior_shape` — shape for all inverse-gamma priors (default: 1.0)
    * `:prior_scale` — scale for all inverse-gamma priors (default: 1.0)

  ## Examples

      iex> x = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
      iex> spec = BstsNx.Components.regression_spec(x)
      iex> length(spec.q_specs)
      2
      iex> is_list(spec.h)
      true
  """
  @spec regression_spec(Nx.t(), keyword()) :: ModelSpec.t()
  def regression_spec(regressors, opts \\ []) do
    regressors_t =
      case regressors do
        %Nx.Tensor{} -> regressors
        list when is_list(list) -> Nx.tensor(list)
      end

    {t_len, p} = Nx.shape(regressors_t)
    obs_var = Keyword.get(opts, :obs_var, 1.0)
    prior_shape = Keyword.get(opts, :prior_shape, 1.0)
    prior_scale = Keyword.get(opts, :prior_scale, 1.0)

    # Per-coefficient process variances
    var_beta_opt = Keyword.get(opts, :var_beta, 0.01)

    var_betas =
      case var_beta_opt do
        v when is_number(v) -> List.duplicate(v, p)
        list when is_list(list) -> list
      end

    # Initial betas
    initial_betas_opt = Keyword.get(opts, :initial_betas, List.duplicate(0.0, p))

    initial_betas =
      case initial_betas_opt do
        %Nx.Tensor{} -> initial_betas_opt
        list when is_list(list) -> Nx.tensor(list)
      end

    # Initial covariance for betas
    initial_cov_opt = Keyword.get(opts, :initial_cov_beta, 1.0)

    initial_covs =
      case initial_cov_opt do
        v when is_number(v) -> List.duplicate(v, p)
        list when is_list(list) -> list
      end

    # Time-varying H: list of {1, p} tensors, one per timestep
    h_list =
      Enum.map(0..(t_len - 1), fn i ->
        Nx.slice(regressors_t, [i, 0], [1, p])
      end)

    q_specs =
      Enum.map(0..(p - 1), fn d ->
        %{
          dim_index: d,
          initial: Enum.at(var_betas, d),
          prior_shape: prior_shape,
          prior_scale: prior_scale
        }
      end)

    %ModelSpec{
      f: Nx.eye(p),
      h: h_list,
      x0: initial_betas,
      p0: BstsNx.StateSpace.block_diag(initial_covs),
      obs_var: obs_var,
      q_specs: q_specs,
      obs_prior_shape: prior_shape,
      obs_prior_scale: prior_scale
    }
  end

  @doc """
  Creates a `%ModelSpec{}` for a seasonal component with `num_seasons` seasons.

  The dummy-variable seasonal model uses `num_seasons - 1` state dimensions.
  The transition matrix enforces that seasonal effects approximately sum to
  zero over a full cycle, with innovation noise on the first state dimension
  only.

  This spec is designed to be composed with other specs (e.g., local level
  or trend) via `compose_specs/2` to build richer models.

  ## Options

    * `:process_var` — initial seasonal innovation variance (default: 0.1)
    * `:obs_var` — initial observation variance (default: 1.0)
    * `:initial_cov` — prior variance for each seasonal state (default: 1.0)
    * `:prior_shape` — shape parameter for inverse-gamma prior (default: 1.0)
    * `:prior_scale` — scale parameter for inverse-gamma prior (default: 1.0)

  ## Examples

      iex> spec = BstsNx.Components.seasonal_spec(7)
      iex> Nx.shape(spec.f)
      {6, 6}
      iex> length(spec.q_specs)
      1

      iex> spec = BstsNx.Components.seasonal_spec(4, process_var: 0.5)
      iex> Nx.shape(spec.f)
      {3, 3}
      iex> hd(spec.q_specs).initial
      0.5
  """
  @spec seasonal_spec(pos_integer(), keyword()) :: ModelSpec.t()
  def seasonal_spec(num_seasons, opts \\ [])

  def seasonal_spec(num_seasons, opts) when is_integer(num_seasons) and num_seasons >= 2 do
    dim = num_seasons - 1
    process_var = Keyword.get(opts, :process_var, 0.1)
    obs_var = Keyword.get(opts, :obs_var, 1.0)
    initial_cov = Keyword.get(opts, :initial_cov, 1.0)
    prior_shape = Keyword.get(opts, :prior_shape, 1.0)
    prior_scale = Keyword.get(opts, :prior_scale, 1.0)

    f = build_seasonal_transition(dim)

    # H: observe only the current seasonal effect
    h = Nx.broadcast(0.0, {1, dim}) |> Nx.put_slice([0, 0], Nx.tensor([[1.0]]))

    # Initial state: zeros (no prior seasonal preference)
    x0 = Nx.broadcast(0.0, {dim})

    # Initial covariance: diagonal
    p0 = Nx.multiply(Nx.eye(dim), initial_cov)

    # Only one Q entry: seasonal innovation on dim 0
    q_specs = [
      %{dim_index: 0, initial: process_var, prior_shape: prior_shape, prior_scale: prior_scale}
    ]

    %ModelSpec{
      f: f,
      h: h,
      x0: x0,
      p0: p0,
      obs_var: obs_var,
      q_specs: q_specs,
      obs_prior_shape: prior_shape,
      obs_prior_scale: prior_scale
    }
  end

  def seasonal_spec(num_seasons, _opts) do
    raise ArgumentError, "num_seasons must be an integer >= 2, got: #{inspect(num_seasons)}"
  end

  @doc """
  Composes two `%ModelSpec{}` structs into a single combined model.

  The transition matrices and process covariances are placed on the block
  diagonal, observation matrices are concatenated horizontally, and initial
  states/covariances are concatenated.  The q_specs from the second spec
  have their `dim_index` values shifted by the state dimension of the first
  spec.

  The observation variance and its inverse-gamma prior parameters are
  taken from `spec1` (the primary model).  Since the composed model has
  a single observation equation, there is only one observation variance
  to resample.  If you need different priors, set them on `spec1` before
  composing or override them on the returned struct.

  Handles mixed static/time-varying H: if one spec has static H and the
  other has time-varying H, the static one is broadcast to match.

  ## Examples

      iex> s1 = BstsNx.Components.local_level_spec()
      iex> x = Nx.tensor([[1.0], [2.0], [3.0]])
      iex> s2 = BstsNx.Components.regression_spec(x)
      iex> combined = BstsNx.Components.compose_specs(s1, s2)
      iex> length(combined.q_specs)
      2
  """
  @spec compose_specs(ModelSpec.t(), ModelSpec.t()) :: ModelSpec.t()
  def compose_specs(%ModelSpec{} = spec1, %ModelSpec{} = spec2) do
    n1 = Nx.axis_size(spec1.f, 0)

    # Block-diagonal F
    f = BstsNx.StateSpace.block_diag([spec1.f, spec2.f])

    # Compose H (handle mixed static/time-varying)
    h = compose_h(spec1.h, spec2.h, n1)

    # Concatenate x0
    x0 = Nx.concatenate([Nx.flatten(spec1.x0), Nx.flatten(spec2.x0)])

    # Block-diagonal p0
    p0 = BstsNx.StateSpace.block_diag([spec1.p0, spec2.p0])

    # Use spec1's observation variance and priors (single observation equation)
    obs_var = spec1.obs_var
    obs_prior_shape = spec1.obs_prior_shape
    obs_prior_scale = spec1.obs_prior_scale

    # Shift q_specs from spec2
    shifted_q2 =
      Enum.map(spec2.q_specs, fn qs ->
        %{qs | dim_index: qs.dim_index + n1}
      end)

    %ModelSpec{
      f: f,
      h: h,
      x0: x0,
      p0: p0,
      obs_var: obs_var,
      q_specs: spec1.q_specs ++ shifted_q2,
      obs_prior_shape: obs_prior_shape,
      obs_prior_scale: obs_prior_scale
    }
  end

  # Composes two H specifications. Each may be a tensor (static) or a list
  # of tensors (time-varying). Returns a single static tensor or a list.
  defp compose_h(h1, h2, _n1) when is_list(h1) and is_list(h2) do
    if length(h1) != length(h2) do
      raise ArgumentError,
            "time-varying H lists must have same length (got #{length(h1)} and #{length(h2)})"
    end

    Enum.zip(h1, h2)
    |> Enum.map(fn {h1_t, h2_t} ->
      Nx.concatenate([ensure_row(h1_t), ensure_row(h2_t)], axis: 1)
    end)
  end

  defp compose_h(%Nx.Tensor{} = h1, %Nx.Tensor{} = h2, _n1) do
    Nx.concatenate([ensure_row(h1), ensure_row(h2)], axis: 1)
  end

  defp compose_h(%Nx.Tensor{} = h1, h2_list, _n1) when is_list(h2_list) do
    h1_row = ensure_row(h1)

    Enum.map(h2_list, fn h2_t ->
      Nx.concatenate([h1_row, ensure_row(h2_t)], axis: 1)
    end)
  end

  defp compose_h(h1_list, %Nx.Tensor{} = h2, _n1) when is_list(h1_list) do
    h2_row = ensure_row(h2)

    Enum.map(h1_list, fn h1_t ->
      Nx.concatenate([ensure_row(h1_t), h2_row], axis: 1)
    end)
  end

  # Ensures a tensor is a {1, n} row matrix
  defp ensure_row(%Nx.Tensor{} = t) do
    case Nx.rank(t) do
      0 -> Nx.reshape(t, {1, 1})
      1 -> Nx.reshape(t, {1, Nx.axis_size(t, 0)})
      2 -> t
    end
  end

  # Builds the seasonal transition matrix of size dim × dim.
  #
  # For S seasons (dim = S-1):
  #   Row 0:        [-1, -1, -1, ..., -1]   (new effect = -sum of past S-1 effects)
  #   Rows 1..dim-1: identity shift          (carry history forward)
  #
  # For dim=1 (S=2): F = [[-1]]
  defp build_seasonal_transition(1) do
    Nx.tensor([[-1.0]])
  end

  defp build_seasonal_transition(dim) when dim > 1 do
    # First row: all -1's
    first_row = Nx.broadcast(-1.0, {1, dim})

    # Bottom rows: [I_{dim-1} | 0_{(dim-1)×1}]
    eye_part = Nx.eye(dim - 1)
    zero_col = Nx.broadcast(0.0, {dim - 1, 1})
    shift = Nx.concatenate([eye_part, zero_col], axis: 1)

    Nx.concatenate([first_row, shift], axis: 0)
  end
end
