defmodule BstsNx.ModelSpec do
  @moduledoc """
  Struct defining a state-space model specification for the structured Gibbs sampler.

  A `ModelSpec` encodes the complete parametric form of a linear Gaussian
  state-space model:

      x_t = F x_{t-1} + w_t,   w_t ~ N(0, Q)
      y_t = H_t x_t  + v_t,    v_t ~ N(0, R)

  where Q is a diagonal matrix whose entries are independently resampled
  from inverse-gamma posteriors during Gibbs sampling.

  ## Fields

    * `:f` — transition matrix, `{n, n}` tensor
    * `:h` — observation matrix. Either a single `{1, n}` tensor (static)
      or a list of `{1, n}` tensors (time-varying, one per observation)
    * `:x0` — initial state mean, `{n}` vector
    * `:p0` — initial state covariance, `{n, n}` matrix
    * `:obs_var` — initial observation variance (positive float)
    * `:q_specs` — list of maps, one per diagonal Q entry to resample.
      Each map has keys: `:dim_index` (int), `:initial` (float),
      `:prior_shape` (float), `:prior_scale` (float)
    * `:obs_prior_shape` — shape parameter of the inverse-gamma prior on
      observation variance (default: 1.0)
    * `:obs_prior_scale` — scale parameter of the inverse-gamma prior on
      observation variance (default: 1.0)
  """

  @enforce_keys [:f, :h, :x0, :p0, :obs_var, :q_specs]
  defstruct [
    :f,
    :h,
    :x0,
    :p0,
    :obs_var,
    :q_specs,
    obs_prior_shape: 1.0,
    obs_prior_scale: 1.0
  ]

  @type q_spec :: %{
          dim_index: non_neg_integer(),
          initial: float(),
          prior_shape: float(),
          prior_scale: float()
        }

  @type t :: %__MODULE__{
          f: Nx.t(),
          h: Nx.t() | [Nx.t()],
          x0: Nx.t(),
          p0: Nx.t(),
          obs_var: float(),
          q_specs: [q_spec()],
          obs_prior_shape: float(),
          obs_prior_scale: float()
        }
end
