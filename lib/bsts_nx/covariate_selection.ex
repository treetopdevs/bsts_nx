defmodule BstsNx.CovariateSelection do
  @moduledoc """
  Pre-period correlation-based covariate selection for BSTS models.

  Selects control time series that are most correlated with the target
  during the pre-intervention period. This helps build better
  counterfactual predictions by including only relevant predictors.
  """

  @doc """
  Selects the best covariates from a pool based on pre-period correlation.

  ## Parameters
    * `target` - list or {n} tensor of target values (pre-period only)
    * `candidates` - {n, p} tensor of candidate covariate values
    * `opts` - keyword options

  ## Options
    * `:threshold` - minimum absolute correlation (default: 0.1)
    * `:max_controls` - maximum number to select (default: 10)

  ## Returns
    * `:selected_indices` - 0-based column indices of selected covariates
    * `:correlations` - list of {index, correlation} for ALL candidates (sorted by abs corr desc)
    * `:selected_matrix` - {n, k} tensor of selected covariates, or nil if none selected

  ## Examples

      iex> target = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      iex> cands = Nx.tensor([[2.0, 5.0], [4.0, 5.0], [6.0, 5.0], [8.0, 5.0], [10.0, 5.0]])
      iex> result = BstsNx.CovariateSelection.select(target, cands, threshold: 0.5)
      iex> result.selected_indices
      [0]
  """
  @spec select([number()] | Nx.t(), Nx.t(), keyword()) :: map()
  def select(target, candidates, opts \\ [])

  def select(target, candidates, opts) do
    threshold = Keyword.get(opts, :threshold, 0.1)
    max_controls = Keyword.get(opts, :max_controls, 10)

    target_t =
      case target do
        %Nx.Tensor{} -> target
        list when is_list(list) -> Nx.tensor(list)
      end

    # Flatten target to 1-D
    target_t = Nx.flatten(target_t)
    n = Nx.axis_size(target_t, 0)

    {n_cand, p} = Nx.shape(candidates)

    if n_cand != n do
      raise ArgumentError,
            "candidates rows (#{n_cand}) must match target length (#{n})"
    end

    if p == 0 do
      %{
        selected_indices: [],
        correlations: [],
        selected_matrix: nil
      }
    else
      # Compute correlation for each candidate column
      correlations =
        Enum.map(0..(p - 1), fn j ->
          col = Nx.slice(candidates, [0, j], [n, 1]) |> Nx.flatten()
          corr = pearson_correlation(target_t, col)
          {j, corr}
        end)

      # Sort by absolute correlation descending
      sorted = Enum.sort_by(correlations, fn {_j, corr} -> abs(corr) end, :desc)

      # Filter by threshold and take top max_controls
      selected =
        sorted
        |> Enum.filter(fn {_j, corr} -> abs(corr) >= threshold end)
        |> Enum.take(max_controls)

      selected_indices = Enum.map(selected, fn {j, _corr} -> j end)

      selected_matrix =
        if selected_indices == [] do
          nil
        else
          cols =
            Enum.map(selected_indices, fn j ->
              Nx.slice(candidates, [0, j], [n, 1])
            end)

          Nx.concatenate(cols, axis: 1)
        end

      %{
        selected_indices: selected_indices,
        correlations: sorted,
        selected_matrix: selected_matrix
      }
    end
  end

  @doc """
  Computes Pearson correlation between two equal-length series.

  Returns 0.0 if either series has zero variance.

  ## Examples

      iex> x = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      iex> y = Nx.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
      iex> BstsNx.CovariateSelection.pearson_correlation(x, y)
      1.0

      iex> x = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      iex> y = Nx.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
      iex> BstsNx.CovariateSelection.pearson_correlation(x, y)
      -1.0
  """
  @spec pearson_correlation(Nx.t(), Nx.t()) :: float()
  def pearson_correlation(x, y) do
    x_mean = Nx.mean(x)
    y_mean = Nx.mean(y)

    x_centered = Nx.subtract(x, x_mean)
    y_centered = Nx.subtract(y, y_mean)

    numerator = Nx.multiply(x_centered, y_centered) |> Nx.sum() |> Nx.to_number()

    x_ss = x_centered |> Nx.multiply(x_centered) |> Nx.sum() |> Nx.to_number()
    y_ss = y_centered |> Nx.multiply(y_centered) |> Nx.sum() |> Nx.to_number()

    denominator = :math.sqrt(x_ss * y_ss)

    if denominator < 1.0e-15 do
      0.0
    else
      numerator / denominator
    end
  end
end
