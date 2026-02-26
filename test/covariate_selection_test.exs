defmodule BstsNx.CovariateSelectionTest do
  use ExUnit.Case, async: true

  alias BstsNx.CovariateSelection

  # ---------------------------------------------------------------
  # 1. pearson_correlation/2
  # ---------------------------------------------------------------

  describe "pearson_correlation/2" do
    test "perfect positive correlation (r = 1.0)" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      y = Nx.tensor([2.0, 4.0, 6.0, 8.0, 10.0])

      assert_in_delta CovariateSelection.pearson_correlation(x, y), 1.0, 1.0e-10
    end

    test "perfect negative correlation (r = -1.0)" do
      x = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
      y = Nx.tensor([10.0, 8.0, 6.0, 4.0, 2.0])

      assert_in_delta CovariateSelection.pearson_correlation(x, y), -1.0, 1.0e-10
    end

    test "zero correlation for orthogonal series" do
      # sin and cos over a full period are orthogonal
      n = 100
      angles = Enum.map(0..(n - 1), fn i -> 2.0 * :math.pi() * i / n end)
      x = Nx.tensor(Enum.map(angles, &:math.sin/1))
      y = Nx.tensor(Enum.map(angles, &:math.cos/1))

      assert_in_delta CovariateSelection.pearson_correlation(x, y), 0.0, 0.05
    end

    test "constant series returns 0.0" do
      x = Nx.tensor([3.0, 3.0, 3.0, 3.0, 3.0])
      y = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

      assert CovariateSelection.pearson_correlation(x, y) == 0.0
    end

    test "both constant series returns 0.0" do
      x = Nx.tensor([5.0, 5.0, 5.0, 5.0])
      y = Nx.tensor([3.0, 3.0, 3.0, 3.0])

      assert CovariateSelection.pearson_correlation(x, y) == 0.0
    end

    test "near-zero correlation for uncorrelated data" do
      # Alternating +1/-1 vs linear trend — low correlation expected
      x = Nx.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
      y = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

      corr = CovariateSelection.pearson_correlation(x, y)
      assert abs(corr) < 0.3
    end

    test "identical series gives r = 1.0" do
      x = Nx.tensor([10.0, 20.0, 30.0])
      assert_in_delta CovariateSelection.pearson_correlation(x, x), 1.0, 1.0e-10
    end
  end

  # ---------------------------------------------------------------
  # 2. select/3
  # ---------------------------------------------------------------

  describe "select/3" do
    test "selects highly correlated covariates" do
      target = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

      # Column 0: perfectly correlated, Column 1: weakly correlated noise
      candidates =
        Nx.tensor([
          [2.0, 10.0],
          [4.0, 10.1],
          [6.0, 10.2],
          [8.0, 10.3],
          [10.0, 10.4]
        ])

      result = CovariateSelection.select(target, candidates, threshold: 0.5)

      assert 0 in result.selected_indices
      assert Nx.shape(result.selected_matrix) == {5, length(result.selected_indices)}
    end

    test "filters out low-correlation covariates below threshold" do
      target = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

      # Both columns have only weak correlation with the target (constant)
      candidates =
        Nx.tensor([
          [5.0, 3.0],
          [5.0, 3.0],
          [5.0, 3.0],
          [5.0, 3.0],
          [5.0, 3.0]
        ])

      result = CovariateSelection.select(target, candidates, threshold: 0.5)

      assert result.selected_indices == []
      assert result.selected_matrix == nil
    end

    test "respects max_controls limit" do
      target = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

      # 4 columns all perfectly correlated
      candidates =
        Nx.tensor([
          [2.0, 3.0, 4.0, 5.0],
          [4.0, 6.0, 8.0, 10.0],
          [6.0, 9.0, 12.0, 15.0],
          [8.0, 12.0, 16.0, 20.0],
          [10.0, 15.0, 20.0, 25.0]
        ])

      result = CovariateSelection.select(target, candidates, max_controls: 2)

      assert length(result.selected_indices) == 2
      assert Nx.shape(result.selected_matrix) == {5, 2}
    end

    test "returns correct selected_matrix dimensions" do
      target = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

      candidates =
        Nx.tensor([
          [2.0, 100.0, 3.0],
          [4.0, 100.0, 6.0],
          [6.0, 100.0, 9.0],
          [8.0, 100.0, 12.0],
          [10.0, 100.0, 15.0],
          [12.0, 100.0, 18.0],
          [14.0, 100.0, 21.0],
          [16.0, 100.0, 24.0]
        ])

      result = CovariateSelection.select(target, candidates, threshold: 0.5)

      {rows, cols} = Nx.shape(result.selected_matrix)
      assert rows == 8
      assert cols == length(result.selected_indices)
    end

    test "all below threshold returns empty selection" do
      target = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

      # Constant columns have zero correlation
      candidates =
        Nx.tensor([
          [7.0, 3.0],
          [7.0, 3.0],
          [7.0, 3.0],
          [7.0, 3.0],
          [7.0, 3.0]
        ])

      result = CovariateSelection.select(target, candidates, threshold: 0.1)

      assert result.selected_indices == []
      assert result.correlations != []
      assert result.selected_matrix == nil
    end

    test "custom threshold works" do
      target = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

      # Column 0: r=1.0, Column 1: moderate correlation
      candidates =
        Nx.tensor([
          [2.0, 5.0],
          [4.0, 4.5],
          [6.0, 6.0],
          [8.0, 7.5],
          [10.0, 9.0]
        ])

      # With high threshold, only the perfectly correlated column passes
      result_high = CovariateSelection.select(target, candidates, threshold: 0.99)
      assert length(result_high.selected_indices) == 1
      assert 0 in result_high.selected_indices

      # With low threshold, both pass
      result_low = CovariateSelection.select(target, candidates, threshold: 0.1)
      assert length(result_low.selected_indices) == 2
    end

    test "correlations list is sorted by absolute correlation descending" do
      target = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

      candidates =
        Nx.tensor([
          [2.0, 5.0, 3.0],
          [4.0, 4.0, 3.1],
          [6.0, 3.0, 3.2],
          [8.0, 2.0, 3.3],
          [10.0, 1.0, 3.4]
        ])

      result = CovariateSelection.select(target, candidates)

      abs_corrs = Enum.map(result.correlations, fn {_j, corr} -> abs(corr) end)
      assert abs_corrs == Enum.sort(abs_corrs, :desc)
    end

    test "correlations list contains all candidates" do
      target = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

      candidates =
        Nx.tensor([
          [2.0, 5.0, 3.0],
          [4.0, 4.0, 6.0],
          [6.0, 3.0, 9.0],
          [8.0, 2.0, 12.0],
          [10.0, 1.0, 15.0]
        ])

      result = CovariateSelection.select(target, candidates)

      assert length(result.correlations) == 3
      indices = Enum.map(result.correlations, fn {j, _} -> j end) |> Enum.sort()
      assert indices == [0, 1, 2]
    end

    test "accepts list as target" do
      target = [1.0, 2.0, 3.0, 4.0, 5.0]
      candidates = Nx.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])

      result = CovariateSelection.select(target, candidates)

      assert result.selected_indices == [0]
    end

    test "integration: selected_matrix can be fed to Components.regression_spec/2" do
      target = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

      candidates =
        Nx.tensor([
          [2.0, 100.0],
          [4.0, 100.0],
          [6.0, 100.0],
          [8.0, 100.0],
          [10.0, 100.0]
        ])

      result = CovariateSelection.select(target, candidates, threshold: 0.5)

      assert length(result.selected_indices) > 0

      spec = BstsNx.Components.regression_spec(result.selected_matrix)
      assert %BstsNx.ModelSpec{} = spec
      assert is_list(spec.h)
    end

    test "negative correlations are also selected" do
      target = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

      candidates =
        Nx.tensor([
          [10.0],
          [8.0],
          [6.0],
          [4.0],
          [2.0]
        ])

      result = CovariateSelection.select(target, candidates, threshold: 0.5)

      assert result.selected_indices == [0]
      [{0, corr}] = Enum.filter(result.correlations, fn {j, _} -> j == 0 end)
      assert corr < -0.9
    end

    test "handles single candidate column" do
      target = Nx.tensor([1.0, 2.0, 3.0])
      candidates = Nx.tensor([[10.0], [20.0], [30.0]])

      result = CovariateSelection.select(target, candidates)

      assert result.selected_indices == [0]
      assert Nx.shape(result.selected_matrix) == {3, 1}
    end
  end

  describe "select_spike_slab/3" do
    test "works via select/3 method dispatch" do
      target = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

      candidates =
        Nx.tensor([
          [1.0, 0.2],
          [2.0, -0.1],
          [3.0, 0.1],
          [4.0, -0.2],
          [5.0, 0.0]
        ])

      result =
        CovariateSelection.select(target, candidates,
          method: :spike_and_slab,
          num_samples: 50,
          burn_in: 50,
          seed: 123
        )

      assert result.method == :spike_and_slab
      assert is_list(result.pip)
      assert is_list(result.posterior_mean_beta)
      assert length(result.pip) == 2
    end

    test "recovers sparse signal with high PIP for true regressors" do
      :rand.seed(:exsss, {900, 901, 902})

      n = 220
      p = 100
      true_idxs = [5, 23, 77]
      true_betas = %{5 => 4.0, 23 => -3.5, 77 => 3.0}
      noise_sd = 0.5

      x_rows =
        Enum.map(1..n, fn _ ->
          Enum.map(1..p, fn _ -> :rand.normal() end)
        end)

      target =
        Enum.map(x_rows, fn row ->
          signal =
            Enum.reduce(true_idxs, 0.0, fn j, acc ->
              acc + Map.fetch!(true_betas, j) * Enum.at(row, j)
            end)

          signal + :rand.normal() * noise_sd
        end)

      result =
        CovariateSelection.select_spike_slab(Nx.tensor(target), Nx.tensor(x_rows),
          num_samples: 350,
          burn_in: 350,
          thin: 1,
          seed: 42,
          prior_inclusion: 0.03,
          spike_sd: 0.03,
          slab_sd: 1.5,
          pip_threshold: 0.5
        )

      pip_map = Map.new(result.pip)

      Enum.each(true_idxs, fn j ->
        assert pip_map[j] > 0.95,
               "true regressor #{j} should have high PIP; got #{pip_map[j]}"
      end)

      max_noise_pip =
        result.pip
        |> Enum.reject(fn {j, _pip} -> j in true_idxs end)
        |> Enum.map(fn {_j, pip} -> pip end)
        |> Enum.max()

      assert max_noise_pip < 0.25,
             "noise regressors should have low PIP; max was #{max_noise_pip}"
    end
  end
end
