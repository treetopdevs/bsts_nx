defmodule BstsNxMissingDataCorrectnessTest do
  use ExUnit.Case, async: true

  alias BstsNx.Components
  alias BstsNx.GibbsSampler
  alias BstsNx.KalmanFilter
  alias BstsNx.Smoother

  # Builds a rank-1 {n} tensor equal to `values` but with NaN at `nan_index`.
  defp with_nan(values, nan_index) do
    n = length(values)
    base = Nx.tensor(values, type: {:f, 64})
    nan = Nx.broadcast(Nx.Constants.nan() |> Nx.as_type({:f, 64}), {n})
    mask_list = Enum.map(0..(n - 1), fn i -> if i == nan_index, do: 1, else: 0 end)
    mask = Nx.tensor(mask_list, type: {:u, 8})
    Nx.select(mask, nan, base)
  end

  defp finite?(v) when is_number(v), do: v == v and v not in [:infinity, :neg_infinity]
  defp finite?(_), do: false

  describe "compiled filter agrees with eager filter on missing data" do
    # The eager path encodes missing as nil; the compiled path encodes it as NaN.
    # Both must produce the same filtered means/covariances.
    test "filter_defn (NaN) matches filter (nil) when an interior obs is missing" do
      f = 1.0
      h = 1.0
      q = 0.1
      r = 0.5
      x0 = 0.0
      p0 = 1.0

      eager = KalmanFilter.filter([1.0, nil, 3.0], f, h, q, r, x0, p0)
      eager_x = Enum.map(eager, fn {x, _p} -> Nx.to_number(x) end)
      eager_p = Enum.map(eager, fn {_x, p} -> Nx.to_number(p) end)

      obs_defn = with_nan([1.0, 0.0, 3.0], 1)
      {xs, ps} = KalmanFilter.filter_defn(obs_defn, f, h, q, r, x0, p0)
      defn_x = Nx.to_flat_list(xs)
      defn_p = Nx.to_flat_list(ps)

      Enum.zip(eager_x, defn_x)
      |> Enum.each(fn {a, b} -> assert_in_delta(a, b, 1.0e-6) end)

      Enum.zip(eager_p, defn_p)
      |> Enum.each(fn {a, b} -> assert_in_delta(a, b, 1.0e-6) end)
    end
  end

  describe "compiled smoother stays finite through a missing observation" do
    test "rts_defn produces finite smoothed states when filter saw a NaN obs" do
      obs_defn = with_nan([1.0, 0.0, 3.0, 4.0, 5.0], 2)
      {xs, ps} = KalmanFilter.filter_defn(obs_defn, 1.0, 1.0, 0.1, 0.5, 0.0, 1.0)
      {sxs, sps} = Smoother.rts_defn(xs, ps, 1.0, 0.1)

      Enum.each(Nx.to_flat_list(sxs), fn v -> assert finite?(v) end)
      Enum.each(Nx.to_flat_list(sps), fn v -> assert finite?(v) end)
    end
  end

  describe "structured Gibbs is reproducible with missing data" do
    test "same seed yields identical samples when an interior obs is missing" do
      observations = [1.0, 2.0, Nx.Constants.nan(), 4.0, 5.0, 6.0]
      spec = Components.local_level_spec(process_var: 0.5, obs_var: 1.0)

      run = fn ->
        GibbsSampler.sample_structured(observations, spec, 5, burn_in: 2, seed: 99)
      end

      a = run.()
      b = run.()

      assert length(a) == length(b)

      Enum.zip(a, b)
      |> Enum.each(fn {sa, sb} ->
        assert Nx.to_flat_list(sa.q_matrix) == Nx.to_flat_list(sb.q_matrix)
        assert Nx.to_number(sa.obs_var) == Nx.to_number(sb.obs_var)
      end)
    end
  end
end
