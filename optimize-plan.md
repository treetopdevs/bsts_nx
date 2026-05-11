# BstsNx Performance Optimization Plan (Integrated)

## Context

Thorough code review of the entire BstsNx codebase identified ~75 optimization opportunities across all core modules. The dominant performance bottleneck pattern is Nx tensor materialization (`Nx.to_number`, `Nx.to_flat_list`) inside MCMC hot loops, which forces synchronous device-to-host transfers on every iteration. Secondary patterns include Elixir list arithmetic where Nx vectorized ops should be used, redundant rank dispatch, unnecessary intermediate tensor allocations, and O(n²) list-access anti-patterns in diagnostics and selection code.

Changes are organized into 5 tiers by impact. Each tier can be implemented and tested independently. All changes preserve mathematical precision unless explicitly noted.

---

## Execution Checklist

This section is the tracker of record for completion status.

### Milestones

- [x] PR 1: Tracking + benchmark harness
- [~] PR 2: Finish Tier 1 + Tier 2 core
- [~] PR 3: Finish Tier 2D/2E + remaining Tier 4E batching
- [~] PR 4: Close remaining Tier 3/4 medium work
- [~] PR 5: Tier 5A compiled multi-dimensional filter
- [~] PR 6: Tier 5B compiled multi-dimensional RTS smoother
- [x] PR 7: Tier 5C pure-Nx gamma / inv-gamma defn path
- [ ] PR 8: Tier 5D spike-and-slab rank-1 updates + closeout

### Tier Snapshot

- [~] Tier 1: Mostly complete
- [~] Tier 2: Partial
- [~] Tier 3: Mostly complete
- [~] Tier 4: Mostly complete
- [~] Tier 5: Partial; PR 5/6/7 paths are wired, PR 8 remains deferred

Legend: `[x]` done, `[~]` partial, `[ ]` pending.

---

## Tier 1: MCMC Hot-Loop Materializations (Highest Impact)

These changes target code that runs `iterations × T` times (thousands to millions of invocations).

### 1A. Vectorize sum-of-squares in GibbsSampler scalar path

**File:** `lib/bsts_nx/gibbs_sampler.ex` (lines 312-315, 1491-1522)
- Replace `Nx.to_flat_list(sampled_xs)` + Elixir list arithmetic in `process_sum_of_squares` and `obs_sum_of_squares` with defn tensor operations
- `process_ss_defn`: `diffs = x[1..] - f * x[0..-2]`, then `Nx.sum(diffs * diffs)`
- `obs_ss_defn`: `diffs = obs - h * states`, then `Nx.sum(diffs * diffs)` (skip NaN entries via mask)
- Pre-compute missing observation bitmask once before MCMC loop

### 1B. Vectorize structured residual computation

**File:** `lib/bsts_nx/gibbs_sampler.ex` (lines 1271-1303, 1307-1322)
- `process_residuals_structured`: Replace per-timestep `Nx.to_flat_list(residual)` + `compat_dot(f_t, prev_state)` with:
  ```
  states = Nx.stack(sampled_states)           # {T, state_dim}
  predicted = Nx.dot(states[0:T-1], Nx.transpose(f_t))
  residuals = states[1:T] - predicted
  ss_vec = Nx.sum(residuals², axes: [0])      # {state_dim}
  ```
  Replaces T matrix-vector mults per iteration with 3 tensor ops.
- `obs_residuals_structured`: Replace per-timestep `Nx.to_number(compat_dot(h_row, x_t))` with batched element-wise multiply + row-sum:
  ```
  states_t = Nx.stack(sampled_states)         # {T, state_dim}
  h_t = Nx.stack(h_rows)                      # {T, state_dim}
  preds = Nx.sum(Nx.multiply(h_t, states_t), axes: [1])
  ```

### 1C. Eliminate Nx.to_number(s) in eager Kalman filter

**File:** `lib/bsts_nx/kalman_filter.ex` (line 151)
- Replace `if abs(Nx.to_number(s)) < 1.0e-15` with `Nx.select(Nx.less(Nx.abs(s), 1.0e-15), ...)` pattern (already used in `filter_defn_vec`)

### 1D. Eliminate Nx.to_number in eager Smoother backward loops

**File:** `lib/bsts_nx/smoother.ex` (lines 517-521, 476-488)
- In `smoother_gain`: replace `Nx.to_number(p_pred_next)` threshold check with `Nx.select`
- In `simulate_with_key`: replace `Nx.to_number(cov)` negative check with `Nx.max(cov, 1.0e-12)`

### 1E. Fast scalar path for inverse-gamma sampling

**File:** `lib/bsts_nx/distributions.ex` (lines 140-186)
- Add `when is_number(alpha) and is_number(beta)` guard clause that skips tensor round-trip entirely
- Fix `Nx.Random.split(key, parts: 2)` → `Nx.Random.split(key)` tuple return (lines 105-107, 126-128, 170-179)
- Merge `validate_prng_key!` into `rand_state_from_key` to avoid double materialization
- **[NEW]** Replace per-draw `split_key_at` (Nx.slice_along_axis + Nx.squeeze) loop in `sample_with_key/4` with single `Nx.to_list(split_keys)` before loop — eliminates N slice ops per batch draw

### 1F. [NEW] Switch sample_general to use defn filter+smoother for scalar case

**File:** `lib/bsts_nx/gibbs_sampler.ex` (lines 300-344)
- `sample_general/5` currently uses eager `filter_with_pred` + `Smoother.rts` inside the MCMC loop
- The compiled `filter_defn` + `rts_defn` paths already exist
- Switch to: `KalmanFilter.filter_defn` + `Smoother.rts_defn` per iteration, keeping only IG sampling in Erlang
- Converts O(T) eager Nx ops per filter/smooth pass into two compiled defn calls per iteration
- This is the single highest-impact change for the scalar sampler path

### 1G. [NEW] Normalize H series without O(T) Nx.slice calls

**File:** `lib/bsts_nx/kalman_filter.ex` (lines 393-402)
- `normalize_h_series/2`: For rank-1 time-varying H, iterates `0..(T-1)` calling `Nx.slice(h, [idx], [1]) |> Nx.squeeze()` per step — T individual kernel dispatches
- Replace with single `Nx.to_flat_list(h)` followed by `Enum.map(&Nx.tensor/1)` — one device transfer for all of H
- Similarly for rank-2 case: use `Nx.to_batched(h, 1)` or `Nx.to_list`

---

## Tier 2: Compiled Path Expansion (High Impact, More Effort)

### 2A. Fuse rts_defn + simulate_defn into single backward pass

**File:** `lib/bsts_nx/smoother.ex`
- Create `rts_and_simulate_defn/5` that does smoothing + sampling in one backward loop
- Eliminates second O(T) pass and two intermediate T-length tensors (sxs, sps)
- Smoother gain computed once instead of twice
- Update `gibbs_sampler.ex` scalar path (lines 307-310) to call fused version
- **[NEW]** Also unifies the duplicated backward loop between `rts_defn_impl` and `rts_defn_with_lag1_impl` (lines 49-90 vs 132-177) — same logic, only lag1 computation differs. Single defn with compile-time boolean parameter.

### 2B. Remove read-only tensors from while-loop state (Nx 0.11+)

**File:** `lib/bsts_nx/smoother.ex` (lines 64-87, 149-174, 217-237)
- Remove `xs_in`, `ps_in`, `f_in`, `q_in` from while-loop carried state — capture from outer defn scope
- Reduces loop state from 7→3 tensors (rts_defn) and 10→4 (simulate_defn)
- **[NEW]** Also applies to `take_scalar_at` helper (line 179) called 4-6 times per backward step — currently does `Nx.slice(vec, [idx], [1]) |> Nx.squeeze()` inside while loop. These gather ops are unavoidable in the while loop but reducing carried state improves XLA optimization.
- Verify closure capture works on Nx 0.11 before committing

### 2C. Hoist invariant operations out of eager filter/smoother loops

**File:** `lib/bsts_nx/kalman_filter.ex` (lines 150-200), `lib/bsts_nx/smoother.ex`
- Determine `is_scalar_state` once before loop, dispatch to specialized scalar vs matrix inner functions
- Hoist `Nx.eye(n)` allocation outside loop (line 181) — currently allocates `n×n` identity per time step
- **[NEW]** Pre-compute `Nx.rank` branching flags before the reduce (lines 128-206 check `Nx.rank(s)`, `Nx.rank(p_pred)`, `Nx.rank(k)`, `Nx.rank(p_filt)` on every step — values never change for fixed-dim models)
- Remove `mul_or_dot` rank dispatch — call `Nx.dot` directly (works for scalars in Nx 0.11)

### 2D. [NEW] Compile forward_simulate as defn for RollingBaseline

**File:** `lib/bsts_nx/rolling_baseline.ex` (lines 518-554)
- `forward_simulate` does S×H iterations with ~8 Nx ops each (including `Nx.transpose(f)` recomputed every step)
- Immediate fix: hoist `Nx.transpose(f)` and `Nx.transpose(h_row)` outside the loop
- Better: compile the entire horizon propagation as a `Nx.Defn.while` loop — eliminates S×H Elixir→Nx dispatch overhead
- For S=200, H=96: eliminates 19,200 redundant `Nx.transpose` calls + reduces 19,200 Elixir iterations to compiled XLA while-loops
- Same optimization applies to `forward_simulate_h` (line 538)

### 2E. [NEW] Batch predict_structured inner loop

**File:** `lib/bsts_nx/forecaster.ex` (lines 322-341)
- Currently ~10 Nx boundary crossings per step per sample (3 `split_key_at`, 2 `Nx.Random.normal`, 2 `Nx.to_number`, 1 `Nx.dot`, etc.)
- Pre-generate all noise as `{horizon, n_state}` and `{horizon}` tensors outside the reduce (matching what `predict_scalar` already does correctly at lines 362-377)
- Apply `q_sds` to entire noise matrix at once: `Nx.multiply(proc_noise, q_sds)` instead of per-step
- For static H: compute all y_mean predictions as single `Nx.dot(mean_state_tensor, h_row)`

---

## Tier 3: Vectorization of Supporting Computations (Medium Impact)

### 3A. Covariate selection: Elixir lists → Nx tensors

**File:** `lib/bsts_nx/covariate_selection.ex` (lines 140-223, 439-474, 557-564)
- Replace `dot/2` (`Enum.zip`+`reduce`) and `add_scaled/3` (`Enum.zip`+`map`) with `Nx.dot` and `Nx.add(vec, Nx.multiply(x, scale))`
- Store `y`, `x_cols`, `residual` as Nx tensors throughout Gibbs sweep
- Replace `Enum.at` column extraction (O(n×j)) with Nx slicing
- Standardize columns via vectorized `Nx.mean`/`Nx.standard_deviation`
- **[NEW]** Also vectorize `select_pearson/3` (lines 283-289): Replace p sequential per-column correlations (3p host-device syncs) with batched matrix correlation:
  ```
  t_centered = Nx.subtract(target, Nx.mean(target))
  c_centered = Nx.subtract(candidates, Nx.mean(candidates, axes: [0]))
  numerators = Nx.dot(t_centered, c_centered)           # {p} — all cross-products at once
  col_ss = Nx.sum(c_centered * c_centered, axes: [0])   # {p}
  corrs = numerators / Nx.sqrt(t_ss * col_ss)
  ```
  Reduces ~6p tensor ops + 3p host syncs to 4 batched Nx ops + 1 sync.
- **[NEW]** Convert `x_rows` to tuples via `List.to_tuple/1` for O(1) positional access in spike-and-slab column extraction (lines 155-164) — reduces O(n×p²) to O(n×p)
- **[NEW]** Use `Enum.zip_reduce` instead of `Enum.zip` + `Enum.reduce` in `dot/2` helper to avoid allocating intermediate zipped list

### 3B. Diagnostics: O(n²) → O(n·L) autocovariance + O(n²) HPD fix

**File:** `lib/bsts_nx/diagnostics.ex` (lines 104-127, 376-406)
- Precompute autocovariances via Nx dot products: `Nx.dot(centered[0..n-lag], centered[lag..n])` for all needed lags up front
- Store in `:array` for O(1) lookup in truncation loop
- **[NEW]** Extract shared `autocovariance_at/3` helper to deduplicate between `ess_single` and `spectral_density_zero` (both implement identical `:array`-based gamma closure — lines 104-127 and 381-404)
- Fix `hpd_interval` O(n²) `Enum.at` → `:array`/tuple for O(1) access (lines 289-305)
  - Or better: Nx-based approach — `Nx.sort` + `Nx.slice` widths + `Nx.argmin` for O(n) total
- **[NEW]** Fix `chain_stats/1` (lines 338-354):
  - Chain mean computed twice (once in `Enum.map`, again inside W variance reduce) — reuse precomputed means via `Enum.zip(chains_n, means)`
  - Replace `:math.pow(x - mean_c, 2)` with `d * d` in tight inner loop (line 351) — avoids C `pow()` function call overhead

### 3C. Batch rebuild_q in structured sampler

**File:** `lib/bsts_nx/gibbs_sampler.ex` (lines 1339-1346)
- Batch all `Nx.indexed_put` calls into single operation with stacked indices/updates
- **[NEW]** Since Q is always diagonal, simplest approach: `Nx.make_diagonal(Nx.tensor(new_diag_values))` — one allocation instead of k sequential `indexed_put` copies. Requires `q_specs` to cover all dims.

### 3D. Shapley: eliminate redundant sorting + cache coalitions

**File:** `lib/bsts_nx/shapley.ex` (lines 142-154, 206-215)
- Pre-sort `others` so `subset_from_mask` returns sorted subsets
- Use sorted insertion instead of `Enum.sort` per step in MC path
- **[NEW]** Cache all 2^n coalition values by bitmask in `exact_shapley`:
  ```
  coalition_cache = Map.new(0..(2^n - 1), fn mask -> {mask, value_fn.(sorted_subset)} end)
  ```
  Reduces value_fn calls from n×2^(n-1) to 2^n (for n=12: 24,576 → 4,096 — 6× reduction)
- **[NEW]** In `monte_carlo_shapley`: carry `sorted_coalition` forward and reuse `v_with` from step k as `v_without` at step k+1 — halves value_fn calls and replaces O(k log k) sorts with O(k) insertions
- **[NEW]** Fix `phash2` entropy loss in PRNG bridge (line 199-200): use `BstsNx.Utils.derive_exsss_seed/1` instead of raw `:erlang.phash2` which collapses two 64-bit integers to one 27-bit hash
- **[NEW]** Precompute `decay_powers` lookup table in `default_value_function/2` (line 273) to avoid `:math.pow(decay, rank)` per coalition element

### 3E. [NEW] AR Forecaster: Elixir matrix multiply → Nx

**File:** `lib/bsts_nx/bct/ar_forecaster.ex` (lines 192-228)
- `estimate_coefficients` implements X'X and X'y with nested `Enum.with_index` + `Enum.at` (O(i) per access)
- For AR(10) with n=500: ~55M list-index operations
- Replace with:
  ```
  x = Nx.tensor(rows, type: {:f, 64})
  xtx = Nx.dot(Nx.transpose(x), x)              # single BLAS DGEMM call
  xty = Nx.dot(Nx.transpose(x), y)
  coeffs = BstsNx.Utils.safe_solve(Nx.add(xtx, ridge_mat), xty)
  ```
- **[NEW]** Also fix `summarize_paths` (line 337-343): Replace `Enum.map(0..(horizon-1), fn idx -> Enum.map(paths, &Enum.at(&1, idx)) end)` with `Enum.zip_with(paths, & &1)` — O(horizon²×n_samples/2) → O(horizon×n_samples)

### 3F. [NEW] Vectorize posterior diagonal extraction in RollingBaseline

**File:** `lib/bsts_nx/rolling_baseline.ex` (lines 378-445)
- `posterior_q_means/2` calls `diagonal_value_at` (which does `Nx.take_diagonal` + `Nx.take` + `Nx.squeeze` + `Nx.to_number`) D×S times — one per q_spec dimension per sample
- `extract_process_var_chain/2` repeats the same pattern
- Fix: Extract all diagonals once per sample, cache as Elixir lists:
  ```
  all_diags = Enum.map(samples, fn s -> Nx.take_diagonal(s.q_matrix) |> Nx.to_flat_list() end)
  ```
  Then index with `Enum.at(diag_list, dim)` — pure Elixir, no Nx. Reduces D×S×4 Nx calls to S×2.

### 3G. [NEW] Spike-and-slab covariance rebuild optimization

**File:** `lib/bsts_nx/gibbs_sampler.ex` (lines 1064-1088, 1107-1155)
- `build_full_q_matrix/3`: O(n²) Elixir list comprehension + `Nx.tensor` per saved sample
- `build_full_covariances/5`: nested O(T×n²) rebuild per retained sample — dominant cost in spike-and-slab
- Fix: Use `Nx.put_slice` for the entire structural block if indices are contiguous (they usually are for structural components), or `Nx.indexed_put` with batch of indices
- Also fix `submatrix/3` (lines 1233-1250): `Enum.at(row_data, r)` is O(n) — convert `row_data` to tuple via `List.to_tuple` for O(1) access

---

## Tier 4: Setup & Aggregation Efficiency (Lower Impact)

### 4A. Time-varying H construction

**File:** `lib/bsts_nx/components.ex` (lines 635-637, 694-697)
- Replace per-row `Nx.slice` with `Nx.to_batched(1)` for H list construction
- In `compose_h` list+list path (lines 542-546): if both stored as stacked tensors, single `Nx.concatenate` on axis=2 instead of T separate concatenations
- **[NEW]** In `compose_h` static+time-varying path (lines 552-557): broadcast static H and concatenate once instead of T per-step concatenations
- **[NEW]** `build_future_h/3` in model_builder.ex (lines 348-354): Broadcast `static_h` to `{horizon, n_non_reg}`, stack regressor rows, concatenate once — replaces horizon tensor allocations with one op. Same fix for `rolling_baseline.ex:618-621`.

### 4B. Summary statistics consolidation

**File:** `lib/bsts_nx/causal_impact.ex` (lines 328-385), `lib/bsts_nx/forecaster.ex` (lines 380-416)
- Combine sort + mean + sd + percentile into single `compute_stats` function (1 sort + 2 traversals instead of 1 sort + 3 traversals)
- Convert sorted list to tuple for O(1) percentile access in `Utils.percentile_interval`
- **[NEW]** `aggregate_trajectories` in forecaster.ex (lines 385-406): Stack all trajectories into `{S, horizon}` tensor and use `Nx.sort(t, axis: 0)` + `Nx.mean` + `Nx.standard_deviation` — one batched sort instead of horizon separate Elixir sorts
  - **Precision note:** Confirm `Nx.standard_deviation` uses `ddof: 1` (Bessel's correction) to match current `ss/(n-1)` formula
- **[NEW]** `summary/1` in causal_impact.ex (lines 351-367): Per-time-step `Enum.sort(vals)` for interval computation — same vectorization applies: stack into `{m, n_post}` tensor and compute percentiles along axis 0
- **[NEW]** Per-spot statistics in spot_attributor.ex (lines 232-260): Same pattern — collect per-spot lifts into `{n_draws, n_spots}` tensor, batch sort + mean + stats
- **[NEW]** Consider Welford's online algorithm for single-pass mean+variance in `aggregate/3` (demand_forecaster.ex:383-398) and `compute_*_predictions` (anomaly_detector.ex:391-429) to eliminate second pass

### 4C. Forecaster.decompose tensor round-trips

**File:** `lib/bsts_nx/forecaster.ex` (lines 237-276)
- Replace per-step `Nx.to_flat_list` (20,000 calls for 200 samples × 100 steps) with `Nx.stack` → `Nx.mean(stacked, axes: [0])`
- **[NEW]** Also vectorize fitted value computation (lines 265-269): For static H, `Nx.dot(mean_state_tensor, h_row)` replaces T individual `Nx.tensor(state_list)` + `compat_dot` + `Nx.to_number` calls

### 4D. Utils/small fixes

**Files:** `lib/bsts_nx/utils.ex`, `lib/bsts_nx/state_space.ex`
- `to_tensor(v)`: `Nx.tensor(v)` instead of `Nx.tensor([v]) |> Nx.squeeze()` (line 14)
- `safe_cholesky`: hoist `Nx.eye(dim)` outside jitter retry loop (lines 50-51)
- `block_diag`: fast-path for all-scalar lists via `Nx.make_diagonal`
- **[NEW]** `block_diag` 2-block special case (most common for BSTS composition):
  ```
  def block_diag([a, b]) do
    top    = Nx.concatenate([a, Nx.broadcast(0.0, {n1, n2})], axis: 1)
    bottom = Nx.concatenate([Nx.broadcast(0.0, {n2, n1}), b], axis: 1)
    Nx.concatenate([top, bottom], axis: 0)
  end
  ```
  Two concatenations instead of 2 `put_slice` + copy.
- Consolidate 5 copies of `compat_dot` into `BstsNx.Utils`
- **[NEW]** `has_non_finite?/1` (lines 130-137): Replace `Nx.is_nan` + `Nx.is_infinity` + OR with single `Nx.is_finite` — one kernel instead of three
- **[NEW]** `normalize_number/1` in model_builder.ex (lines 479-490): Replace `:erlang.float_to_binary(f, [:compact])` string-based NaN/Inf check with arithmetic `f != f` or `f - f == 0.0` — zero allocation
- **[NEW]** Replace `Nx.pow(2)` with `Nx.multiply(x, x)` throughout (validation.ex:49, diagnostics, etc.) — `pow` dispatches to generic C power function; self-multiply is a direct elementwise op

### 4E. Pre-batch PRNG noise in counterfactual generation

**Files:** `lib/bsts_nx/causal_impact.ex` (lines 705-745), `lib/bsts_nx/forecaster.ex` (lines 308-345)
- Generate all state/obs noise upfront via `Nx.Random.normal(key, shape: {n_steps, ...})` instead of per-step key splitting
- **[NEW]** Also applies to `demand_forecaster.ex` (lines 239-243, 328-339): Replace per-sample `split_key_at(keys, idx)` with single `Nx.to_list(keys)` before the map — eliminates n_samples × horizon × 3 tensor slice operations
- **[NEW]** Same fix for `generate_structured_counterfactual` in causal_impact.ex (lines 717-741): pre-extract all subkeys via `Nx.to_list(keys)` before the reduce

### 4F. [NEW] AnomalyDetector optimizations

**File:** `lib/bsts_nx/applications/anomaly_detector.ex`
- `score_one` (lines 201-224): For scalar Kalman filter, all values are 0-dim tensors. Store and operate on plain Elixir floats instead — eliminates ~10 `Nx.tensor`/`Nx.to_number` round-trips per observation in streaming mode
- `compute_structured_predictions` (lines 373-405): `Nx.rank(h_t)` + `Nx.squeeze`/`Nx.flatten` called T×S times. Pre-normalize all H once before the loop; for static H, convert to flat list and do dot product in pure Elixir

### 4G. [NEW] Code quality: eliminate duplicates

- `intervention_analysis.ex` (lines 341-358): `is_significant?/2` and `is_significant_filter?/1` have identical bodies — merge
- `diagnostics.ex`: Extract shared `autocovariance_at/3` from duplicated `ess_single`/`spectral_density_zero` closures
- `covariate_selection.ex` `build_selected_matrix/3` (lines 319-326): Same k-slice+concatenate pattern as `model_builder.ex` `build_selected_regressors` — both should use `Nx.take(tensor, indices, axis: 1)`
- `spot_attributor.ex` `evaluate_attribution` (line 292): `overlap_groups` recomputed per draw even though `groups` is constant — compute once before the reduce

### 4H. [NEW] Minor construction optimizations

**File:** `lib/bsts_nx/components.ex`
- `build_seasonal_transition` (lines 588-597): 5 intermediate tensor allocations for a constant-structure matrix. Replace with `Nx.tensor(rows)` from list-of-lists — single allocation, run-once cost
- `seasonal/2` and `seasonal_spec/2` (lines 151-154, 422-423): `Nx.broadcast(0.0, {dim, dim}) |> Nx.put_slice(...)` for sparse Q/H. Replace with `Nx.tensor([[1.0 | List.duplicate(0.0, dim-1)]])` and `Nx.make_diagonal(...)` — eliminates intermediate allocation
- `mv_normal_sample/3` in distributions.ex (lines 122-138): Cholesky recomputed every call. Expose `mv_normal_sample_with_chol/4` variant for callers that reuse the same covariance (e.g., fixed observation noise in the simulation smoother)

### 4I. [NEW] RollingBaseline counterfactual accumulation

**File:** `lib/bsts_nx/rolling_baseline.ex` (lines 256-309)
- Per-sample accumulation with `add_lists` (Enum.zip_with) does S×3 O(horizon) list passes
- Collect means/vars as tensors per sample, then `Nx.stack` + `Nx.mean`/`Nx.variance` for law of total variance
- **Precision note:** Use `type: {:f, 64}` for accumulator tensors if f32 loss (~7 decimal digits) is unacceptable for MCMC posterior statistics

---

## Tier 5: Structural / Long-term (Highest Effort)

### 5A. Compiled multi-dimensional Kalman filter (filter_defn_multi)

**File:** `lib/bsts_nx/kalman_filter.ex` (new function)
- defn filter for multi-dim state using matrix ops in while-loop
- Would replace eager `filter_with_pred` in structured Gibbs sampler (5-20× potential speedup)
- **[NEW]** The `filter_defn_vec` while loop (lines 308-338) already uses `Nx.put_slice` in a while loop — on XLA/EXLA this prevents vectorization/unrolling. The multi-dim version should use the same pattern but with matrix ops, accepting this XLA limitation. Investigate whether restructuring as separate output streams improves throughput.
- Tradeoff: Requires fixed state dimension at compile time; cannot handle `nil` missing obs (must use NaN sentinel)

### 5B. Compiled multi-dimensional RTS smoother (rts_defn_matrix)

**File:** `lib/bsts_nx/smoother.ex` (new function)
- Companion to 5A for the backward pass
- Stacked tensor layout: xs shape `{T, n}`, ps shape `{T, n, n}`
- Tradeoff: Must replace `safe_solve` try/rescue with defn-compatible regularization (always-add-jitter strategy)
- **[NEW]** The `safe_cholesky_or_zero` fallback (lines 522-545) runs full `Nx.LinAlg.eigh` eigendecomposition — O(n³). For composed models with 20+ states this matters. The defn version should use always-add-jitter without try/rescue.

### 5C. Pure-Nx defn gamma sampler

**Status:** Complete in PR 7.

**File:** `lib/bsts_nx/distributions.ex` (new implementation)
- [x] Added `gamma_sample_defn/4` using the Marsaglia-Tsang algorithm with `Nx.Random.normal` + `Nx.Random.uniform`
- [x] Reworked `inv_gamma_sample_defn/4` to reuse the pure-Nx gamma kernel and apply inverse-gamma scaling/truncation in defn
- [x] Uses explicit `{:f, 64}` typing for gamma and inverse-gamma parameters/results, including small-shape coverage below `alpha < 0.5`
- [x] Wired scalar and structured Gibbs variance resampling through the defn inverse-gamma path, batching scalar `{Q, R}` and structured Q-component draws
- [x] Added targeted distribution and Gibbs regression tests
- [ ] Deferred: full end-to-end MCMC-loop fusion remains out of scope for PR 7; posterior sufficient statistics are still computed by the surrounding sampler code

### 5D. [NEW] Spike-and-slab XtX rank-1 updates

**File:** `lib/bsts_nx/gibbs_sampler.ex` (lines 955-968)
- `resample_gamma_g_prior`: `log_marginal_g_prior` called twice per variable per iteration with near-identical XtX (only column j differs)
- Pre-compute `XtX` and `Xty` for the full active set, then apply Woodbury rank-1 updates for each variable toggle
- For p > 10 this would be a significant speedup; for small p the overhead is justified
- This is algorithmically complex but pays off for high-dimensional regression

---

## Decisions

- **Scope:** All 5 tiers included (Tiers 1-5, including compiled multi-dim filter/smoother and pure-Nx gamma sampler)
- **Branch:** Single feature branch, one commit per tier
- **f32 vs f64:** Use `{:f, 64}` for accumulator tensors in Tiers 4B/4I where Elixir f64 is being replaced. Default f32 is acceptable for Tier 5A/5B defn paths given inherent Monte Carlo noise, but provide `type: {:f, 64}` option.
- **Nx.standard_deviation ddof:** Verify `ddof: 1` behavior in Nx 0.11 before replacing manual variance computations

## Implementation Order

```
Tier 4D (trivial fixes) → Tier 1A-1G (hot loop fixes) → Tier 2A-2E (compiled paths)
→ Tier 3A-3G (vectorization) → Tier 4A-4I (setup/aggregation) → Tier 5A-5D (structural)
```

Each tier should be a separate commit. Run `mix test` after each tier.

## Verification

After each tier:
1. `mix test` — full suite must pass
2. `mix test --include slow` — include slow tests for numerical accuracy
3. For Tiers 1-2, create a quick benchmark script:
```elixir
# Benchmark scalar sampler (Tier 1A target)
obs = BstsNx.Synthetic.Generator.local_level(200, seed: 42)
:timer.tc(fn -> BstsNx.GibbsSampler.sample(obs, 1.0, 1.0, 0.1, 0.1, 500, seed: 1) end)

# Benchmark structured sampler (Tier 1B target)
spec = BstsNx.Components.local_linear_trend_spec(0.1, 0.01)
:timer.tc(fn -> BstsNx.GibbsSampler.sample_structured(obs, spec, 200, seed: 1) end)
```
4. For Tier 5A/5B: compare filter/smoother output between eager and defn implementations to verify numerical equivalence (within f32 tolerance)

## Precision Notes

- **No precision impact:** Tiers 1-4 (identical arithmetic, just in tensor form)
- **Minor f32 vs f64:** Tier 5A/5B defn paths default to f32; explicitly use `Nx.as_type(:f64)` for inputs if precision matters
- **Potential accumulation error:** Tier 3D Shapley value caching — none (exact same computation, just cached)
- **Tier 5C (defn gamma):** complete in PR 7 with explicit f64 typing and targeted small-alpha validation
- **Welford's algorithm** (4B): More numerically stable than two-pass for large n (avoids catastrophic cancellation)
- **Vectorized Pearson** (3A): Numerically equivalent; uses same formula, just batched

## Findings Cross-Reference

Items marked [NEW] were discovered by the parallel code review agents and were not in the original plan. Key additions:
- **1F** (sample_general → defn paths): Highest single-impact change for scalar sampler
- **1G** (normalize_h_series): Eliminates T kernel dispatches in filter setup
- **2D** (forward_simulate defn): Eliminates S×H redundant transpose + Elixir→Nx dispatch
- **2E** (predict_structured batching): Mirrors existing predict_scalar optimization
- **3E** (AR forecaster Nx rewrite): Replaces O(n×p³) Elixir with single BLAS call
- **3F** (diagonal extraction caching): Eliminates D×S×4 redundant Nx calls
- **3G** (spike-and-slab covariance rebuild): Fixes O(T×n²) per retained sample
- **5D** (XtX rank-1 updates): Algorithmic improvement for high-dim spike-and-slab
