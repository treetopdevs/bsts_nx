# Code Review Synthesis — origin/main..HEAD

**Date:** 2026-05-15
**Branch:** `main` (7 commits ahead of `origin/main`)
**Scope:** 26 lib files, +3,228 / -739 LOC (tests/livebooks/docs excluded)
**Method:** Four parallel reviewers (holistic severity, security, math/logic correctness, duplication/maintenance), deduplicated and cross-validated.

**Headline:** No CRITICAL bugs. Six HIGH issues. R sidecar is materially safe-by-design — two MEDIUM hygiene fixes only. Significant copy-paste in the CausalImpact estimator family is the single biggest maintenance liability.

---

## HIGH

### H1 — `safe_to_number` silently corrupts Gibbs posteriors and self-suppresses warnings forever
`lib/bsts_nx/gibbs_sampler.ex:1398-1428` (used at `:632-633, :784-785, :1313-1318`)

Coerces `:nan → 0.0`, `:inf → 1.0e10` and any unknown → `0.0`, then feeds the value into the inverse-gamma posterior shape/scale. A non-finite residual collapses `obs_ss = 0`, drawing `R` from the prior with falsely tight uncertainty. The first warning sets a `:persistent_term` flag that suppresses every future warning **for the BEAM lifetime** — not per-call.

**Fix:** fail hard or skip the iteration on non-finite input; remove the persistent-term suppression.

### H2 — Mixed `f32`/`f64` precision between scalar and structured filter entry points
`lib/bsts_nx/causal_impact.ex:427, :641` (f32) vs `lib/bsts_nx/operational.ex:434` (f64) vs `lib/bsts_nx/applications/anomaly_detector.ex:357` (f32)

Same logical computation produces different numeric results depending on which entry point the caller uses. Tests pinned to one path won't catch regressions in the other.

**Fix:** standardize observation tensors at `{:f, 64}` everywhere ingestion happens.

### H3 — RSidecar `String.to_integer` on `BSTS_NX_R_TIMEOUT_MS` raises, gets squashed by blanket rescue
`lib/bsts_nx/r_sidecar.ex:226-232` + `:113-118`

Malformed env (`""`, `"abc"`, `"1.5"`) raises `ArgumentError`, caught by the `rescue e -> {:error, %{reason: :r_system_error, ...}}` block — user sees a generic system error instead of "invalid env var". Same blanket-rescue swallows the stacktrace from genuine bugs.

**Fix:** `Integer.parse/1` with a typed error; `Logger.error(Exception.format(:error, e, __STACKTRACE__))` before wrapping.

### H4 — RSidecar inherits the BEAM env into Rscript (secret leakage to a future R defect or package)
`lib/bsts_nx/r_sidecar.ex:88, :168` — CWE-526

`System.cmd/3` does not clear env by default. AWS keys, DB URLs, API tokens in the parent env are inherited by the R subprocess (and by every R package executed inside it). Today's `r_code/0` doesn't use them, but it's defense-in-depth that costs ~10 lines.

**Fix:** `clear_env: true` plus an explicit minimal env (`HOME`, `LANG`, `R_LIBS_USER`, `TMPDIR`, `PATH`, plus the `BSTS_NX_*` vars).

### H5 — RSidecar error tuple echoes full R stdout+stderr (info disclosure via logs)
`lib/bsts_nx/r_sidecar.ex:107` — CWE-209

With `stderr_to_stdout: true`, the error tuple contains the full R traceback, `.libPaths()`, the temp payload path, and any inherited env vars R prints in package warnings. If callers log errors to a centralized service, this leaks internal layout.

**Fix:** truncate, log verbose output at `debug` level, return a fixed `:r_failed` reason with a bounded `output_excerpt`.

### H6 — PRNG keys round-trip through Erlang lists in tight forecasting loops
`lib/bsts_nx/forecaster.ex:285-322, :378-396`; same pattern at `lib/bsts_nx/applications/demand_forecaster.ex:249, :339-346`

```elixir
[key_state_row, key_obs_row] = Nx.Random.split(sample_key, parts: 2) |> Nx.to_list()
key_state = Nx.tensor(key_state_row, type: ...)
```

Two BEAM lists allocated per posterior sample, full type re-validation, copy back across the EXLA boundary. `BstsNx.Utils.split_key_at/2` already exists and avoids all of this. There's also a latent correctness risk on backends that distinguish key tensors from generic `{2}` integer tensors.

**Fix:** replace every occurrence with `split_key_at/2`.

---

## MEDIUM

### M1 — Four CausalImpact estimators are 80% copy-paste
`lib/bsts_nx/causal_impact.ex` — `estimate/4` (73-161) ≈ `estimate_structured/5` (189-271); `estimate_from_filter/3` (423-613) ≈ `estimate_structured_from_filter/4` (637-802).

Already produced one inconsistency: `cross_cov_included: false` is set in one empty-path return and entirely omitted from the sibling. Highest-leverage refactor in the branch — extract `build_impact_result/...` and `build_filter_summary/...` helpers.

### M2 — Smoother defn `_solve` / `_pinv` pairs are textual clones
`lib/bsts_nx/smoother.ex` — `rts_defn_matrix_impl` (119-164) vs `_pinv` (166-211); `simulate_from_filtered_defn_matrix_impl_solve` (458-515) vs `_pinv` (517-570); plus `simulate_defn_impl` (380-417) vs `simulate_from_filtered_defn_impl` (419-456).

46-58 line bodies kept in lockstep manually. Numerical fixes (e.g., the `near_zero_p` guard at :403/:442) must be applied to both copies or one backend silently diverges.

### M3 — `safe_cholesky_or_zero_defn` runs ~3× the necessary linear algebra
`lib/bsts_nx/smoother.ex:881-896`

Inside defn there's no short-circuit, so every smoother step executes raw Cholesky + retry Cholesky + eigh-free fallback, then `Nx.select`s among them. ~3× per-step cost vs the eager version. Painful for composed trend+seasonal+regression specs.

### M4 — `sample_chains` and `sample_structured_chains` duplicate ~80 LOC of orchestration
`lib/bsts_nx/gibbs_sampler.ex:145-227` vs `:499-567` — same `cond` (seeds/key derivation), same `Task.async_stream`, same warning-and-drop on chain failure. Extract `run_chains(num_chains, opts, fun, label)`.

### M5 — Forward-simulation loop inlined five times
`forecaster.ex:305-343, :377-408` · `applications/demand_forecaster.ex:329-373` · `applications/anomaly_detector.ex:463-501, :503-521` · `causal_impact.ex:938-1006` · `operational.ex:299-349` (the only defn version).

Each computes `obs_sd`, `q_sds`, then `x_{t+1} = F·x_t + w_t`, `y_t = H·x_t + v_t`. Five places to keep in sync. Extract `BstsNx.Forward`.

### M6 — `intervention_analysis.ex` forwards unknown keys silently
`lib/bsts_nx/intervention_analysis.ex:299-301` — `:control_selection*`, `:fallback`, `:return`, etc. survive the `Keyword.drop` and flow into `GibbsSampler.sample_structured/4`, which silently ignores unknown options. Typos like `:control_seclection` produce no signal. Switch to allow-list `Keyword.take`.

### M7 — `Pipeline.split_options` and `Operational.attribution_opts` reimplement the same shaping with an undocumented semantic difference
`pipeline.ex:152-166` uses `Keyword.put` (overwrites caller's `:n_samples`); `operational.ex:583-594` uses `Keyword.put_new` (preserves it). Probably a bug. Centralize in one helper, pick `put_new`.

### M8 — Period validation triplicated with subtly different error wording
`intervention_analysis.ex:151-169` · `causal_impact.ex:1143-1161` · `operational.ex:565-574`. Direct `CausalImpact.estimate` callers see different messages than `analyze` callers for the same condition. Centralize in `BstsNx.Validation`.

### M9 — `gibbs_sampler.ex` is a 1,700-line god module
Mixes chain orchestration, structured sampler, spike-and-slab, generic linear-algebra primitives, and a `:persistent_term`-backed coercion shim. Extract `BstsNx.GibbsSampler.SpikeAndSlab` (~350 LOC) and move LinAlg helpers (~270 LOC) to `BstsNx.Utils`/new `BstsNx.LinAlg`.

---

## LOW

- **Dead code:** `Execution.backend_lu_missing?/1`, `raise_structured_filter_unsupported!/1`, `explicit_mcmc_fallback?/1` (`execution.ex:95-120`) — referenced nowhere in `lib/`. Same for `Smoother.rts_and_simulate_defn/5` (`smoother.ex:374-378`) and `_n_steps` parameter (`causal_impact.ex:971`).
- **Duplicate validation:** `gibbs_sampler.ex:1486-1502` reruns `validate_q_specs!` already done by `ModelSpec.validate!` 28 lines earlier.
- **Two private `transpose_rows/1`** — `applications/anomaly_detector.ex:548-554` and `gibbs_sampler.ex:1198-1204`. Centralize in `Utils`.
- **`MarketingLift.measure_overlapping_group/3` reimplements `ModelBuilder.build_effect`** — `applications/marketing_lift.ex:313-369`.
- **Magic numbers** (`1.0e-15`, `1.0e-12`, `±1.0e10`, `±35.0`) scattered across `causal_impact.ex`, `smoother.ex`, `gibbs_sampler.ex`, `distributions.ex`. Centralize in `BstsNx.Numerics`.
- **`estimate_from_filter` mask reconstruction** does three full passes (`:array` → list → tensor) — `causal_impact.ex:1129-1141`. Will hurt at minute-level horizons.
- **RSidecar `validate_rscript_override`** accepts any executable named `Rscript` and follows symlinks — CWE-426/CWE-59. Low because it requires env-var control.
- **RSidecar `cleanup_payload`** uses `File.rmdir` (fails silently on non-empty dir, no log). Use `File.rm_rf` constrained to the per-call dir.
- **`Distributions.gamma_sample_defn_impl` silent fallback** to `d` after 50,000 rejection iters — degenerate point mass with no observability.
- **`Pipeline.run` validates spot windows twice** in `:operational` mode (`pipeline.ex:104` + `operational.ex:142`).

---

## What was checked and **not** found

- ✅ **No command injection / shell escape** in `r_sidecar.ex` — `System.cmd/3` (no shell), R parameters via `Sys.getenv` only, all numeric payload values pre-validated by `normalize_numeric_list/2`.
- ✅ **No unsafe deserialization** — no `:erlang.binary_to_term`, `Code.eval_*`, `String.to_atom` introduced.
- ✅ **No regex / ReDoS surface** added.
- ✅ **PRNG key threading** in `gibbs_sampler.ex` chain `key_prev → key_after_smooth → key_after_q → key_after_gamma → key_after_beta → key_next` — no reuse.
- ✅ **`smooth_state_cross_covariance` recurrence** — `G_i · G_{i+1} · ... · G_{j-1} · P_{j|T}` is the correct lag-`(j-i)` cross-covariance; gains map covers the full contiguous range so `Map.fetch!` cannot raise.
- ✅ **Inverse-gamma shape/scale** (`α + n/2`, `β + SS/2`) — correct for the SSM signal-variance prior.
- ✅ **g-prior log marginal** in `gibbs_sampler.ex:1065-1082` — equals the standard Zellner formula up to constants that cancel in the Bernoulli ratio.
- ✅ **Burn-in / thin indexing** in `gibbs_sampler.ex:331` — collects exactly `num_samples` post-burn samples for all `(burn_in, thin)` combinations checked.
- ✅ **Counterfactual observation timing** (initially flagged HIGH by the logic reviewer, then verified manually) — `final_state` represents `x_{T_pre}`, so `y_{T_pre+1} = H · (F · x_{T_pre} + w)` is the correct first post-period observation. Both scalar (`generate_counterfactual` :938) and structured (`generate_structured_counterfactual_defn` :1012) match this convention. **Not a bug.**
- ✅ **`mix compile --warnings-as-errors`** is clean on this branch; no Credo configured.

---

## Suggested execution order

1. **H1** — data-correctness foot-gun, low effort
2. **H3, H4, H5** — R sidecar hardening, bundle into one PR
3. **M6, M7, M8** — option-key hygiene, prevents silent option drops
4. **M1, M5** — extract shared estimator + forward simulator, biggest long-term ROI
5. **H2** — precision standardization, needs a test sweep
6. **H6, M2, M3** — perf, measure first
7. Dead code cleanup + magic-number centralization (mechanical)

---

## Severity counts

| Severity | Count |
|----------|-------|
| CRITICAL | 0 |
| HIGH     | 6 |
| MEDIUM   | 9 |
| LOW      | 10 |

## Files reviewed

`lib/bsts_nx.ex`, `lib/bsts_nx/applications/{anomaly_detector,demand_forecaster,marketing_lift,policy_evaluator,tv_attribution}.ex`, `lib/bsts_nx/bct/ar_forecaster.ex`, `lib/bsts_nx/causal_impact.ex`, `lib/bsts_nx/components.ex`, `lib/bsts_nx/covariate_selection.ex`, `lib/bsts_nx/distributions.ex`, `lib/bsts_nx/execution.ex`, `lib/bsts_nx/forecaster.ex`, `lib/bsts_nx/gibbs_sampler.ex`, `lib/bsts_nx/intervention_analysis.ex`, `lib/bsts_nx/kalman_filter.ex`, `lib/bsts_nx/model_builder.ex`, `lib/bsts_nx/model_spec.ex`, `lib/bsts_nx/operational.ex`, `lib/bsts_nx/pipeline.ex`, `lib/bsts_nx/r_sidecar.ex`, `lib/bsts_nx/shapley_allocator.ex`, `lib/bsts_nx/smoother.ex`, `lib/bsts_nx/spot_attributor.ex`, `lib/bsts_nx/synthetic/generator.ex`, `lib/bsts_nx/utils.ex`, `lib/bsts_nx/validation.ex`.
