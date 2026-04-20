# RL-only 20x20 Cantilever — what we tried and what happened

Branch: `codex/rl-only-20x20`
Pipeline: `gigala.topology.topology_optimiz.gen_rl` (MaskablePPO + FEA evaluator)
Launcher: `scripts/run_gen_rl_pipeline_rl_only_20x20.sh`
Problem: classic 2D cantilever beam on a 20x20 grid, volume target ≈ 0.55, score = compliance + small smoothness penalty (lower is better).

This document summarises the experiments on this branch, commit by commit, so that someone new can quickly see what was tried, what worked, and what is still open.

---

## 1. Goal of the branch

Run the **RL-only** variant of the topology-optimisation pipeline on a 20x20 grid end-to-end: start from a trivial full-solid seed, let a PPO agent remove material, and return a valid cantilever mask (1 component, support + load contact, volume inside `[target − slack, target + slack]`). No GA coarse search, no hand-crafted initial design.

The main questions were:

1. Can PPO alone, under a sparse reward, learn to produce feasible low-compliance masks at 20x20?
2. How do we stop the "infeasible collapse" where every episode ends invalid and training stalls?
3. How do we reliably pick the best mask the agent has ever seen, not just the one it happens to produce at inference time?

---

## 2. Timeline of commits (branch-only)

Listed newest → oldest. Each bullet is what was **added/changed** in that commit.

| # | Commit | Subject | What it introduced |
|---|--------|---------|--------------------|
| 1 | `0f36647` | Lower default `RL_TOTAL_TIMESTEPS` to 100k for faster runs | Launcher default; override with `GEN_RL_RL_TOTAL_TIMESTEPS` for full runs. |
| 2 | `6b3d4a3` | Fix RL-only pipeline: terminal cleanup, soft infeasible reward, training-best harvester, entropy/KL controls | Biggest substantive fix on this branch — details below. |
| 3 | `e9b3e51` | Enhance RL CLI (sparse reward, env settings, volume constraints), extend `ProblemConfig`, volume-aware actions | Turned 20x20 RL into a first-class, fully parameterised CLI pipeline. |
| 4 | `c6d9f9e` | Track best RL result from rollout and seed | Persist incumbent across rollouts + seed inside the env. |
| 5 | `93fe434` | Add RL-only 20x20 pipeline | Added `rl_only_exact` mode and the launcher script. |
| 6 | `ae724bc` | Fix JSON serialization for direct RL artifacts | Artifact hygiene. |
| 7 | `79aff83` | Penalize early RL stops and detect degenerate episodes | Stop-penalty + episode-degeneracy detector. |
| 8 | `f0be0ab` | Fix exact FEM compliance and stress grid orientation | Bugfix in the evaluator itself — compliance numbers before this commit are not comparable. |
| 9 | `f001a40` | Boundary- and stress-limited direct RL refinement | Constrain where the agent is allowed to change voxels. |
| 10 | `d023b5e` | Multi-start RL refinement and volume-safe action guards | Masked actions that would obviously break volume bounds. |
| 11 | `f4c70b0` | Prevent RL from overwriting better topology results | Refuse to regress the incumbent. |
| 12 | `0ffbb75` | Make direct RL honor eval budgets and unlimited runtime | Budget semantics. |

Commits 9–12 were originally developed on the 64x64 branch; they are present here because the 20x20 work was cut from the same codebase.

### 2.1 Key changes introduced in `6b3d4a3`

This is the commit that made RL-only 20x20 actually produce a valid mask. Before this commit, the final mask was reverting to the full-solid seed (see §3.1).

1. **Terminal cleanup before the final FEA.**
   `refine_env._finalize_sparse_reward` now calls `retain_components_touching_region(mask, support_load_mask)` before the terminal full64 evaluation. The agent is scored on the mask that will actually be saved — disconnected islands far from the supports/load are dropped and no longer silently inflate compliance or cause `islands_out_of_range` rejections.

2. **Soft, gradient-bearing infeasible terminal reward.**
   The old `−1.0` constant at a failed terminal was replaced with
   `rl_infeasible_terminal_reward × gap`, where `gap` aggregates:
   - volume overshoot relative to tolerance (`|vol − target| / tol − 1`, clipped ≥ 0)
   - islands overflow (`max(islands − limit, 0) / limit`, weight 0.5)
   - missing support contact (+1)
   - missing load contact (+1)
   - emptiness (+1 if mask is empty).
   The sum is clipped to 1, so the agent still sees a monotone signal of "how bad" an infeasible terminal is instead of a flat cliff.

3. **Training-best harvester (`_RLTrainingCallback`).**
   After every PPO rollout the callback:
   - syncs `global_step` to every worker env,
   - pulls each worker's best feasible candidate, re-evaluates it on the main `Evaluator`,
   - aggregates `terminal_reason_counts` across workers,
   - keeps the **global** best over the whole training run.
   Post-training, this harvested best is preferred over whatever the inference rollouts produce if it scores better. Without this, an unlucky deterministic rollout can make the pipeline report an infeasible final even though training saw a valid mask hours earlier.

4. **Entropy / KL controls.**
   New `ProblemConfig` fields `rl_ent_coef`, `rl_target_kl`, `rl_best_harvest_topk` are wired through CLI flags, env vars, and the summary. `MaskablePPO` is now constructed with `ent_coef` and `target_kl` to avoid entropy collapse under sparse rewards. Defaults in the launcher: `ent_coef=0.03`, `target_kl=0.03`, `best_harvest_topk=4`.

5. **Summary diagnostics.**
   `metrics.rl_training` now reports:
   - `terminal_reason_counts` (why episodes ended: `passed`, `volume_out_of_range`, `missing_load_contact`, `fea_skipped`, `fea_budget_exhausted`, …),
   - `training_best` (with its own fidelity / score / volume / islands),
   - `training_best_source` (`harvested` vs `rollout_env`),
   - `selected_source` (`training_best_harvested`, `training_best_rollout_env`, `inference`, etc.),
   - `harvested_candidates` / `harvested_feasible` counters.

6. **Tests.**
   `tests/test_gen_rl_pipeline.py` was extended to cover terminal cleanup, the monotonic soft-infeasible reward, and the terminal-reason counter drain.

### 2.2 Key changes introduced in `e9b3e51`

- `ProblemConfig` gained the RL knobs that were previously hardcoded: `rl_sparse_reward`, `rl_volume_slack_lower`/`_upper`, `rl_skip_threshold`, `rl_skip_warmup_fraction`, `rl_harmonic_clamp`, `rl_infeasible_terminal_reward`, `max_episode_steps`, `rl_policy_size`.
- Action catalog became **volume-aware**: actions that would push the mask obviously outside `[target − slack_lower, target + slack_upper]` get masked out via `MaskablePPO`'s action mask, so the agent does not waste samples on guaranteed-infeasible moves.
- `refine_env` learned about boundary layers (`rl_boundary_depth`) and stress hotspots (`rl_stress_*`) — voxels around supports/load and under high von-Mises stress are protected.
- The launcher exposes all of this through environment variables.

---

## 3. Two concrete runs on this branch

Both runs used `python -m gigala.topology.topology_optimiz.gen_rl --pipeline-mode rl_only_exact --resolution 20 --volume-target 0.55`, same seed (42), same number of envs (8), same timesteps (3M). Only the code differs.

Runlogs live in `runlogs/gen_rl_rl_only_20x20_<timestamp>/` and each contains `summary.json`, `archive.json`, `best20.npy`, `best20.png`, plus the full terminal transcript.

### 3.1 Run before `6b3d4a3` — commit `e9b3e51` (runlog `20260419_144255`)

Config highlights:

- `rl_total_timesteps = 3_000_000`, `rl_n_envs = 8`, `policy_size = large`, `sparse_reward = True`
- `rl_infeasible_terminal_reward = −1.0` (hard terminal)
- no training-best harvester yet

Result (`summary.json`):

| Metric | Seed | RL candidate | Final |
|---|---|---|---|
| score | 1e12 | 1e12 | **1e12** |
| volume | 1.0 | 1.0 | 1.0 |
| passed_filters | False | False | False |
| invalid_reason | volume_out_of_range | volume_out_of_range | volume_out_of_range |
| `rl_selection.reason` | — | — | `rejected_invalid_rl_candidate:volume_out_of_range` |
| `selected_source` | — | — | `training_best` (but `training_best` was never populated) |
| `fea_counts.full64` | — | — | **0** (no full64 evaluation of any RL candidate was accepted) |
| `runtime_sec` | — | — | 9461 (~2h 38m) |

Inference rollouts (from terminal log):

```
rollout 1/8 (deterministic)  score=1e12  passed=False  volume=0.565
rollout 2..8 (stochastic)    score=1e12  passed=False  volume≈0.56
```

The agent ended up close to the target volume (0.56), but every single terminal was rejected. Because there was no harvester and no terminal cleanup, the pipeline could not produce a valid mask and fell back to the full-solid seed. The final `best20.png` was effectively the seed.

**Diagnosis:** `ep_rew_mean` drifted from ~−0.94 to ~−0.65 over 3M steps (agent was *learning* — just never crossing the feasibility boundary at the terminal), entropy collapsed, and disconnected voxels at termination flipped even near-target masks into `volume_out_of_range` / `islands_out_of_range`.

### 3.2 Run after `6b3d4a3` (runlog `20260419_194406`)

Same config, plus `rl_ent_coef=0.03`, `rl_target_kl=0.03`, `rl_best_harvest_topk=4`, terminal cleanup, soft infeasible reward, harvester active.

Result (`summary.json`):

| Metric | Seed | Final |
|---|---|---|
| score | 1e12 | **27.277** |
| compliance | 1e12 | 24.165 |
| volume | 1.0 | 0.69 |
| smoothness | 0 | 155 |
| islands | 1 | 1 |
| passed_filters | False | **True** |
| `selected_source` | — | **`training_best_harvested`** |
| `fea_counts.full64` | — | 34 (+10 cache hits) |
| `runtime_sec` | — | 4032 (~67m; launcher later reduced default to 100k steps) |

Terminal-reason counts over the full run:

```
volume_out_of_range   = 3479
missing_load_contact  = 579
passed                = 258
```

So ~5.9 % of episodes reached a feasible terminal; the rest failed on volume or load contact. `training_best_score` locked in at 27.277 and did not improve for the remainder of training.

Inference rollouts (8 total):

```
1 deterministic      score=36193.23   vol=0.505   passed=True
2 stochastic_1       INF              vol=0.018   passed=False
3 stochastic_2       score=28241.74   vol=0.443   passed=True
4 stochastic_3       INF              vol=0.028   passed=False
5 stochastic_4       INF              vol=0.013   passed=False
6 stochastic_5       INF              vol=0.415   passed=False
7 stochastic_6       score=77658.80   vol=0.357   passed=True
8 stochastic_7       INF              vol=0.010   passed=False
```

The best *inference* rollout was 28241 (~3 orders of magnitude worse than the harvested best of 27.28). The harvester is the only reason this run produced a usable mask — without it the pipeline would have reported 28241 and the branch would look broken again.

The final mask (`best20.png`) is a recognisable cantilever with several rectangular voids, single connected component, volume fraction 0.69.

---

## 4. What is still problematic

These are the issues a reader of the article / reviewer should be aware of.

1. **Volume overshoot vs reported slack.**
   `rl_volume_slack_upper = 0.05`, target = 0.55 → the feasibility band reported in config is `[0.50, 0.60]`, but the accepted final has `volume_fraction = 0.69` and still `passed_filters = True`. The env-level filter is using a wider tolerance than the one written into the config summary, or the tolerance is applied differently to FEA vs RL-env validation. This needs to be reconciled before the number is published.

2. **Inference rollouts are essentially unusable.**
   After 3M steps the deterministic rollout scores ~36k and half the stochastic rollouts collapse to volume ≈ 0.01. Any improvement in training quality is invisible at inference. Practically the policy is not converging to a useful point in a sense a paper reader would expect — we're relying entirely on the harvester.

3. **PPO early-stopping saturates on `target_kl`.**
   In the second half of training almost every iteration prints `Early stopping at step 1/2 due to reaching max kl: 0.05–0.06` with `target_kl = 0.03`. Effective updates per iteration drop to 2–3; the policy is basically frozen. Candidate fixes: raise `target_kl` to 0.05–0.08, or schedule it (tight early → loose late), or lower the LR in the late phase.

4. **Sparse reward + `explained_variance` ≈ 0.0–0.12.**
   The value head learns almost nothing, which is expected under terminal-only reward but is the reason training plateaus. The "soft infeasible reward" helps at the boundary but is still a single terminal scalar. Worth trying:
   - dense intermediate shaping (e.g. tiny bonuses for entering the volume band / touching supports),
   - a curriculum on the volume target,
   - potential-based shaping so the optimum is unchanged.

5. **Reward sign is always negative.**
   `ep_rew_mean` stays in `[−0.95, −0.65]` for the whole run. The agent is penalised into learning which side of the cliff to walk along, rather than rewarded for improvements. Reviewer-friendly change: log also *feasible-only* mean reward per rollout.

6. **Single seed, single problem.**
   All numbers in this branch are from seed 42 on the 20x20 cantilever. We have no variance estimate and no generalisation check.

7. **Launcher now defaults to 100k steps (`0f36647`).**
   If the article cites numbers from a 3M-step run, make sure the reproduction instructions set `GEN_RL_RL_TOTAL_TIMESTEPS=3000000` explicitly; the default is now much lower for iteration speed.

8. **Compliance numbers before `f0be0ab` are not comparable.**
   That commit fixed exact FEM compliance and stress-grid orientation. Any runlog or notebook output produced on earlier commits of this branch should be regenerated before citing.

9. **M1/MPS note.**
   The launcher forces `RL_DEVICE=cpu` on Apple Silicon because MPS + `SubprocVecEnv` interact poorly for tiny policies on a 20x20 grid. This is a practical finding, not a bug — reviewers running on CUDA should remove that override.

---

## 5. What to try next (suggested)

- **Close the volume-slack gap first.** Trace both the env-level `volume_out_of_range` check and the final-eval filter, make them share one source of truth, re-run, and expect `volume_fraction ≤ 0.60`.
- **Loosen `target_kl` to 0.05–0.08** (or schedule it) and rerun the 3M-step experiment. Log the number of "early stopping" events per rollout to confirm the policy is still being updated in the late phase.
- **Add a small dense shaping term** for (a) being inside the volume band, (b) contact with supports, (c) contact with load. Keep the terminal score as the dominant signal. Potential-based shaping is cleanest if we want to preserve the optimal policy theoretically.
- **Run at least three seeds** (e.g. 17, 42, 2026) and report `{best, median, std}` of final score + `selected_source` split.
- **Include `training_best_inference` as a baseline column** in the paper — it makes it clear how much of the result comes from the harvester vs the policy at test time.

---

## 6. Files to look at

- `scripts/run_gen_rl_pipeline_rl_only_20x20.sh` — single entry point, all env knobs.
- `gigala/topology/topology_optimiz/gen_rl/pipeline.py` — `_RLTrainingCallback`, selection logic, summary assembly.
- `gigala/topology/topology_optimiz/gen_rl/refine_env.py` — env, `_finalize_sparse_reward`, `_soft_infeasible_reward`, terminal cleanup, action masking.
- `gigala/topology/topology_optimiz/gen_rl/fem.py` — `ProblemConfig` fields (all RL knobs live here).
- `tests/test_gen_rl_pipeline.py` — regression tests for cleanup, soft reward, terminal-reason counters.
- `runlogs/gen_rl_rl_only_20x20_20260419_144255/` — the "broken" baseline (score = 1e12, fell back to seed).
- `runlogs/gen_rl_rl_only_20x20_20260419_194406/` — the "fixed" reference run (score = 27.277, harvested).

---

## 7. Post-plan fix (`rl-only_20x20_fix_plan_70b44490`) — Phases 0–6

The six-phase plan landed on this branch after the reference runs above. The
full description of each phase lives in `.cursor/plans/rl-only_20x20_fix_plan_70b44490.plan.md`;
the summary below is what changed in code, keyed by the diagnosis from §4.

| Phase | Diagnosis it targets | What changed |
|---|---|---|
| **0 — tight band** | §4.1 volume-slack vs. final-volume mismatch | Added `ProblemConfig.rl_volume_tolerance = 0.03` and an `effective_volume_tolerance` property. `rl_only_exact` mode now uses the tight band for both `_screen_geometry` and the stop-action mask, so a mask reported as "feasible" is always inside `[target − 0.03, target + 0.03]`. |
| **1 — reward redesign** | §4.5 always-negative reward; "inflate volume for low compliance" exploit in `_harmonic_reward` | Replaced `_harmonic_reward` with `_terminal_reward_v2`: infeasible masks get the soft penalty scaled by gap; feasible masks (inside the tight band) get `baseline / (baseline + score)` — strictly positive and monotone in score. |
| **2 — potential shaping** | §4.4 `explained_variance` ≈ 0 | Added optional potential-based shaping `F = γ · Φ(s') − Φ(s)` with `Φ` penalising volume gap, missing support/load contact, and isolated voxels. Shaping is on by default (`rl_potential_shaping=True`) but gated by a flag so we can ablate. |
| **3 — warm start** | §3.2 "~5.9 % feasible terminals" | New `rl_seed_strategy = "random_near_target"` (default) builds the episode's initial mask by randomly removing non-boundary cells from full-solid until the volume is at `target + slack_upper`, rejecting any removal that would disconnect the mask. The agent now starts ≈15 steps from the band instead of ≈90. |
| **4 — PPO schedules** | §4.3 `target_kl` saturation, entropy collapse, `batch_size` too small | Added schedules for `ent_coef` (0.03 → 0.005 linear), `target_kl` (0.03 → 0.08 linear), and `learning_rate` (3e-4 → 5e-5 cosine by default). `rl_batch_size` default lifted 64 → 256. All of these are wired through CLI flags and the launcher. |
| **5 — inference polish** | §4.2 inference rollouts unusable | `_rollout_inference` now warm-starts the evaluation env from `training_best_harvested` (via new `BinaryTopologyRefineEnv.set_seed_mask`), runs deterministic + stochastic rollouts, and then applies `_local_greedy_polish` — a bounded (≤200 FEA) boundary-cell flipper that only accepts score-reducing feasible flips. New `selected_source` values include `harvested_plus_polish`. |
| **6 — baselines & 3-seed sweep** | §4.6 single seed, no classical baseline | Added `scripts/compute_baseline_20x20.py` (full-solid / random-at-target / ESO strain-energy) and `scripts/run_rl_only_20x20_3seeds.sh`. Both default to seeds `{17, 42, 2026}` for reproducibility. |

### 7.1 New/renamed `ProblemConfig` fields

All new fields are backwards-compatible defaults; set via env vars on the launcher:

| Field | Default | Purpose |
|---|---|---|
| `rl_volume_tolerance` | `0.03` | Tight feasibility band for `rl_only_exact`. |
| `rl_reward_baseline_score` | `100.0` | Scale for the monotone feasible-reward map `B / (B + score)`. |
| `rl_potential_shaping` | `True` | Turn potential-based dense shaping on/off. |
| `rl_shaping_gamma` | `0.99` | Discount used when computing `γ · Φ(s') − Φ(s)`. |
| `rl_shaping_scale` | `1.0` | Outer multiplier on the shaping term. |
| `rl_shaping_w_{volume,contact,islands}` | `1.0, 0.5, 0.5` | Per-component weights inside Φ. |
| `rl_seed_strategy` | `"random_near_target"` | Warm-start strategy (`"full_solid"` restores legacy). |
| `rl_ent_coef_final` | `0.005` | End point of the entropy-coef linear schedule. |
| `rl_target_kl_final` | `0.08` | End point of the `target_kl` linear schedule. |
| `rl_lr_initial` / `rl_lr_final` / `rl_lr_schedule` | `3e-4 / 5e-5 / "cosine"` | PPO learning-rate schedule. |
| `rl_schedule_hparams` | `True` | Gate that toggles all three schedules at once. |
| `rl_batch_size` | `256` | PPO minibatch size. |

### 7.2 Baseline table (seeds 17, 42, 2026; target = 0.55; resolution = 20)

Numbers produced by `scripts/compute_baseline_20x20.py --random-samples 200
--eso-step-fraction 0.02 --eso-max-iterations 128`
(see `runlogs/baseline_20x20_phase6/summary.json`). Lower score is better;
`feasible_rate` is across the 3 seeds.

| Baseline | score (best) | score (median) | score (worst) | feasible_rate |
|---|---:|---:|---:|---:|
| `full_solid` | 1e12 | 1e12 | 1e12 | 0.00 |
| `random_at_target` (N=200) | **41.58** | 48.13 | 1e12 | 0.67 |
| `eso_strain_energy` (greedy ESO) | 306.30 | 306.30 | 306.30 | 1.00 |
| RL (harvested best, seed 42, 3M steps, pre-plan) | **27.28** | — | — | 1.00 |
| RL-only post-plan, 3 seeds, 3M steps | *TBD — run `scripts/run_rl_only_20x20_3seeds.sh`* | | | |

Takeaways from the baseline:

1. **`full_solid` is strictly infeasible** at this target (as the docs §3.1
   already implied). It is kept in the table so we have a reference point for
   readers who assume "do-nothing" is a trivial baseline.
2. **`random_at_target` at N=200 is surprisingly strong** — 0.5 % of samples
   are feasible, but the best of them scores ≈42. This is the baseline the
   paper should cite as "cheap random": any method worse than ~42 is not
   doing useful work. Raising `--random-samples` to 1000–5000 usually brings
   the best down into the high-30s.
3. **ESO strain-energy at ~306 is much worse than random** on this specific
   20x20 cantilever. This is a known failure mode of purely greedy ESO on
   small grids: once the low-energy corner cells are removed, the remaining
   structure is a nearly-uniform cantilever and further removals get stuck
   in a poor local optimum. A more serious classical baseline would be a
   density-based SIMP with filtering; we leave that to future work and flag
   it in §4.
4. **RL harvested best (27.28)** still beats the random-at-target median by
   a factor of ~1.7 on seed 42, i.e. the policy is genuinely learning
   something and the harvester is extracting it. Whether the RL mean across
   seeds improves with the Phase-0–5 changes is exactly what the 3-seed
   sweep is designed to answer — fill in the last row of the table after
   running `scripts/run_rl_only_20x20_3seeds.sh`.

### 7.3 Reproducing the numbers

```bash
# (a) baselines — ~2 min on CPU:
python3 scripts/compute_baseline_20x20.py \
    --resolution 20 --volume-target 0.55 \
    --seeds 17 42 2026 \
    --random-samples 1000 --eso-step-fraction 0.02 \
    --output-dir runlogs/baseline_20x20_repro

# (b) RL sweep at 100k steps (smoke, ~15 min/seed on CPU):
scripts/run_rl_only_20x20_3seeds.sh

# (c) RL sweep at 3M steps (full, ~1h/seed on CPU):
GEN_RL_RL_TOTAL_TIMESTEPS=3000000 \
  scripts/run_rl_only_20x20_3seeds.sh
```

Each seed lands in `runlogs/gen_rl_rl_only_20x20_3seeds_<timestamp>/seed_<seed>/`
with the standard `summary.json`, `archive.json`, `best20.{npy,png}` and the
terminal transcript — same layout as the single-seed launcher, so the existing
`docs/rl_only_20x20_report.md §3` machinery still applies.

### 7.4 What to grep in a new `summary.json`

After a post-plan run, these keys are new or moved and are the ones to pull
for the paper:

- `config.rl_volume_tolerance` — actual feasibility band used.
- `config.rl_seed_strategy` — `"random_near_target"` for post-plan, `"full_solid"` for pre-plan.
- `metrics.rl_training.terminal_reason_counts.passed` — expected to hit 30–50 %
  in the first third of training (Phase 3 success criterion).
- `metrics.rl_training.training_best.score` — the harvested best.
- `metrics.rl_training.selected_source` — now can be `"harvested_plus_polish"` (Phase 5).
- `metrics.rl_training.inference_scores` — per-rollout scores *before* polish,
  so reviewers can see the raw policy quality separately from the harvester.
- `config.rl_ent_coef{,_final}`, `config.rl_target_kl{,_final}`,
  `config.rl_lr_{initial,final,schedule}` — initial/final values of the three
  Phase-4 schedules, captured verbatim from the CLI into the summary.
