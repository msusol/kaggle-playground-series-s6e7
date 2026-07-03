# TODO

## Phase 0 - Setup
- [x] ~/.kaggle/kaggle.json present
- [x] `zsh scripts/download_data.sh` exits 0 (halts + prints a rules-acceptance URL if
      the competition rules haven't been accepted yet — accept them and re-run)
- [ ] Trivial submission scores at the floor (sanity)

## Phase 1 - Baseline (v0.1)
- [x] EDA (`notebooks/v0.1-eda.ipynb` — see findings summary + implementation-plan.md)
- [x] Baseline model + 5-fold CV (`notebooks/v0.1-baseline.ipynb` — OOF balanced
      accuracy 0.9389 +/- 0.0012, recorded in leaderboard.md)
- [x] First offline submission; record in leaderboard.md (submitted 2026-07-02,
      public LB 0.94051 vs. OOF 0.9389 — good CV<->LB correlation)

## Phase 2 - Feature engineering (v0.2) — done, negative result
- [x] Training-budget ablation on v0.1 feature set (more rounds / lower LR) — OOF
      0.9290, worse than v0.1's 0.9389
- [x] Root-cause `sleep_duration`'s #2 feature importance (binned/interaction view) —
      genuine non-monotonic signal + strong `stress_level` interaction, confirmed
- [x] Engineered features: missingness indicators, categorical interactions, OOF
      smoothed target encoding — OOF 0.9255, worse than both v0.1 and the budget ablation
- [x] Retrain 5-fold stratified CV; compare OOF vs. v0.1 (0.9389) — v0.1 remains best
- [x] Feature importance check on engineered features (`sleepbin_x_stress` dominant,
      confirms the interaction is informative even though the overall model regressed)
- [x] Candidate submission.csv written (Section A config, budget-only — best of the
      two v0.2 candidates, still worse than v0.1)

## Phase 3+ - Stronger models
- [ ] CatBoost bake-off (native categorical + NaN handling — may be less prone to
      the overfitting seen in v0.2's engineered-feature model)
- [ ] Rung 3: per-class threshold tuning on OOF predictions (argmax isn't necessarily
      balanced-accuracy-optimal under imbalance)
- [ ] Rung 4: ensemble / squeeze (see implementation-plan.md)
