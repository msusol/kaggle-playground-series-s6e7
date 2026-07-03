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

## Phase 3 - CatBoost bake-off (v0.3) — done, POSITIVE result, new best model
- [x] Notebook built (`notebooks/v0.3-catboost-bakeoff.ipynb`): Variant 1 (base
      features) + Variant 2 (v0.2 engineered features), same 5-fold split as
      v0.1/v0.2, custom `balanced_accuracy_score`-based eval metric for early
      stopping (directly addresses the v0.2 lesson that LightGBM's early stopping
      tracked `multi_logloss`, not the competition metric)
- [x] Smoke-tested full pipeline on a data sample before the full run
- [x] Full run (5-fold, both variants), live in JupyterLab
- [x] Compare vs. v0.1 (0.9389) and v0.2 (0.9290 / 0.9255); recorded in leaderboard.md
      — **Variant 1 OOF 0.9493 (new best), Variant 2 OOF 0.9491 (tied)**
- [x] Candidate submission.csv written from Variant 1 (best model overall)
- [x] Submitted both variants to Kaggle: Variant 1 LB 0.94885, Variant 2 LB
      **0.94913** (higher despite lower OOF) — confirms the two are genuinely tied
      on both CV and LB, either is a defensible Final Submission pick

## Phase 4 - Threshold tuning (v0.4) — done, negative result, cleanly explained
- [x] Notebook built (`notebooks/v0.4-threshold-tuning.ipynb`): reproduce v0.3
      Variant 2 (engineered features, since it scored higher on LB) capturing OOF
      probabilities this time; weighted-argmax grid search over per-class weights;
      nested validation (fit weights on 4/5 folds, evaluate on the held-out 5th) to
      get an honest improvement estimate rather than trusting the same-data fit
- [x] Smoke-tested full pipeline on a data sample before the full run
- [x] Full run, live in JupyterLab — reproduction PASS (exact match vs. v0.3 Variant 2)
- [x] Recorded nested-validated result in leaderboard.md: **honest improvement
      -0.0001 — no real gain**, full-OOF grid search found plain argmax (w=1,1)
      already optimal
- [x] No new submission.csv (nested validation correctly found no real improvement)
- [x] Investigated Kaggle discussion threads 717018/717222 for context — explains
      the negative result (stacking class-weighting + post-hoc correction
      double-corrects) and suggests ~0.95 may be near the practical ceiling for
      this dataset (noised synthesis of a near-deterministic depth-4 rule)

## Phase 5+ - Stronger models
- [ ] Rung 4: ensemble / squeeze (see implementation-plan.md) — worth tempering
      expectations given the likely synthesis-noise ceiling at ~0.95
- [ ] Consider: does the depth-4 rule from discussion 717222 suggest any concrete
      feature-engineering angle we haven't tried, or is ~0.949 genuinely close to
      the ceiling regardless of approach?
