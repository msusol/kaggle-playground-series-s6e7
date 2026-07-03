# Leaderboard

Update after **every** completed run + validation pass. OOF = out-of-fold CV.

| Version | Model | Key change | OOF | Kaggle LB | Notes |
|---|---|---|---|---|---|
| floor | - | all-majority-class (`at-risk`) | 0.333 (analytic) | _tbd_ | must-beat |
| v0.1 | LightGBM multiclass | class_weight='balanced', native categoricals w/ NaN-as-level, 5-fold stratified | 0.9389 (+/- 0.0012) | 0.94051 | `notebook/v0.1-baseline.ipynb`; per-class recall at-risk 0.956 / fit 0.929 / unhealthy 0.932 |

## Run log

### 2026-07-02 — v0.1 baseline (LightGBM)
- Config: `lgb.LGBMClassifier(objective='multiclass', class_weight='balanced', n_estimators=2000, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8)`, early stopping on `multi_logloss` (all 5 folds ran the full 2000 rounds — best_iteration never triggered early stop, worth revisiting: either more rounds or a lower LR would probably still help).
- OOF balanced accuracy: 0.9389 (+/- 0.0012 across folds) — very stable.
- Feature importance (gain): `stress_level` (4.59M) >> `sleep_duration` (4.10M) > `physical_activity_level` (1.31M) >> rest. `sleep_duration` ranking #2 was not obvious from the EDA's univariate histograms — likely a nonlinear/threshold or interaction effect.
- Takeaway: class weighting + native categorical/NaN-as-level handling works well out of the box; imbalance is not the blocker it could have been. Next lever is probably early-stopping/LR tuning (since no fold early-stopped) and Rung 2 feature engineering around `stress_level` x `sleep_duration` x `physical_activity_level` interactions.
- Submitted to Kaggle 2026-07-02 (submission id 54284663): **public LB 0.94051** vs.
  OOF 0.9389 — LB slightly *higher* than CV, good CV<->LB correlation, no sign of
  overfitting to the training folds.
