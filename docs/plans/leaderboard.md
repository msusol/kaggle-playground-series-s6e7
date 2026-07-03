# Leaderboard

Update after **every** completed run + validation pass. OOF = out-of-fold CV.

| Version | Model | Key change | OOF | Kaggle LB | Notes |
|---|---|---|---|---|---|
| floor | - | all-majority-class (`at-risk`) | 0.333 (analytic) | _tbd_ | must-beat |
| v0.1 | LightGBM multiclass | class_weight='balanced', native categoricals w/ NaN-as-level, 5-fold stratified | **0.9389 (+/- 0.0012)** | **0.94051** | `notebooks/v0.1-baseline.ipynb`; **current best** — per-class recall at-risk 0.956 / fit 0.929 / unhealthy 0.932 |
| v0.2-A | LightGBM multiclass | same v0.1 features, n_estimators=5000/lr=0.03 (budget-only ablation) | 0.9290 | 0.93155 | `notebooks/v0.2-feature-engineering.ipynb`; **worse than v0.1** — more rounds/lower LR hurt |
| v0.2-D | LightGBM multiclass | v0.2-A budget + missingness indicators, categorical interactions, OOF target encoding | 0.9255 | not submitted | `notebooks/v0.2-feature-engineering.ipynb`; **worse than both v0.1 and v0.2-A** |

## Run log

### 2026-07-02 — v0.1 baseline (LightGBM)
- Config: `lgb.LGBMClassifier(objective='multiclass', class_weight='balanced', n_estimators=2000, learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8)`, early stopping on `multi_logloss` (all 5 folds ran the full 2000 rounds — best_iteration never triggered early stop, worth revisiting: either more rounds or a lower LR would probably still help).
- OOF balanced accuracy: 0.9389 (+/- 0.0012 across folds) — very stable.
- Feature importance (gain): `stress_level` (4.59M) >> `sleep_duration` (4.10M) > `physical_activity_level` (1.31M) >> rest. `sleep_duration` ranking #2 was not obvious from the EDA's univariate histograms — likely a nonlinear/threshold or interaction effect.
- Takeaway: class weighting + native categorical/NaN-as-level handling works well out of the box; imbalance is not the blocker it could have been. Next lever is probably early-stopping/LR tuning (since no fold early-stopped) and Rung 2 feature engineering around `stress_level` x `sleep_duration` x `physical_activity_level` interactions.
- Submitted to Kaggle 2026-07-02 (submission id 54284663): **public LB 0.94051** vs.
  OOF 0.9389 — LB slightly *higher* than CV, good CV<->LB correlation, no sign of
  overfitting to the training folds.

### 2026-07-02 — v0.2 feature engineering (Rung 2) — negative result
- **Section A (training-budget ablation)**: same v0.1 features, `n_estimators=5000,
  learning_rate=0.03` (vs. v0.1's `2000`/`0.05`), patience 100. Result: **OOF 0.9290**,
  *worse* than v0.1's 0.9389. 4/5 folds ran the full 5000 rounds without early
  stopping. Takeaway: v0.1's original config was closer to optimal for *balanced
  accuracy* specifically than the "no fold early-stopped" observation suggested —
  early stopping was tracking `multi_logloss`, not the competition metric, so pushing
  training further with a lower LR drifted the decision boundary away from balanced
  per-class recall even as logloss kept improving.
- **Section B (sleep_duration root-cause)**: confirmed a genuine, strong non-monotonic
  interaction. Short sleep (<6h) -> ~60% at-risk / ~37% unhealthy (vs. ~8% baseline).
  Mid-range sleep (6-8.5h) -> ~90-99% at-risk, near-zero unhealthy/fit. Longer sleep
  -> `fit` share up to ~11-13%. Interaction with `stress_level`: at `stress_level=low`,
  at-risk share drops from ~99% (short sleep) to ~57-69% (longer sleep) — low stress +
  adequate sleep strongly predicts `fit`. At `stress_level=high`, at-risk stays ~99.5%
  regardless of sleep except at very short sleep, where it collapses toward
  `unhealthy`. This fully explains `sleep_duration`'s v0.1 #2 importance ranking.
- **Section D (engineered features)**: missingness indicators, `stress_level` x
  `physical_activity_level` cross, `sleep_quality` x `smoking_alcohol` cross,
  `sleep_duration`-decile x `stress_level` cross, OOF smoothed multiclass target
  encoding (4 columns x 3 classes = 12 encoded features), same budget as Section A.
  Result: **OOF 0.9255** — worse than *both* v0.1 and Section A. Per-class recall:
  at-risk 0.966 (up from v0.1's 0.956), but `fit` 0.902 (down from 0.929) and
  `unhealthy` 0.908 (down from 0.932) — the engineered features traded minority-class
  recall for majority-class recall, net negative for balanced accuracy.
- Feature importance on the Section D model: `sleepbin_x_stress` (the engineered
  cross) became the single dominant feature (9.0M gain, more than double v0.1's raw
  `stress_level` at 4.59M), and raw `stress_level` collapsed to near-zero importance
  (99.7) — its signal got fully absorbed into the cross/target-encoded features. The
  interaction feature IS informative (confirms Section B), but the overall model still
  regressed, most likely from added variance/overfitting risk with target encoding and
  35 features vs. v0.1's 13 (Section D's per-fold `best_iteration` varied widely —
  4661 to 5000 — vs. Section A's near-uniform ~5000, suggesting less stable training).
- **Neither Section A nor Section D beat v0.1 — v0.1 remains the best model.**
  Submitted the Section A candidate anyway (submission id 54287439) to confirm
  CV<->LB correlation holds directionally even for a regression: **public LB
  0.93155** vs. OOF 0.9290 — closely tracks CV, and confirms it's worse than v0.1's
  0.94051 as expected. CV<->LB correlation remains trustworthy in both directions.
