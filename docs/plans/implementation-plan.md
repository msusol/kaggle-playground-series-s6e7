# Implementation Plan - Strategy Ladder

Tabular, 3-class classification (`at-risk` / `unhealthy` / `fit`) scored on **balanced
accuracy**. Each rung yields a valid `submission.csv` + a CV score, so a working
pipeline exists before adding model complexity. See `docs/plans/competition-overview.md`
for full data/metric detail.

Two things dominate this ladder more than usual for a Playground tabular competition:

- **Class imbalance** (85.9% / 8.4% / 5.8%) — balanced accuracy punishes ignoring the
  minority classes, so `class_weight='balanced'` / `scale_pos_weight`-style handling
  (or resampling) is not optional, it's the main lever.
- **Missingness across nearly every column** (1-12% per feature, both numeric and
  categorical) — imputation strategy and native-NaN-aware model choice matter more
  than exotic feature engineering.

**EDA update (`notebook/v0.1-eda.ipynb`):** `stress_level` and `physical_activity_level`
are unusually strong, near-deterministic signals (e.g. `stress_level == medium` -> 99.4%
`at-risk`; `physical_activity_level == active` -> 17.2% `fit` vs. ~0.2-0.3% for
`moderate`/`sedentary`). `sleep_quality` and `smoking_alcohol` are secondary signals;
`diet_type` and `gender` show almost no class separation. Missingness matches train/test
almost exactly (no leakage signal). Full findings in the notebook's summary cell.

## Rung 0 - Pipeline skeleton
- Download data; emit an all-majority-class (`at-risk`) submission; confirm it scores
  ~0.333 balanced accuracy (the floor).

## Rung 1 - Cheap baseline — DONE (`notebook/v0.1-baseline.ipynb`)
- LightGBM multiclass on raw features, categoricals as native categorical
  dtype (NaN as its own explicit level — do not mode-impute `stress_level` /
  `physical_activity_level`, see EDA note above), numeric NaNs left as-is (tree
  splits handle missing natively).
- `class_weight='balanced'`.
- 5-fold stratified CV harness + submission format validated end-to-end.
- **Result: OOF balanced accuracy 0.9389 (+/- 0.0012)**. Per-class recall: at-risk
  0.956, fit 0.929, unhealthy 0.932 — class weighting is working, no minority-class
  collapse. Full run log in `leaderboard.md`.
- Feature importance confirmed `stress_level` (top) and `physical_activity_level`
  (top-3) as predicted. **Unexpected**: `sleep_duration` (numeric) is the #2 signal —
  not obvious from univariate EDA histograms, likely a nonlinear/threshold or
  interaction effect. `diet_type`/`gender` confirmed lowest-importance.
- **Not yet submitted to Kaggle** — `data/submission.csv` is written and validated,
  pending an explicit submit to get the first LB score.
- **Open question carried to Rung 2**: none of the 5 folds triggered early stopping
  (all hit the `n_estimators=2000` cap) — worth trying more rounds / a lower learning
  rate before adding feature engineering, to know how much headroom is left in the
  current feature set alone.

## Rung 2 - Stronger model
- First, cheaply check whether Rung 1 was under-trained: more `n_estimators` /
  lower learning rate, since no fold early-stopped at 2000 rounds.
- Feature engineering: missingness indicators per column (is the NaN itself
  informative — e.g. from a skipped survey question), simple interactions
  (activity level x exercise duration, sleep quality x sleep duration,
  `stress_level` x `sleep_duration` x `physical_activity_level` given how much
  signal those three carry individually), target encoding for categoricals with
  smoothing.
- Compare LightGBM vs. CatBoost (CatBoost has strong native categorical + NaN
  handling, worth a direct bake-off here).

## Rung 3 - Contender
- Threshold/decision-rule tuning per class on OOF predictions to directly optimize
  balanced accuracy (argmax of raw probabilities does not necessarily maximize it
  under imbalance) — e.g. searching per-class decision thresholds or a
  cost-sensitive argmax.
- _The model that wins; see versioned plan vX.Y._

## Rung 4 - Squeeze
- Ensemble LightGBM + CatBoost + a regularized linear/NN baseline (blend or stack).
- Pseudo-labeling from high-confidence test predictions if CV/LB stays well-correlated.

## Cross-validation
- 5-fold **stratified** (by target) given the imbalance; trust CV->LB correlation.
- Track balanced accuracy per class (not just the aggregate) in `leaderboard.md`
  takeaways so minority-class regressions are visible early.

## Submission
- Not a Code Competition — no runtime cap or internet-disabled requirement.
- Generate `submission.csv` locally or in a Kaggle notebook and submit via
  `kaggle competitions submit -c playground-series-s6e7 -f submission.csv -m "<msg>"`
  or the web UI. Max 10 submissions/day, 2 selected as Final Submissions.
