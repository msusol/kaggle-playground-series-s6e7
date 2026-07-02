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

## Rung 0 - Pipeline skeleton
- Download data; emit an all-majority-class (`at-risk`) submission; confirm it scores
  ~0.333 balanced accuracy (the floor).

## Rung 1 - Cheap baseline
- LightGBM/XGBoost multiclass on raw features, categoricals as native categorical
  dtype, numeric NaNs left as-is (tree splits handle missing natively).
- `class_weight='balanced'` (or per-class weights matching inverse frequency).
- Validate the 5-fold stratified CV harness + submission format end-to-end.

## Rung 2 - Stronger model
- Feature engineering: missingness indicators per column (is the NaN itself
  informative — e.g. from a skipped survey question), simple interactions
  (activity level x exercise duration, sleep quality x sleep duration), target
  encoding for categoricals with smoothing.
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
