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

**EDA update (`notebooks/v0.1-eda.ipynb`):** `stress_level` and `physical_activity_level`
are unusually strong, near-deterministic signals (e.g. `stress_level == medium` -> 99.4%
`at-risk`; `physical_activity_level == active` -> 17.2% `fit` vs. ~0.2-0.3% for
`moderate`/`sedentary`). `sleep_quality` and `smoking_alcohol` are secondary signals;
`diet_type` and `gender` show almost no class separation. Missingness matches train/test
almost exactly (no leakage signal). Full findings in the notebook's summary cell.

## Rung 0 - Pipeline skeleton
- Download data; emit an all-majority-class (`at-risk`) submission; confirm it scores
  ~0.333 balanced accuracy (the floor).

## Rung 1 - Cheap baseline — DONE (`notebooks/v0.1-baseline.ipynb`)
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
- Submitted to Kaggle 2026-07-02 (submission id 54284663): **public LB 0.94051** —
  still the current best model.

## Rung 2 - Stronger model — DONE, negative result (`notebooks/v0.2-feature-engineering.ipynb`)
- **Training-budget ablation** (`n_estimators=5000, learning_rate=0.03` vs. v0.1's
  `2000`/`0.05`, same features): **OOF 0.9290 — worse than v0.1's 0.9389.** The "no
  fold early-stopped" observation from Rung 1 was misleading: early stopping tracks
  `multi_logloss`, not balanced accuracy, so more rounds at a lower LR kept improving
  logloss while drifting the decision boundary away from balanced per-class recall.
  **Takeaway: don't chase training budget further on this feature set** — v0.1's
  config was closer to a real optimum for the actual competition metric.
- **Root-caused `sleep_duration`'s importance** (binned/quantile + interaction view):
  genuine non-monotonic signal — short sleep (<6h) associates with ~37% `unhealthy`
  (vs. ~8% baseline); mid-range sleep (6-8.5h) is ~90-99% `at-risk`; longer sleep
  raises `fit` share to ~11-13%. Strong interaction with `stress_level`: at
  `stress_level=low`, at-risk share drops sharply as sleep increases (favoring `fit`);
  at `stress_level=high`, at-risk stays ~99.5% regardless of sleep except at very
  short sleep, where it collapses toward `unhealthy`. Fully explains the v0.1 feature
  importance ranking.
- **Engineered features** (missingness indicators, `stress_level` x
  `physical_activity_level` and `sleep_quality` x `smoking_alcohol` crosses,
  `sleep_duration`-decile x `stress_level` cross, OOF smoothed multiclass target
  encoding): **OOF 0.9255 — worse than both v0.1 and the budget-only ablation.**
  Per-class recall traded minority-class accuracy for majority-class accuracy
  (`fit` 0.929 -> 0.902, `unhealthy` 0.932 -> 0.908, `at-risk` 0.956 -> 0.966) — net
  negative for balanced accuracy. The `sleepbin_x_stress` interaction feature itself
  *is* highly informative (became the single top feature by gain, 9.0M, absorbing
  nearly all of raw `stress_level`'s prior importance) — the interaction hypothesis
  was correct — but the larger feature set (35 vs. 13) added more variance/overfitting
  risk than it removed, net negative.
- **v0.1 remains the best model going into Rung 3.** Full run log and per-fold detail
  in `leaderboard.md` and `docs/investigate/notebook-runs.md`.
- **Carried lessons**: (1) tune/validate directly against the competition metric, not
  a training-loss proxy, before assuming more training helps; (2) an individually
  informative engineered feature does not guarantee a better model once it's added to
  a larger feature set — validate the net CV effect, not just the feature's own
  importance ranking; (3) CatBoost bake-off (native categorical + NaN handling, maybe
  less prone to the overfitting seen here) is still untried and worth a shot before
  concluding the feature set is saturated.

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
