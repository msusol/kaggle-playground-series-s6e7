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

## Rung 2.5 - CatBoost bake-off — DONE, positive result (`notebooks/v0.3-catboost-bakeoff.ipynb`)
- Directly acted on the two Rung 2 lessons above: implemented a custom Python
  `balanced_accuracy_score`-based CatBoost eval metric (instead of tracking
  `multi_logloss`), and tested whether CatBoost absorbs v0.2's engineered feature set
  without the regression LightGBM showed.
- **Variant 1 (CatBoost, v0.1's exact 13 base features)**: `auto_class_weights='Balanced'`,
  custom balanced-accuracy eval metric, early stopping. **best_iterations: [428, 950,
  605, 339, 779]** — fired well before the 3000-round cap in every fold, confirming
  the hypothesis that tracking the real metric fixes the "never early-stops" pattern.
  **OOF balanced accuracy: 0.9493 — beats v0.1's 0.9389 by +0.0104. New best model.**
  Per-class recall: at-risk 0.933, fit 0.950, unhealthy 0.965 — more evenly balanced
  across classes than v0.1's 0.956 / 0.929 / 0.932.
- **Variant 2 (CatBoost, v0.2's full 35-feature engineered set)**: same config, OOF
  **0.9491** — essentially tied with Variant 1 (-0.0002). Unlike LightGBM's Section D
  (which regressed from 0.9389 to 0.9255 on the identical feature set), CatBoost
  handled the larger, more collinear feature set without a net loss — supports the
  ordered-boosting hypothesis. The engineered features are genuinely used (feature
  importance: `te_stress_x_activity_k1` #2, `sleepbin_x_stress` present) but don't
  add value over the simpler base-feature model here.
- **v0.3 (CatBoost) is now the best model — v0.1 no longer holds that spot.** Both
  variants submitted to Kaggle: Variant 1 scored LB 0.94885, Variant 2 scored LB
  **0.94913** (higher, despite slightly lower OOF) — a +0.00028 flip confirming the
  two variants are genuinely statistically tied on both CV and LB, not just CV.
  Either is a defensible choice for a Final Submission slot.
- **Carried lesson confirmed**: the "no early stopping" pattern from Rung 1 was a
  genuine training-setup issue, not a feature-set limitation — fixing the eval metric
  (not adding features or training budget) is what actually moved the needle.

## Rung 3 - Contender — DONE, negative result, cleanly explained (`notebooks/v0.4-threshold-tuning.ipynb`)
- Weighted-argmax (`predict = argmax(proba * w)`) grid search on v0.3 Variant 2's
  reproduced OOF probabilities (reproduction verified exact vs. v0.3: OOF 0.9491,
  identical `best_iterations`). **Full-OOF grid search found w=(1.0, 1.0) —
  i.e. plain argmax was already optimal, zero improvement.**
- **Nested validation** (fit weights on 4/5 folds, evaluate on the held-out 5th,
  cycle across folds): honest improvement estimate **-0.0001** — within noise, no
  real gain. No new submission written.
- **Why, per Kaggle discussion 717018 (Georgy Mamarin)**: stacking training-time
  class-weighting with a separate post-hoc threshold/prior correction is a known
  pitfall — it double-corrects and can actively hurt (he measured 0.9047 vs. ~0.950
  for either alone). Our models were trained with `auto_class_weights='Balanced'`,
  so the balance correction was already spent during training; there was nothing
  left for a post-hoc adjustment to capture. Confirms our own prediction ("may be
  less headroom here than originally expected") with a concrete mechanism, not just
  an empirical shrug.
- **External context (same discussion + 717222, broccoli beef)**: the competition
  data is likely a noised synthesis of a near-deterministic depth-4 decision rule on
  `sleep_duration`/`stress_level`/`physical_activity_level` (100% accuracy on the
  original pre-synthesis dataset). Multiple independent competitor pipelines
  (different model families) converge to the same ~0.948-0.950 OOF / ~0.9498 LB
  range regardless of approach — consistent with a synthesis-noise ceiling rather
  than an easy gain being left unclaimed. Full details in
  `docs/investigate/notebook-runs.md`.
- **v0.3 (either variant, still tied) remains the best model going into Rung 4.**

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
