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

## Rung 4 - Squeeze — DONE, negative result, cleanly explained (`notebooks/v0.5-ensemble.ipynb`)
- Blended 4 members, each reproducing an exact validated config: LightGBM (v0.1),
  CatBoost-V1 (v0.3 base features), CatBoost-V2 (v0.3 engineered features), and a new
  regularized logistic regression (genuine architectural diversity, one-hot +
  median-impute + standardize preprocessing). All reproductions PASS exact match.
  LogisticRegression solo: 0.8994 (notably weaker, as expected for a linear model).
- **4-way blend weight search (simplex grid) + nested validation**: full-OOF best
  blend degenerates to 100% weight on CatBoost-V1 alone (0.9493, zero improvement
  even on the optimistic same-data fit). **Nested-validated honest improvement:
  -0.0002** — no real gain. Every subset blend that includes CatBoost-V1 caps out at
  exactly its solo score; no combination beats it. No new submission written.
- **Why**: directly confirms the Rung 3 prediction. Per discussion 717222, the
  competition data is a noised synthesis of a near-deterministic depth-4 rule on the
  same 3 features every one of our models already keys on — there are no
  complementary errors left for an ensemble to correct. The architecturally-distinct
  logistic regression adding nothing is stronger evidence for a synthesis-noise
  ceiling than Rung 3 alone (rules out "wrong model family" as the gap's cause).
- **v0.3 (either variant) remains the best model.** Two independent experiments
  (Rung 3 threshold tuning, Rung 4 ensembling) both point at the same ceiling —
  further squeeze attempts should be weighed against this before investing more
  effort chasing marginal gains on this dataset.

## Rung 5 - External research — DONE, confirms the ceiling with a larger sample
- Investigated Kaggle discussion 718258 (Masaya Kawamata): a CV-vs-Public-LB
  comparison across **13 independently-trained model families** (XGBoost variants,
  RepLeafGBM, 3 factorization-machine flavors, and 4 tabular/neural architectures —
  GRN, GANDALF, LNN, ResNet), all landing in the same ~0.946-0.951 band with CV/LB
  gaps within ±0.0011. A much larger, more diverse confirmation of the
  synthesis-noise ceiling than our own Rung 3/4 experiments alone.
- Pulled and read the actual notebooks behind two headline scores in that
  comparison (rather than trusting the summary table): `georgymamarin`'s
  prior-correction writeup and `masayakawamata`'s `XGB_OvR` notebook. Both
  independently confirm the double-correction pitfall (stacking training-time
  class weighting with a separate post-hoc decision-rule correction hurts) already
  found in Rung 3 — now a well-corroborated, cross-notebook finding, not an
  isolated result of our own.
- Full details in `docs/investigate/2026-07-03-kaggle-discussion-findings.md`.

## Rung 6 - XGBoost one-vs-rest — DONE, flat result, corroborated by external research (`notebooks/v0.6-xgboost-ovr.ipynb`)
- Surfaced by Rung 5's research: `XGB_OvR` scored highest of the 13 model families
  (CV 0.95036). Built our own version — 3 independent binary XGBoost classifiers
  (one per class, `scale_pos_weight` for imbalance, native categorical handling),
  combined via argmax, blended with reproduced CatBoost-V1 (base features) and
  CatBoost-V2 (engineered features, added after a scoping gap was caught: XGB-OvR
  uses engineered features, so CatBoost-V1 alone was an apples-to-oranges peg).
- **Solo XGB-OvR: 0.9493 — an exact tie with CatBoost-V1.** Full-OOF 3-way blend
  best: 0.9495 (well within noise). **Nested-validated honest improvement: +0.0001**
  — the first-ever positive nested blend result in this project, but far below the
  0.0005 submit threshold. No candidate submission written.
- Submitted the best pairwise blend (`xgb_ovr+catboost_v1`, 0.9495 same-data OOF)
  anyway out of curiosity, despite not clearing the threshold: **public LB
  0.94937** — the highest LB score in this project, but explicitly treated as
  statistically tied with v0.3 (same conclusion as v0.3-V1 vs. V2), not a confirmed
  new best model, given it sits inside the same noise band as every other
  Rung 3-6 result.
- **Why the flat result is expected, not a shortfall**: pulled the actual notebook
  behind Rung 5's `XGB_OvR` headline score. Its author ran a ~20-arm ablation
  campaign and concluded **"the OvR decomposition itself — a no-op vs. the
  multiclass flagship"** — the 0.95036 score comes from the same prior-correction
  decision rule applied across their whole notebook series, not from OvR
  structure. The same campaign also found per-class `scale_pos_weight` (which our
  own XGB-OvR used) actively harmful when stacked with a separate correction — a
  third independent confirmation of Rung 3's double-correction pitfall.
- **v0.3 (either variant) remains the best model with a stable, credible OOF
  behind it.** Four independent squeeze attempts (Rung 3, Rung 4, Rung 5, Rung 6)
  now all point at the same synthesis-noise ceiling.

## Rung 7 - HistGradientBoosting + exact-value target encoding — DONE, POSITIVE result, new best model (`notebooks/v0.7-hgbc-te.ipynb`)
- Surfaced by investigating `redamountassir/ps-s6e7-hgbc-baseline-lb-0-95034-cv-0-95026`
  ("TE-HGBC"). Two new ingredients: exact-value target encoding of the 7 numeric
  features (cast to string, target-encoded via sklearn's native
  `TargetEncoder(cv=5, target_type='multiclass')` — not just categoricals, unlike
  every prior rung's feature engineering), and `HistGradientBoostingClassifier`
  (sklearn's native GBM, a 4th distinct tree-boosting implementation) with native
  `class_weight='balanced'`. Reused the source notebook's tuned hyperparameters;
  our own 5-fold split; no post-hoc correction (plain argmax). Run on Kaggle's
  own compute.
- **HGBC-TE solo OOF: 0.9502 — beats CatBoost-V1 (0.9493) by +0.0009, the first
  genuine non-noise-level improvement across the entire squeeze phase** (Rungs
  3-6 were all within ±0.0005 of CatBoost-V1). CatBoost-V1 reproduction PASS
  (exact match) confirmed the comparison is trustworthy.
- **Blend check**: nested-validated blend with CatBoost-V1 adds only +0.0002 over
  HGBC-TE solo — below threshold, not worth the added complexity. The decision
  logic correctly submitted the **solo** HGBC-TE predictions.
- **Submitted to Kaggle: public LB 0.95036** vs. OOF 0.9502 — tight correlation,
  no haircut. **New best LB in this project**, beating the previous best
  (v0.6's curiosity submission, 0.94937 — a noise-level result) by +0.00099, and
  unlike that submission this one clears our own honest-improvement threshold.
- **This revises the "synthesis-noise ceiling" conclusion from Rungs 3-6**: the
  ceiling wasn't really at ~0.949 — a sufficiently different feature
  representation (exact-value numeric target encoding, not binned/qcut'd) found
  real additional signal that class-weighting, ensembling, one-vs-rest
  decomposition, and threshold tuning had all missed. The lesson generalizes:
  when several different *decision-rule* and *model-structure* levers all plateau
  at the same score, that's evidence the *feature representation* is the binding
  constraint, not necessarily the data's intrinsic noise floor.
- **v0.7 (HGBC-TE) is now the best model — v0.3 CatBoost no longer holds that
  spot.** Worth revisiting whether applying the same exact-value target encoding
  to CatBoost (rather than only pairing it with a new model family) pushes
  further, as a follow-up.

## Cross-validation
- 5-fold **stratified** (by target) given the imbalance; trust CV->LB correlation.
- Track balanced accuracy per class (not just the aggregate) in `leaderboard.md`
  takeaways so minority-class regressions are visible early.

## Submission
- Not a Code Competition — no runtime cap or internet-disabled requirement.
- Generate `submission.csv` locally or in a Kaggle notebook and submit via
  `kaggle competitions submit -c playground-series-s6e7 -f submission.csv -m "<msg>"`
  or the web UI. Max 10 submissions/day, 2 selected as Final Submissions.
