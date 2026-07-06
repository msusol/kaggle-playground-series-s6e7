# Leaderboard

Update after **every** completed run + validation pass. OOF = out-of-fold CV.

**All 7 notebooks (v0.1-v0.7) are published publicly on Kaggle** (2026-07-04), each
executing cleanly end-to-end and reproducing the OOF/LB numbers below exactly. A
[discussion thread](https://www.kaggle.com/competitions/playground-series-s6e7/discussion/719199)
summarizing the full research trail links all 7 and the GitHub repo. Public kernels:
[v0.1 EDA](https://www.kaggle.com/code/gdataranger/s6e7-v0-1-eda) /
[v0.1 LightGBM baseline](https://www.kaggle.com/code/gdataranger/s6e7-v0-1-lightgbm-baseline) ·
[v0.2 Feature engineering](https://www.kaggle.com/code/gdataranger/s6e7-feature-engineering-notebook-v2) ·
[v0.3 CatBoost bake-off](https://www.kaggle.com/code/gdataranger/s6e7-v0-3-catboost-bake-off) ·
[v0.4 Threshold tuning](https://www.kaggle.com/code/gdataranger/s6e7-v0-4-threshold-tuning) ·
[v0.5 Ensemble](https://www.kaggle.com/code/gdataranger/s6e7-ensemble-rung-4) ·
[v0.6 XGBoost One-vs-Rest](https://www.kaggle.com/code/gdataranger/s6e7-xgboost-one-vs-rest-rung-6) ·
[v0.7 HistGradientBoosting + target encoding](https://www.kaggle.com/code/gdataranger/s6e7-v0-7-histgradientboosting-target-encoding).
The public republish runs of v0.2/v0.5/v0.6 were not resubmitted to the competition
(they reproduce already-recorded scores below, not new results) — no OOF/LB numbers
below changed as a result of this publish.

| Version | Model | Key change | OOF | Kaggle LB | Notes |
|---|---|---|---|---|---|
| floor | - | all-majority-class (`at-risk`) | 0.333 (analytic) | **0.33333** | must-beat; confirmed 2026-07-03 (submission 54310528) — exact match to the analytic floor |
| v0.1 | LightGBM multiclass | class_weight='balanced', native categoricals w/ NaN-as-level, 5-fold stratified | 0.9389 (+/- 0.0012) | 0.94051 | `notebooks/v0.1-baseline.ipynb`; per-class recall at-risk 0.956 / fit 0.929 / unhealthy 0.932 |
| v0.2-A | LightGBM multiclass | same v0.1 features, n_estimators=5000/lr=0.03 (budget-only ablation) | 0.9290 | 0.93155 | `notebooks/v0.2-feature-engineering.ipynb`; **worse than v0.1** — more rounds/lower LR hurt |
| v0.2-D | LightGBM multiclass | v0.2-A budget + missingness indicators, categorical interactions, OOF target encoding | 0.9255 | not submitted | `notebooks/v0.2-feature-engineering.ipynb`; **worse than both v0.1 and v0.2-A** |
| v0.3-V1 | CatBoost multiclass | v0.1's exact 13 base features, `auto_class_weights='Balanced'`, custom balanced-accuracy eval metric for early stopping | 0.9493 | 0.94885 | `notebooks/v0.3-catboost-bakeoff.ipynb`; per-class recall at-risk 0.933 / fit 0.950 / unhealthy 0.965 |
| v0.3-V2 | CatBoost multiclass | v0.3-V1 config + v0.2's 35-feature engineered set | 0.9491 | **0.94913** | `notebooks/v0.3-catboost-bakeoff.ipynb`; **statistically tied with v0.3-V1** — slightly lower OOF but slightly *higher* LB, confirming the two are within noise of each other either way |
| v0.4 | CatBoost (v0.3-V2) + weighted-argmax | reproduces v0.3-V2, tunes per-class argmax weights | 0.9491 (nested: -0.0001 vs. plain argmax) | not submitted | `notebooks/v0.4-threshold-tuning.ipynb`; **negative result** — plain argmax already optimal, double-correction pitfall (see run log) |
| v0.5 | 4-way blend: LightGBM + CatBoost-V1 + CatBoost-V2 + LogReg | reproduces v0.1/v0.3's exact configs + new LogReg baseline, nested-validated blend weight search | 0.9493 (nested blend: -0.0002 vs. solo CatBoost-V1) | not submitted | `notebooks/v0.5-ensemble.ipynb`; **negative result** — best blend degenerates to 100% CatBoost-V1, no diversity gain from any member |
| v0.6 | XGBoost one-vs-rest (engineered feats) + CatBoost-V1/V2 | 3 binary XGB classifiers, `scale_pos_weight`, combined via argmax; blended with reproduced CatBoost-V1/V2 | 0.9493 solo (nested 3-way blend: +0.0001 vs. solo) | **0.94937** (2-way `xgb_ovr+catboost_v1` blend, weights 0.46/0.54) | `notebooks/v0.6-xgboost-ovr.ipynb`; **flat/negative per our own threshold** (honest improvement below 0.0005), but the curiosity-submission is the **highest LB score in this project so far** — see run log for the important caveat |
| v0.7 | HistGradientBoostingClassifier + exact-value target encoding | all 13 raw features (incl. numerics at exact value) target-encoded via sklearn's native `TargetEncoder(cv=5)`; `class_weight='balanced'`, native categorical handling | **0.9502** (nested vs. solo CatBoost-V1: +0.0009) | **0.95036** | `notebooks/v0.7-hgbc-te.ipynb`; **POSITIVE result — new best model**, first genuine non-noise-level improvement in the squeeze phase (all of Rung 3-6 were within ±0.0005 of CatBoost-V1). Blend with CatBoost-V1 adds only +0.0002 more (not worth the complexity); HGBC-TE solo submitted directly |
| v0.8 | RealMLP (neural net) + multi-resolution numeric target encoding | from-scratch PyTorch port of `yunsuxiaozi/pss6e7-realmlp-cv-0-95063`; periodic numeric embeddings, NTK-parametrized linears, 16-way ensemble-in-one-model, EMA; our own 5-fold split, single training-time class-weight correction only (no post-hoc reweighting); run locally (Apple M3 Pro, PyTorch MPS) | **0.95062 solo** (nested vs. current best: +0.0001, below threshold) | **0.95048** (curiosity submission, submission 54376269) | `notebooks/v0.8-realmlp.ipynb`; **flat result by our own strict threshold, but the highest raw solo OOF of any model in this project** — essentially a statistical tie with v0.7 (+0.0004 raw, nested-validated honest margin only +0.0001, short of the 0.0005 bar). First non-tree-boosting model family tried; blend with CatBoost-V1 only reaches 82/18 weighting (not degenerate to one member like v0.5), suggesting some real diversity, but not enough to clear threshold on its own. LB narrowly edges v0.7's LB (0.95036) by +0.00012 — within the same noise band documented throughout this project, not treated as a confirmed new best (same caveat as v0.6's curiosity submission) |

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

### 2026-07-03 — v0.3 CatBoost bake-off — POSITIVE result, new best model
- Directly acted on two Rung 2 lessons: (1) implemented a custom Python
  `balanced_accuracy_score`-based CatBoost eval metric so early stopping tracks the
  actual competition metric, not a `multi_logloss` proxy; (2) tested whether
  CatBoost's ordered boosting + native categorical handling absorbs v0.2's engineered
  feature set without the variance/regression LightGBM showed.
- **Variant 1 (CatBoost, v0.1's exact 13 base features)**: `auto_class_weights='Balanced'`,
  `iterations=3000, learning_rate=0.05, depth=6`, early stopping (patience 100) on the
  custom balanced-accuracy metric. **best_iterations: [428, 950, 605, 339, 779]** —
  early stopping fired well before the 3000-round cap in every fold (unlike v0.1/v0.2-A,
  which never did under the `multi_logloss` proxy), confirming the hypothesis.
  **OOF balanced accuracy: 0.9493 — beats v0.1's 0.9389 by +0.0104, the best model so
  far.** Per-class recall: at-risk 0.933, fit 0.950, unhealthy 0.965 (vs. v0.1's
  0.956 / 0.929 / 0.932) — CatBoost balances the classes more evenly, exactly what
  the metric rewards, trading some at-risk recall for large minority-class gains.
- **Variant 2 (CatBoost, v0.2's full 35-feature engineered set)**: same config.
  **best_iterations: [544, 765, 542, 354, 628]**. **OOF balanced accuracy: 0.9491** —
  essentially tied with Variant 1 (-0.0002, within fold-to-fold noise). Feature
  importance shows the engineered features (`te_stress_x_activity_k1` #2 at 17.45,
  `sleepbin_x_stress` at 5.32) are genuinely used, but — unlike LightGBM's Section D,
  which regressed from 0.9389 to 0.9255 with the same feature set — CatBoost handled
  the larger, more collinear feature set without a net loss. Supports the ordered
  boosting hypothesis from the plan doc.
- **Variant 1 auto-selected as best overall** (0.9493 > Variant 2's 0.9491 > all prior
  runs); `data/submission.csv` written from it.
- Submitted to Kaggle 2026-07-03 (submission id 54301243): **public LB 0.94885** vs.
  OOF 0.9493 — closely tracks CV, and beats v0.1's 0.94051 by +0.00834. Confirms the
  CatBoost improvement holds on the leaderboard, not just in CV.
- **v0.1 is no longer the best model — v0.3 CatBoost (base or engineered features,
  they're statistically tied) is.**
- Also submitted Variant 2 (submission id 54301348, `data/submission_v2.csv`) to
  check whether the tiny CV gap (Variant 1 +0.0002) held on the LB: it did not — LB
  actually favored Variant 2 (**0.94913** vs. Variant 1's 0.94885, a +0.00028 flip in
  the other direction). Confirms the two variants are genuinely statistically tied on
  both CV and LB; the engineered feature set neither helps nor hurts CatBoost in any
  reliable way here. Either candidate is a reasonable Final Submission pick.

### 2026-07-03 — v0.4 threshold tuning (Rung 3) — negative result, cleanly explained
- Reproduced v0.3 Variant 2 (engineered features) capturing OOF probabilities this
  time (v0.3's harness only stored hard labels). **Reproduction PASS: exact match**
  (`best_iterations [544, 765, 542, 354, 628]`, OOF 0.9491, identical to v0.3).
- Weighted-argmax grid search (`predict = argmax(proba * w)`, log-scale grid over
  `w[fit]`/`w[unhealthy]`): **full-OOF best weights were w=(1.0, 1.0) — i.e. plain
  argmax was already optimal, zero improvement found** even on the same-data fit
  that's normally mildly optimistic.
- **Nested validation** (fit weights on 4/5 folds, evaluate on the held-out 5th,
  cycle across all folds): nested plain-argmax 0.9491 (+/- 0.0011), nested
  tuned-argmax 0.9490 (+/- 0.0011). **Honest improvement estimate: -0.0001** — within
  noise, no real gain. No new `submission.csv` written (correctly, per the notebook's
  own decision threshold of 0.0005).
- **Why**: per Kaggle discussion thread 717018 (Georgy Mamarin, see
  `docs/investigate/notebook-runs.md`), stacking training-time class-weighting with a
  *separate* post-hoc threshold/prior correction is a known pitfall — the second
  correction double-corrects probabilities the first one already shifted, and can
  actively hurt (he measured a drop to 0.9047 from ~0.950 for either alone). Our
  CatBoost models were trained with `auto_class_weights='Balanced'`, so the balance
  correction was already "spent" during training — there was nothing left for a
  post-hoc weighted argmax to capture. This cleanly explains the flat result.
- **v0.3 (either variant, still statistically tied) remains the best model.** Per the
  same discussion thread, other competitors' independent pipelines land in the same
  ~0.948-0.950 OOF / ~0.9498 LB range regardless of model family — likely because the
  competition data is a noised synthesis of an underlying near-deterministic depth-4
  decision rule (see `docs/investigate/notebook-runs.md`), meaning ~0.95 may be close
  to the practical ceiling here, not a sign of being left on the table.

### 2026-07-03 — v0.5 ensemble (Rung 4) — negative result, cleanly explained
- Four members, each reproducing an exact validated config (no re-tuning): LightGBM
  (v0.1's config, base features), CatBoost-V1 (v0.3's config, base features),
  CatBoost-V2 (v0.3's config, v0.2's 35-feature engineered set), and a new regularized
  logistic regression (base features, one-hot + median-impute + standardize
  preprocessing — genuine architectural diversity from the tree models).
- **All four reproductions PASS** (exact match to known results): LightGBM 0.9389,
  CatBoost-V1 0.9493 (`best_iterations [428, 950, 605, 339, 779]`, identical to v0.3),
  CatBoost-V2 0.9491 (`best_iterations [544, 765, 542, 354, 628]`, identical to v0.3).
  LogisticRegression (new, no prior baseline): **0.8994** — notably weaker, as
  expected for a linear model on this data, with the same recall pattern as other
  models (at-risk 0.816 lower, fit 0.952 / unhealthy 0.930 higher — class-weighting
  works even for the weak learner).
- **4-way blend weight search (simplex grid, step 0.1, 286 combinations)**: full-OOF
  best = **0.9493, found at weights {lgbm: 0, catboost_v1: 1.0, catboost_v2: 0,
  logreg: 0}** — i.e. even the optimistic same-data search degenerates to using
  CatBoost-V1 alone; zero improvement found even before nested validation.
- **Nested validation** (fit weights on 4/5 folds, evaluate on the held-out 5th):
  nested solo CatBoost-V1 0.9493 (+/- 0.0011), nested 4-way blend 0.9492 (+/- 0.0011).
  **Honest improvement estimate: -0.0002** — no real gain. No new `submission.csv`
  written (correctly, per the notebook's 0.0005 decision threshold).
- **Subset blend comparison** (all pairs + triples): every combination that includes
  CatBoost-V1 caps out at exactly 0.9493 (matching CatBoost-V1 solo); no combination
  beats it. `lgbm+logreg` (the two members *without* CatBoost-V1) tops out at just
  0.9442 — notably worse, confirming CatBoost-V1 alone already captures what the
  other members can offer.
- **Why**: directly confirms the Rung 3 finding's prediction. Per discussion thread
  717222 (`docs/investigate/2026-07-03-kaggle-discussion-findings.md`), the
  competition data is a noised synthesis of a near-deterministic depth-4 rule over
  `sleep_duration`/`stress_level`/`physical_activity_level` — all four of our models
  (including the architecturally-distinct logistic regression) already capture
  essentially all the recoverable signal via these same features, so there are no
  complementary errors left for an ensemble to correct. This is stronger evidence for
  the synthesis-noise-ceiling hypothesis than Rung 3 alone: even a linear model with a
  fundamentally different decision boundary shape adds nothing, which argues against
  "wrong model family" as an explanation for the ~0.949-0.951 plateau.
- **v0.3 (either variant) remains the best model at OOF ~0.949 / LB ~0.949.** Given
  two independent Rung 3/4 experiments both cleanly point at a synthesis-noise
  ceiling rather than a modeling gap, further squeeze attempts (Rung 5+) should be
  weighed against this — see `docs/investigate/2026-07-03-kaggle-discussion-findings.md`'s
  follow-ups for the standing recommendation on this.

### 2026-07-03/04 — v0.6 XGBoost one-vs-rest (Phase 6) — flat result, but highest LB submission
- Surfaced by Kaggle discussion 718258 (Masaya Kawamata): across 13 model
  families, `XGB_OvR` scored highest (CV 0.95036 / LB 0.95040). Built our own
  version: 3 independent binary XGBoost classifiers (one per class,
  `scale_pos_weight` for imbalance, native categorical via `enable_categorical`),
  combined via argmax, on the 35-feature engineered set.
- Required adding `xgboost` to the project for the first time — needed a one-time
  macOS/MacPorts environment fix (`libomp` via MacPorts + an `install_name_tool`
  rpath patch, since the pip wheel expects Homebrew); documented in
  `docs/process/xgboost-macos-setup.md`.
- **First pass (2 members: XGB-OvR + CatBoost-V1)**: XGB-OvR solo **0.9493**, an
  exact tie with CatBoost-V1. 2-way blend nested-validated honest improvement:
  **+0.0001** — first-ever positive nested blend result in this project, though
  below the 0.0005 submit threshold.
- **User caught a scoping gap**: XGB-OvR uses engineered features (35, matching
  v0.2/v0.3-V2) but the only comparison peg was CatBoost-V1 (base features, 13) —
  an apples-to-oranges blend. Extended to 3 members, adding CatBoost-V2 (same
  engineered features as XGB-OvR) for a cleaner comparison.
- **Full 3-member result**: solo scores xgb_ovr 0.9493, catboost_v1 0.9493,
  catboost_v2 0.9491. Full-OOF 3-way blend best: 0.9494 at weights
  (xgb_ovr=0.55/0.4, catboost_v1=0.2/0.6, catboost_v2=0.25/0.0, varies by fold).
  **Nested 3-way honest improvement: +0.0001** — same flat/negative-per-threshold
  result as the 2-member pass. No submission written by the notebook's own logic.
- **Pairwise breakdown** (same-data): `xgb_ovr+catboost_v1` 0.9495 (weights
  0.46/0.54) was the single best pairwise combination — better than the full 3-way
  blend or either other pair. `catboost_v1+catboost_v2` capped at 0.9493 (matching
  solo), consistent with v0.5's finding that the two CatBoost variants are too
  correlated to help each other.
- **Curiosity submission**: user asked to submit the `xgb_ovr+catboost_v1` blend
  (0.9495 same-data OOF) purely to see the actual LB number, despite it not
  clearing our own honest-improvement threshold. Built directly from the live
  kernel's in-memory test-set probabilities (via `jupyter_client`, no notebook
  re-run needed) and submitted: **public LB 0.94937** (submission 54313483) — the
  **highest LB score in this project so far**, edging out v0.3-V2's 0.94913 by
  +0.00024 and v0.3-V1's 0.94885 by +0.00052.
- **Important caveat, not a new best-model claim**: this LB delta (+0.0002 to
  +0.0005 over prior submissions) sits squarely inside the same noise band this
  whole investigation has repeatedly characterized as unreliable/non-actionable
  (Rung 3's -0.0001, Rung 4's -0.0002, this notebook's own nested estimate of only
  +0.0001). A single LB submission at this margin is not distinguishable from
  public-slice noise (per discussion 718258's own analysis: minority-class LB
  noise is on the order of ±0.001-0.002, since the public slice only has ~3.4k
  `fit` + ~5.0k `unhealthy` rows). Treat this as "the two are statistically tied,
  same as v0.3-V1 vs. V2," not as "the blend is confirmed better."
- **Investigated the actual notebook behind discussion 718258's top-scoring row**
  (`masayakawamata/s6e7-xgb-ovr-cv-0-95036`, pulled via `kaggle kernels pull`) —
  the author's own ~20-arm ablation campaign concludes **"the OvR decomposition
  itself — a no-op vs. the multiclass flagship"** and that per-class
  `scale_pos_weight` (which our own XGB-OvR used) is actively harmful when
  stacked with a separate decision-rule correction — a third independent
  confirmation of the double-correction pitfall first seen via Georgy Mamarin's
  notebook and our own v0.4. Full details in
  `docs/investigate/2026-07-03-kaggle-discussion-findings.md`. This means our own
  flat result is corroborated by, not contradicted by, the more rigorous
  literature — the "0.95036, highest of 13" framing from the summary table was
  misleading in isolation.
- **v0.3 (either variant) remains the best model with a stable, credible
  0.9493-0.9491 OOF backing it.** The v0.6 blend's marginally higher LB number
  should not be treated as a confirmed improvement given everything above.

### 2026-07-04 — v0.7 HistGradientBoosting + exact-value target encoding (Rung 7) — POSITIVE result, new best model
- Surfaced by investigating `redamountassir/ps-s6e7-hgbc-baseline-lb-0-95034-cv-0-95026`
  ("TE-HGBC") — see `docs/investigate/2026-07-03-kaggle-discussion-findings.md`.
  Two genuinely new ingredients tested: exact-value target encoding of the 7
  numeric features (not just categoricals, cast to string and target-encoded via
  sklearn's native `TargetEncoder(cv=5, target_type='multiclass')`), and
  `HistGradientBoostingClassifier` (sklearn's native GBM, a 4th distinct
  tree-boosting implementation in this project) with native `class_weight='balanced'`.
  Reused the source notebook's tuned hyperparameters as-is; used our own
  established 5-fold split; no post-hoc prior/threshold correction (plain argmax).
- Run on Kaggle's own compute (not local), per user request — required a one-time
  environment setup (`xgboost` was added to the project in v0.6; this run needed
  no new dependency, `HistGradientBoostingClassifier`/`TargetEncoder` are both
  core sklearn). Took ~86 minutes total on Kaggle's shared CPU (HGBC-TE itself
  finished in ~4.5 min; the CatBoost-V1 reproduction peg took the bulk of the
  time, consistent with earlier CatBoost runs on Kaggle in this project).
- **HGBC-TE solo OOF: 0.9502** — matches the source notebook's own reported CV
  (0.95026) closely despite a different fold split, and **beats CatBoost-V1
  (0.9493) by +0.0009 — the first genuine, non-noise-level improvement in the
  entire squeeze phase** (Rung 3-6 were all within ±0.0005 of CatBoost-V1, i.e.
  indistinguishable from noise). Per-class recall: at-risk 0.9373, fit 0.9500,
  unhealthy 0.9633 (vs. CatBoost-V1's at-risk 0.933 / fit 0.950 / unhealthy 0.965
  — very similar profile, slightly better balanced across all three classes).
- **CatBoost-V1 reproduction PASS** (exact match, `best_iterations [428, 950,
  605, 339, 779]`, OOF 0.9493) — confirms the OOF probability matrix used for the
  blend check is trustworthy.
- **Blend check**: full-OOF 2-way grid search best 0.9504 at weights
  (hgbc_te=0.78, catboost_v1=0.22) — only +0.0002 over HGBC-TE solo. **Nested
  validation**: nested solo (hgbc_te) 0.9502 (+/- 0.0012), nested blend 0.9503
  (+/- 0.0011) — honest improvement **+0.0002**, below the 0.0005 threshold for
  the *blend* specifically. But HGBC-TE's *solo* score independently clears
  `current best + 0.0005` on its own merits, so the notebook's decision logic
  correctly submitted the **solo** HGBC-TE predictions, not the blend (the small
  blend gain doesn't justify the added complexity of maintaining two models).
- **Submitted to Kaggle** (submission 54321699): **public LB 0.95036** vs. OOF
  0.9502 — tight correlation, no haircut, consistent with this project's
  established CV-LB trustworthiness. **New best LB score, beating the previous
  best (v0.6's curiosity submission at 0.94937) by +0.00099** — and unlike that
  submission, this one clears our own honest-improvement threshold, so it's a
  confirmed new best, not a noise-level curiosity.
- **v0.7 (HGBC-TE) is now the best model in this project — v0.3 CatBoost no
  longer holds that spot.** This also meaningfully updates the "synthesis-noise
  ceiling" narrative from Rung 3-6: the ceiling wasn't at ~0.949 after all — a
  sufficiently different feature representation (exact-value numeric target
  encoding) found real additional signal that class-weighting, ensembling, OvR
  decomposition, and threshold tuning had all missed. Worth revisiting whether
  applying the same exact-value target encoding to CatBoost (rather than only to
  a new model family) could push further, as a follow-up.

### 2026-07-05 — v0.8 RealMLP (neural net) — flat result by our own threshold, highest raw OOF yet
- Surfaced by investigating `yunsuxiaozi/pss6e7-realmlp-cv-0-95063` — see
  `docs/investigate/2026-07-05-kaggle-discussion-findings.md`. First
  neural-net model family in this project (all prior work: LightGBM,
  CatBoost, XGBoost, HistGradientBoosting — all tree-boosting). From-scratch
  PyTorch port: periodic numeric embeddings (`PBLDEmbedding`), NTK-parametrized
  linear layers (`NTPLinear`), a 16-way ensemble-in-one-model trick, EMA of
  weights, 5-parameter-group AdamW schedule. Our own `StratifiedKFold(5)`
  (not the source's naive fold split); single training-time class-weight
  correction only (the architecture's own `compute_class_weight('balanced')`
  with a fixed `[0.9, 1.1, 1.0]` tweak) — the source's post-hoc Optuna
  reweighting was **not** reproduced (our own investigation found it added
  only +0.00006, negligible, a 5th confirmation of "tunable second correction
  finds ~nothing extra").
- Run **locally** on Apple M3 Pro (PyTorch MPS backend), not Kaggle. Found and
  fixed a real bug during smoke-testing: pandas 3.0.3's `.astype(str)` now
  produces a native `str` dtype rather than `object`, breaking the
  categorical/numeric column classification (`dtype == object` no longer
  matches) — fixed via `not pd.api.types.is_numeric_dtype(...)`.
- **RealMLP solo OOF: 0.95062** — the **highest raw solo OOF of any model in
  this project**, edging out v0.7 (0.9502) by +0.0004, and closely matching
  the source notebook's own raw CV (0.95057, different fold split).
  CatBoost-V1 reproduction: **PASS** (0.9493, exact match to v0.3 Variant 1).
- **Blend check**: full-OOF grid search best 0.9507 at weights (realmlp=0.82,
  catboost_v1=0.18) — notably NOT degenerate to 100% one model (unlike v0.5's
  all-tree-boosting blend), suggesting some genuine diversity between a
  neural net and a GBDT. **Nested validation**: nested solo (realmlp) 0.9506
  (+/- 0.0012), nested blend 0.9507 (+/- 0.0012) — honest improvement only
  **+0.0001**.
- **Decision: NO REAL IMPROVEMENT, no submission written.** RealMLP solo's
  raw margin over the current best (+0.0004) falls just short of the project's
  0.0005 threshold for "real" (0.95062 vs. `CURRENT_BEST_OOF + 0.0005` =
  0.9507) — essentially a statistical tie with v0.7, not a confirmed new best.
  Nothing was submitted to Kaggle by the notebook's own decision logic.
- **v0.7 (HGBC-TE) remains the best model.** This is the closest any
  alternative has come to displacing it, and the first time a genuinely
  different model family (not just a different tree-boosting library) reached
  parity — worth keeping in mind if future blend attempts revisit RealMLP as a
  diversity source, even though it doesn't clear the bar solo.
- **Curiosity submission (2026-07-05)**: connected to the user's live local
  Jupyter kernel (`jupyter_client.BlockingKernelClient`, via the running
  server's `/api/sessions` to find the right kernel ID) to build a
  submission.csv from the already-computed `test_proba_realmlp` predictions,
  avoiding a wasteful full re-run, then submitted purely to see the actual LB
  number (submission 54376269): **public LB 0.95048** — narrowly edges out
  v0.7's LB (0.95036) by +0.00012, and closely tracks the OOF (0.95062, a
  small -0.00014 haircut, consistent with this project's tight CV-LB
  correlation). Per this project's own discipline, this margin is well within
  the noise band documented extensively throughout (public LB scored on only
  ~20% of test; competing models have swapped rank by ±0.0005 or more between
  CV and LB elsewhere in this project). **Not treated as a confirmed new
  best** — same caveat applied to v0.6's curiosity submission previously.
