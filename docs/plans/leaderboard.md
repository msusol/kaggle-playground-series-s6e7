# Leaderboard

Update after **every** completed run + validation pass. OOF = out-of-fold CV.

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
