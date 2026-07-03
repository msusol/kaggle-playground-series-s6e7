# Notebook Run Log

Tracks run results, errors, and follow-up questions for each notebook under
`notebooks/`, per the project's notebook workflow convention (one `##` section per
version slug). See `docs/plans/leaderboard.md` for the scored-run summary table and
`docs/plans/implementation-plan.md` for how each run's findings feed the strategy
ladder.

## v0.1-eda

### Context

- Notebook: `notebooks/v0.1-eda.ipynb`.
- Purpose: initial EDA on `train.csv`/`test.csv` for Playground Series S6E7
  ("Predicting Student Health Risk") — confirm target imbalance, missingness, and
  feature/class relationships before committing to a Rung 1 modeling approach.
- Run: executed locally via `jupyter nbconvert --execute --inplace` against the
  shared venv kernel (`/Users/marksusol/LosusAI/Projects/Kaggle/.venv`).

### Investigation Checklist

- [x] Confirm target class distribution matches `competition-overview.md`.
- [x] Confirm train/test missingness rates are consistent (no leakage-by-missingness).
- [x] Look at numeric feature distributions split by target class.
- [x] Look at categorical feature distributions split by target class.
- [x] Check numeric feature inter-correlations.
- [x] Check `id` vs. target for an ordering/leakage artifact.

### Findings

- Target: `at-risk` 85.87%, `unhealthy` 8.36%, `fit` 5.77% — confirms the
  analytic floor of ~0.333 balanced accuracy (majority-class-only prediction).
- Missingness matches train/test almost exactly (largest gap 0.0002pp, on
  `stress_level`) — no missingness-pattern leakage signal.
- `stress_level` and `physical_activity_level` are unusually strong, near-deterministic
  signals (e.g. `stress_level == medium` -> 99.4% `at-risk`; `physical_activity_level
  == active` -> 17.2% `fit` vs. ~0.2-0.3% for `moderate`/`sedentary`).
- `sleep_quality` and `smoking_alcohol` are secondary signals; `diet_type` and `gender`
  show almost no class separation in their marginal distributions.
- Numeric features show only mild inter-correlation (`calorie_expenditure` /
  `step_count` / `exercise_duration` at 0.39-0.44, all proxying "activity volume").
- No obvious `id`-ordering artifact.

### Actions Taken

- Installed `ipykernel`, `nbconvert`, `matplotlib` into the shared venv and added them
  to `requirements.txt` so the notebook could execute (not just be authored as JSON).
- Ran the notebook end-to-end via `nbconvert --execute --inplace`.
- Saved 3 plots to `docs/images/` (`v0.1-target-distribution.png`,
  `v0.1-numeric-by-class.png`, `v0.1-numeric-correlation.png`).
- Carried findings into `docs/plans/implementation-plan.md` (Rung 1 categorical/NaN
  handling) and checked off the EDA task in `docs/plans/TODO.md`.

### Resolution

**resolved** — EDA complete; findings validated against the actual data (not assumed)
and used to shape the Rung 1 baseline design.

### Follow-ups

- The EDA's univariate view did not flag `sleep_duration` as a strong signal — the
  v0.1-baseline run below found it to be the #2 feature by importance. Worth a
  binned/quantile or 2D-interaction view of `sleep_duration` in a future EDA pass
  (see v0.1-baseline follow-ups).

## v0.1-baseline

### Context

- Notebook: `notebooks/v0.1-baseline.ipynb`.
- Purpose: Rung 1 of `docs/plans/implementation-plan.md` — a LightGBM multiclass
  baseline with `class_weight='balanced'` and native categorical/NaN handling, to
  establish a real CV/LB reference point beyond the analytic floor.
- Run: executed locally via `nbconvert --execute --inplace` (backgrounded; ~a few
  minutes for 5-fold LightGBM training on 690k rows).

### Investigation Checklist

- [x] Categorical `NaN` encoded as an explicit `"missing"` category (not imputed away),
      with train/test sharing one category list so LightGBM's native category codes
      line up between fit and predict.
- [x] 5-fold **stratified** CV harness runs end-to-end.
- [x] OOF balanced accuracy clears the ~0.333 analytic floor by a wide margin.
- [x] `submission.csv` format validated against `sample_submission.csv` (columns,
      row count, label set, no nulls).
- [x] Feature importance checked against the EDA's `stress_level` /
      `physical_activity_level` hypothesis.
- [x] Submitted to Kaggle; CV vs. LB correlation checked.

### Findings

- **OOF balanced accuracy: 0.9389 (+/- 0.0012)** across 5 folds — stable, well above
  the 0.333 floor.
- Per-class recall: `at-risk` 0.956, `fit` 0.929, `unhealthy` 0.932 — class weighting
  is working; no minority-class collapse.
- **None of the 5 folds triggered early stopping** — all hit the `n_estimators=2000`
  cap without the 50-round-patience early stop firing, meaning validation loss was
  still improving at round 2000. The model may be under-trained at the current
  learning rate/round budget.
- Feature importance (gain): `stress_level` (4.59M) >> `sleep_duration` (4.10M) >
  `physical_activity_level` (1.31M) >> rest. `diet_type` and `gender` confirmed
  lowest-importance, consistent with the EDA.
- **Unexpected**: `sleep_duration` (a numeric feature) ranked #2 by importance — the
  EDA's univariate histograms didn't show obvious class separation for it. Likely a
  nonlinear/threshold effect or an interaction the tree splits pick up that a marginal
  histogram can't surface; not yet root-caused.
- Public LB: **0.94051** vs. OOF 0.9389 — LB slightly *higher* than CV, indicating
  healthy CV<->LB correlation and no overfitting to the training folds.

### Actions Taken

- Built `notebooks/v0.1-baseline.ipynb`: preprocessing (categorical NaN ->
  `"missing"` -> shared-category `pandas.Categorical`), 5-fold `StratifiedKFold`,
  per-fold `lgb.LGBMClassifier(objective='multiclass', class_weight='balanced',
  n_estimators=2000, learning_rate=0.05, num_leaves=63, subsample=0.8,
  colsample_bytree=0.8)` with early stopping (patience 50) on `multi_logloss`.
  averaged the 5 fold models' test-set probabilities (soft-voting) for final
  predictions.
- Executed via `nbconvert --execute --inplace`.
- Wrote and validated `data/submission.csv` (295,753 rows).
- Submitted via `kaggle competitions submit -c playground-series-s6e7 -f
  data/submission.csv -m "v0.1 LightGBM baseline, class_weight=balanced, 5-fold
  stratified CV, OOF balanced_accuracy=0.9389"` (submission id 54284663).
- Recorded the run in `docs/plans/leaderboard.md`, updated
  `docs/plans/implementation-plan.md` (Rung 1 marked done, open questions carried to
  Rung 2) and `docs/plans/TODO.md`.
- Committed (`918dcc3`) and pushed to `origin/main`.

### Resolution

**resolved** — Rung 1 baseline complete and submitted; CV/LB correlation confirmed
healthy, so the CV harness can be trusted for iterating on Rung 2 without needing to
burn a submission per experiment.

### Follow-ups

- **Why did no fold early-stop at `n_estimators=2000`?** Try a higher round budget /
  lower learning rate before adding feature engineering, to know how much headroom is
  left in the current feature set alone — cheaper than jumping straight to Rung 2.
- **Root-cause `sleep_duration`'s importance**: a binned/quantile view of
  `sleep_duration` vs. `health_condition`, or a 2D interaction plot against
  `stress_level`/`physical_activity_level`, would clarify whether this is a genuine
  nonlinear effect worth engineering around or a redundant proxy for one of the two
  dominant categoricals.
- Rung 2 candidates per `implementation-plan.md`: feature engineering around
  `stress_level` x `sleep_duration` x `physical_activity_level` interactions, and a
  LightGBM vs. CatBoost bake-off.

## v0.2-feature-engineering

### Context

- Notebook: `notebooks/v0.2-feature-engineering.ipynb`.
- Purpose: Rung 2 of `docs/plans/implementation-plan.md` — (A) a training-budget
  ablation on v0.1's exact feature set, (B) root-cause `sleep_duration`'s unexpected
  #2 importance, (C) engineered features (missingness indicators, categorical
  interactions, OOF smoothed multiclass target encoding), (D) retrain with the
  engineered features at the tuned budget, compared against v0.1 (OOF 0.9389, LB
  0.94051).
- Run: first attempt via headless `nbconvert --execute --inplace` (background task
  `bbgs5nu0v`) failed on a real bug (see Findings). After the fix, re-run live in
  JupyterLab (user-driven, watched via `tqdm` progress bars added for this purpose)
  so the user could monitor training directly rather than polling a headless process.

### Investigation Checklist

- [x] Reproduce and root-cause the `bbgs5nu0v` failure (`TypeError: '<' not supported
      between instances of 'float' and 'str'`) before re-running the full ~40 min job.
- [x] Verify the fix on a data sample before re-running at full scale (smoke test).
- [x] Section A: training-budget ablation vs. v0.1's exact config.
- [x] Section B: root-cause `sleep_duration` via binned/interaction view.
- [x] Section D: retrain with engineered features at the Section A budget.
- [x] Feature importance check on the engineered-feature model.
- [x] Candidate `submission.csv` written and format-validated.

### Findings

- **Bug (first run, `bbgs5nu0v`)**: `pd.cut(test['sleep_duration'], bins=sleep_bin_edges)`
  produces `NaN` not only for genuinely-missing `sleep_duration` values but also for
  test values falling **outside train's bin range** (test min/max differ from train's).
  The notebook's original NaN-sanitization only checked `test['sleep_duration'].isnull()`,
  missing the out-of-range case. `astype(str)` on the resulting
  `Categorical(Interval)` column does not reliably stringify those NaNs, so
  `make_cross()`'s string concatenation mixed real float `NaN` with `str` values in the
  `sleepbin_x_stress` cross feature, and `sorted(set(train[col].unique()) |
  set(test[col].unique()))` raised `TypeError: '<' not supported between instances of
  'float' and 'str'`. This wasted the full ~37 minutes of Section A compute from the
  first run (headless `nbconvert` does not write partial results to disk on a cell
  error — nothing was recoverable).
- **Fix**: clip test's `sleep_duration` into train's bin range
  (`test['sleep_duration'].clip(lower=sleep_bin_edges[0], upper=sleep_bin_edges[-1])`)
  *before* cutting, so the only remaining `NaN` source is genuine missingness, which
  the existing null-mask sanitization then correctly catches. Verified via a
  standalone reproduction and a full-pipeline smoke test on a data sample before
  committing to the full ~40+ minute re-run.
- **A second clobbering incident**: after fixing the bug, a JupyterLab browser tab
  that had the notebook open *before* the fix (loaded at first server start) had its
  stale in-memory copy autosaved back to disk, silently reverting the file to the
  pre-fix (and pre-tqdm) state. Caught by re-checking the file content for the fix's
  signature strings before trusting it was actually applied. Resolved by having the
  user close and reopen the tab fresh (not "revert to checkpoint", since the
  checkpoint was also the stale version) after the fixes were reapplied.
- **Section A (training-budget ablation)**: `n_estimators=5000, learning_rate=0.03`
  (vs. v0.1's `2000`/`0.05`), same features, patience 100. **OOF 0.9290** — worse
  than v0.1's 0.9389. 4/5 folds ran the full 5000 rounds. Early stopping tracks
  `multi_logloss`, not balanced accuracy — more rounds at a lower LR kept improving
  logloss while the decision boundary drifted away from balanced per-class recall.
- **Section B (sleep_duration root-cause)**: confirmed a genuine non-monotonic signal.
  Short sleep (<6h) -> ~60% at-risk / ~37% unhealthy (vs. ~8% baseline). Mid-range
  sleep (6-8.5h) -> ~90-99% at-risk, near-zero unhealthy/fit. Longer sleep -> `fit`
  share up to ~11-13%. Strong `stress_level` interaction: at `stress_level=low`,
  at-risk share drops from ~99% (short sleep) to ~57-69% (longer sleep) — low stress +
  adequate sleep strongly predicts `fit`. At `stress_level=high`, at-risk stays ~99.5%
  regardless of sleep except at very short sleep, where it collapses toward
  `unhealthy`. Fully explains v0.1's #2 importance ranking for this feature.
- **Section D (engineered features)**: missingness indicators, `stress_level` x
  `physical_activity_level` cross (16 categories), `sleep_quality` x
  `smoking_alcohol` cross (16 categories), `sleep_duration`-decile x `stress_level`
  cross (44 categories), OOF smoothed multiclass target encoding (4 columns x 3
  classes = 12 features), same budget as Section A. **OOF 0.9255** — worse than
  *both* v0.1 and Section A. Per-class recall shifted rather than improved: `at-risk`
  0.956 -> 0.966, but `fit` 0.929 -> 0.902 and `unhealthy` 0.932 -> 0.908 — net
  negative for balanced accuracy.
- Feature importance on the Section D model: `sleepbin_x_stress` became the single
  dominant feature (9.0M gain, more than double v0.1's raw `stress_level` at 4.59M),
  and raw `stress_level` collapsed to near-zero importance (99.7) — fully absorbed
  into the engineered feature. The interaction hypothesis from Section B was
  confirmed correct at the feature-importance level, but the larger feature set
  (35 vs. 13 columns) net-hurt CV, most likely from added variance — Section D's
  per-fold `best_iteration` varied widely (4661-5000) vs. Section A's near-uniform
  ~5000, suggesting less stable training.
- The notebook's own logic correctly selected Section A (0.9290) over Section D
  (0.9255) as the better of the two candidates for the written `submission.csv` — but
  both still underperform v0.1's already-submitted 0.9389 OOF / 0.94051 LB.

### Actions Taken

- Root-caused and fixed the `sleep_duration_bin` NaN bug (clip-before-cut), verified
  with a standalone script and a full-pipeline smoke test on a data sample.
- Added `tqdm`/`ipywidgets`-based progress bars (outer bar per CV fold, inner bar per
  boosting round via a custom LightGBM callback) to both `v0.2-feature-engineering.ipynb`
  and `v0.1-baseline.ipynb` (for consistency on future re-runs), and to `requirements.txt`.
- Detected and recovered from the JupyterLab stale-tab clobbering incident by
  reapplying all fixes and instructing the user to reopen the notebook fresh.
- Ran the full notebook live via JupyterLab (user-driven `Run All`), monitored
  progress via `ps`/CPU-time checks and by reading live tqdm output in the `.ipynb`.
- Recorded both v0.2 candidates in `docs/plans/leaderboard.md`, updated
  `docs/plans/implementation-plan.md` (Rung 2 marked done, negative result, lessons
  carried to Rung 3) and `docs/plans/TODO.md`.

### Resolution

**resolved** — Rung 2 complete with a real, informative negative result: neither the
training-budget ablation nor the engineered features beat v0.1. v0.1 remains the best
model. The `sleep_duration` x `stress_level` interaction hypothesis was confirmed
correct in isolation (Section B, and Section D's feature importance) even though the
overall engineered model regressed — a useful distinction between "this feature is
informative" and "adding it nets a better model."

### Follow-ups

- **CatBoost bake-off** (per `implementation-plan.md`) is now more motivated: CatBoost's
  native categorical/NaN handling and ordered boosting may be less prone to the
  variance/overfitting pattern seen in Section D.
- **Rung 3 threshold tuning** on v0.1's OOF predictions (not Section D's) is the next
  concrete lever, since v0.1 remains the best base model.
- If feature engineering is revisited, consider it in smaller increments (e.g. just
  the `sleepbin_x_stress` cross alone, without the full target-encoding set) to
  isolate which specific addition drives the regression, rather than evaluating the
  whole engineered set as one bundle.
- Submitted the Section A candidate to Kaggle despite the negative CV result, at the
  user's request, to confirm CV<->LB correlation holds directionally for a regression
  too — see `leaderboard.md` for the resulting score.

## v0.3-catboost-bakeoff

### Context

- Notebook: `notebooks/v0.3-catboost-bakeoff.ipynb`.
- Purpose: test whether CatBoost beats LightGBM v0.1 (OOF 0.9389, LB 0.94051), acting
  directly on two Rung 2 lessons: (1) implement a real `balanced_accuracy_score`-based
  custom CatBoost eval metric so early stopping tracks the actual competition metric,
  not a `multi_logloss` proxy; (2) test whether CatBoost's ordered boosting absorbs
  v0.2's engineered feature set without the regression LightGBM showed.
- Run: built and smoke-tested (data-sample dry run) headlessly, then handed off for
  live execution in JupyterLab per the user's standing preference to watch training
  runs interactively rather than as a backgrounded `nbconvert` process.

### Investigation Checklist

- [x] Implement and verify a custom CatBoost eval metric (`is_max_optimal` /
      `evaluate` / `get_final_error` protocol) computing real `balanced_accuracy_score`.
- [x] Implement and verify a `TqdmCatBoostCallback` using CatBoost's `after_iteration(info)`
      callback protocol (analogous to the LightGBM `TqdmCallback` from v0.1/v0.2).
- [x] Smoke-test the full pipeline (both variants) on a data sample before the full run.
- [x] Variant 1 (CatBoost, v0.1's 13 base features): full 5-fold run.
- [x] Variant 2 (CatBoost, v0.2's 35 engineered features): full 5-fold run.
- [x] Compare both variants against every prior run (v0.1, v0.2-A, v0.2-D).
- [x] Feature importance check on both variants.
- [x] Candidate `submission.csv` written from the actual best model across all runs
      (not just the newest experiment).

### Findings

- **Numba investigation (tangential, during setup)**: installed `numba` hoping to
  speed up the custom eval metric's per-iteration evaluation. It doesn't help here —
  confirmed via two separate implementations that CatBoost's numba JIT can't compile
  the `evaluate` method: the sklearn-based version fails because `numba` can't type
  arbitrary Python library calls, and a pure-numpy rewrite still fails because
  CatBoost passes `approxes` as a tuple-of-arrays shape numba's `nopython` mode
  can't convert via `np.array()`. Both fall back safely to interpreted Python (correct
  results, just no JIT speedup) — confirmed via smoke tests before concluding this,
  not assumed. Not a real bottleneck anyway: the eval metric runs once per boosting
  iteration on the validation fold, not in a hot per-row training loop, so CatBoost's
  tree-building dominates actual training cost. Kept `numba` in `requirements.txt`
  for potential future use elsewhere, documented honestly as a non-benefit here.
- **Variant 1 (CatBoost, base features)**: `auto_class_weights='Balanced'`,
  `iterations=3000, learning_rate=0.05, depth=6`, custom balanced-accuracy eval
  metric, early stopping patience 100. **best_iterations: [428, 950, 605, 339, 779]**
  — fired well before the 3000-round cap in every fold, unlike v0.1/v0.2-A which
  never early-stopped under the `multi_logloss` proxy. **OOF balanced accuracy:
  0.9493** — beats v0.1's 0.9389 by +0.0104, the best model so far. Per-class recall:
  at-risk 0.933, fit 0.950, unhealthy 0.965 — more evenly balanced than v0.1's
  0.956 / 0.929 / 0.932, consistent with directly optimizing the real metric during
  training instead of a proxy.
- **Variant 2 (CatBoost, engineered features)**: same config on v0.2's 35-feature set.
  **best_iterations: [544, 765, 542, 354, 628]**. **OOF balanced accuracy: 0.9491** —
  essentially tied with Variant 1 (-0.0002, within fold-to-fold noise). This directly
  contrasts with LightGBM's Section D, which regressed from 0.9389 to 0.9255 on the
  identical feature set — CatBoost's ordered boosting / native categorical handling
  hypothesis (stated in the plan doc) held up: the larger, more collinear feature set
  did not hurt CatBoost the way it hurt LightGBM. Feature importance shows the
  engineered features are genuinely used (`te_stress_x_activity_k1` #2 at 17.45,
  `sleepbin_x_stress` present at 5.32) but don't add net value over the simpler
  base-feature model here.
- **Variant 1 selected as the overall best model** (0.9493 > Variant 2's 0.9491 >
  v0.1's 0.9389 > v0.2-A's 0.9290 > v0.2-D's 0.9255). `data/submission.csv` written
  from Variant 1's averaged fold predictions.

### Actions Taken

- Built `notebooks/v0.3-catboost-bakeoff.ipynb` and `docs/plans/v0.3-catboost-bakeoff-plan.md`.
- Smoke-tested the full pipeline (both variants, custom metric, tqdm callback,
  feature-engineering reconstruction, comparison/submission logic) on a 20k/10k-row
  data sample before committing to the full run — no bugs found this time (unlike
  v0.2's `pd.cut` NaN bug), likely because the feature-engineering code was reused
  verbatim from the already-debugged v0.2 notebook.
- Investigated and ruled out a numba speedup for the custom eval metric (see Findings).
- Handed off to the user for live execution in JupyterLab; monitored via `ps`/CPU-time
  checks and by reading live tqdm/print output from the `.ipynb` JSON.
- Sent an interim SMS update after Variant 1 finished (before Variant 2 completed),
  given the significance of the result.
- Recorded both variants in `docs/plans/leaderboard.md`, updated
  `docs/plans/implementation-plan.md` (new Rung 2.5 section, Rung 3 updated to
  reference the new best model) and `docs/plans/TODO.md`.

### Resolution

**resolved** — a genuine positive result. CatBoost with a metric-aware early-stopping
setup beats LightGBM's best v0.1 model by a meaningful margin (+0.0104 OOF), and
handles the previously-regressive engineered feature set without penalty. v0.3
Variant 1 is the new best model going into Rung 3.

### Follow-ups

- ~~Submit Variant 1's `submission.csv` to Kaggle~~ — done: LB 0.94885 (Variant 1),
  LB 0.94913 (Variant 2, submitted separately to check whether the tiny CV gap held
  on LB — it didn't, see the "Both variants submitted" note above).
- **Rung 3 threshold tuning** should now target v0.3's OOF predictions, not
  v0.1's — the current best model changed. See `v0.4-threshold-tuning` below.
- Given Variant 2 (engineered features) didn't beat Variant 1 despite not regressing,
  the engineered feature set doesn't appear to hold real additional signal beyond
  what the base features already capture for CatBoost specifically — not an obviously
  productive direction to keep pushing on without a new hypothesis.
- Worth considering whether Variant 1's early-stopping `best_iteration` spread
  (339-950, a ~3x range across folds) indicates fold-to-fold variance worth
  investigating, or is simply expected given `auto_class_weights='Balanced'`
  reweighting shifts the loss landscape per fold.

## v0.4-threshold-tuning

### Context

- Notebook: `notebooks/v0.4-threshold-tuning.ipynb`.
- Purpose: Rung 3 — check whether per-class weighted-argmax threshold tuning beats
  plain argmax on v0.3's CatBoost predictions (argmax over raw probabilities isn't
  necessarily balanced-accuracy-optimal under imbalance).
- Base model: v0.3 Variant 2 (engineered features), chosen over Variant 1 because it
  scored slightly higher on LB (0.94913 vs. 0.94885), even though the two are
  statistically tied.
- Run: built, then hit a real notebook-corruption bug while editing (see Findings),
  fixed by a full rewrite, then executed live in JupyterLab per the user's standing
  preference.

### Investigation Checklist

- [x] Root-cause and fix the cell-corruption bug before handing back to the user.
- [x] Reproduce v0.3 Variant 2 capturing OOF probabilities (not just hard labels),
      verified against v0.3's known exact result before trusting anything downstream.
- [x] Weighted-argmax grid search on the full OOF set.
- [x] Nested validation (fit weights on 4/5 folds, evaluate on the held-out 5th) for
      an honest improvement estimate, not the optimistic same-data fit.
- [x] Investigate relevant Kaggle discussion threads for external context on why the
      result came out the way it did.

### Findings

- **Notebook-corruption bug**: while editing the notebook to switch its base model
  from Variant 1 to Variant 2 (per user request), a sequence of `NotebookEdit` insert
  + replace calls left the file with cells in the wrong order and some content
  duplicated — traced to the fact that this notebook's cells had no real nbformat
  `id` fields (unlike v0.1-v0.3, which were written cell-by-cell and picked up ids
  along the way), so the Read tool's positional placeholder ids (`cell-0`, `cell-1`,
  ...) silently pointed at different cells after earlier inserts shifted positions.
  A user re-run hit `NameError: name 'ENGINEERED_FEATURES' is not defined` as a
  direct symptom. Fixed by a full rewrite of the notebook file (verified cell order
  and that every code cell compiles) rather than patching further via `NotebookEdit`.
- **Reproduction**: exact match against v0.3 Variant 2 (`best_iterations [544, 765,
  542, 354, 628]`, OOF 0.9491) — the OOF probability matrix is trustworthy.
- **Full-OOF grid search**: best weights found were **w=(1.0, 1.0)** — plain argmax
  was already optimal. Zero improvement, even on the same-data fit that's normally
  mildly optimistic.
- **Nested validation**: nested plain-argmax 0.9491 (+/- 0.0011), nested
  tuned-argmax 0.9490 (+/- 0.0011). **Honest improvement estimate: -0.0001** — within
  noise, not a real effect.
- **External context**: see `docs/investigate/2026-07-03-kaggle-discussion-findings.md`
  for the full research (Kaggle discussion threads 717018 and 717222). Short version:
  Georgy Mamarin (717018) independently found that stacking training-time
  class-weighting with a *separate* post-hoc correction actively hurts (double-
  correction), which directly explains our flat result — `auto_class_weights='Balanced'`
  already spent the available correction during training. broccoli beef (717222)
  found the original pre-synthesis dataset has a near-deterministic depth-4 decision
  rule on `sleep_duration`/`stress_level`/`physical_activity_level`, suggesting our
  ~0.95 ceiling is likely synthesis noise rather than a modeling gap.

### Actions Taken

- Fixed the notebook-corruption bug via a full rewrite (Write tool), verified cell
  order and that every code cell compiles before handing back to the user.
- Ran the full notebook live in JupyterLab (user-driven), monitored via `ps`/CPU-time
  checks and by reading live print/tqdm output from the `.ipynb` JSON.
- Investigated Kaggle discussion threads 717018 and 717222 via the `kaggle` CLI
  (`competitions topics list` / `topic-messages`) after being corrected away from
  browser automation — added `~/.cline/rules/kaggle-discussions.md` and a matching
  memory entry so this preference persists across sessions. Full findings recorded
  separately in `docs/investigate/2026-07-03-kaggle-discussion-findings.md`, per the
  same rule file's convention of keeping external research out of notebook run logs.
- Recorded the negative result and its explanation in `docs/plans/leaderboard.md`,
  updated `docs/plans/implementation-plan.md` (Rung 3 marked done with mechanism)
  and `docs/plans/TODO.md`.

### Resolution

**resolved** — a clean negative result with a concrete, externally-corroborated
mechanism (double-correction of an already-balanced model), not just an empirical
shrug. v0.3 (either variant, still tied) remains the best model. The discussion
threads additionally suggest ~0.95 may be close to this dataset's practical ceiling
given its likely origin as a noised synthesis of a near-deterministic rule.

### Follow-ups

- See `docs/investigate/2026-07-03-kaggle-discussion-findings.md` for the Rung 4
  expectation-tempering and depth-4-rule-as-feature follow-ups that stem from the
  discussion threads — kept there since the reasoning lives in that file.
- The notebook-corruption bug is worth remembering as a general pattern: when editing
  a notebook via `NotebookEdit` insert operations, re-`Read` the file immediately
  after each insert to get real cell ids before issuing further edits, rather than
  reusing ids/positions from an earlier Read — especially for notebooks whose cells
  lack real nbformat `id` fields.
