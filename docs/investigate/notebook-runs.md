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

- Built `notebooks/v0.3-catboost-bakeoff.ipynb` and `docs/plans/archive/v0.3-catboost-bakeoff-plan.md`.
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

## v0.5-ensemble

### Context

- Notebook: `notebooks/v0.5-ensemble.ipynb`.
- Purpose: Rung 4 — check whether blending diverse model families beats the current
  best single model (v0.3 CatBoost, OOF ~0.949).
- Originally scoped as a 3-way blend (LightGBM + CatBoost-V1 + a new logistic
  regression). User asked "should we use v0.3 Variant 2's exact config" before the
  first run — expanded to a 4-way blend adding CatBoost-V2 (engineered features) as
  a 4th member, since a different feature view can count as real diversity even
  between two CatBoost runs.
- Run: hit a system-wide disk-full incident (`ENOSPC`) mid-run on the first (3-way)
  attempt, which silently stalled the kernel; user stopped it once disk space
  recovered, notebook was rewritten to the 4-way version, and the full run completed
  cleanly on the second attempt.

### Investigation Checklist

- [x] Reproduce LightGBM v0.1, CatBoost v0.3 Variant 1, and CatBoost v0.3 Variant 2,
      capturing OOF + test probabilities (not just hard labels) this time, verified
      against each's known exact result before trusting anything downstream.
- [x] Build and fit a new regularized logistic regression baseline with proper
      preprocessing (one-hot categoricals, median-impute + standardize numerics).
- [x] 4-way blend weight search (simplex grid) + nested validation (fit weights on
      4/5 folds, evaluate on the held-out 5th) for an honest improvement estimate.
- [x] Subset blends (all pairs + triples) for comparison against the full 4-way.
- [x] Monitor disk space throughout, given the mid-run incident.

### Findings

- **Disk-full incident**: mid-run on the first (3-way) attempt, `df -h` and even
  trivial shell commands started failing with `ENOSPC: no space left on device` —
  the volume containing `/private/tmp` (and likely the whole container) ran out of
  free space. The live kernel (mid-training on the LightGBM member) went idle with
  no further progress and no error captured in the notebook's own outputs —
  consistent with the disk-full condition silently interrupting a background write
  (CatBoost/LightGBM logging, autosave) rather than raising a visible Python
  exception. User confirmed manually stopping the stalled kernel. Disk space
  recovered on its own between checks (15GB -> 13GB -> 11GB free, stabilizing) by
  the time the rewritten 4-way notebook was run.
- **All 3 reproductions PASS** (exact match to known results): LightGBM 0.9389
  (fold scores identical to v0.1: `[0.9397, 0.9402, 0.9389, 0.9389, 0.9367]`),
  CatBoost-V1 0.9493 (`best_iterations [428, 950, 605, 339, 779]`, identical to
  v0.3), CatBoost-V2 0.9491 (`best_iterations [544, 765, 542, 354, 628]`, identical
  to v0.3).
- **LogisticRegression (new member, no prior baseline)**: OOF **0.8994** — notably
  weaker than the tree models, as expected for a linear model on this data. Recall
  pattern matches the tree models' direction (at-risk 0.816 lower, fit 0.952 /
  unhealthy 0.930 higher) — class-weighting works even for the weak learner, but its
  decision boundary is much less expressive than the tree ensembles'.
- **4-way blend weight search** (simplex grid, step 0.1, 286 combinations):
  full-OOF best = **0.9493**, found at weights `{lgbm: 0, catboost_v1: 1.0,
  catboost_v2: 0, logreg: 0}` — the optimistic same-data search degenerates to using
  CatBoost-V1 alone. Zero improvement found even before nested validation.
- **Nested validation**: nested solo CatBoost-V1 0.9493 (+/- 0.0011), nested 4-way
  blend 0.9492 (+/- 0.0011). **Honest improvement estimate: -0.0002** — no real
  gain. Per-fold blend weights fit on the other 4 folds mostly landed near
  `(0, 1, 0, 0)` (pure CatBoost-V1), with one fold assigning some weight to LightGBM
  and CatBoost-V2 — noise, not a stable pattern.
- **Subset blends** (all pairs + triples): every combination that includes
  CatBoost-V1 caps out at exactly 0.9493 (matching its solo score); no combination
  beats it. `lgbm+logreg` (the two members *without* CatBoost-V1) tops out at just
  0.9442 — notably worse, confirming CatBoost-V1 alone already captures what the
  other members can offer.

### Actions Taken

- Smoke-tested the full pipeline (all 4 members + blend search + nested validation)
  on a data sample before the full run; found and fixed a real bug —
  `LogisticRegression(multi_class='multinomial', ...)` raised `TypeError` since
  `multi_class` was removed in the installed sklearn 1.9 (the `lbfgs` solver now
  handles multinomial softmax automatically for multi-class problems without the
  parameter). Dropped the argument; verified the fix with a targeted repro before
  rerunning the full smoke test.
- Per the new notebook-corruption-avoidance rule in `kaggle-notebook-workflow.md`,
  rewrote the whole notebook file via `Write` (rather than chaining `NotebookEdit`
  inserts) when expanding from 3-way to 4-way, since this notebook's cells lacked
  real nbformat ids.
- Monitored disk space and kernel CPU/process state throughout via scheduled
  check-ins; flagged the `ENOSPC` incident to the user immediately when first
  observed rather than continuing to poll blindly.
- Ran the full 4-way notebook live in JupyterLab (user-driven) after the disk issue
  cleared; verified all reproduction-check cells passed before trusting the blend
  results.
- Recorded the negative result in `docs/plans/leaderboard.md`, updated
  `docs/plans/implementation-plan.md` (Rung 4 marked done with mechanism) and
  `docs/plans/TODO.md`.

### Resolution

**resolved** — a clean negative result, consistent with and stronger evidence for
the synthesis-noise-ceiling hypothesis from Rung 3 (`v0.4-threshold-tuning` above):
even an architecturally-distinct logistic regression adds nothing to the blend,
arguing against "wrong model family" as an explanation for the ~0.949-0.951
plateau. v0.3 (either variant) remains the best model.

### Follow-ups

- If further squeeze attempts are made (Rung 5+), weigh them against two independent
  negative results (Rung 3 threshold tuning, Rung 4 ensembling) both pointing at the
  same ceiling — see `docs/investigate/2026-07-03-kaggle-discussion-findings.md`'s
  follow-ups for the standing recommendation.
- The disk-full incident was investigated only as far as confirming it happened and
  that it recovered on its own; the root cause (what filled the disk) was not
  identified. Worth a closer look if it recurs, since a silently-stalled kernel with
  no error output is easy to miss.

## v0.7-hgbc-te

### Context

- Notebook: `notebooks/v0.7-hgbc-te.ipynb`.
- Purpose: Rung 7 — test whether `HistGradientBoostingClassifier` (sklearn's
  native GBM, a 4th distinct tree-boosting implementation beyond
  LightGBM/CatBoost/XGBoost) plus exact-value target encoding of numeric
  features beats CatBoost-V1 (current best, OOF 0.9493). Surfaced by
  investigating `redamountassir/ps-s6e7-hgbc-baseline-lb-0-95034-cv-0-95026`
  ("TE-HGBC") — see `docs/investigate/2026-07-03-kaggle-discussion-findings.md`.
- Run venue: **Kaggle**, not local JupyterLab, per user request — built with
  Kaggle-input-first data loading from the start, pushed publicly, ran on
  Kaggle's own shared CPU compute.

### Investigation Checklist

- [x] Smoke-test the pipeline locally on a data sample before pushing to Kaggle.
- [x] Push and run on Kaggle.
- [x] Monitor via both `kaggle kernels status` (CLI) and the Kaggle web log
      viewer (CLI returns nothing for an in-progress kernel; the web UI streams
      live logs).
- [x] Extract HGBC-TE solo OOF, CatBoost-V1 reproduction check, blend/nested
      validation results, and the final decision once complete.
- [x] Submit the resulting `submission.csv` if the notebook's own decision logic
      wrote one.

### Findings

- **Smoke test** (reduced scale/budget): HGBC-TE solo 0.9428 vs. CatBoost-V1's
  0.9422 — the first new model to edge out CatBoost solo at reduced scale,
  though not conclusive there.
- **Full run took ~86 minutes on Kaggle's shared CPU** (5144.5s per the
  platform's own reported runtime). HGBC-TE itself finished quickly (~4.5 min,
  270.3s per the log) — the CatBoost-V1 reproduction peg took the bulk of the
  time, consistent with how long CatBoost training took on Kaggle for earlier
  notebooks in this project (v0.3/v0.4 also took unusually long there).
- **HGBC-TE solo OOF: 0.9502** — closely matches the source notebook's own
  reported CV (0.95026) despite using a different fold split, and **beats
  CatBoost-V1 (0.9493) by +0.0009 — the first genuine, non-noise-level
  improvement across the entire squeeze phase** (Rungs 3-6 were all within
  ±0.0005 of CatBoost-V1, i.e. statistically indistinguishable from it).
  Per-class recall: at-risk 0.9373, fit 0.9500, unhealthy 0.9633.
- **CatBoost-V1 reproduction: PASS** (exact match, `best_iterations [428, 950,
  605, 339, 779]`, OOF 0.9493) — confirms the blend comparison is trustworthy.
- **Blend check**: full-OOF 2-way grid search best 0.9504 at weights
  (hgbc_te=0.78, catboost_v1=0.22) — only +0.0002 over HGBC-TE solo same-data.
  **Nested validation**: nested solo (hgbc_te) 0.9502 (+/- 0.0012), nested blend
  0.9503 (+/- 0.0011) — honest improvement **+0.0002** for the blend
  specifically, below the 0.0005 threshold. But HGBC-TE's solo score
  independently clears `current_best + 0.0005` on its own, so the notebook's
  decision logic correctly wrote a submission from the **solo** HGBC-TE
  predictions rather than the (marginally better but not worth the added
  complexity) blend.
- **Submitted to Kaggle: public LB 0.95036** vs. OOF 0.9502 — tight correlation,
  no haircut, consistent with this project's established CV-LB trustworthiness.
  New best LB in this project.
- **Monitoring note**: the Kaggle CLI's `kernels output` command returns nothing
  for an in-progress kernel (confirmed repeatedly — same limitation hit with
  earlier long-running kernels in this project). The Kaggle web UI's Logs tab,
  by contrast, streams live output and was essential for confirming genuine
  progress (vs. a stall) during the long CatBoost-V1 wait, and for reading the
  final results the moment the run completed.

### Actions Taken

- Smoke-tested the full pipeline on a 15k/8k-row data sample before pushing.
- Pushed to Kaggle publicly via `kaggle kernels push` with a dedicated
  `kernel-metadata.json` (`enable_gpu: false` — both `HistGradientBoostingClassifier`
  and `TargetEncoder` are CPU-only, no benefit from GPU).
- Monitored via periodic `kaggle kernels status` checks plus the Kaggle web log
  viewer (loaded Chrome browser tools mid-session specifically for this) once
  the CLI's lack of in-progress output became a limiter.
- Downloaded the completed kernel's output (`kaggle kernels output`), verified
  `submission.csv`, and submitted it via `kaggle competitions submit`.
- Recorded the positive result in `leaderboard.md` and `implementation-plan.md`
  (Rung 7 marked done, revising the Rung 3-6 "synthesis-noise ceiling"
  conclusion), and `TODO.md`.

### Resolution

**resolved** — a genuine positive result, the first of this project's entire
squeeze phase (Rungs 3-6 were all flat/negative or within noise). v0.7 (HGBC-TE)
is now the best model, superseding v0.3 CatBoost.

### Follow-ups

- **Revisit the "synthesis-noise ceiling" framing from Rungs 3-6**: it wasn't
  really a ceiling at ~0.949 — it was specific to the feature representations
  and model structures tried up to that point. Exact-value numeric target
  encoding found real additional signal. The generalizable lesson: when several
  different *decision-rule* and *model-structure* levers all plateau at the same
  score, that's evidence the *feature representation* is the binding constraint,
  not necessarily the data's intrinsic noise floor.
- **Consider applying exact-value target encoding to CatBoost** (our previously
  best-performing library) rather than only pairing it with a new model family —
  worth testing whether the encoding itself, or the HGBC/encoding combination
  specifically, is what drives the improvement.
- No `notebook-runs.md` entry exists for v0.6-xgboost-ovr — it was covered in
  `leaderboard.md` and `TODO.md` but this file's per-notebook convention was
  missed for that one. Not backfilled here to stay in scope; worth doing if a
  future session revisits v0.6.

## v0.8-realmlp

### Context

- Notebook: `notebooks/v0.8-realmlp.ipynb`.
- Purpose: Rung 8 — test whether RealMLP, a from-scratch PyTorch neural-net
  architecture, adds real signal vs. our current best (v0.7 HGBC-TE, OOF
  0.9502), either solo or as a blend member. First non-tree-boosting model
  family in this project. Surfaced by investigating
  `yunsuxiaozi/pss6e7-realmlp-cv-0-95063` — see
  `docs/investigate/2026-07-05-kaggle-discussion-findings.md`.
- Run venue: **local** (Apple M3 Pro, PyTorch MPS backend), per user request
  — not Kaggle, unlike v0.7. Executed interactively by the user in their own
  JupyterLab session (`http://localhost:8888`), not launched by the assistant.

### Investigation Checklist

- [x] Adapt the pulled source notebook: our own `StratifiedKFold(5)`, MPS
      device detection, drop the source's post-hoc Optuna reweighting
      (already found negligible), add the standard CatBoost-V1 reproduction +
      blend/nested-validation/decision-logic cells matching v0.5/v0.6/v0.7.
- [x] Smoke-test on a reduced sample before the full run.
- [x] Monitor the user's live local run (no CLI/API for local Jupyter kernel
      status — periodic autosave file reads instead) and report at milestones.
- [x] Extract final solo/blend/nested-validation numbers and the decision
      once complete.

### Findings

- **Smoke-test bug found and fixed**: pandas 3.0.3's `.astype(str)` now
  produces a native `str` dtype rather than `object`, which broke the
  `CATS = [c for c in test.columns if test[c].dtype == object or ...]`
  categorical/numeric column classification (the binned `_cat`/`_cat2`
  columns were silently misclassified as numeric, causing `np.median` to
  fail on string data). Fixed via
  `not pd.api.types.is_numeric_dtype(test[c])`. This was an environment
  version-compatibility issue in the adaptation, not a bug in the source
  notebook (presumably run under an older pandas where `.astype(str)` still
  produced `object` dtype).
- **Smoke-test progress-display iteration**: initial architecture-faithful
  smoke run at reduced scale (15k rows, 2 folds, 2 epochs) showed RealMLP
  accuracy near-random (0.43) — not a bug, but an artifact of the EMA decay
  (0.997) needing far more gradient steps than a tiny smoke test provides to
  converge away from random initialization. Confirmed by re-running at a
  larger reduced scale (60k rows, 3 folds, 5 epochs), which reached a
  sensible 0.9411 with visibly improving per-epoch accuracy. Also added
  (then refined) `tqdm` progress bars for the training loop, which had none
  initially (only per-fold/per-epoch `print` statements) — first added
  nested fold/epoch/batch bars with `tqdm.write()` for status text, but the
  user found this still produced separate print-like output blocks
  interleaved with the progress widgets in Jupyter (each `tqdm.write()` call
  and each new `tqdm()` instance creates its own output area in the
  ipywidgets-based renderer). Final fix: collapsed to 2 nested levels
  (folds, epochs) and moved all status info (class weights, loss, best
  accuracy) into the bars' own `set_postfix()`/description instead of any
  print/write call — eliminates all interleaved text blocks.
- **Full local run results**:
  - **RealMLP solo OOF: 0.95062** — the highest raw solo OOF of any model in
    this project, edging out v0.7 (0.9502) by +0.0004, and closely matching
    the source notebook's own raw CV (0.95057, different fold split).
    Per-class recall: at-risk 0.9355, fit 0.9507, unhealthy 0.9657.
  - **CatBoost-V1 reproduction: PASS** (0.9493 OOF, matching v0.3 Variant 1
    exactly to 4 decimals). Per-fold `best_iterations` differed from v0.3's
    original run (`[765, 461, 703, 516, 576]` vs. `[428, 950, 605, 339,
    779]`) despite the same fold seed/config — plausibly CatBoost's
    CPU floating-point nondeterminism across different hardware (local M3
    Pro vs. wherever v0.3/v0.7 ran); the OOF score matching exactly confirms
    this doesn't affect the actual reproduction validity.
  - **Blend check**: full-OOF grid search best 0.9507 at weights
    (realmlp=0.82, catboost_v1=0.18) — notably not degenerate to 100% one
    member (unlike v0.5's all-tree-boosting blend), suggesting genuine
    diversity between a neural net and a GBDT. Nested validation: nested
    solo (realmlp) 0.9506 (+/- 0.0012), nested blend 0.9507 (+/- 0.0012) —
    honest improvement only **+0.0001**.
  - **Decision: NO REAL IMPROVEMENT, no submission written.** RealMLP
    solo's raw margin over the current best (+0.0004) falls just short of
    the project's 0.0005 threshold (0.95062 vs. `0.9502 + 0.0005` = 0.9507)
    — essentially a statistical tie with v0.7, not a confirmed new best.
    Nothing was submitted to Kaggle for this run.
- **Monitoring note**: with no CLI/API to query a local Jupyter kernel's
  live execution status, progress was tracked by periodically re-reading the
  saved `.ipynb` file's cell `execution_count`/`outputs` (Jupyter autosaves
  periodically) and by asking the user directly when the file hadn't changed
  in an unusually long time (~52 minutes at one point) — confirmed by the
  user that the kernel was still genuinely progressing (progress bar moving)
  despite the autosave lag, i.e. file-mtime gaps are not a reliable
  stuck/stalled signal on their own for a local session.

### Actions Taken

- Adapted the pulled `yunsuxiaozi/pss6e7-realmlp-cv-0-95063` notebook into
  `notebooks/v0.8-realmlp.ipynb` with the scope decisions in
  `docs/plans/archive/v0.8-realmlp-plan.md`.
- Installed `torch` and `optuna` into the shared venv; confirmed PyTorch MPS
  backend availability on the local Apple M3 Pro.
- Ran multiple smoke tests at increasing scale to isolate and fix the pandas
  dtype bug and validate the training loop converges correctly given enough
  steps.
- Iterated on `tqdm` progress-bar design based on direct user feedback while
  they ran the full notebook themselves.
- Monitored the user's live full run via periodic `.ipynb` file reads,
  texted SMS updates at the RealMLP-training-done and full-decision
  milestones.
- Recorded the result in `leaderboard.md`, `implementation-plan.md`
  (Rung 8), and `TODO.md` (Phase 10).

### Resolution

**resolved** — a flat result by the project's own strict threshold, but a
genuinely close one: the highest raw OOF of any model tried, and the first
non-tree-boosting model family to reach parity with the best tree-based
model. No submission made (correctly, per the decision logic). v0.7
(HGBC-TE) remains the best model.

### Follow-ups

- **RealMLP is the best-positioned diversity source found so far** if blend
  work is revisited — the 82/18 non-degenerate blend weight (vs. v0.5's
  100/0 collapse) suggests it captures somewhat different errors than the
  tree-boosting models, even though it doesn't clear the solo-improvement
  bar on its own. A worthwhile next experiment would be blending RealMLP
  with v0.7 (HGBC-TE, the actual current best) rather than CatBoost-V1 (this
  run's comparison peg), in case that pairing clears the threshold where
  RealMLP+CatBoost-V1 didn't.
- The pandas 3.0.3 `.astype(str)` dtype-detection issue is worth remembering
  for any future notebook that classifies columns by `dtype == object` —
  prefer `pd.api.types.is_numeric_dtype()`/`is_string_dtype()` checks instead,
  since they're robust across pandas versions.

## v0.9-realmlp-hgbc-blend

### Context

- Notebook: `notebooks/v0.9-realmlp-hgbc-blend.ipynb`.
- Purpose: Rung 9 — direct follow-up flagged in v0.8's own entry above. v0.8
  only blended RealMLP against CatBoost-V1 (its comparison peg); this tests
  RealMLP blended against v0.7's HGBC-TE (the actual current best model)
  instead, since RealMLP found a genuinely non-degenerate blend weight
  against CatBoost-V1 (unlike every prior blend attempt).
- Run venue: Kaggle (GPU T4), pushed directly (not local — the local
  JupyterLab server was down after a machine reboot). Both RealMLP and
  HGBC-TE retrained fresh in this run (not reused from v0.7/v0.8's prior
  results) so their OOF arrays share a single `StratifiedKFold(5,
  random_state=42)` split and are directly comparable/blendable.
- Per explicit user request, submits **unconditionally** — no 0.0005
  real-improvement threshold gate this time (this run was framed as an
  exploration/curiosity pass, not a gated experiment like v0.1-v0.8).

### Investigation Checklist

- [x] Build a combined notebook retraining RealMLP (from v0.8) and HGBC-TE
      (from v0.7, exact config) with a shared fold split.
- [x] Smoke-test before pushing to Kaggle.
- [x] Push to Kaggle GPU, monitor to completion.
- [x] Extract solo/blend/nested-validation results.
- [x] Submit unconditionally and record the score.

### Findings

- **Smoke-test bug found and fixed**: reusing v0.7's original categorical-view
  code (`pd.Categorical` over raw `train_model[col]` without any `.astype(str)`
  conversion) crashed with `TypeError: '<' not supported between instances of
  'str' and 'float'` when building the sorted category list — the raw
  categorical columns have substantial real missingness (thousands of NaN
  rows even in the full 690k dataset), and pandas 3.0.3's category-sorting
  path chokes on a set containing both real NaN (float) and strings. This
  same code worked fine when v0.7 was originally built, confirming the local
  venv's pandas was silently upgraded (likely as a side effect of installing
  `torch`/`optuna` for v0.8) past a version where this comparison succeeded.
  Fixed by filtering NaN out of the category list before sorting (`categories
  = sorted(x for x in (...) if pd.notna(x))`), preserving v0.7's original
  intent of leaving NaN in the data for `HistGradientBoostingClassifier` to
  route natively, without the sort-crash on the (unrelated) category-list
  construction step.
- **Full Kaggle run results**:
  - **RealMLP solo OOF: 0.9506** (fresh retrain, close to v0.8's originally
    recorded 0.95062 — small run-to-run variance expected).
  - **HGBC-TE solo OOF: 0.9503** (fresh retrain, matches v0.7's recorded
    0.9502 almost exactly).
  - **Blend check**: full-OOF grid search best 0.9507 at weights
    (realmlp=0.86, hgbc_te=0.14). **Nested validation**: nested solo
    (realmlp) 0.9506 (+/- 0.0012), nested blend 0.9506 (+/- 0.0012) — honest
    improvement **~+0.0000**, i.e. essentially no measurable value added by
    blending in HGBC-TE, once cross-validation is done honestly. Nearly
    identical pattern to v0.8's RealMLP+CatBoost-V1 blend result (+0.0001) —
    two independent tree-boosting partners now both fail to add real
    diversity to RealMLP.
  - **Submitted unconditionally (per request): public LB 0.95065** — the
    **highest public LB score in this project so far**, ahead of v0.8's
    curiosity submission (0.95048) and v0.7's confirmed best (0.95036). Tight
    OOF-LB correlation (blend OOF 0.9506 → LB 0.95065, no meaningful haircut).
- **Important interpretive caveat**: despite being the highest LB number
  observed, this is **not treated as a confirmed new best model** — the
  nested-validated honest improvement is ~0, meaning the blend doesn't
  meaningfully outperform RealMLP solo under honest cross-validation. The
  higher LB number reflects RealMLP's own already-strong solo performance
  carrying through this particular blend combination, not new information
  extracted from the blend itself.

### Actions Taken

- Built `notebooks/v0.9-realmlp-hgbc-blend.ipynb` combining RealMLP's
  architecture/training loop (from v0.8) and HGBC-TE's exact training code
  (from v0.7) with a single shared fold split, plus an unconditional
  (non-thresholded) decision/submission cell per explicit request.
- Smoke-tested at reduced scale locally, found and fixed the pandas NaN/str
  sorting bug above.
- Pushed to Kaggle (GPU T4) publicly, monitored to completion (~37 min total
  — RealMLP ~10 min, HGBC-TE ~5 min, blend/nested fast, plus queue/startup
  overhead).
- Downloaded the completed run's submission.csv and log via `kaggle kernels
  output`, submitted via `kaggle competitions submit`, confirmed the score via
  `kaggle competitions submissions -v`.
- Recorded the result in `leaderboard.md`, `implementation-plan.md` (Rung 9),
  and `TODO.md` (Phase 11).

### Resolution

**resolved** — new highest LB score in the project (0.95065), but explicitly
**not** treated as a confirmed new best model given the ~0 nested-validated
improvement over RealMLP solo. v0.7 (HGBC-TE) and v0.8 (RealMLP) remain
statistically tied for best model; this run adds a data point confirming
RealMLP hasn't yet found a real ensemble partner among the tree-boosting
models tried.

### Follow-ups

- **RealMLP has now failed to find real ensemble lift with two different
  tree-boosting partners** (CatBoost-V1 in v0.8, HGBC-TE in v0.9). If blend
  work continues, the next candidate would need to be an even more
  differently-biased model (not another GBDT variant) to have a real chance
  at adding diversity — e.g. a linear/logistic model (already tried in v0.5
  and found unhelpful there too, though not blended with RealMLP
  specifically) or a genuinely different neural architecture.
- Given two independent blend attempts have both landed at ~0 honest
  improvement, further RealMLP-blend experiments are likely low-value unless
  a structurally different partner model is tried.
