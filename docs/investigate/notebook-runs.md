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
