# Competition Overview

## Objective

Predict `health_condition` for each student in the test set — a 3-class label:
`at-risk`, `unhealthy`, `fit`.[cite:1] The dataset (train and test) is synthetically
generated, inspired by the "College Student Health Behavior Dataset"; feature
distributions are close to, but not identical to, the original.[cite:2]

## Metric

**Balanced accuracy** — the average of per-class recall across the three classes,
not raw accuracy.[cite:1] This matters because the target is heavily imbalanced
(see below); a model that only predicts the majority class scores ~0.333, not ~0.86.

**Floor (must-beat):** ~0.333 (predict majority class `at-risk` for every row).

## Data

| File | Rows | Columns |
|---|---|---|
| `train.csv` | 690,088 | 15 (14 features + `health_condition` target) |
| `test.csv` | 295,753 (hidden labels) | 14 (features only) |
| `sample_submission.csv` | 295,753 | 2 (`id`, `health_condition`) |

### Target distribution (train)

| Class | Share |
|---|---|
| `at-risk` | 85.9% |
| `unhealthy` | 8.4% |
| `fit` | 5.8% |

### Features

| Column | Type | Notes |
|---|---|---|
| `sleep_duration` | float | ~11% missing |
| `heart_rate` | float | ~1% missing |
| `bmi` | float | ~2% missing |
| `calorie_expenditure` | float | ~8% missing |
| `step_count` | float | ~2% missing |
| `exercise_duration` | float | ~1% missing |
| `water_intake` | float | ~6% missing |
| `diet_type` | categorical (`veg`/`non-veg`/`balanced`) | ~1% missing |
| `stress_level` | categorical (`low`/`medium`/`high`) | ~12% missing |
| `sleep_quality` | categorical (`poor`/`average`/`good`) | ~8% missing |
| `physical_activity_level` | categorical (`sedentary`/`moderate`/`active`) | ~5% missing |
| `smoking_alcohol` | categorical (`no`/`occasional`/`yes`) | ~4% missing |
| `gender` | categorical (`male`/`female`/`other`) | ~3% missing |

Missingness rates are near-identical between train and test (verified by direct
inspection of both files) — no leakage signal from missingness patterns is expected,
but missing-value handling (imputation vs. native NaN support in GBM libraries) is a
first-class modeling decision here, not an edge case.

## Competition type

This is a **standard prediction-file competition**, not a Code Competition — there is
no notebook runtime cap or internet-disabled scoring requirement. Submissions are
CSV uploads via the Kaggle UI/CLI (`kaggle competitions submit`).[cite:3]

- Max 10 submissions/day; up to 2 selected as Final Submissions.[cite:3]
- Max team size: 3; team mergers allowed up to the Team Merger Deadline.[cite:3]
- License: CC BY 4.0 (competition data).[cite:3]

## Timeline

- Start Date: July 1, 2026
- Final Submission Deadline: July 31, 2026 (11:59 PM UTC)[cite:1]
- Entry Deadline / Team Merger Deadline: same as Final Submission Deadline[cite:1]

## Prizes

Kaggle merchandise for 1st/2nd/3rd place (no points or medals awarded); merch is
awarded once per person across the Playground Series.[cite:1]
