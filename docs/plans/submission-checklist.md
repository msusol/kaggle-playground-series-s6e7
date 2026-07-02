# Submission Checklist

Not a Code Competition — submit via `kaggle competitions submit` or the web UI, no
notebook runtime cap or internet-disabled requirement.

## File format
- [ ] File named exactly `submission.csv`.
- [ ] Header + columns match sample_submission.csv (`id,health_condition`).
- [ ] One row per test id; row count matches test set (295,753 rows).
- [ ] No NaN / invalid values.

## Target format
- [ ] **Categorical / hard label:** predicted values are an exact match to
      `at-risk` / `unhealthy` / `fit` (case, spelling, no unseen labels).

## Modeling sanity
- [ ] CV is 5-fold **stratified** (class imbalance: 85.9% / 8.4% / 5.8%).
- [ ] CV metric is balanced accuracy (not raw accuracy) — matches the leaderboard metric.
- [ ] CV recorded in leaderboard.md and beats current best.
- [ ] No leakage features used.

## Submission limits
- [ ] Under 10 submissions today.
- [ ] Final Submission selection (up to 2) set before July 31, 2026 deadline.
