# Playground Series - Season 6, Episode 7 - Predicting Student Health Risk

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

https://www.kaggle.com/competitions/playground-series-s6e7

Predict a 3-class student health risk label (`at-risk` / `unhealthy` / `fit`) from
synthetic student lifestyle/health features, scored on balanced accuracy.

## Goal

Beat the majority-class floor (~0.333 balanced accuracy) with a model that handles
heavy class imbalance and near-universal per-column missingness well. See
`docs/plans/implementation-plan.md` for the strategy ladder.

## Status

| Item | Value |
|---|---|
| Metric | Balanced accuracy |
| Baseline to beat | ~0.333 (all-majority-class) |
| Current best | _see docs/plans/leaderboard.md_ |
| Deadline | July 31, 2026 |

## Competition constraints

| Constraint | Value | Implication |
|---|---|---|
| Type | Standard prediction-file competition (not a Code Competition) | No runtime cap, no internet-off requirement |
| Output | `submission.csv` with `id,health_condition` | Exact filename + header, label must match `at-risk`/`unhealthy`/`fit` |
| Submissions | Max 10/day, 2 selected as Final | Budget experiments accordingly |
| External data | Allowed under Kaggle's Reasonableness Standard | Pretrained models permitted |

## Layout

```
docs/   plans, adr, investigate, images
scripts/  download_data.sh, ...
notebooks/ Kaggle kernel + kernel-metadata.json
configs/  training YAML
data/     train/test (gitignored)
```

## Quick start

```zsh
zsh scripts/download_data.sh      # needs ~/.kaggle/kaggle.json + accepted rules
```

See `docs/plans/implementation-plan.md` for the strategy ladder.

---

Scaffolded by the [kaggle plugin](https://github.com/msusol/claude-code-plugins/tree/main/kaggle)
for Claude Code.
