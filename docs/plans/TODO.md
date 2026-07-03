# TODO

## Phase 0 - Setup
- [x] ~/.kaggle/kaggle.json present
- [x] `zsh scripts/download_data.sh` exits 0 (halts + prints a rules-acceptance URL if
      the competition rules haven't been accepted yet — accept them and re-run)
- [ ] Trivial submission scores at the floor (sanity)

## Phase 1 - Baseline (v0.1)
- [x] EDA (`notebooks/v0.1-eda.ipynb` — see findings summary + implementation-plan.md)
- [x] Baseline model + 5-fold CV (`notebooks/v0.1-baseline.ipynb` — OOF balanced
      accuracy 0.9389 +/- 0.0012, recorded in leaderboard.md)
- [x] First offline submission; record in leaderboard.md (submitted 2026-07-02,
      public LB 0.94051 vs. OOF 0.9389 — good CV<->LB correlation)

## Phase 2+ - Stronger models
- [ ] _fill from implementation-plan.md_
