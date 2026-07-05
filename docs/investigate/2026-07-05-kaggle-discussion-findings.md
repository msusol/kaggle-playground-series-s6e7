# Kaggle Discussion Findings — Playground Series S6E7 (2026-07-05)

External research findings from the competition's Discussion tab and public
notebooks, read via the `kaggle` CLI per `kaggle-discussions.md`. Continues
`docs/investigate/2026-07-03-kaggle-discussion-findings.md`.

## 718921 — "Adversarial Validation Insight: Mild Drift detected between Train and Test distributions (AUC ~0.63)"

### Context

- URL: https://www.kaggle.com/competitions/playground-series-s6e7/discussion/718921
- User asked to investigate this thread for context on train/test distribution
  drift, relevant to how much we should trust our own CV.

### Investigation Checklist

- [x] Read the full thread via
      `kaggle competitions topic-messages playground-series-s6e7 718921 -s old -n -1`.
- [x] Determine whether the drift is concentrated in features our models rely on,
      or in low-importance ones.
- [x] Assess relevance to our own CV-trust methodology.

### Findings

- **Original post**: an adversarial-validation XGBoost classifier (train=1,
  test=0) reaches ROC-AUC ~0.632 — a mild but real distribution shift. Top
  drivers: `physical_activity_level` (0.1199), `diet_type=veg` one-hot (0.1069),
  `smoking_alcohol` (0.1050), `water_intake` (0.0900), `calorie_expenditure`
  (0.0781), `bmi` (0.0752). No single feature dominates (no 80%+ importance
  leak) — reads as a collective lifestyle-cohort shift, not a broken column.
- **Follow-up comment (the more useful finding)**: a commenter reran the
  adversarial-validation AUC while incrementally adding features **in descending
  permutation-importance order** (most-important-for-the-real-task first) and
  found the AUC curve stays low until the *least* important features are added —
  i.e. **the drift is concentrated in features that barely matter for predicting
  `health_condition`, not in the ones our models actually lean on**
  (`sleep_duration`, `stress_level`, `physical_activity_level`, `bmi` per our own
  and Georgy Mamarin's/broccoli beef's feature-importance findings). This
  directly softens the original post's implied concern.
- **Consistent with our own project's stance**: we've already established (via
  717018/717222/718258, see 2026-07-03's file) that CV tracks public LB tightly
  and non-systematically across every model we've built (v0.1 OOF 0.9389/LB
  0.94051; v0.3-V1 OOF 0.9493/LB 0.94885; v0.7 OOF 0.9502/LB 0.95036 — no
  haircut in any of these). A mild AUC~0.63 drift concentrated in
  low-importance features is consistent with that observed tightness, not in
  tension with it.

### Actions Taken

- Read via `kaggle competitions topic-messages playground-series-s6e7 718921 -s old -n -1`.
- Cross-checked the drift's top features against our own established
  feature-importance rankings (sleep/stress/activity/bmi as the real drivers,
  matching all prior investigation entries).

### Resolution

**resolved** — the drift is real but concentrated in features that don't drive
our models' predictions. No change to our own methodology: continue trusting
CV over the public LB, as we already do.

### Follow-ups

- None — this reinforces existing practice rather than surfacing a new action
  item.

## yunsuxiaozi/pss6e7-realmlp-cv-0-95063 (notebook)

### Context

- URL: https://www.kaggle.com/code/yunsuxiaozi/pss6e7-realmlp-cv-0-95063
- User asked to investigate this notebook. Pulled via `kaggle kernels pull`
  (CLI, not browser) and its completed run's log downloaded via
  `kaggle kernels output` to get the actual numbers, not just the title's
  claimed score.
- Notable up front: this is our first encounter with a **neural-network**
  approach to this competition (all of our own work, and everything
  investigated on 2026-07-03, has been tree-boosting — LightGBM/CatBoost/
  XGBoost/HistGradientBoosting). RealMLP is a published architecture
  ("Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular
  Data", Holzmüller et al.) specifically designed to be competitive with
  boosted trees on tabular data.

### Investigation Checklist

- [x] Pull and read the full notebook via the `kaggle` CLI.
- [x] Download the completed run's log to verify the actual CV number and
      whether any post-hoc correction is stacked on top of training-time
      class weighting (our project's recurring double-correction concern).
- [x] Identify what, if anything, is genuinely new vs. our own v0.7 (target
      encoding of exact-value numerics) and the 2026-07-03 findings.
- [x] Assess reproduction cost/feasibility within our own project's scope.

### Findings

- **Architecture**: a from-scratch PyTorch implementation of RealMLP, with
  several non-trivial pieces: periodic/Fourier-style numeric embeddings
  (`PBLDEmbedding` — a learned periodic-basis + MLP layer per numeric
  feature), a `ScalingLayer` with learnable per-feature scale parameters,
  `NTPLinear` layers (NTK-style parametrization, weights normalized by
  `1/sqrt(in_features)`), and a residual bottleneck-GLU classification head.
  Trained with **`n_ens=16`** — 16 independent "ensemble member" weight sets
  batched into one model/training run (a single forward pass processes all 16
  in parallel via an extra tensor dimension), plus EMA (exponential moving
  average, decay 0.997) of weights for eval, cosine-annealed label smoothing,
  and 5 separate parameter groups each with their own learning-rate schedule
  (scale/embedding/first-layer/other-weights/biases) — reflects the RealMLP
  paper's specific tuning recommendations, not a generic MLP.
- **Feature engineering — independently confirms our own v0.7 direction, with
  a twist**: like our v0.7, all numeric features are target-encoded via
  sklearn's native `TargetEncoder`. Unlike our single exact-value encoding,
  this notebook additionally creates **two extra binned copies per numeric
  feature at different granularities** (e.g. `step_count // 10` and
  `step_count // 20`) and target-encodes those too, alongside the raw
  categoricals — i.e. it target-encodes numerics at 3 different resolutions
  (exact value implicitly via the raw float going into the embedding, plus 2
  binned granularities as extra TE columns) rather than just one. A second,
  independent data point that target-encoding numeric features (not just
  categoricals) carries real signal for this dataset.
- **GPU-dependent**: `enable_gpu: true`, `machine_shape: NvidiaTeslaT4` — the
  16-way batched ensemble and custom tensor ops require GPU to train in
  reasonable time (total run: ~617s / ~10 minutes on a T4, per the log — much
  faster than our own CPU-only CatBoost/HGBC runs on Kaggle, which is a point
  in its favor if we ever reproduce it).
- **Actual numbers, from the completed run's log** (not just the title):
  per-fold best-epoch balanced accuracy 0.94997-0.95095 across 5 folds ->
  **raw 5-fold CV: 0.95057** (before any post-hoc correction) — genuinely
  slightly above our own v0.7 HGBC-TE's OOF (0.9502). Class weights used:
  `compute_class_weight('balanced', ...)` further scaled by a hand-picked
  `[0.9, 1.1, 1.0]` per-class multiplier (a manual tweak on top of the
  standard balanced formula — not a free parameter search, just a fixed
  adjustment baked into the config).
- **Post-hoc correction — a fifth independent confirmation of "stacking
  doesn't help"**: the notebook then runs an Optuna search (200 trials) for a
  per-class multiplicative weight applied to the OOF probabilities before
  argmax (`adjusted_probs = oof_probs * [w1, w2, w3]`, `w1..w3` each searched
  in `[0, 1]`). Best weights found: `W1=0.7942, W2=0.7676, W3=0.7811` —
  **CV moves from 0.95057 to 0.95063, a gain of only +0.00006.** This is
  another case of a *tunable* second correction landing near a near-uniform
  ratio (the three weights are within 0.03 of each other, i.e. close to a
  no-op relative rescaling) rather than a large, harmful, or even meaningfully
  helpful shift — consistent with the "tunable second correction finds ~do
  nothing" pattern already documented for Georgy Mamarin's and our own v0.4's
  experiments (2026-07-03 file), now observed a 5th time, in a genuinely
  different model family (neural net, not GBDT).
- **Reproduction cost assessment**: this is a substantially heavier lift than
  anything in our own project so far — a ~300-line custom PyTorch
  architecture with several non-standard components (periodic numeric
  embeddings, NTK-parametrized linears, batched 16-way ensemble-in-one-model,
  EMA, 5-group LR scheduling) vs. our tree-boosting-library calls with mostly
  off-the-shelf configs. It would also be our first GPU-dependent notebook
  (all 7 of ours so far are CPU-only, `enable_gpu: false`).

### Actions Taken

- Pulled the notebook via `kaggle kernels pull` (CLI).
- Downloaded the completed run's output/log via `kaggle kernels output` and
  extracted the actual per-fold and final CV numbers, plus the Optuna
  best-weights result, rather than trusting the notebook title's headline
  score alone.
- Cross-checked the "post-hoc reweighting barely moves the needle" finding
  against the 4 already-documented instances of the same pattern
  (2026-07-03 file: Georgy Mamarin's notebook, Kawamata's `XGB_OvR` ablation,
  redamountassir's TE-HGBC, our own v0.4).

### Resolution

**resolved** — read in full, numbers verified from the actual run log (not
just the title). Genuinely competitive (raw CV 0.95057, essentially tied with
our own v0.7's 0.9502) alternative model family, and independently confirms
both (a) target-encoding numeric features carries real signal, and (b) a
tunable post-hoc per-class reweighting on top of already-balanced training
finds essentially nothing extra to correct. No changes made to our own
pipeline as a result of this investigation.

### Follow-ups

- **A candidate v0.8, if further squeeze work continues**: a from-scratch
  RealMLP port is a legitimate, differently-biased model family (neural net
  vs. tree boosting) that could add real blend diversity, unlike v0.4-v0.6's
  attempts (all tree-based, all too correlated with CatBoost to help). Cost:
  requires GPU (`enable_gpu: true`, `machine_shape: NvidiaTeslaT4`, a first for
  this project) and substantially more implementation effort than any prior
  version. Not started — flagging as the most promising untried direction,
  not committing to build it.
- Multi-resolution target encoding of numerics (exact-value + 2 binned
  granularities, vs. our v0.7's exact-value-only) is a cheap, low-risk
  variant worth trying on our existing HGBC-TE/CatBoost pipeline before
  attempting a full RealMLP port, if incremental squeeze effort continues.
