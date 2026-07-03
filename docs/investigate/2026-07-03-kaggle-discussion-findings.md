# Kaggle Discussion Findings — Playground Series S6E7

External research findings from the competition's Discussion tab, read via the
`kaggle` CLI (`kaggle competitions topics list` / `topic-messages`) per
`kaggle-discussions.md`. Findings here informed interpretation of
`docs/investigate/notebook-runs.md`'s `v0.4-threshold-tuning` entry — see that file
for our own run results; this file tracks the external research itself.

## 717018 — "An almost-exhaustive S6E7 read: why 86% accuracy scores 0.33" (Georgy Mamarin)

### Context

- URL: https://www.kaggle.com/competitions/playground-series-s6e7/discussion/717018
- Prompted by the user asking to investigate this thread while our own v0.4
  threshold-tuning notebook was running, to check for external context on the
  competition's metric behavior and modeling approaches.

### Investigation Checklist

- [x] Read the full thread (original post + all comments) via
      `kaggle competitions topic-messages playground-series-s6e7 717018 -s old -n -1`.
- [x] Cross-check the author's reported numbers against our own results.
- [x] Determine relevance to our currently-running v0.4 experiment.

### Findings

- Confirms the balanced-accuracy floor mechanics we already derived: a majority-class
  model scores ~0.86 plain accuracy but ~0.33 balanced accuracy.
- **Missingness carries no class signal**: an ablation training on nothing but the 13
  `is_missing_*` flags scored balanced accuracy 0.333 (still 0.339 after
  prior-correction) — i.e. right at the floor. Matches our own EDA/feature-importance
  finding that `is_missing_*` columns ranked lowest in every model we've built.
- **What separates the classes**: sleep, activity, and stress carry most of the
  signal; heart rate is near-noise — matches our own EDA and feature-importance
  results across v0.1/v0.3 almost exactly.
- **Leak-free LightGBM baseline**: OOF balanced accuracy 0.878 unweighted; the lever
  that moves it to ~0.95 is **prior-correction on class probabilities** (a post-hoc
  Bayes-rule correction using train/target class-frequency ratios), not more
  modeling. A separate commenter ("ymemo") independently landed at ~0.948 OOF /
  0.9496 LB using plain `class_weight='balanced'` alone — model-side tweaks (HPO,
  features, external data) added ~0 on top.
- **Critical finding — stacking corrections hurts**: the author ran the ablation most
  pipelines here imply — class weights vs. prior-correction vs. both at once. Either
  correction alone lands at ~0.950. **Stacking both together drops to 0.9047**,
  because the second correction over-corrects probabilities the first one already
  shifted.
  - **This is the key piece of context for our own v0.4 result.** Our CatBoost models
    were trained with `auto_class_weights='Balanced'` (a training-time correction),
    and v0.4 then applied a *separate* post-hoc weighted-argmax correction on top —
    structurally the same "stack both" scenario Georgy found hurts. Our own nested
    validation found a flat/negative result (-0.0001), consistent with having
    already spent the available correction during training, with no residual
    imbalance left to fix post-hoc (rather than actively regressing to Georgy's
    0.9047 — plausibly because our post-hoc search still allows w=(1,1), i.e. "do
    nothing," as a candidate, whereas Georgy's pipeline may have applied a fixed,
    non-optional prior-correction on top of already-weighted training).
- **CV-LB tightness**: the author submitted the exact leak-free pipeline and got LB
  0.94988 vs. OOF 0.9498 — no haircut, consistent with our own CV-LB correlation
  findings (v0.1: OOF 0.9389 / LB 0.94051; v0.3 Variant 1: OOF 0.9493 / LB 0.94885).
- A cited thread comment references broccoli beef's mechanism thread (717222, see
  below) as the likely explanation for why ~3 different model families all plateau
  at ~0.950.
- A tangent on validating against weighted logloss instead of balanced accuracy
  (citing a prior competition's 8th-place writeup) was later **retracted by the
  author himself** after measuring it: on this dataset, prior-corrected balanced
  accuracy is already stable fold-to-fold (0.13% relative spread) while weighted
  logloss varies more (1.5%), so the surrogate metric isn't needed here. Not directly
  actionable for us (we already track balanced accuracy directly via our custom
  CatBoost eval metric), but a useful general lesson: measure a metric's fold-spread
  before importing a fix from a different competition/dataset.

### Actions Taken

- Read via `kaggle competitions topic-messages playground-series-s6e7 717018 -s old -n -1`.
- Cross-referenced findings against our own leaderboard.md history.
- Findings folded into interpretation of `docs/investigate/notebook-runs.md`'s
  `v0.4-threshold-tuning` entry (cross-referenced from there, not duplicated).

### Resolution

**resolved** — read in full, cross-checked against our own results, and used to
explain the v0.4 threshold-tuning negative result with a concrete mechanism rather
than treating it as an unexplained null result.

### Follow-ups

- See "Follow-ups" under 717222 below — the two threads' implications are combined
  there since they point at the same underlying conclusion (a synthesis-noise
  ceiling around ~0.95).

## 717222 — "Plausible generation model of the 'original' dataset" (broccoli beef)

### Context

- URL: https://www.kaggle.com/competitions/playground-series-s6e7/discussion/717222
- Referenced from within 717018 as the likely mechanism behind the ~0.95 plateau
  multiple competitors (including us) are observing.

### Investigation Checklist

- [x] Read the full thread via
      `kaggle competitions topic-messages playground-series-s6e7 717222 -s old -n -1`.
- [x] Verify the claimed decision rule's accuracy claim is substantiated in-thread
      (not just asserted) before treating it as reliable.
- [x] Assess relevance to our own feature set / model choices.

### Findings

- The competition's underlying "original" dataset (before Kaggle's synthetic
  generation) is the "College Student Health Behavior Dataset". The thread author
  found that of its many features, only **`sleep_duration`, `stress_level`, and
  `physical_activity_level`** actually drive the `health_condition` target.
- A **hand-built rule** using just these three features (`sleep_duration < 6` and a
  small set of `stress_level`/`physical_activity_level` conditions, thresholded into
  3 buckets) scores **0.9906 balanced accuracy** on the original (pre-synthesis)
  dataset — substantiated with runnable code in-thread, not just asserted.
- A **`DecisionTreeClassifier` fit directly** to `sleep_duration` (at two thresholds,
  ~6 and ~7), `stress_level`, and `physical_activity_level` reaches **100% accuracy**
  on the original dataset, with a depth-4 tree structure:
  ```
  sleep_duration < 6:
      stress_level == high -> unhealthy, else -> at-risk
  sleep_duration >= 6:
      stress_level == low is False -> at-risk
      stress_level == low is True:
          physical_activity_level == active is False -> at-risk
          physical_activity_level == active is True:
              sleep_duration < 7 -> at-risk, else -> fit
  ```
- A mutual-information-over-thresholds scan (vectorized trick shared in-thread)
  independently confirms **sleep_duration ~6-7h** as the strongest cut point — this
  matches our own v0.2 Section B finding that the `at-risk` share of `sleep_duration`
  deciles transitions sharply right around the 5.92-6.37h boundary.
- The creator of the original dataset also published an "enhanced" version (not what
  this competition is based on) with a similar depth-6 tree using two more features
  (`screen_time`, `academic_pressure`, `mental_health_status`) — not directly
  applicable to us since the competition data doesn't include those columns.
- **Competition data is a noised synthesis of this deterministic original**: since
  our actual train/test data is NOT 100%-recoverable by any of our models (best OOF
  ~0.949, not ~0.99+), the gap between the near-perfect original-data rule and our
  ~0.95 ceiling is attributable to noise injected during Kaggle's synthetic-data
  generation, not to us missing an exploitable modeling improvement.
- A follow-up commenter (in 717018, replying to this thread) independently observed
  "three different model families all plateau at ~0.950 balanced accuracy" in their
  own EDA — consistent with our own LightGBM (v0.1: 0.9389) and CatBoost (v0.3:
  0.9491-0.9493) results landing in the same general range, and with our own v0.4
  threshold-tuning finding no further headroom via decision-rule adjustment alone.

### Actions Taken

- Read via `kaggle competitions topic-messages playground-series-s6e7 717222 -s old -n -1`.
- Verified `sleep_duration`, `stress_level`, `physical_activity_level` are already
  our top-3 (or top-4, with `bmi`) features by importance in every model we've built
  (v0.1 LightGBM, v0.3 CatBoost Variants 1 & 2) — consistent with our models already
  implicitly capturing this same signal via native tree splits on the same columns,
  rather than needing an explicit hard-coded rule feature.

### Resolution

**resolved** — a substantiated, code-backed explanation (not speculation) for why our
own models plateau around 0.95. Directly informs expectations for Rung 4.

### Follow-ups

- **Temper Rung 4 (ensemble/squeeze) expectations**: if ~0.95 is a synthesis-noise
  ceiling rather than a modeling gap, an ensemble's realistic upside is small — worth
  attempting (cheap) but budget effort accordingly rather than expecting a large gain.
- **Whether to add the depth-4 rule as an explicit feature**: our models already use
  `sleep_duration`, `stress_level`, `physical_activity_level` as their top features by
  importance in every run so far, which suggests they're already implicitly
  capturing this rule via native tree splits. A hard-coded "rule-predicted class"
  feature is unlikely to add anything a tree-based model can't already construct
  from the raw columns — **not planned as a next step** unless a future experiment
  gives a concrete reason to revisit this assumption.
- If Rung 4 experiments also plateau near ~0.949-0.951 regardless of approach, that
  should be read as further confirmation of the noise-ceiling hypothesis, not as a
  sign to keep searching for a missed lever.
