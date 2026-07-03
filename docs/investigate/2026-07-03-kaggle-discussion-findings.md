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
  - **Update 2026-07-03**: confirmed — see `notebook-runs.md`'s `v0.5-ensemble` entry.
    Our own 4-way blend (LightGBM + 2 CatBoost variants + logistic regression)
    plateaued at exactly the solo CatBoost score, no lift from diversity.

## georgymamarin/student-health-risk-why-86-accuracy-scores-0-33 (notebook, companion to 717018)

### Context

- URL: https://www.kaggle.com/code/georgymamarin/student-health-risk-why-86-accuracy-scores-0-33
- This is the actual notebook behind discussion 717018 above (same author). Pulled
  via `kaggle kernels pull georgymamarin/student-health-risk-why-86-accuracy-scores-0-33`
  (CLI, not browser) to read the real code and numbers behind the discussion post's
  prose summary.

### Investigation Checklist

- [x] Pull and read the full notebook via the `kaggle` CLI.
- [x] Extract the exact prior-correction mechanism (code, not just description).
- [x] Cross-check the "stacking corrections hurts" claim's exact numbers.
- [x] Read the cross-competitor comparison table (§10) in full.
- [x] Assess whether anything here changes our own pipeline recommendations.

### Findings

- **The prior-correction mechanism, precisely**: `argmax(oof_proba / class_prior)` —
  divide each row's predicted probabilities by the class's overall frequency in
  train (`pi = np.bincount(y) / len(y)`), then argmax. This is a **fixed, principled,
  zero-tuning** Bayes-rule correction, not a grid-searched free parameter. It is
  meaningfully different from our own v0.4's `weighted_argmax` approach, which
  grid-searched `w[fit]`/`w[unhealthy]` as free parameters — Georgy's correction has
  no parameters to overfit at all.
- **Quantified lift**: unweighted LightGBM plain argmax OOF **0.878** ->
  prior-corrected OOF **0.9498** (a +0.072 lift from the decision rule alone, zero
  new modeling) -> **public LB 0.94988**, matching OOF almost exactly (no haircut).
  An exponent sweep (`argmax(p / prior^b)` for `b` in `[0, 2]`) shows the curve peaks
  near `b=1` (full prior division) and is flat around it — the correction isn't a
  fragile knife-edge optimum.
- **"Stacking corrections hurts" — exact numbers** (all 4 combinations, same OOF
  folds): unweighted+plain-argmax ~0.878, unweighted+prior-correction ~0.9498,
  weighted+plain-argmax ~0.950, **weighted+prior-correction: costs ~0.045** (drops
  to roughly 0.905) — confirms the number our 717018 summary already cited (0.9047),
  now traced to its exact source code. Mechanism: weight-trained probabilities are
  already prior-shifted, so dividing by the prior a second time over-corrects.
- **Nuance on the "double correction" pattern relevant to our own v0.4**: Georgy
  notes "a common pattern in the public pipelines here is weights in training plus a
  *tuned* multiplier search on top; the tuner can walk the second correction back
  toward 'do nothing', but a *fixed* second correction cannot." This is exactly why
  our own v0.4 saw a small flat result (-0.0001) rather than a large drop like
  Georgy's fixed-prior-on-top-of-weights experiment (-0.045): our weighted-argmax
  grid search always included `w=(1,1)` (no-op) as a candidate and correctly found
  it optimal, whereas Georgy's ablation forced the second correction on
  unconditionally. Same underlying mechanism, different magnitude because of a
  tunable vs. fixed second correction.
- **Threshold-scan nuance**: a global mutual-information scan over `sleep_duration`
  cut points finds the strongest split at **exactly 6.0h**, matching broccoli beef's
  top-level `sleep < 6` rule. But the rule's *second* `sleep < 7` split (nested
  inside the low-stress/active branch) is barely visible in the *global* scan (MI
  0.084 vs. 0.132 at the 6h peak) — a branch-conditional split needs a
  conditional/local rescan to show up clearly, not a global one. Matches our own
  v0.2 Section B finding of a sharp `at-risk` share transition near 5.92-6.37h.
- **Cross-competitor comparison table (§10), read in full — stronger evidence than
  our 717018 summary conveyed**: five different approaches by four different people,
  all within ~0.0006 of each other:
  | approach | model family | score |
  |---|---|---|
  | XGBoost + prior-correction (Masaya Kawamata) | gradient-boosted trees | CV 0.94986 |
  | RepLeafGBM + prior-correction (Masaya Kawamata) | GBDT w/ linear leaves | CV 0.94964 |
  | RealMLP + class weights (Sohail Khan) | neural net | CV 0.94972 |
  | LightGBM + prior-correction (this notebook) | gradient-boosted trees | OOF 0.9498 / LB 0.94988 |
  | majority vote over 5 public submissions, **no model trained** (Vadim Irtlach) | ensemble-of-predictions only | LB 0.95025 |
  Four genuinely distinct model families (tree boosting, linear-leaf GBDT, neural
  net) *and* a pure prediction-vote with no model at all, all landing in the same
  ~0.0006 band — this is materially stronger confirmation of the synthesis-noise
  ceiling than "LightGBM and CatBoost converge," since it includes a neural net and
  a model-free ensemble too. Directly corroborates our own v0.5 finding (4-way blend
  including a logistic regression also plateaued at the same score as CatBoost
  alone) from an independent, much larger sample of the public leaderboard.
- **Explicit guidance on further squeeze attempts**: "If you want to climb
  meaningfully above ~0.950, the winning move is unlikely to be a fancier single
  model; historically in Playground it comes from ensembling diversity and luck on
  the private split." Notably, even Vadim Irtlach's pure-vote ensemble (no model
  training at all) only reached 0.95025 — a +0.0004 to +0.0009 edge over the single
  models here, i.e. the *practical* ceiling for "ensembling diversity" appears to be
  on the order of a few ten-thousandths, not a meaningful jump.
- **Surrogate-metric tangent, quantified**: prior-corrected balanced accuracy's
  fold-to-fold relative spread is ~0.13% here, vs. ~1.5% for class-weighted logloss
  (a proxy borrowed from a prior competition's 8th-place writeup) — because even the
  rarest class has ~40,000 rows per fold. Not actionable for us (we already track
  balanced accuracy directly), but confirms the general lesson already noted under
  717018: check a metric's fold-spread before importing a fix from elsewhere.

### Actions Taken

- Pulled the notebook via `kaggle kernels pull` (CLI, not browser automation or
  WebFetch — the notebook is large and rendering it in a browser would have been
  slower and less reliable than getting the raw `.ipynb` source directly).
- Read all 46 cells; extracted code for the prior-correction mechanism, the
  double-correction ablation, and the cross-competitor table above.
- Cross-checked the "costs ~0.045" and "0.94988 LB" numbers against our existing
  717018 summary — consistent, now traced to exact source code and numbers rather
  than a discussion-post prose description.

### Resolution

**resolved** — read in full via the CLI, findings extracted and cross-checked. No
changes needed to our own pipeline: we're already using one of the two valid
"doors" (training-time class weighting via `auto_class_weights='Balanced'`) rather
than stacking both, which is why our own v0.4 saw a small flat result instead of a
large regression. The cross-competitor table and the explicit "diminishing returns
from ensembling" guidance both directly corroborate our own Rung 4 (v0.5) finding.

### Follow-ups

- No new action items for this competition — this notebook confirms and sharpens
  (with exact numbers and code) conclusions we'd already reached independently via
  717018/717222 and our own v0.4/v0.5 experiments. Treat further squeeze attempts
  (Rung 5+) as chasing a ceiling on the order of ~0.0005-0.001, per the
  cross-competitor table, not a meaningful gap.
- See the `vad13irt/ps-s6e7-eda-ensemble` entry below — the table's "majority vote,
  no model trained" row needed a correction after reading its actual source code.

## vad13irt/ps-s6e7-eda-ensemble-lb-0-95075 (notebook, the "majority vote" row from Georgy's table)

### Context

- URL (current revision): https://www.kaggle.com/code/vad13irt/ps-s6e7-eda-ensemble-lb-0-95075
- Georgy's §10 table cited this notebook at an older slug/score
  (`ps-s6e7-eda-ensemble-lb-0-95025`) as "majority vote over 5 public submissions,
  no model trained." The user asked to look closer at this specific row.
- Premise check: the cited slug 404'd on `kaggle kernels pull` — the notebook has
  since been updated (title now reports LB 0.95075, last run 2026-07-03, today).
  Searched `kaggle kernels list --user vad13irt` to find the current slug. Per the
  standing rule to verify AI/external-source claims against ground truth before
  treating them as fact, pulled and read the *current* live version rather than
  trusting the older cited number.

### Investigation Checklist

- [x] Locate the current notebook slug (the cited one is stale) via
      `kaggle kernels list --user vad13irt`.
- [x] Pull and read the actual blending code, not just the table's one-line summary.
- [x] Verify whether "majority vote over 5 submissions" and "no model trained" are
      accurate descriptions of what the code actually does.
- [x] Assess whether this technique is something we could replicate ourselves.

### Findings

- **"Majority vote over 5 public submissions" is not an accurate description of the
  code.** The notebook loads 5 competitors' prediction CSVs (`kosprintr`,
  `yekenot_pl`, `yekenot`, `kirill0212`, `hmnshudhmn24` — each filename-tagged with
  that person's own solo LB score), but the actual blend line only uses 2 of them:
  ```python
  blend = 0.6*kosprintr + 0.4*yekenot  # *yekenot #+ 0*kirill0212 + 0*hmnshudhmn24
  preds = [cols[p] for p in np.argmax(blend.values, axis=1)]
  ```
  `kirill0212` and `hmnshudhmn24` are loaded but multiplied by an explicit `0*` (kept
  as commented-out dead code, apparently from manual experimentation); `yekenot_pl`
  is loaded but never referenced in the blend at all. This is a **weighted
  probability average** (soft blend, then argmax) of exactly 2 files at a
  hand-picked 0.6/0.4 ratio — not a discrete majority vote over 5.
- **"No model trained" is accurate, but the mechanism is different from what it
  suggests.** This notebook does not train anything itself — it reads other
  competitors' *already-computed* prediction probability files from a private
  Kaggle dataset the author assembled (`vad13irt/ps-s7e6-02072026`), presumably
  aggregated from files those competitors shared publicly (Discord, forums, or
  similar). The "no model trained" framing makes it sound like a clever
  training-free technique; in practice it's **entirely dependent on having access to
  other specific individuals' raw output files** — there's no transferable modeling
  or feature-engineering lesson here for our own pipeline.
- **The cited score is stale**: Georgy's table cites 0.95025 from an earlier
  revision; the live notebook (today, 2026-07-03) reports 0.95075 from a different
  2-file blend combination than whatever produced 0.95025. The exact files/weights
  behind either number weren't preserved in the notebook's own history that we could
  see — this number will likely keep moving as the author swaps in newer public
  files.
- **Not replicable by us**: this technique requires having other competitors' raw
  prediction CSVs, which we don't have and have no channel to obtain (we're not
  drawing from the same shared-file pool). It's a "leaderboard blending" meta-tactic
  specific to short synthetic-data Playground competitions where competitors
  informally share raw submission files, not a methodology we can adopt within our
  own single-pipeline effort.

### Actions Taken

- Attempted `kaggle kernels pull vad13irt/ps-s6e7-eda-ensemble-lb-0-95025` — got a
  404, confirming the cited slug was stale.
- Ran `kaggle kernels list --user vad13irt` to find the current slug
  (`ps-s6e7-eda-ensemble-lb-0-95075`), pulled and read it in full (81 cells).
- Read cell 72 (the actual blend code) in full rather than trusting the table row's
  one-line description.

### Resolution

**resolved** — corrected our own record: the table row's "majority vote over 5
submissions, no model trained" is an imprecise gloss over the actual mechanism (a
hand-weighted 2-file probability blend of other competitors' outputs, sourced from
a shared dataset). Doesn't change our own Rung 4 conclusions, since this technique
isn't something our pipeline has access to or could replicate — the cross-
competitor ceiling evidence still stands on the other 4 rows (independently-trained
models), which remain unaffected by this correction.

### Follow-ups

- None — this was a one-off clarification of a single table row at the user's
  request, not a new avenue for our own pipeline.
