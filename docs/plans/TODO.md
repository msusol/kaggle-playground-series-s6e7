# TODO

_Archived 2026-07-05 — all per-version plan docs (v0.1 through v0.8) moved to
`docs/plans/archive/`; every phase below is complete and each version's plan
is superseded by its recorded results in `leaderboard.md`/`implementation-plan.md`._

## Phase 0 - Setup
- [x] ~/.kaggle/kaggle.json present
- [x] `zsh scripts/download_data.sh` exits 0 (halts + prints a rules-acceptance URL if
      the competition rules haven't been accepted yet — accept them and re-run)
- [x] Trivial submission scores at the floor (sanity) — all-majority-class
      (`at-risk`) submission scored public LB **0.33333**, exact match to the
      analytic floor (submitted 2026-07-03, submission 54310528)

## Phase 1 - Baseline (v0.1)
- [x] EDA (`notebooks/v0.1-eda.ipynb` — see findings summary + implementation-plan.md)
- [x] Baseline model + 5-fold CV (`notebooks/v0.1-baseline.ipynb` — OOF balanced
      accuracy 0.9389 +/- 0.0012, recorded in leaderboard.md)
- [x] First offline submission; record in leaderboard.md (submitted 2026-07-02,
      public LB 0.94051 vs. OOF 0.9389 — good CV<->LB correlation)

## Phase 2 - Feature engineering (v0.2) — done, negative result
- [x] Training-budget ablation on v0.1 feature set (more rounds / lower LR) — OOF
      0.9290, worse than v0.1's 0.9389
- [x] Root-cause `sleep_duration`'s #2 feature importance (binned/interaction view) —
      genuine non-monotonic signal + strong `stress_level` interaction, confirmed
- [x] Engineered features: missingness indicators, categorical interactions, OOF
      smoothed target encoding — OOF 0.9255, worse than both v0.1 and the budget ablation
- [x] Retrain 5-fold stratified CV; compare OOF vs. v0.1 (0.9389) — v0.1 remains best
- [x] Feature importance check on engineered features (`sleepbin_x_stress` dominant,
      confirms the interaction is informative even though the overall model regressed)
- [x] Candidate submission.csv written (Section A config, budget-only — best of the
      two v0.2 candidates, still worse than v0.1)

## Phase 3 - CatBoost bake-off (v0.3) — done, POSITIVE result, new best model
- [x] Notebook built (`notebooks/v0.3-catboost-bakeoff.ipynb`): Variant 1 (base
      features) + Variant 2 (v0.2 engineered features), same 5-fold split as
      v0.1/v0.2, custom `balanced_accuracy_score`-based eval metric for early
      stopping (directly addresses the v0.2 lesson that LightGBM's early stopping
      tracked `multi_logloss`, not the competition metric)
- [x] Smoke-tested full pipeline on a data sample before the full run
- [x] Full run (5-fold, both variants), live in JupyterLab
- [x] Compare vs. v0.1 (0.9389) and v0.2 (0.9290 / 0.9255); recorded in leaderboard.md
      — **Variant 1 OOF 0.9493 (new best), Variant 2 OOF 0.9491 (tied)**
- [x] Candidate submission.csv written from Variant 1 (best model overall)
- [x] Submitted both variants to Kaggle: Variant 1 LB 0.94885, Variant 2 LB
      **0.94913** (higher despite lower OOF) — confirms the two are genuinely tied
      on both CV and LB, either is a defensible Final Submission pick

## Phase 4 - Threshold tuning (v0.4) — done, negative result, cleanly explained
- [x] Notebook built (`notebooks/v0.4-threshold-tuning.ipynb`): reproduce v0.3
      Variant 2 (engineered features, since it scored higher on LB) capturing OOF
      probabilities this time; weighted-argmax grid search over per-class weights;
      nested validation (fit weights on 4/5 folds, evaluate on the held-out 5th) to
      get an honest improvement estimate rather than trusting the same-data fit
- [x] Smoke-tested full pipeline on a data sample before the full run
- [x] Full run, live in JupyterLab — reproduction PASS (exact match vs. v0.3 Variant 2)
- [x] Recorded nested-validated result in leaderboard.md: **honest improvement
      -0.0001 — no real gain**, full-OOF grid search found plain argmax (w=1,1)
      already optimal
- [x] No new submission.csv (nested validation correctly found no real improvement)
- [x] Investigated Kaggle discussion threads 717018/717222 for context — explains
      the negative result (stacking class-weighting + post-hoc correction
      double-corrects) and suggests ~0.95 may be near the practical ceiling for
      this dataset (noised synthesis of a near-deterministic depth-4 rule)

## Phase 5 - Ensemble (v0.5) — done, negative result, cleanly explained
- [x] Notebook built (`notebooks/v0.5-ensemble.ipynb`): 4-way blend — LightGBM v0.1,
      CatBoost v0.3 Variant 1, CatBoost v0.3 Variant 2 (added after user asked
      whether to include it — different feature view counts as diversity), and a
      new regularized logistic regression baseline (genuine architectural
      diversity); 4-way blend weight search + nested validation; subset blends for
      comparison
- [x] Smoke-tested full pipeline on a data sample before the full run (found and
      fixed a real bug: `LogisticRegression(multi_class=...)` was removed in
      sklearn 1.9 — dropped the arg, `lbfgs` handles multinomial automatically)
- [x] Full run, live in JupyterLab (survived a mid-session disk-full incident that
      silently killed the first attempt — user stopped the stalled kernel, disk
      space recovered, notebook rewritten to 4-way and rerun cleanly)
- [x] All 4 reproductions PASS exact match; LogReg solo 0.8994 (new, no baseline)
- [x] Recorded nested-validated result in leaderboard.md: **honest improvement
      -0.0002 — no real gain**, full-OOF grid search degenerates to 100% weight on
      CatBoost-V1 alone; every subset blend containing CatBoost-V1 caps at its solo
      score
- [x] No new submission.csv (nested validation correctly found no real improvement)
- [x] Stronger confirmation of the synthesis-noise-ceiling hypothesis from Rung 3 —
      even the architecturally-distinct logistic regression adds nothing, arguing
      against "wrong model family" as the cause of the ~0.949-0.951 plateau

## Phase 6+ - Further ideas (implementation-plan.md's Rung 5)
- [x] Consider: does the depth-4 rule from discussion 717222 suggest any concrete
      feature-engineering angle we haven't tried, or is ~0.949 genuinely close to
      the ceiling regardless of approach? — **answered**: investigated discussion
      718258 (Masaya Kawamata, 13 model families incl. XGBoost variants,
      factorization machines, 4 tabular/neural architectures), all landing in the
      same ~0.946-0.951 band with CV/LB gaps within ±0.0011. Confirms the ceiling
      with a much larger sample than before. See
      `docs/investigate/2026-07-03-kaggle-discussion-findings.md`.

## Phase 7 - XGBoost one-vs-rest (v0.6, implementation-plan.md's Rung 6) — done, flat result (per our threshold), highest LB submission
- [x] Notebook built (`notebooks/v0.6-xgboost-ovr.ipynb`): 3 binary XGBoost
      classifiers per fold (one per class, `scale_pos_weight` for imbalance,
      native categorical via `enable_categorical=True`), combined via argmax;
      reproduces CatBoost-V1 alongside for a same-data peg + blend check
- [x] Added `xgboost` to `requirements.txt`
- [x] Environment fix: XGBoost's macOS wheel needs OpenMP. Installed via MacPorts
      (`sudo port install libomp`), then patched the installed `libxgboost.dylib`'s
      rpath from the wheel's hardcoded Homebrew path to the real MacPorts path via
      `install_name_tool` + `codesign` — permanent, no environment variable needed
      (the earlier `DYLD_LIBRARY_PATH` approach doesn't actually work for
      JupyterLab kernels; see `docs/process/xgboost-macos-setup.md`)
- [x] Smoke-tested full pipeline on a data sample before the full run — all checks
      passed; encouraging early signal (nested blend +0.0038 over solo at reduced
      scale/budget, stronger than anything seen in v0.5), not conclusive at this
      scale
- [x] Full run #1 (2 members: XGB-OvR + CatBoost-V1 only) — **XGB-OvR solo 0.9493,
      exact tie with CatBoost-V1**. 2-way blend: nested honest improvement
      **+0.0001** — first-ever positive nested blend result in this project, but
      below the 0.0005 submit threshold. No submission written.
- [x] User caught a scoping gap: XGB-OvR uses engineered features (35) but the
      only comparison peg was CatBoost-V1 (base features, 13) — apples-to-oranges.
      Extended to 3 members, adding CatBoost-V2 (engineered features, same set as
      XGB-OvR) for a cleaner comparison; added pairwise blend breakdown too.
      Smoke-tested the extended pipeline before the full rerun.
- [x] Full run #2 (3 members), live in JupyterLab — solo xgb_ovr 0.9493, catboost_v1
      0.9493, catboost_v2 0.9491. Nested 3-way honest improvement: **+0.0001** —
      same flat/negative-per-threshold result as run #1. Best pairwise same-data
      combo: `xgb_ovr+catboost_v1` at 0.9495 (weights 0.46/0.54).
- [x] Recorded final result in leaderboard.md — no submission written by the
      notebook's own decision logic (correctly, given the flat honest-improvement
      estimate)
- [x] User asked to submit the `xgb_ovr+catboost_v1` blend anyway "for giggles" —
      built directly from the live kernel's in-memory test probabilities (no
      re-run needed) and submitted: **public LB 0.94937** — highest LB in this
      project, but explicitly caveated in leaderboard.md as within the same noise
      band as every other Rung 3-7 result, not a confirmed new best model
- [x] Investigated the actual notebook behind discussion 718258's top `XGB_OvR`
      row (`masayakawamata/s6e7-xgb-ovr-cv-0-95036`) — the author's own ~20-arm
      ablation concludes OvR itself is a no-op vs. multiclass, and confirms
      per-class `scale_pos_weight` (harmful) as a third independent instance of
      the double-correction pitfall. Corroborates rather than contradicts our own
      flat result. Full details in
      `docs/investigate/2026-07-03-kaggle-discussion-findings.md`.

## Phase 8 - HistGradientBoosting + exact-value target encoding (v0.7, implementation-plan.md's Rung 7) — done, POSITIVE result, new best model
- [x] Investigated `redamountassir/ps-s6e7-hgbc-baseline-lb-0-95034-cv-0-95026`
      ("TE-HGBC") — its "weights" turned out to be training-time `sample_weight`
      balancing, not a post-hoc per-class prediction reweighting (a 4th
      independent confirmation of the single-correction finding); the genuinely
      new techniques were exact-value target encoding of numeric features via
      sklearn's native `TargetEncoder`, and `HistGradientBoostingClassifier`
      (sklearn's native GBM, a 4th distinct tree-boosting library in this
      project). Full details in
      `docs/investigate/2026-07-03-kaggle-discussion-findings.md`.
- [x] Notebook built (`notebooks/v0.7-hgbc-te.ipynb`): target-encodes all 13 raw
      features (including exact-value numerics) via `TargetEncoder(cv=5,
      target_type='multiclass')` per fold, feeds into
      `HistGradientBoostingClassifier` with native `class_weight='balanced'` and
      native categorical handling; reproduces CatBoost-V1 alongside for a blend
      check + nested validation. Built with Kaggle-input-first data loading from
      the start, since this one runs on Kaggle's own compute rather than locally.
- [x] Smoke-tested full pipeline on a data sample before running — all checks
      passed; encouraging early signal (HGBC-TE solo 0.9428 vs. CatBoost-V1's
      0.9422 at reduced scale/budget — the first new model to edge out CatBoost
      solo, though not conclusive at this scale)
- [x] Pushed and run on Kaggle publicly (~86 min total; HGBC-TE itself finished
      in ~4.5 min, CatBoost-V1 reproduction peg took the bulk of the time,
      consistent with earlier CatBoost runs on Kaggle in this project)
- [x] **HGBC-TE solo OOF 0.9502 — beats CatBoost-V1 (0.9493) by +0.0009, the
      first genuine non-noise-level improvement across the whole squeeze phase**
      (Rungs 3-6 were all within ±0.0005 of CatBoost-V1). CatBoost-V1
      reproduction PASS (exact match).
- [x] Blend check: nested-validated blend with CatBoost-V1 adds only +0.0002
      over solo — below threshold, not worth the complexity. Decision logic
      correctly submitted the solo HGBC-TE predictions instead.
- [x] Submitted to Kaggle: **public LB 0.95036** (submission 54321699) vs. OOF
      0.9502 — tight correlation, no haircut. **New best LB in this project**,
      beating the previous best (v0.6's noise-level curiosity submission,
      0.94937) by +0.00099, and this one actually clears our honest-improvement
      threshold.
- [x] Recorded in leaderboard.md, implementation-plan.md (Rung 7) — revises the
      "synthesis-noise ceiling" conclusion from Rungs 3-6: the ceiling wasn't at
      ~0.949 after all, a different feature representation (exact-value numeric
      target encoding) found real signal the other levers missed
- [x] **v0.7 (HGBC-TE) is now the best model — v0.3 CatBoost no longer holds
      that spot.**

## Phase 9 - Publish all notebooks publicly on Kaggle — done
- [x] Cleaned up notebooks v0.1-v0.6 for public release (removed internal
      session/doc-path references, fixed local-only data paths to a
      Kaggle-input-first pattern with local fallback, fixed v0.2's broken
      kernelspec that caused a NoSuchKernel error on Kaggle)
- [x] Pushed all 7 notebooks (v0.1-v0.7) publicly to Kaggle; all 7 executed
      cleanly end-to-end and reproduced their recorded OOF/LB numbers exactly
      (v0.2's LightGBM reproductions, v0.5's LightGBM/CatBoost-V1 reproductions,
      v0.6/v0.7's CatBoost-V1 reproductions all PASSed). None of the three
      republish runs (v0.2/v0.5/v0.6) were resubmitted to the competition —
      they reproduce already-recorded scores, not new results.
- [x] Diagnosed and fixed a Kaggle API 409 conflict blocking the v0.2/v0.5/v0.6
      republishes: Kaggle derives the actual public slug from the notebook
      **title**, not the `id` field in kernel-metadata.json — reusing a title
      already claimed by an earlier (even failed/orphaned) push attempt causes
      a genuine slug collision. Fixed by giving each a distinct, previously
      unused title.
- [x] Posted a [Kaggle discussion thread](https://www.kaggle.com/competitions/playground-series-s6e7/discussion/719199)
      summarizing the full 7-notebook research trail (what each version tried,
      OOF/LB table) and linking the GitHub repo (committed + pushed first so
      the repo link reflects current state).
- [x] Committed and pushed all of the above work to GitHub (commit 9c98d5d).

## Phase 10 - RealMLP neural net (v0.8, implementation-plan.md's Rung 8) — done, flat result, highest raw OOF yet
- [x] Investigated `yunsuxiaozi/pss6e7-realmlp-cv-0-95063` — first neural-net
      model family in this project, raw CV 0.95057, essentially tied with v0.7.
      See `docs/investigate/2026-07-05-kaggle-discussion-findings.md`.
- [x] Built `notebooks/v0.8-realmlp.ipynb`: from-scratch PyTorch RealMLP port
      (periodic numeric embeddings, NTK-style linears, 16-way
      ensemble-in-one-model, EMA), our own StratifiedKFold(5), single
      training-time class-weight correction only (source's post-hoc Optuna
      reweighting not reproduced -- already found negligible). Run locally
      (Apple M3 Pro, PyTorch MPS backend), not Kaggle.
- [x] Smoke-tested before the full run; found and fixed a real bug (pandas
      3.0.3's `.astype(str)` produces native `str` dtype not `object`, broke
      categorical/numeric column classification) and cleaned up tqdm progress
      display (per-batch bar was too noisy; moved to fold+epoch bars with
      loss/accuracy shown via postfix instead of interleaved prints).
- [x] Monitored the user's live local run via periodic autosave file checks
      (no CLI/API for a local Jupyter kernel's status) with SMS updates at
      milestones.
- [x] **RealMLP solo OOF: 0.95062 — highest raw OOF of any model in this
      project**, +0.0004 over v0.7 (0.9502). CatBoost-V1 reproduction PASS.
- [x] Blend (82/18 realmlp/catboost_v1) not degenerate to one member --
      genuine diversity signal -- but nested-validated honest improvement
      only +0.0001.
- [x] **Decision: NO REAL IMPROVEMENT** -- +0.0004 raw margin over v0.7 falls
      short of the 0.0005 threshold. No submission written, nothing submitted
      to Kaggle. Recorded in leaderboard.md and implementation-plan.md
      (Rung 8) regardless of the negative-by-threshold outcome.
- [x] **v0.7 (HGBC-TE) remains the best model** -- v0.8 is the closest
      challenger yet and the first non-tree-boosting family to reach parity.
