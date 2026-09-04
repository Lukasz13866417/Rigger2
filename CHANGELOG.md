# Changelog

All meaningful project changes are recorded here.

## Unreleased

### Fixed

- Calibration tutorials no longer inherit the scored-trial mandatory-playback gate or autoplay all
  five severity rungs with disabled controls. Rung, speed, camera, zoom, replay, and Continue
  controls unlock immediately; scored first-pass enforcement remains unchanged. Playback timing
  now starts after the server start acknowledgement, and stalled API requests fail visibly after a
  bounded timeout instead of leaving the interface on `Confirming full viewing…` indefinitely.
- Current-protocol subjective analysis now requires canonical freeze-recorded evidence paths and
  authenticates every downstream-trusted trial, lineage, population, response, viewing, and
  telemetry field before fitting. Duplicate, malformed, unknown, unserved, or inconsistent rows
  fail closed, and outputs must be fresh and outside the active pilot tree.
- Post-label forensics reuses strict observation authentication and remains locked until every
  observed rater completes the blinded pilot, preventing diagnostic disclosure before hidden
  repeats or adaptive judgments.
- Relative-comparator variant selection now uses validation cross-entropy with a deterministic
  tie-break. Held-out labels, audit policy, targets, features, and metrics are excluded from the
  selection decision and are applied or evaluated only afterward.

### Added

- A content-addressed freeze for the active Phase 15 evaluation protocol, covering exact mannequin
  and renderer implementations, camera/playback, rating semantics, anchors/repeats, scheduler and
  randomization, observation schema, 133 immutable files, and append-only observation/playlist
  prefixes. Validation permits suffix growth during collection but rejects asset, prefix, schema,
  or unversioned implementation drift.
- A separate 625-motion Phase 16 perceptual candidate pool: 560 records remain `UNLABELED` and 65
  are `CALIBRATION_ONLY`; all are frozen-protocol renderable, have null subjective preferences, and
  are critic-training-ineligible. It covers clean, subtle, mechanically valid stress, smoothing/
  rigidity, coordination/phase, pelvis/torso, phase-aligned style incoherence, equivalence controls,
  all 18 adversarial CMA outputs, and calibration ladders with preserved lineage, parameters,
  feasibility, and hashes. The adversarial rows retain content-addressed seed/configuration,
  source-run, trajectory, optimization, and spline-coefficient evidence bound to their corpus ID,
  normalization, original/final motions, history, and control-point configuration; no Phase 16
  perceptual CMA was invoked.
- A bounded balanced active-query scheduler for `RATE_SINGLE` and `COMPARE`, with uncertainty,
  repeat, coverage, critic-disagreement/confirmation, adversarial, comparator, and anchor priorities;
  only within-rater explicit hidden repeats drive repeat disagreement, global prior coverage and
  session-local balancing remain separate, and item/pair batch caps are hard. A dummy-only fixture
  selects 12 singles/eight comparisons from 3,026 bounded candidate pairs (558 eligible stimuli)
  without mutating or enqueueing the active pilot.
- Subjective-analysis v3 direct-pair-versus-ordinal evidence with pair rows excluded from the
  ordinal fit, complementing latent intervals/source effects, Davidson/Bradley–Terry estimates,
  repeat/drift/category/time/replay diagnostics, human/synthetic agreement, and underjudged-item
  selection. All new analysis tests use synthetic observations only.
- Frozen-log/served-trial-gated, analysis-only motion forensics with synchronized articulated/
  skeleton and source/variant views, root/foot/contact/pelvis curves, angular velocity/jerk,
  deterministic metrics, and optional temporal critic outputs; it cannot write into the active
  pilot tree and the blinded collection UI does not expose it.
- Variable-rig data infrastructure comprising canonical `SkeletonGraph`, `MotionGraph`, separate
  universal/rig feature streams, helper-aware anatomical topology, coordinate/basis metadata, and
  masked variable-frame/joint batching, plus permutation/padding/helper/basis/unit-axis/count
  invariance tests. No graph/attention model or training path was added.
- Evaluation-only comparator readiness for absolute, frozen-relative, and fine-tuned-relative
  precomputed predictions, with held-out source/mechanism, direct-human, equivalence, adversarial,
  swap-consistency, and Wilson-interval reports. Relative-experiment v3 freezes hashed train/
  validation supervision before held-out labels, policy, features, or metrics are applied, then
  evaluates those held-out partitions only after selection. It exports native forward/swapped
  probabilities and adds a completion/authentication-gated post-label three-ablation command;
  human ties never become equivalence controls. A guarded MotionCritic path records the current
  representation incompatibility and defers instead of adapting labels or inventing an SMPL map.
- Additional 100STYLE deterministic repair stress coverage over 14 cases from seven styles absent
  from the prior production benchmark, preserving successful-specialist-first selection,
  deterministic CMA fallback, clean-reference withholding, and a hard ban on learned/perceptual
  objectives. Every executed case also receives a generic CMA head-to-head; nine comparison-only
  runs are non-selectable and five fallback runs double as the baseline. The
  production-eligible portfolio remains unchanged at 11/12 successes; loop windows remain
  diagnostic-only without cyclic-task metadata, and joint-limit coverage is explicitly skipped
  without trustworthy authored limits.

- A separately versioned Phase 15 mannequin subjective pilot while Phase 14 remains paused and
  immutable: its skeleton render hashes, pause sentinel, served playlist, and raw observations are
  preserved, and fresh-output guards refuse to replace any existing human evidence.
- A neutral articulated, orientation-revealing mannequin evaluator with mandatory first playback
  at the standardized camera, shared-camera A/B toggling, post-playback speed/zoom/view/orbit
  controls, optional skeleton overlay and post-rating difficulty, plus replay, speed, camera,
  zoom, and inspection-time telemetry.
- A visible five-rung tutorial backed by a separate 13-family/65-stimulus calibration bank,
  tracked-render representability checks, and explicit CALIBRATION_ONLY,
  HARD_FEASIBLE_PERCEPTUAL, and EVALUATION_ANCHOR purpose metadata. The revised schedule has
  40 scored stimuli, cross-session anchor and hard-feasible repeats, and 12 response-adaptive
  same-viewport comparisons with enforced global repeat spacing.
- Strict critic-training eligibility that excludes tutorial and anchor stimuli, content- and
  source-window separation between tutorial and scored cohorts, and revised cohort-aware pilot
  analysis for ratings, reliability, inspection controls, difficulty, and defect perceptibility.
  Critic retraining remains disabled until the revised human pilot is reviewed.
- Repository-local Playwright/Brave headless coverage for the calibrated subjective UI, using
  isolated temporary observation and served-playlist files. Real single, same-viewport-toggle,
  and sequential-neutral-gap flows verify blinding, the exact anchored prompt, canvas rendering,
  server-acknowledged playback locks, persistence, and session completion. Submission is guarded
  client-side and committed atomically server-side to reject concurrent duplicate observations.
- Versioned `motionlab.fixed_skeleton_canvas.v2` rendering with a corrected perspective ground
  grid; the Phase 14 manifest was regenerated before collection so all render, stimulus, and
  presented-trial hashes describe the pixels shown. Its four later observations remain bound to
  those exact legacy hashes.
- A 73-pair stratified human audit covering every held-out source/style derivative, all
  perturbation families, equivalence controls, both admissible adversarial pairs, and the five
  confident absolute-critic misrankings first.
- Append-only A/B/equal labels with 1–5 confidence, family-level human/synthetic agreement with
  Wilson intervals, and an audited policy that suspends unverified directional labels as
  `HUMAN_REQUIRED`.
- An exactly swap-equivariant fixed-rig temporal comparator over direct aligned pair features,
  matched frozen and low-learning-rate backbone-finetuned ablations, category-separated metrics,
  and confidence intervals.
- Gate-controlled anchor-relative perceptual CMA using local comparison logits and recentering,
  without assuming a global transitive quality score. The current run is withheld pending human
  labels.

- Satisficing production thresholds with raw metric evidence and lexicographic candidate ordering,
  plus preserved specialized, semantic-CMA, and generic repair portfolios.
- A 265-pair hard-feasible fixed-rig perceptual dataset with explicit CERTAIN, HUMAN_LABELED, and
  UNORDERED categories, equivalence controls, subtle constraint-preserving perturbations, and
  feasibility-screened reuse of the 18-pair adversarial corpus.
- A frozen-GroupNorm residual preference head, pairwise/tie losses, label-supported auxiliary
  dimensions, source/mechanism/adversarial/equivalence/calibration evaluation, and a strict
  pre-optimization ranking gate.
- A localhost A/B/approximately-equal animation labeler with reason tags and prioritized review
  queues, plus gate-controlled iterative trust-region perceptual CMA and a consolidated milestone
  report. The initial held-out-source gate fails, so no perceptual optimization is run.

- Explicit production benchmark speed semantics separating external task targets, measured clean
  window references, inherited parent speeds, declared tolerances, and eligibility; every eligible
  synthetic pair now asserts a feasible clean endpoint.
- Multi-source, three-seed production CMA-ES benchmarks for foot slide, floating contact, and
  joint pop, with deterministic objectives/constraints, clean-only evaluation, oracle projections,
  collateral metrics, and specialized repair comparisons.
- Local DCT/SO(3) jitter corrections, a seven-parameter cutoff/strength/blend/per-joint smoothing
  block, and two-case oracle representability evidence without increasing generic spline density.
- Autonomous task-directed deterministic repair benchmark and CLI commands for production CMA,
  jitter spectral audit, and the closed-loop evaluation.

- Expanded fixed-rig hardening corpus with 12 qualifying 100STYLE sources, 24 clean windows,
  48 heading-equivalent controls, deterministic 50/50 control/defect task sampling, and fixed
  train/validation/test lineage splits.
- Per-family ranking heads and masked within-family ranking loss; no synthetic global quality
  ordering is trained or exposed.
- Full-sham forensic audit with reconstructed clean/sham clips, all learned heads, deterministic
  deltas, motion distances, comparison strips, and shortcut-versus-other-artifact classification.
- Two-stage 4-to-8-control cubic B-spline BIPOP-CMA-ES with selected local SO(3) residuals, root
  translation/yaw, multi-family severity objectives, separately enforced red-team/production
  policies, per-generation state, automatic outcome categories, and adversarial-data eligibility.
- Multi-seed learned optimization matrix for foot slide, jitter, pop, and eligible floating
  contact; complete findings are in `docs/critic_hardening_cma_experiment.md`.
- Exact-residual oracle representability diagnostics with 4/8-control fits, residual-driven local
  knots, exposed-joint energy analysis, unchanged production constraints, and a per-run CMA failure
  classification matrix.
- Exact SO(3) repair-path monotonicity and aligned overlapping-window consistency evaluations.
- Controlled GroupNorm versus per-frame LayerNorm critic ablation, including full held-out,
  sham, ranking, repair-path, window-consistency, and twelve-run CMA red-team comparison.
- Lineage-preserving adversarial-quality packages for all six original bizarre CMA candidates,
  including full critic tensors, deterministic metrics, optimizer trajectories, and explicit
  `source > exploit` preferences without existing-family labels.
- Separate CMA objective policies: learned hard-defect severity for red-team probing and direct
  deterministic target metrics by default in production; learned hard-defect production
  objectives are rejected.

- Consolidated implementation plan.
- Python packaging, CLI, validated YAML configuration, structured logging, and CPU CI.
- Quaternion, rotation-6D, immutable skeleton/motion, NumPy/Torch FK, and versioned NPZ support.
- Strict BVH parsing, explicit coordinate conversion, SLERP resampling, facing canonicalization,
  virtual-marker evaluation, synthetic fixtures, and static skeleton previewing.
- Confidence-aware heel/toe contacts, gait events, continuous phase, cycle candidates, and
  physical-unit sliding, ground, speed, smoothness, and loop-seam metrics.
- Rest-relative retargeting, analytic two-bone IK, seeded root-drift corruption, foot locking with
  bounded pelvis compensation, and non-duplicate-frame loop closure.
- A deterministic contact-authored walk and end-to-end clean/corrupt/repair demo with reports,
  previews, immutable NPZ outputs, and provenance history.
- Cadence, stride-time, stride-length, step-width, bilateral-asymmetry, and confidence-aware
  Euler/swing-twist joint-limit diagnostics.
- Compact diagnostic JSON plus checksummed, motion-linked dense NPZ artifacts.
- A target-reference retarget CLI path and non-identity rest-pose/proportion validation fixture.
- Separate foot-lock reporting for IK target residual and actual post-repair marker slip.
- Lossless forward-metadata adapters covering explicit coordinate transforms, rest/bind/anchor
  distinctions, fixed anatomical semantics, joint flags, modality/confidence masks, and ground
  provenance.
- Immutable source/split/operation lineage with automatic BVH source fingerprints and propagation
  through current transforms, retargeting, corruptions, and repairs.
- An immutable dense corruption contract separating exact interventions, measured symptoms, and
  confidence-weighted functional responsibility, with backward-compatible vertical-slice aliases.
- Measured hard-negative postconditions, verified ordinal chains, four initial defect families,
  and a second foot-slide mechanism reserved for held-out evaluation.
- Checksummed, split-lineage-safe corruption bundles containing compact JSON, clean/corrupted
  motions, pickle-free dense labels, and explicit privileged-contact/neural-input exclusions.
- The full Phase 6 corruption vocabulary: floating contact, smooth held-out joint pops, seeded
  train/held-out jitter, limb-phase shifts, pose/root time warps, root/leg speed mismatch, held-out
  root-velocity seams, pelvis curves, swing-foot clearance loss, authored joint-limit excess, and
  aligned donor-region style blending.
- Bounded two-to-three-defect composition with final-clip remeasurement of every constituent and
  rejection when a later edit erases an earlier defect.
- Leakage-audited fixed-rig neural inputs and separate dense intervention, symptom,
  responsibility, weak joint-defect, clip-defect, preference, and clean reconstruction targets.
- Deterministic 60 fps fixed-window corruption dataset generation with verified ordinal
  preferences, content-addressed compressed NPZ samples, JSONL manifests, take-level split
  lineage, held-out-corruptor isolation, train-only normalization, exact-state resume, checksums,
  rejection/pass-rate summaries, and end-to-end validation.
- `generate-dataset` and `validate-dataset` commands plus BVH `--clean-confidence` assignment.
- Five family/metric-specific observed-severity bins with deterministic bounded parameter search;
  persisted severity is the remeasured symptom delta rather than the chosen curve parameter.
- Matched contact-compensated foot-slide sham controls, explicit no-target-defect and
  approximate-quality-equivalence targets, counterfactual group IDs, and targeted
  cross-mechanism consistency/contrast records.
- Declared primary/collateral metric policies for automatic hard samples, including rejection or
  dense multi-labeling of significant secondary defects.
- Privileged clean-source gait event/side/phase targeting metadata and expanded dataset diagnostics
  for severity population, pass/acceptance rates, rejections, collateral failures, source/event/
  side/mechanism coverage, and sham counts.
- Version 2 fixed-rig sample, normalization, and corruption-dataset persistence contracts.
- Numerical, validation, serialization, CLI, BVH, processing, metric, corruption, repair, and demo
  tests.
