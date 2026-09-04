# Implementation status

## Current stage

Inspection is now explicit opt-in: after saving the primary answer, the default controls are
**Next animation** and **Inspect (optional)**. No animation, inspection panel or inspected
question appears unless Inspect is clicked. Continuing creates no inspection amendment;
inspection telemetry starts only on opening. The active path is
`artifacts/phase17_opt_in_inspection_pilot`, with a separately frozen presentation revision.
The previous repeated-view pilot remains intact with one primary answer, its inspection
amendment and eight served/playback events; inherited completed trials are skipped, not relabeled.
The live opt-in collector returns HTTP 200 with the exact frozen HTML. All previous evidence
hashes and copied motion assets match. The new frozen protocol ID is
`notice-protocol:sha256:735ea94a096d8724dd98b3d62a055c295e2a547fe348b367ed6331e94dd75b69`.
Opt-in verification: **283 tests pass in 168.47 seconds**, including six new browser cases
covering hidden inspection, direct continuation without an amendment, explicit opening/timing,
and lossless progress across both earlier protocols and camera versions. Repository-wide Ruff,
format checks and mypy (136 source files) pass. No human labels were submitted by these checks.

Current labeling uses the user-requested **bounded repeated-view noticeability** continuation:
up to three pre-answer plays at 0.85x, 0.90x, 0.95x or 1x. The same localhost endpoint now serves
the latest opt-in-inspection continuation. Its target is explicitly `repeated_view_notice`, not the
one-view `spontaneous_notice` construct. Per-view speed, duration and timestamps are recorded;
the server enforces both the three-play cap and speed-adjusted full-view requirement. The old
single-view collector, frozen manifest and three serving/playback events remain intact, with
zero saved answers at cutover. Tests cover preserved progress, early-answer rejection, invalid
speeds, a rejected fourth play, immediate answering after one play, inspection and reloads under
both camera versions. Calibration and exports do not silently combine the two targets. No
training or learned production optimization is enabled. See `docs/noticeability_pilot.md`.

Historical repeated-view verification: **277 tests passed in 148.45 seconds**, including 27
noticeability tests covering both camera versions. Ruff and mypy (135 package files) passed.
That now-paused collector responded with HTTP 200 and was frozen under
`notice-protocol:sha256:077cc2ea775b4d7942cc51f277dd15666e9a0f4c2d02e72b2a5ebbae83ccf861`.
Every recorded original evidence hash still matches; repeated-view pre-collection analysis and
export contain zero labels and explicitly name the new target.

Phases 0–6, fixed-rig critic hardening, learned spline red-team diagnostics, adaptive search-space
work, deterministic production repair, and the first fixed-rig perceptual-quality experiment are
complete. Deterministic CMA succeeds on all 18 production runs and the autonomous mechanical
closed loop succeeds on 6/6 cases. The residual preference model fails its held-out-source ranking
gate at 41.7%. An exactly symmetric relative-comparator experiment and 73-pair human audit remain
prepared, but their human-label prerequisite is incomplete, so comparator ablations and perceptual
CMA remain correctly unrun.

Phase 15 replaces the initial skeleton pilot with a calibrated neutral-mannequin pilot. The Phase
14 pilot is paused and immutable after four raw observations and five served-playlist events. The
Phase 15 artifact contains 13 five-rung calibration ladders, 40 scored stimuli, eight hidden repeat
presentations, and 12 adaptive same-viewport comparisons. Its completed first session is now
paused and preserved under the Phase 17 noticeability transition described below. The earlier
zero-observation analysis remains a historical pre-collection snapshot, not a claim about the
completed log.

Phase 16 completes the work that is independent of those pending labels: a separate 625-motion
unlabeled/calibration candidate pool, dummy-tested analysis and bounded query scheduling,
post-label-only forensics, variable-rig data contracts/invariance tests, guarded comparator
evaluation, a documented MotionCritic representation deferral, and additional deterministic
100STYLE repair stress testing. Critic retraining, perceptual CMA, and variable-rig graph/attention
model work remain stopped until the completed human-reliability report is reviewed.

The subsequent user-requested camera update is a versioned continuation, not an in-place edit of
the frozen Phase 15 protocol. `phase15_free_camera_pilot` uses collection v4/render v2 with wheel
zoom (0.1x–5x), screen-space panning, unrestricted yaw, and ±89-degree pitch. The mandatory first
viewing and post-rating locks remain intact. Original labels and their exact serving
implementation remain preserved; authenticated completion history skips already submitted trials
without mixing old labels into the new observation log. No human-rating analysis or learning was
run for this camera update.

The cutover preserved 13 submitted scores and 21 served events under the original protocol, and
copied all 129 NPZ assets with identical hashes. Only a pause marker was added to the original
pilot; no original manifest, motion, or log was rewritten. Both protocol freezes validate. The
continuation was served at `http://127.0.0.1:8765`, with protocol ID
`human-evaluation-protocol:sha256:c1b4d0f8f6af65c0734b576bff2e8d92df65b0a254c1b0071e2012554f40a78a`
and render hash
`render:sha256:9304a3c1d2f62892bfa927527569fa6f800c2d85a46942d954a1dc02f2b2d99e`.
The split cohort will require explicit cross-protocol completion accounting before post-label
analysis/forensics; inherited scheduling rows must not be treated as new measurements.

Camera-cutover verification: all 249 repository tests pass, including real-browser wheel/pan/
orbit/lock/persistence checks and authenticated progress-resumption tests. Ruff and formatting
pass repository-wide; mypy passes all 119 package source files. Both original and continuation
protocol freezes validate. This is historical camera-cutover verification; the collector is now
paused and replaced by the separately frozen noticeability protocol.

## Phase 17: spontaneous noticeability transition (human-pilot review stop)

The primary future learned perceptual target is now **P(a human spontaneously notices an
unintended motion problem during one normal-speed presentation)**, not OK-versus-good quality.
The implementation and local pilot are described in `docs/noticeability_pilot.md`.

- Both old Phase 15 collectors are paused. Exactly 30 completed-session observations are
  preserved in place and in `artifacts/phase17_legacy_quality_freeze`: 24 single-animation
  ordinal judgments and six pairwise judgments. Scheduling-only inherited rows are not counted
  again. Scores 1 through 7 have counts **0, 0, 10, 3, 8, 2, 1**. Original question/scale, hashes,
  rater/session/trial, inspection metadata, provenance and timestamps remain unchanged. The four
  older Phase 14 skeleton observations also remain untouched and separate.
- New collection protocol `motionlab.noticeability.v1` asks NO/MAYBE/YES only after one complete
  initial 1x/default-camera viewing. The first answer is committed before replay, speed changes,
  zoom, pan, orbit, alternate views or skeleton inspection unlock. Optional inspectability and
  diagnostic information are separate append-only amendments, never replacements. Interrupted
  first presentations are skipped without a label; saved primary answers survive reloads.
- Style policy B is explicit: human and future critic receive the same intended style/goal.
  Source aliases that reveal adversarial provenance are not shown as style instructions. The
  original mannequin renderer, viewport geometry and both camera-version hashes are retained;
  render strata are not silently pooled for migration. Historical browser environment variation
  cannot be reconstructed and is not claimed to be identical.
- `artifacts/phase17_noticeability_pilot` contains **68 trials / 64 assets**: 24 exact-render
  overlaps (13 original-camera, 11 free-camera), 36 physical-strength ladder samples across
  Neutral/Proud and three mechanisms, four new clean/sham controls and four hidden repeats.
  Fewer than 30 old singles exist, so no overlap labels are invented. A declared 24-hour gap
  unlocks old overlaps at **2026-09-05 23:16:56 Europe/Warsaw**; the 44 other trials are available
  first. Overlaps and known repeated exposures are flagged rather than called naive viewing.
- Population counts are 48 NOTICEABILITY_THRESHOLD, 11 CLEAN_SHAM_CONTROL, three
  CALIBRATION_ONLY and two ADVERSARIAL_CRITIC; DEPLOYMENT_CANDIDATE remains available as a
  separate category. Construction severity is not a human label. Catastrophic calibration-only
  cases are excluded from the direct training export.
- Implemented regularized monotone ordinal calibration, render-separated leave-source-out
  evaluation, confusion/Brier/log-loss reports, per-category bootstrap uncertainty and
  source/style/family breakdowns. Weak proxies are derived, versioned, idempotent/reversible,
  source-linked and gated on adequate support plus held-source predictive improvement.
  Pairwise judgments never become detection labels; direct human evidence outranks proxies.
  The conservative 30-overlap-per-render gate cannot be met by this small pilot alone; that
  limitation is not interpreted as proof that old quality evidence is useless.
- Implemented raw response/false-alarm/miss/repeat/inspection-transition analysis, a modest
  shared-threshold latent detectability model with supported-only session effects, and guarded
  psychometric threshold/JND estimation. Optional interleaved staircases use YES-down/NO-up/
  MAYBE-hold with clean/sham catches. Independent fixed validation plans require supported
  human threshold estimates and a separately prepared render batch.
- Added the minimal GroupNorm fixed-rig clip noticeability head with explicit style/goal context,
  three response probabilities and P(YES) as the primary output. Clip labels are never copied
  across frames; exports retain exactly the displayed cycle window. Detector metrics and
  held-out-validation operating-point helpers are implemented without a default tau=0.5.
  Production's staged contract is deterministic feasibility AND p_notice <= tau, with
  max(0, p_notice - tau) penalty; aggressive red-team probing is separate and inactive.
- **First-stop results:** direct new labels = 0; completed overlap re-labels = 0; weak proxies = 0.
  Calibration curve/usefulness, clean/sham false-alarm rate, obvious misses, repeat reliability,
  family thresholds/JND, NO/MAYBE/YES rates and spontaneous/inspected disagreement are all
  **awaiting human collection**, not zero-error results. Complete the small pilot and review it
  before selecting the next threshold-focused batch or retraining.
- No learned training experiment, perceptual CMA, normalization change or variable-rig
  graph/attention model was run. GroupNorm remains the baseline, LayerNorm an experimental
  artifact, and all prior adversarial-quality pairs, absolute-q and relative-comparator ablations
  are preserved. `train-noticeability-critic` / `evaluate-noticeability-critic` are explicit
  first-stop guards; full training/evaluation orchestration is deliberately deferred, not claimed
  complete. Existing deterministic repair production behavior is unchanged.

Phase 17 verification: **266 repository tests passed in 131.63 seconds**, including 16 new
noticeability cases with real-browser checks under both historical mannequin camera versions.
Ruff passes repository-wide; mypy passes all 131 package files. The active pilot freezes 69 assets
under protocol ID
`notice-protocol:sha256:51657c073faa4ae6fb29983e79c487f44f8f15ec2d7be348fbd488f76bd5a1d7`.
Its loopback endpoint returns HTTP 200. Pre-collection analysis, calibration, zero-proxy migration,
direct-only export and the required first-stop report are saved separately under
`artifacts/phase17_noticeability_precollection`; none claims human performance from dummy tests.

## Implemented behavior

- Python packaging, CLI, validated YAML configuration, structured logging, optional visualization,
  CI configuration, and locked dependencies.
- Quaternion/log-exp/SLERP/6D math, physical-time finite differences, immutable `Skeleton` and
  `MotionClip`, NumPy/Torch FK, rig schemas, and versioned pickle-free NPZ serialization.
- Strict recursive BVH parser with preserved per-joint channel order, ZXY/XYZ order tests, end-site
  metadata, explicit source-axis/handedness/unit conversion, 60 fps SLERP resampling, and facing
  canonicalization.
- Heel/toe marker contacts with hysteresis/confidence, gait events, continuous phase, and cycle
  candidates.
- Physical-unit foot sliding, penetration/floating, speed, SO(3) smoothness, loop seam, cadence,
  stride time/length, step width, bilateral asymmetry, and configured joint-limit metrics.
- Joint-limit results include per-joint validity, authored-limit confidence, local-axis confidence,
  rig coverage, maximum excess, integrated excess, and localized half-open intervals.
- Compact deterministic JSON reports linked by motion ID and SHA-256 to versioned, pickle-free
  dense NPZ arrays. Dense artifact manifests declare every key, shape, and dtype.
- Rest-relative semantic retargeting, two-bone IK, root-drift foot-slide corruption, foot locking,
  and loop closure.
- A target-reference retarget CLI plus non-identity target proportions/rest-orientation validation.
- Foot-lock results distinguish the IK target residual from measured post-repair sole-marker slip.
- A lossless `MotionClip` metadata adapter exposing explicit source-to-canonical coordinate/unit
  transforms, rest/bind/pose-anchor distinctions, anatomical role/side/part assignments, joint
  flags, modality validity/confidence, and ground source/confidence.
- Immutable source, take, split, importer/retargeter, operation, and corruption lineage. Current
  transformations append typed parent-addressed steps only for tagged clips, preserving legacy
  hashes and outputs.
- BVH file import assigns stable source-file fingerprints and split-lineage IDs automatically;
  joints without rotation channels are explicitly rotation-invalid in the adapter.
- Immutable corruption results separate exact `[T,J,channel]` interventions, fixed-part symptoms,
  and confidence-weighted functional responsibility instead of treating generator edits as exact
  causal blame. Legacy corruption accessors remain available.
- Hard-negative preference labels are emitted only after family-specific metrics measurably
  worsen. Ordinal chains are accepted only when post-generation metric values verify the order.
- The versioned catalog contains 20 mechanisms across 13 defect families. Foot slide, joint pop,
  jitter, limb phase, cadence, speed, and loop seam each have a separately declared held-out
  mechanism; pelvis-curve and donor-style edits are explicitly soft rather than automatic
  clean-is-better preferences.
- Physical/temporal corruptors cover penetration and floating, rotation pops, white/correlated
  jitter, unilateral/bilateral phase shifts, pose/root time warps, root/leg speed mismatch,
  rotation/root-velocity seams, pelvis curves, IK-based foot-clearance loss, authored joint-limit
  excess, and aligned donor-region style blending.
- Bounded two-to-three-defect composition admits only training-partition hard negatives and
  remeasures every constituent on the final motion, rejecting interactions that erase a defect.
- Source-reference contacts and inference-available clean/corrupted estimates are stored as
  separate modalities with contact/ground confidence. Detector-derived references are labeled as
  proxies instead of authored truth.
- Split-lineage-safe corruption bundles store checksummed clean/corrupted motion, compact JSON,
  and declared pickle-free dense NPZ labels; metadata-only and privileged-contact fields are
  explicitly excluded from neural inputs.
- A fixed-rig `TrainingSample` contract exposes an audited neural-input whitelist separately from
  intervention, symptom, semantic responsibility, weak joint-defect, clip-defect, ranking, and
  clean-reconstruction targets. Contact inputs are re-estimated on the encoded motion rather than
  copied from privileged source truth.
- `generate-dataset` resamples to 60 fps and creates deterministic 160-frame/32-stride windows,
  five family/metric-specific observed-severity bins found by bounded parameter search, verified
  ordinal preference edges, optional bounded composites,
  content-addressed compressed NPZ samples, sorted JSONL manifests, take-level split registry,
  held-out-corruptor isolation, train-only normalization, checksums, pass/rejection summaries, and
  exact-state idempotent resume. `validate-dataset` verifies the full persisted contract.
- Every automatic hard sample declares and runs a primary/collateral metric policy. Disallowed
  collateral symptoms are rejected; explicitly permitted unavoidable collaterals are added to the
  dense labels and the sample is marked non-single.
- Root-drift foot-slide samples have matched contact-compensated sham controls with explicit
  no-target-defect and approximate-quality-equivalence targets. Clean, corrupt, and sham variants
  share a counterfactual group independent of take split lineage.
- Walking corruptions persist privileged clean-source gait event/side/phase targeting metadata
  while neural contact and phase inputs continue to be inferred from the encoded motion.
- Cross-mechanism consistency records align only the justified per-defect measured-severity target;
  they explicitly do not align whole-motion latent representations.
- Dataset summaries report attempts, final acceptance, primary postcondition pass rate, rejection
  causes, observed severity distributions/bin population, collateral rejection, source/gait/side
  coverage, mechanism coverage, and sham counts without equating acceptance with generalization.
- The real-motion audit imports authoritative 100STYLE frame cuts, qualifies 12 of 14 expanded
  forward walks without changing thresholds, normalizes them to the fixed 23-joint rig and 60 fps,
  and verifies complete
  severity-bin coverage, bilateral contact/event targeting, sham symptom absence, held-out-key
  isolation, persistence validation, and take-level source lineage.
- Every generation attempt is recorded in `audit.jsonl`; aggregate audit tables cover family,
  mechanism, severity bin, gait event, side, source, acceptance/rejection reason, collateral
  rejection, sham result, and actual observed severity.
- Exact deterministic diagnostic summaries are metadata-only inputs to the metric and summary-MLP
  baselines. They are forbidden from the TCN neural array whitelist. Expected speed is fixed from
  the clean source rather than recomputed after corruption.
- Clean windows have two global-heading equivalent controls with explicit zero-defect targets.
  Deterministic epoch sampling draws half controls balanced by role and half defects balanced by
  family, independently of the much larger persisted corruption population.
- `FixedRigFlatTCN` uses a fixed-joint flatten, 128-channel projection, five noncausal residual
  Conv1D blocks with dilations 1/2/4/8/16, no downsampling/attention/graph layers, and frame,
  anatomical-part, clip, severity, per-family ranking, and representation heads. Synthetic order
  supervises only the active family; no global quality scalar exists.
- Training checkpoints include model/optimizer state, next epoch, RNG states, immutable configs,
  and dataset/normalization hashes. Uninterrupted and resumed training match bit-for-bit in the
  exact-resume test.
- Evaluation keeps primary-mechanism and all-labeled-symptom views distinct and reports per-family
  AUROC/AP, temporal and part F1/IoU, severity correlation, calibration, severity bins, shams,
  pairwise ranking, cross-mechanism consistency, mechanism-ID probe, exact metric/MLP baselines,
  saved representations, and concrete failures.
- Sham forensics reconstruct every clean/sham pair across all splits and saves both motions, a
  comparison strip, all head outputs, deterministic before/after/deltas, root/rotation/FK-joint
  distances, and a target-shortcut versus alternate-artifact classification.
- The learned optimizer uses four-control cubic B-spline residuals followed by an eight-control
  refinement. Selected leg/spine/shoulder/elbow joints compose local `R_old * Exp(delta)`, while
  separate smooth channels control root translation and yaw. BIPOP restarts are reproducible.
- Red-team optimization rejects only numerical, absurd-motion, or inference-bound failures and
  never includes deterministic metrics in its objective. Production mode additionally hard-gates
  target speed, penetration, trusted authored limits, declared loops, and task displacement.
- CMA artifacts include original/final and every generation-best motion, spline coefficients,
  CMA mean/sigma/covariance, learned predictions, deterministic evidence, distances, rejection
  counts, automatic outcome category, adversarial eligibility, and comparison preview.
- BVH conversion accepts `--clean-confidence` so imported sources can satisfy the dataset
  cleanliness gate without post-hoc metadata mutation.
- Generated contact-aware walk fixture and deterministic demo exporting clean/corrupted/repaired
  motions, reports, previews, history, and summary.

## Commands run

The Phase 14 commands below are retained as execution history; its collection command now fails
closed because the pause sentinel is present. The Phase 15 preparation commands require fresh
output directories. Its labeling command identifies the active artifact; after collection is
declared sufficient, analysis must write to a fresh directory outside that active pilot tree.

- `uv sync --extra dev --extra visualization`
- `uv run ruff format .`
- `uv run ruff check .`
- `uv run mypy motionlab`
- `uv run pytest --cov=motionlab --cov-report=term-missing`
- `uv run motionlab inspect-bvh tests/fixtures/minimal_zxy.bvh --json`
- `uv run motionlab convert-bvh tests/fixtures/minimal_zxy.bvh --output <tmp>/converted.npz --target-fps 20`
- `uv run motionlab preview <tmp>/converted.npz --output <tmp>/converted.png --frame 2`
- `uv run motionlab run-demo --workspace <tmp> --previews`
- `uv run motionlab metrics <tmp>/repaired.npz --output <tmp>/cli_report.json`
- `uv run motionlab generate-perceptual-pairs artifacts/phase8_critic_hardening/dataset --adversarial-manifest artifacts/phase10_adaptive_optimization/adversarial_quality_corpus/manifest.json --output artifacts/phase12_perceptual_quality/dataset`
- `uv run motionlab train-perceptual-residual artifacts/phase12_perceptual_quality/dataset --source-dataset artifacts/phase8_critic_hardening/dataset --backbone artifacts/phase8_critic_hardening/critic/best.pt --output artifacts/phase12_perceptual_quality/critic`
- `uv run motionlab run-perceptual-cma artifacts/phase12_perceptual_quality/dataset --source-dataset artifacts/phase8_critic_hardening/dataset --checkpoint artifacts/phase12_perceptual_quality/critic/perceptual_residual.pt --evaluation artifacts/phase12_perceptual_quality/critic/evaluation.json --output artifacts/phase12_perceptual_quality/perceptual_cma`
- `uv run motionlab summarize-perceptual-milestone --dataset-summary artifacts/phase12_perceptual_quality/dataset/summary.json --evaluation artifacts/phase12_perceptual_quality/critic/evaluation.json --cma-report artifacts/phase12_perceptual_quality/perceptual_cma/perceptual_cma.json --output artifacts/phase12_perceptual_quality/milestone.json`
- `uv run motionlab prepare-perceptual-human-audit artifacts/phase12_perceptual_quality/dataset --absolute-evaluation artifacts/phase12_perceptual_quality/critic/evaluation.json --absolute-cma-report artifacts/phase12_perceptual_quality/perceptual_cma/perceptual_cma.json --output artifacts/phase13_relative_comparator/human_audit`
- `uv run motionlab analyze-perceptual-human-audit artifacts/phase12_perceptual_quality/dataset --audit-queue artifacts/phase13_relative_comparator/human_audit/human_audit_queue.jsonl --labels artifacts/phase13_relative_comparator/human_audit/human_labels.jsonl --output artifacts/phase13_relative_comparator/human_audit/agreement.json`
- `uv run motionlab run-relative-comparator-experiment artifacts/phase12_perceptual_quality/dataset --source-dataset artifacts/phase8_critic_hardening/dataset --backbone artifacts/phase8_critic_hardening/critic/best.pt --human-labels artifacts/phase13_relative_comparator/human_audit/human_labels.jsonl --human-audit artifacts/phase13_relative_comparator/human_audit/agreement.json --absolute-evaluation artifacts/phase12_perceptual_quality/critic/evaluation.json --output artifacts/phase13_relative_comparator/comparator`
- `uv run motionlab prepare-subjective-pilot artifacts/phase12_perceptual_quality/dataset --audit-queue artifacts/phase13_relative_comparator/human_audit/human_audit_queue.jsonl --output artifacts/phase14_subjective_pilot --seed 9401 --animations 42`
- `uv run motionlab label-perceptual-pairs artifacts/phase12_perceptual_quality/dataset --pilot artifacts/phase14_subjective_pilot/pilot_manifest.json --observations artifacts/phase14_subjective_pilot/raw_observations.jsonl --served-playlist artifacts/phase14_subjective_pilot/served_playlist.jsonl`
- `uv run motionlab analyze-subjective-pilot artifacts/phase12_perceptual_quality/dataset --pilot artifacts/phase14_subjective_pilot/pilot_manifest.json --observations artifacts/phase14_subjective_pilot/raw_observations.jsonl --output artifacts/phase14_subjective_pilot/analysis`
- `MOTIONLAB_REQUIRE_E2E_BROWSER=1 uv run pytest -q tests/test_subjective_protocol.py tests/test_subjective_ui.py`
- `uv run motionlab prepare-subjective-calibration-bank artifacts/phase8_critic_hardening/dataset --output <fresh-calibration-bank-directory> --seed 9501`
- `uv run motionlab prepare-mannequin-subjective-pilot artifacts/phase12_perceptual_quality/dataset --audit-queue artifacts/phase13_relative_comparator/human_audit/human_audit_queue.jsonl --corruption-dataset artifacts/phase8_critic_hardening/dataset --calibration-bank <fresh-calibration-bank-directory>/calibration_bank.json --output <fresh-pilot-directory> --seed 9502 --comparisons 12`
- `uv run motionlab label-perceptual-pairs artifacts/phase12_perceptual_quality/dataset --pilot artifacts/phase15_mannequin_pilot/pilot_manifest.json --observations artifacts/phase15_mannequin_pilot/raw_observations.jsonl --served-playlist artifacts/phase15_mannequin_pilot/served_playlist.jsonl`
- `uv run motionlab analyze-subjective-pilot artifacts/phase12_perceptual_quality/dataset --pilot artifacts/phase15_mannequin_pilot/pilot_manifest.json --observations artifacts/phase15_mannequin_pilot/raw_observations.jsonl --protocol-freeze artifacts/phase15_mannequin_pilot/protocol_freeze.json --served-playlist artifacts/phase15_mannequin_pilot/served_playlist.jsonl --output <fresh-human-analysis-directory-outside-active-pilot>`
- `uv run motionlab validate-active-evaluation-protocol artifacts/phase15_mannequin_pilot/protocol_freeze.json`
- `uv run motionlab freeze-active-evaluation-protocol --help`
- `uv run motionlab validate-active-evaluation-protocol --help`
- `uv run motionlab generate-unlabeled-perceptual-candidates --help`
- `uv run motionlab schedule-active-human-queries --help`
- `uv run motionlab build-post-label-forensics --help`
- `uv run motionlab evaluate-comparator-readiness --help`
- `uv run motionlab evaluate-post-label-comparator-ablations --help`
- `uv run motionlab evaluate-motioncritic-baseline --help`
- `uv run motionlab run-deterministic-repair-stress --help`
- `uv run pytest -q tests/test_calibration_bank.py tests/test_revised_subjective_analysis.py tests/test_subjective_protocol.py`
- `MOTIONLAB_REQUIRE_E2E_BROWSER=1 uv run pytest -q tests/test_subjective_ui.py`
- `uv run motionlab corruption-catalog`
- `uv run motionlab corrupt <lineaged.npz> --output <tmp>/corrupted.npz --artifact-dir <tmp>/example`
- `uv run motionlab generate-dataset <lineaged.npz> --output <tmp>/dataset`
- `uv run motionlab validate-dataset <tmp>/dataset`
- `uv run motionlab audit-100style data/100style --output artifacts/fixed_rig_tcn_experiment_v6/real_audit --maximum-windows-per-source 1`
- `uv run motionlab tiny-overfit-critic artifacts/fixed_rig_tcn_experiment_v6/real_audit/dataset --output artifacts/fixed_rig_tcn_experiment_v6/tiny_overfit_final`
- `uv run motionlab train-fixed-rig-critic artifacts/fixed_rig_tcn_experiment_v6/real_audit/dataset --output artifacts/fixed_rig_tcn_experiment_v6/training --epochs 30 --learning-rate 0.001`
- `uv run motionlab evaluate-fixed-rig-critic artifacts/fixed_rig_tcn_experiment_v6/real_audit/dataset --checkpoint artifacts/fixed_rig_tcn_experiment_v6/training/best.pt --output artifacts/fixed_rig_tcn_experiment_v6/evaluation_v2`
- `uv build`

## Test results

- Ruff: passed.
- mypy: passed for the `motionlab` package using Python 3.11 language semantics.
- pytest: 99 passed in 46.50 seconds.
- Coverage: 86% across 5,748 executable statements.
- Phase 6 deterministic generation, observed-bin targeting, sham/counterfactual/consistency
  records, collateral rejection/multi-labeling, gait metadata, exact resume, held-out isolation,
  normalization scope, leakage checks, composition, and CLI validation tests: passed.
- CLI BVH conversion/validation and PNG preview smoke tests: passed.
- Deterministic demo: foot slip reduced from 7.9454 cm to 0.4374 cm (94.50%) with maximum IK target
  residual `2.01e-16 m`; previous motion IDs and numerical results are unchanged.
- Compact demo reports are 31–32 KB with separate 9–10 KB dense artifacts; the previous combined
  JSON reports were approximately 88–89 KB.
- Real-data gate: passed on `ArmsBySide_FW`, `Neutral_FW`, `Old_FW`, and `Proud_FW`; 255 samples
  comprise 4 clean, 233 hard-corruption, and 18 sham records. Data-split counts are 80 train, 39
  validation, 41 known-mechanism test, and 95 held-out-corruptor. All 12 mechanism keys populate
  all five bins globally, accepted samples remain inside bins, and contact targeting is balanced
  across 47 left and 49 right audit records.
- Real corruption production: 233/240 accepted. Explicit rejects are two root-drift collateral
  failures, three per-source root-velocity seam bin misses, and two per-source leg-pose speed bin
  misses. No acceptance thresholds were loosened.
- Tiny overfit: passed on 18 deliberately selected clean/corrupt/sham samples with 1.000 clip exact
  match, 0.914 temporal F1, 0.920 severity-order Spearman, and loss 3.759 to 0.104.
- Actual training: epoch 1 train/validation loss 3.234/11.628; selected epoch 18 is 0.420/8.217;
  stopped at epoch 26 with 0.224/9.337. The widening split is recorded as overfit/source shift.
- Known primary-mechanism AUROC/AP: floating 1.000/1.000, foot 0.761/0.287, penetration
  0.733/0.519, jitter 0.967/0.760, pop 0.833/0.644, seam 0.578/0.327, speed 1.000/1.000.
- Held-out primary-mechanism AUROC/AP: foot 0.800/0.630, jitter 1.000/1.000, pop 0.800/0.728,
  seam 0.700/0.282, speed 0.863/0.819. Penetration/floating have no second primary mechanism.
- Pairwise severity ranking is 0.400 over 55 pairs. Most temporal and part localization scores are
  poor; held-out temporal F1 is 0.000 foot, 0.109 jitter, 0.103 pop, 0.000 seam, and 0.000 speed.
- Test sham false-positive rate is 1.000 (5/5). The mechanism-ID linear probe is 0.086 versus
  0.143 chance, so no linearly decodable mechanism shortcut is demonstrated.
- Exact-metric/summary-MLP/TCN AUROC comparisons show TCN gains for floating, jitter, and pop;
  mixed foot-slide value; and no advantage for penetration, loop seam, or target-speed mismatch.
  Full values and failure IDs are in `docs/fixed_rig_tcn_experiment.md` and the evaluation JSON.

## Critic hardening and CMA results

- Expanded gate: 12 qualified sources (7 train, 2 validation, 3 test), 24 windows, 24 clean,
  48 equivalent, 114 sham, and 1,388 hard-corruption samples. `GracefulArms` and `LookUp` failed
  the unchanged 2 cm penetration gate and are excluded. All 1,574 samples validate.
- Training selected epoch 9 at train/validation loss 0.757/0.881 and stopped at epoch 17. A
  30-sample memorization stress gate reached temporal F1 0.947 and family-rank rho 0.996 but
  failed exact clip match at 0.933 due loop/speed cross-firing; extra epochs did not fix it.
- Known-source AUROC/AP is 0.996/0.985 floating, 0.888/0.513 foot, 1.000/1.000 penetration,
  jitter, pop, and speed, and 0.976/0.909 seam. Held-out-mechanism AUROC/AP is 0.918/0.773 foot,
  1.000/1.000 jitter and pop, 0.866/0.440 seam, and 0.948/0.882 speed.
- Within-family pair ordering is 0.636 over 1,210 edges. Pop is reversed (0.092 pair accuracy;
  held-out rho -0.978) and speed is weak (0.450; held-out rho -0.899), so detection success is
  not misreported as severity success.
- Target-head sham false positives are 15/114 (0.132); any-head false positives are 33/114
  (0.289). Thirteen target positives have no measured alternate artifact and two have material
  smoothness degradation. Every sham has complete forensic artifacts.
- The CMA matrix ran foot slide, jitter, pop, and floating for three seeds in red-team and
  production modes (24 runs). Six red-team outputs are category-3 adversarial failures, and 18
  runs are category 4. No category-1 production repair was found.
- Foot-slide red-team severity fell from 0.0572 to 0.00035-0.00059; a representative reduced slip
  from 3.95 to 1.75 cm but increased penetration from 1.02 to 12.80 cm. Floating severity fell
  from 0.228 to 0.0093-0.0137 by collapsing support/gait behavior. Jitter/pop severity heads began
  near zero despite high defect probability and provided no useful repair signal.
- All 12 production runs returned no feasible improvement under the hard constraints. The chosen
  windows differed from the full-take speed command by 0.135-0.163 m/s, and the bounded runs did
  not enter the 0.05 m/s speed band while satisfying the other constraints.
- Representative overlay inspection confirms coordinated whole-body/root displacement in the
  adversarial foot/floating solutions and near-exact overlap in numerical sham round trips.

Full evidence is in `docs/critic_hardening_cma_experiment.md` and
`artifacts/phase8_critic_hardening`.

## CMA failure decomposition

- Exact clean residuals were bounded-least-squares projected into the same exposed channels and
  4/8-control spline bases used by CMA. Eight controls recover floating but only 47.9% of the
  foot-slide metric gap and effectively none of the derivative-sensitive jitter/pop gaps.
- Residual-driven knots recover 95.8% foot, 92.6% jitter, and 100% pop without expanding joints.
  Rotation outside exposed joints is only `3.8e-7` deg RMS at most, so joint expansion is not
  justified. The localized fits diagnose temporal bandwidth and do not redefine the current CMA
  search space.
- All oracle candidates fail the unchanged production speed constraint because the chosen windows
  already miss their full-take command by 0.135--0.163 m/s. No case is both current-space
  representable and production-valid, so the conditional oracle-objective CMA test has zero
  eligible cases and optimizer adequacy remains unidentified.
- GroupNorm repair paths are reversed for foot slide and jitter (0/20 adjacent steps), nearly
  reversed for pop (2/20), and mostly correct for floating (18/20). This independently confirms a
  critic-landscape problem.
- The controlled LayerNorm model gives perfect monotonicity for all four paths and bit-identical
  aligned interior-window outputs. It improves all-pair ordering from 0.636 to 0.841, but sham
  target false positives rise from 0.132 to 0.500 and bizarre red-team outcomes rise from 6/12 to
  12/12. It is not adopted as a production critic.
- All six original exploits are persisted as family-unassigned adversarial-quality pairs with full
  tensors, metrics, trajectories, overlays, and `source > exploit` labels. Inspection clusters
  them around strange root compensation, coordination/anatomy distortion, and unusual amplitude.
- Production CMA now defaults to direct deterministic family metrics and refuses learned
  hard-defect heads as production objectives. Learned severity remains available in red-team mode;
  future learned production objectives are reserved for residual perceptual concepts.
- Final classification across the 24 original failed runs is 18 `SEARCH SPACE`, 3
  `CRITIC LANDSCAPE`, 3 `PRODUCTION CONSTRAINT`, 0 `OPTIMIZER`, and 0 `MIXED/UNKNOWN`.

Full evidence is in `docs/cma_failure_decomposition.md` and
`artifacts/phase9_cma_diagnostics`.

## Adaptive search-space diagnostics

- The guaranteed-representable inverse suite passes 3/3 seeds at each of 10, 25, 50, and 100
  active spline dimensions. Median evaluations to the `1e-6` target are 461, 1,282, 2,568, and
  6,347 respectively. CMA-ES is retained.
- Oracle residual energy is measured by joint, time, and scalar rotation/root channel. Coarse
  whole-clip, medium local, and fine per-channel splines improve exact-clean representability from
  floating only (1/4) to floating, foot slide, and joint pop (3/4).
- Joint jitter retains 2.54% residual energy after 684 adaptive parameters and remains outside the
  derivative-sensitive clean target, so it is not reported as representable.
- Contact/IK, speed/cadence, loop-seam, generic SO(3), and adaptive spline blocks share one named,
  bounded interface. CMA can optimize any block or a concatenation.
- Production active regions use deterministic intervention/event localization, optional critic
  symptom fallback, temporal context, and parent/child joint halos.
- The three prior production-constraint failures are one floating-contact target across three
  seeds. Clean, exact interpolation, and the best oracle projection all fail the inherited
  full-take speed command, so the cause is `inconsistent_target_constraints`.
- Production CMA is not rerun because 0/4 oracle endpoints are production-feasible. Learned hard
  heads are absent from the production objective.
- The adversarial-quality corpus now contains 18 family-neutral `source > exploit` pairs: six
  prior GroupNorm cases and 12 manually reviewed LayerNorm global-pose distortions, all with full
  tensors, metrics, trajectories, labels, and lineage.
- GroupNorm and LayerNorm remain experimental artifacts. LayerNorm is not promoted. RMSNorm,
  normalization-free residual blocks, and LayerNorm with adversarial/sham hardening are deferred.

Full evidence is in `docs/adaptive_optimization_diagnostics.md` and
`artifacts/phase10_adaptive_optimization`.

## Deterministic production closed loop

- Synthetic production benchmarks now distinguish external `task_target_speed`, measured
  `clean_window_reference_speed`, inherited `parent_clip_speed`, and the actual `target_speed`
  plus tolerance/source. Without an external task command, the measured clean-window speed is the
  target. An external mismatch makes the pair ineligible.
- The invariant “every eligible synthetic production benchmark has a feasible clean endpoint” is
  asserted and tested. Across 353 train-partition hard-corruption records in the three production
  families, corrected eligibility is 353/353 versus 72/353 under inherited parent targets. The
  correction newly admits 281 cases: 91 foot-slide, 95 floating-contact, and 95 joint-pop.
- Production deterministic CMA-ES uses two distinct clips per family and three seeds per case.
  All 18/18 runs satisfy production constraints and the target-reduction success criterion. Median
  target metrics are 5.848→1.866 cm foot slide, 4.792→0.000 cm floating contact, and
  10,605→5,350 rad/s³ localized pop jerk. Median evaluations/runtime are 92/1.36 s,
  138.5/2.62 s, and 1,202/19.17 s respectively.
- Clean distance is evaluation-only. Median root RMS is 0.0379 m for foot and 0.0574 m for
  floating; median local-rotation RMS is 3.924° for pop. All deterministic before/after and
  collateral deltas are persisted. The floating result exposes metric overcorrection: CMA reaches
  zero while the feasible clean reference/oracle retains a 4.688 cm clip-wide maximum.
- The existing foot-lock/IK repair reaches a 2.021 cm median and beats generic CMA in 4/6 paired
  trials. Specialized local smoothing reaches 58.18 rad/s³ for pop versus 5,350 for generic CMA.
  Specialized deterministic repair remains preferred wherever it exists.
- A local DCT tangent block composed through SO(3) exactly represents two broadband jitter cases
  over their three affected joints. A 95% projection uses 507/514 coefficients and leaves
  0.0618°/0.0484° clean rotation RMS; the full 918-mode basis reaches approximately `1e-6`°.
  Generic spline density was not increased.
- The seven-parameter production smoothing block exposes cutoff, strength, blend-in/out, and
  per-joint strength. Its oracle grid reduces localized jerk p95 from 6,342→1,049 and
  4,963→192 rad/s³, with 0.149°/0.092° clean rotation RMS.
- The autonomous task-directed benchmark removes synthetic labels and clean lineage, selects a
  permitted block from the declared deterministic task objective, localizes it from inference
  diagnostics, and calls an optimizer whose contract has no clean argument. It succeeds on 6/6
  cases (two per family); diagnostic-only recommendations also agree on all six. Median target
  reductions are 3.783 cm foot, 2.788 cm floating, and 5,028 rad/s³ pop.
- CMA-ES remains the optimizer. GroupNorm remains the critic baseline, LayerNorm remains an
  experimental artifact, all adversarial-quality pairs remain preserved, and learned hard-defect
  heads are not production objectives. Normalization and variable-rig graph/attention work were
  not reopened.
- Final verification passes: Ruff, mypy across 91 package files, CLI command discovery, and
  115 pytest tests in 58.02 seconds.

Full evidence is in `docs/deterministic_closed_loop.md` and
`artifacts/phase11_deterministic_closed_loop`.

## Fixed-rig perceptual-quality experiment

- Deterministic repair metrics now have explicit satisficing bands. Reports retain raw values and
  per-constraint satisfied flags; candidate ordering first rejects hard-invalid results, then
  prefers all-threshold-satisfied results, then uses perceptual score, collateral change, and edit
  magnitude.
- Specialized deterministic, semantic parameter-block CMA, and generic adaptive SO(3) CMA methods
  remain in each family portfolio. Generic CMA is a fallback and red-team instrument, not the
  assumed default winner.
- The fixed-rig 100STYLE generator creates 265 pairs whose two sides pass the same production
  constraints: 156 CERTAIN and 109 UNORDERED, including 22 exact equivalence controls. There are
  currently zero HUMAN_LABELED pairs. One generated candidate and two clean windows were excluded
  rather than relaxing constraints.
- Constraint-preserving mechanisms cover excessive rigidification/smoothing, upper-body phase,
  bilateral timing, torso counter-rotation, phase-aligned style inconsistency, a conservative
  weight-transfer proxy, and global-origin equivalence. Subjectively ambiguous constructions carry
  no automatic preference.
- All 18 preserved adversarial-quality pairs are audited. Only 2 satisfy production constraints on
  both sides and enter preference training/evaluation with `adversarial_origin=true`; the other 16
  remain red-team-only evidence.
- The current GroupNorm TCN is loaded unchanged and frozen. A separate residual head learns
  `q(A)-q(B)` with directional and approximately-equal targets. Naturalness, coordination, and
  rigidity/smoothing auxiliaries activate only where reason tags support them; style and
  weight-transfer heads are omitted for lack of labels.
- Final directional accuracy is 41.7% on 24 held-out-source pairs, 100% on 12 held-out-mechanism
  pairs, and 100% on the 2 hard-feasible adversarial pairs. Combined independent accuracy is 63.2%
  over 38 pairs. Held-out-source ECE is 0.584, showing severe overconfidence rather than a marginal
  threshold miss.
- Equivalent-control false preference is 0/22. Twenty-nine held-out subtle pairs are UNORDERED, so
  no unsupported accuracy is reported. Human agreement is unavailable until labels exist.
- The 65% held-out-source/mechanism gate fails. The perceptual-CMA artifact records
  `cma_es_invoked=false`; success and exploit rates remain null rather than fabricated. Five
  highest-confidence held-out misrankings are persisted for review and there are no claimed
  successful refinements.
- The dormant gated optimizer uses three seeds, bounded local upper-body SO(3) edits, iterative
  optimize/evaluate/recenter trust regions, and deterministic acceptance at every round. It cannot
  run while the critic gate fails.
- The phase-12 localhost A/B labeler and its legacy JSONL readers remain reproducible artifacts,
  but the interactive split-screen workflow has since been retired in favor of the calibrated
  hybrid protocol below. Legacy records are not silently rewritten into the new raw schema.
- Final verification passes: Ruff, mypy across 101 package files, CLI discovery for all five new
  commands, 121 pytest tests in 60.28 seconds, package build, and the localhost labeler smoke test.
- The decision is to improve perceptual supervision and the fixed-rig critic next. Variable-rig
  graph/attention and a controller/ranker remain deferred.

Full evidence is in `docs/fixed_rig_perceptual_quality.md` and
`artifacts/phase12_perceptual_quality/milestone.json`.

## Fixed-rig relative-comparator experiment

- A 73-pair audit queue contains all 71 examples from the three held-out source/styles and both
  hard-feasible adversarial pairs. The five most confident absolute-critic misrankings are first.
  Coverage includes all nine perturbation families, 29 subtle, 18 medium, 20 clear, and six exact
  equivalence cases.
- The phase-13 label schema added optional 1–5 confidence and audit-manifest filtering. Its
  append-only parser remains for backward compatibility, while new collection uses the richer
  subjective-observation schema below.
- Human-versus-synthetic agreement is reported per family with a 95% Wilson interval. Current
  agreement is null for every family because 0/73 required human judgments exist. No synthetic
  label is counted as human evidence.
- Until the audit is complete, every directional family is explicitly `HUMAN_REQUIRED`; only
  exact world-origin equivalence remains `CERTAIN`. After audit completion, directional automatic
  supervision requires at least 70% measured human agreement by family.
- The relative comparator uses one shared temporal TCN encoder and ordered per-frame features
  `[EA, EB, EB-EA, abs(EB-EA), EA*EB]`. A shared ordered head is evaluated in both directions and
  antisymmetrized; a separate equality head receives symmetric features. It emits A-better,
  B-better, and approximately-equal probabilities.
- Swapping A and B exactly exchanges the directional probabilities and preserves equal. The
  explicit test passes at `1e-7` tolerance.
- Matched ablations preserve `absolute_q`, compare a fully frozen existing GroupNorm encoder with
  end-to-end temporal-encoder fine-tuning, and use learning rates `2e-3` for the comparator versus
  `2e-5` for the fine-tuned backbone. Normalization and initialization seed are identical.
- Evaluation is implemented separately for held-out source, held-out mechanism, human labels,
  audited synthetic CERTAIN labels, equivalence controls, adversarial pairs, swap consistency,
  perturbation family, human confidence, and reason tag. Accuracy confidence intervals are
  retained; no combined summary score is produced.
- Frozen and fine-tuned result fields are `not_run`, not zero: the human-audit prerequisite is
  incomplete. The primary 65% held-out-source gate cannot yet be evaluated.
- MotionCritic is deferred. Its public checkpoint expects 60 frames of 24 SMPL local axis-angle
  joints plus root XYZ, while the current rig has 23 joints and different rest frames. A defensible
  comparison needs explicit SMPL retarget/rest conversion and separately licensed body assets;
  zero-filling or renaming joints was rejected. No human-labeled pair exists to score yet.
- Anchor-relative CMA is implemented without a global quality scalar: each candidate is compared
  with the current anchor, accepted only when preferred and deterministically feasible, then used
  as the next anchor. The current report records `cma_es_invoked=false`. Generation-local
  Bradley–Terry fitting remains an evidence-triggered optional fallback and was not used.
- `absolute_q`, `relative_comparator_frozen`, and `relative_comparator_finetuned` remain explicit
  ablations. Graph/attention has not started.
- Final verification passes: Ruff, mypy across 105 package files, 125 pytest tests in 60.04
  seconds, CLI discovery, package build, audit-queue/UI smoke tests, and structured no-run gates.

Full evidence is in `docs/relative_comparator_experiment.md` and
`artifacts/phase13_relative_comparator`.

## Phase 14 calibrated hybrid skeleton pilot (paused)

- This is the preserved historical pilot, not the active collection target. Collection was paused
  before substantial labeling by `PILOT_PAUSED.json`. Its immutable evidence comprises four raw
  observations and five served-playlist events; the pause guard rejects both new server startup and
  requests reaching an already-running server.

- The side-by-side-only interface is retired. The primary task is now one blinded, full-size
  animation with the exact style-independent naturalness/internal-coherence question and the
  declared, fully anchored, unselected 1–7 ordinal scale. A separate style-adherence task is
  supported but is not scheduled without an explicit external style target.
- Responses remain locked until every required motion completes a full normal-speed playback.
  Replays, response time, 1–5 optional confidence, perceptual reasons, and an optional marked time
  interval are retained.
- Rendering is fixed and hashed under `motionlab.fixed_skeleton_canvas.v2`: the same fixed-rig
  skeleton, 960x640 viewport, orthographic 25-degree camera, corrected perspective ground grid,
  contrast treatment, 60 Hz, unit playback rate, and two detected gait cycles. The zero-observation
  pilot was regenerated after the grid correction, so every render/stimulus/trial hash identifies
  the pixels now shown. Single trials no longer compute per-frame fit-to-viewport framing.
- Browser payloads contain opaque stimulus/trial identifiers and render data only. Source/style,
  variant, mechanism/family, expected quality, repeat group, anchor status, and selection reason
  remain hidden in the server-side manifest and append-only raw record.
- The seed-9401 pilot contains 42 unique animations, 50 single-stimulus presentations including
  eight hidden repeats, and 12 targeted matched-source comparisons. Each of two sessions has 31
  base trials: 25 singles plus three same-viewport toggle and three sequential-neutral-gap
  comparisons.
- Six hidden anchor candidates span expected poor-to-excellent quality and occur in every session.
  Anchors estimate session offset, threshold scale, drift, and consistency; they are explicitly not
  treated as ground-truth score 7. Within-session exact repeats have at least 12 intervening trials.
- Constrained randomized scheduling distributes source, family, expected quality, and side class,
  prevents avoidable local runs, balances comparison modes, randomizes A/B presentation, and
  records both the seed and exact served playlist. Since the authored unilateral audit edits are
  left-sided, alternating trials apply a declared sagittal reflection; this balances presented
  left/right variants and is retained in the per-trial render hash and exact-repeat definition.
- A capped adaptive tail alternates informative single repeats with matched same-source comparison
  repeats. It prioritizes wide uncertainty, within-rater repeat disagreement, model-human errors,
  close scores, optimization outputs, and underrepresented source/family cells without generating
  an all-pairs matrix.
- Same-viewport comparisons keep one full-size viewport and switch at the same frame/phase with A,
  B, or Space. Sequential trials play A, a neutral interval, then B. Outcomes distinguish A better,
  B better, effectively equal, and not sure; display order is separately retained and canonicalized.
- Every submission appends an immutable `motionlab.subjective_observation.v1` JSONL event containing
  stimulus/source/variant, render hash, rater/session/trial/seed/time, task and response, order/mode,
  response time, replays, repeat/anchor fields, confidence, reasons, interval, and view settings.
- The analysis jointly fits naturalness ordinal judgments and comparisons with source baseline plus
  source-relative item effects, per-session offset/threshold scale, per-rater inconsistency, and a
  Davidson-style four-outcome Bradley–Terry likelihood. Latent item quality, intervals, source
  effects, session thresholds, rater consistency, tie/not-sure parameters, and diagnostic raw means
  are exported; raw means are never the canonical score.
- Reliability output covers ordinal test-retest, exact-repeat agreement, anchor drift, response
  time, replays, A/B swaps, equivalent false preference, model fit/transitivity, family/style/source,
  category use, fatigue, presentation mode, and items needing more judgments.
- Training evidence preserves raw ordinal and pair outcomes, ties, not-sure judgments, fitted latent
  targets, uncertainty weights, source-relative effects, and rater/session provenance. Human and
  synthetic supervision remain separate, with agreement broken down by perturbation family.
- The original pre-collection analysis reported `awaiting_human_pilot`; it predates the four now
  preserved observations and is not a current Phase 15 result. No Phase 14 reliability result is
  promoted into the revised pilot, and critic retraining remains blocked.
- Repository-local Playwright tests drive installed Brave/Chromium through real naturalness,
  same-viewport-toggle, and sequential-neutral-gap sessions. They verify the exact prompt/anchors,
  blinded payload, 960x640 canvas pixels, neutral interval and A/B controls, and keep every response
  disabled until the final playback acknowledgement resolves. Test manifests, screenshots,
  observations, and served playlists are temporary. The production Phase 14 files are preserved
  separately at exactly four observations and five served-playlist events.
- Submission is disabled while persistence is in flight, and the threaded server atomically
  compare-and-commits each observation. An eight-way concurrent regression verifies that exactly
  one request is accepted and that stale duplicates cannot prematurely activate the next trial.
- Final verification passes: Ruff, mypy across 107 package files, 132 pytest tests in 79.24
  seconds (with browser availability required), CLI discovery, and source/wheel builds. CI now
  provisions Playwright Chromium and fails rather than silently skipping browser coverage.

Full protocol details are in `docs/calibrated_subjective_pilot.md`; prepared artifacts are under
`artifacts/phase14_subjective_pilot`.

The preserved Phase 14 checksums are:

- `pilot_manifest.json`: `09828d73de51d79fe4eac029b8279938e711753a799d4afd28a712c64001c2fd`;
- `raw_observations.jsonl`: `3008ca90bcad10d7ddc84c0fc6fd8549be171155fa52a02ae18a4a0ee26dc997`;
- `served_playlist.jsonl`: `834e0aea86f54487ec969221dc7c6793343416565793761ee62ffa5cc87edd06`.

Its skeleton render protocol remains byte-for-byte identified by
`render:sha256:68b9052a8345ebafc8dc761d02bd4ff54c37f675060a30231e51fa84fcbf81e5`.
The legacy `prepare-subjective-pilot` command is now explicitly skeleton-only so it cannot create a
hybrid artifact mislabeled as the revised mannequin protocol.

## Phase 15 calibrated mannequin pilot

- The primary renderer is the fixed neutral articulated mannequin
  `motionlab.fixed_mannequin_canvas.v1`, with render-protocol hash
  `render:sha256:cea7579f211ffb88143ee439f1de9fbdca3af6e4394d9b86a9dd78c69cec1f4b`.
  It uses fixed proportions, appearance, lighting, ground, and camera settings. Rectangular/oriented
  head, chest, pelvis, arm, hand, thigh, shin, and foot solids expose facing and axial rotation; a
  faint skeleton overlay remains optional and defaults off.
- Every scored animation must finish one server-acknowledged playback at 1x with the standardized
  three-quarter camera before any rating or inspection control unlocks. Afterwards the evaluator
  exposes 0.25x, 0.5x, 1x, 1.5x, and 2x playback, front/three-quarter/side views, bounded zoom,
  reset, optional orbit, overlay, and replay. It records replay counts, speed-change events,
  playback time by speed, camera/zoom/view/orbit/overlay events, and total inspection time.
- Same-viewport A/B comparisons share one camera, zoom, playback-rate, phase, and viewport state.
  Switching with buttons or A/B/Space changes only the displayed motion at the same phase. Every
  revised comparison uses this mode; sequential comparison is retained only for the frozen legacy
  protocol.
- The separate calibration bank contains 13 explicit five-rung ladders and 65
  `CALIBRATION_ONLY` stimuli spanning `clean`, `mild`, `medium`, `strong`, and `severe`. Six ladders
  use measured corruption chains and seven use purpose-built perceptual presets. Every non-clean
  rung has unique motion/rendered-geometry/render hashes and a nonzero tracked-position or global-
  rotation difference from its clean reference. All calibration stimuli are explicitly ineligible
  for critic training and scored quality evaluation.
- The visible tutorial copies three complete ladders—foot slide, upper-body phase mismatch, and
  torso counter-rotation mismatch—for 15 tutorial stimuli. It discloses family and severity order,
  permits immediate unrestricted severity/speed/view/zoom/replay inspection without the scored
  first-pass gate, and emits no scored observation. Tutorial and scored assets are disjoint by
  stimulus ID, visual-content hash, and source window.
- The active manifest has 55 directly presented assets: 15 tutorial stimuli plus exactly 40 unique
  scored stimuli. The scored set contains 32 `HARD_FEASIBLE_PERCEPTUAL` stimuli and eight hidden
  `EVALUATION_ANCHOR` stimuli spanning clean through severe quality. Purpose population,
  measurement cohort, and `critic_training_eligible` are persisted on stimuli, trials, and revised
  observations without being exposed in the public scored-trial payload.
- The two sessions each contain 24 base scored presentations. Across them, the 40 unique scored
  stimuli receive eight hidden exact-repeat presentations: four hard-feasible items and four
  evaluation anchors, each repeated once across sessions. The realized minimum repeat index
  distance is 15 (14 intervening base trials), above the declared minimum of 12.
- The response-adaptive tail contains exactly 12 prevalidated matched-source comparisons, split six
  per session. All are same-viewport toggles between two `HARD_FEASIBLE_PERCEPTUAL` items. The
  selector prioritizes close single-stimulus ratings, repeat disagreement, known model/human error,
  and underrepresented families while exhausting only the session-partitioned pool; odd requested
  totals from 10 through 15 remain reachable under the declared per-session cap.
- After the quality response is locked, the pilot optionally asks whether the judgment was easy,
  moderate, or difficult. Analysis reports 1–7 category use, repeat reliability overall and
  separately for `calibration_easy` anchors versus `hard_feasible` items, replay/speed/camera usage,
  post-rating difficulty, anchor drift, and exploratory family perceptibility. Visible tutorial
  ladders remain unscored and therefore cannot masquerade as severity-ranking evidence.
- Training export uses a strict double gate: the observation must explicitly declare eligibility,
  and every referenced manifest stimulus must itself be eligible and belong exclusively to
  `HARD_FEASIBLE_PERCEPTUAL`. `CALIBRATION_ONLY`, `EVALUATION_ANCHOR`, mixed, and legacy-unclassified
  observations cannot enter critic evidence.
- The generated Phase 15 analysis is the preserved pre-collection readiness snapshot: it has zero
  human observations, zero training-evidence records, and status `awaiting_human_pilot`. Collection
  is now active, but the growing append-only log is intentionally not analyzed or summarized until
  the user declares it sufficient. Because the old and new pilots are not a matched single-factor
  experiment, any eventual cross-protocol comparison is exploratory and cannot isolate a causal
  mannequin-only improvement.
- No critic checkpoint, optimizer result, or retraining artifact was created. The manifest and
  analysis both keep retraining blocked until the revised pilot is collected and explicitly
  reviewed. Variable-rig graph/attention work remains deferred.
- Final verification passes: 147 repository tests in 90.24 seconds; 22 focused calibration,
  revised-analysis, protocol, and UI tests, including four required real-browser UI flows; Ruff;
  formatting; mypy across all 108 package source files; CLI discovery; source/wheel builds; and
  direct validation of both generated manifests. The calibration validator verifies all 65 bank
  stimuli, and the pilot validator verifies population separation, blinding, repeat allocation and
  spacing, adaptive comparison mode/count/reachability, copied assets, and immutable hashes.

The prepared pilot, nested calibration bank, and empty pre-collection analysis are under
`artifacts/phase15_mannequin_pilot`.

## Phase 16 label-independent preparation

- The active Phase 15 protocol is frozen as
  `human-evaluation-protocol:sha256:26cd8c93717bc69a80489c5a4afb8e015f50209567162686723eff715e73413a`.
  Its freeze and immutable-snapshot IDs are
  `protocol-freeze:sha256:06fbc1f189aecc5d1d9714329414c64015d91f87e31f238c3b78e7e02f248018`
  and
  `active-pilot-snapshot:sha256:9a840a3481a8d9593d3e358c0c04886c460d75aa034e2912fcb999a1ea532122`.
  The contract hashes 133 immutable files, including 129 NPZ assets, and fixes the mannequin,
  renderer, camera/playback, questions/scales, anchors/repeats, randomization/scheduler, and
  observation schema. Observation and playlist prefixes are immutable while valid suffix appends
  remain permitted. Validation fails closed on drift; no active stimulus or protocol field was
  regenerated.
- The separate Phase 16 pool contains 625 frozen-render-compatible motions across 12 sources: 560
  `UNLABELED` candidates and 65 `CALIBRATION_ONLY` ladder stimuli. Every subjective preference is
  null, every record is currently critic-training-ineligible, and no upstream pair direction was
  imported. All 625 are renderable, 588 pass the deterministic feasibility screen, and 201 inputs
  or attempts were rejected rather than admitted by relaxing constraints. All 18 inherited
  adversarial CMA candidates retain content-addressed copies of seed/configuration, source-run,
  trajectory, optimization, and spline-coefficient evidence: six original GroupNorm exploits and
  12 later LayerNorm additions. The 72 evidence files are bound back to corpus ID, normalization,
  original/final motion, optimizer history, coefficients, and control-point configuration. No
  Phase 16 perceptual CMA was invoked.
- Candidate coverage is 22 clean 100STYLE windows, 150 hard-feasible subtle variants, 20 objective
  equivalence controls, 120 over-smoothed/rigid variants, 100 coordination/phase variants, 90
  pelvis/torso variants, 40 phase-aligned style-incoherence variants, all 18 existing adversarial
  CMA outputs, and 65 calibration ladder rungs. Each retains content/file/visual hashes, source
  lineage, origin and mechanism, parameters/severity, feasibility evidence, phase reference, and
  frozen-protocol renderability.
- The subjective analysis format is now `motionlab.subjective_analysis.v3`. It produces ordinal
  latent-quality estimates and intervals, source-relative effects, session/rater calibration, a
  Davidson-style four-outcome Bradley–Terry fit, and a separately held-out direct-pair-versus-
  ordinal report. Its reliability dashboard covers hidden repeats, anchors/drift, category use,
  response time/replays/inspection, difficulty, order, fatigue/transitivity, family/style/source,
  matched same-anchor cross-session drift, human/synthetic agreement, and items needing more judgments. Only deterministic dummy fixtures
  exercised the new path; no substantive human result is reported. Current-protocol analysis
  requires the canonical freeze-recorded paths and authenticates every downstream-trusted row
  field against the exact served event and manifest trial, rejecting duplicate, malformed,
  unknown, unserved, or inconsistent observations. Analysis outputs must be fresh and outside the
  active pilot tree.
- Active-query scheduling supports `RATE_SINGLE` and `COMPARE`, combines uncertainty/interval
  overlap, repeat disagreement, coverage, critic disagreement/confirmation, adversarial, comparator,
  and required repeat/anchor priorities, and caps item/pair/source/style/family/session exposure.
  Repeat disagreement is restricted to explicit within-rater hidden-repeat groups, global history
  drives undercoverage, and current-session counts drive balancing. It uses a bounded nearest-
  neighbor pair graph rather than all pairs. The dummy fixture considers 3,026 pair candidates for
  558 eligible pool stimuli and returns 12 singles plus eight comparisons; it is not an active
  Phase 15 schedule.
- Post-label forensics fail closed until the requested stimulus or exact pair has a valid persisted
  response linked to the frozen canonical observation and served-playlist logs and every observed
  rater has completed the full blinded pilot, protecting later hidden repeats and adaptive trials.
  Outputs are forbidden inside the active pilot tree. A fresh analysis-only HTML/JSON bundle then synchronizes articulated motion, optional
  skeleton, source/variant, root/feet/contact, pelvis, angular velocity/jerk, deterministic metrics,
  and optional temporal critic outputs. Nothing is imported by or exposed in the blinded rating UI.
- `SkeletonGraph`, `MotionGraph`, and `VariableRigBatch` now provide canonical variable-joint data,
  semantic part/side/role mapping, universal versus rig-coordinate streams, basis/unit/axis
  provenance, helper handling, and explicit padding masks. Permutation, padding, helper insertion,
  local-axis reparameterization, unit/axis canonicalization, and variable-count invariances are
  tested. No message passing, attention, variable-rig critic, or training objective exists.
- MotionCritic is explicitly deferred rather than forced through a representation-confounded
  adapter: the current 23-joint, 160-frame quaternion rig does not match its 24-joint, 60-frame SMPL
  local-axis-angle contract, and the verified mapping/assets/resampling path is absent. No
  checkpoint was loaded and no human labels were adapted.
- Comparator readiness evaluates precomputed `absolute_q`, frozen-backbone relative, and
  fine-tuned-backbone relative predictions without fitting: held-out source/mechanism,
  direct-human-only, equivalence, adversarial, swap consistency, and Wilson intervals are covered.
  Relative-experiment v3 snapshots hashed train/validation supervision before selection, so held-
  out labels and audit policy cannot influence the selected variant; native forward/swapped scores
  and explicit objective-equivalence provenance are retained. The authenticated post-label wrapper
  additionally requires pilot completion, a completed subjective-analysis v3 report, and a completed
  relative-experiment v3 artifact, then aligns direct human observations by motion-content hashes.
  Human ties are never equivalence controls. Complete/incomplete synthetic fixtures test the
  machinery, not model quality; no comparator training ran.
- Additional real-clip deterministic repair testing preserves specialized-first production
  selection with deterministic CMA fallback, withholds the clean source until evaluation, and
  forbids a learned perceptual objective. Fourteen cases span seven styles absent from the Phase 11
  production benchmark; 12 have full production eligibility and 11 succeed (91.7%). Foot slide,
  floating contact, penetration, pop, and jitter each pass 2/2, while speed passes 1/2. Every case
  now also has a generic CMA head-to-head baseline: nine specialist-success comparisons are
  explicitly non-selectable, while five failed-specialist CMA runs double as production fallbacks.
  Specialist/CMA/tie target-objective wins are 7/5/2, and all portfolio-selected outputs remain
  unchanged. Two loop cases remain diagnostic-only without cyclic-task metadata, and joint limits
  are explicitly skipped because no trustworthy authored-limit cases exist. Full evidence is in
  `artifacts/phase16_label_independent/deterministic_repair_stress/deterministic_repair_stress.json`
  (SHA-256 `815163ed7739390efda2845ba56ac35b722ac6b294b97a2cffaf30fe4cfb1ff9`).
- The focused Phase 16 protocol/pool/forensics/query/analysis/comparator/variable-rig/repair suite
  passes 76 tests in 13.33 seconds. All 226 repository tests pass in 107.20 seconds. Ruff formatting
  and checks pass repository-wide, and mypy passes all 117 package source files.

Full Phase 16 evidence and command examples are in `docs/label_independent_phase16.md`; generated
artifacts are under `artifacts/phase16_label_independent`.

## Numerical tolerances

- Quaternion unit-norm validation: `1e-5` absolute.
- Rotation-matrix comparisons: `2e-5` absolute.
- FK position comparisons: `1e-5 m` absolute.
- Time equality floor: `1e-9 s`.
- Loop-seam numerical-noise floor: `1e-6` weighted units.
- Gait cross-rate fixture tolerance: one sampled contact-transition frame (up to `1/30 s`).
- Joint-limit event threshold: `1e-6 deg` beyond an inclusive bound.

## Known unsupported cases

- BVH export and non-root BVH translation channels are unsupported; unsupported channels fail
  explicitly.
- BVH rest orientations are identity because plain BVH does not encode arbitrary joint pre/post
  rotations; richer rig metadata is not inferred.
- Previewing currently produces a static PNG, not MP4/GIF animation.
- Ground planes must be supplied; automatic ground estimation is not implemented.
- Spatial gait geometry infers forward from net plane-tangential root travel and is unavailable for
  in-place clips without a stable travel direction.
- Joint-limit coordinates currently support intrinsic XYZ Euler or local-Y swing/twist. Plain BVH
  local axes default to low confidence because identity rest rotations are not anatomical axes.
- The current production audit remains fixed-rig and forward-walk-only: 12 qualifying 100STYLE
  takes and two 160-frame windows per take. Non-identity retargeting is still validated with a
  generated target rig.
- Report plots are not implemented; they were intentionally kept off the corruption-dataset
  critical path.
- Foot lock handles direct hip-knee-ankle chains. It reports reach clamping and offers bounded root
  compensation but does not yet enforce configured joint limits or protect the opposite foot.
- The deterministic demo uses generated data. A user-provided target-rig Blender round trip is a
  later integration stage.
- Variable-rig learned critics, retrieval/orchestration, HTTP, and Blender integration have not
  started. CMA-ES has passed the narrow deterministic production-repair milestone. The first
  fixed-rig perceptual critic has been evaluated and is not optimization-ready; broader production
  generalization remains unvalidated.
- The metadata adapter still prefers authored semantic roles and does not promise general
  heuristic rig recognition. Canonical variable-topology `SkeletonGraph`/`MotionGraph` tensors and
  padding masks now exist as data infrastructure only; a learned graph/attention critic remains
  intentionally deferred.
- Clips created before the lineage contract remain supported but explicitly report
  `lineage_complete=false`; source identity cannot be reconstructed reliably after the fact.
- Automatic one-source generation excludes aligned style-region blending because it requires a
  compatible phase-aligned donor. Joint-limit samples are rejected when the rig has no valid
  authored limits. These cases are declared in the summary rather than relabeled as hard
  negatives.
- The restricted first-critic corruption core has measured 100STYLE production pass rates. The
  rest of the full catalog remains synthetic-only.
- Some direct curve mechanisms can fail several observed bins after collateral screening. This is
  reported rather than relaxed; production-pipeline fault injection and real-agent failure replay
  remain intentionally deferred until after the fixed-rig baseline.

## Assumptions

- The first supported representation is a single-root humanoid tree.
- Animation quaternions are absolute local orientations, including rest orientation.
- Cycle storage omits a duplicated closing frame; seam checks extrapolate the last pose by one
  sample before comparing it with frame zero.
- Severe synthetic sliding is graded with a deliberately relaxed contact-speed threshold so the
  known stance is not relabeled as swing; the threshold is recorded in report metadata.

## Fixed-rig baseline interpretation

- Synthetic labels are learnable on the tiny gate, and clip recognition crosses mechanisms for
  several families, but the first critic is not operationally ready.
- Frame/part localization, calibration, and severity ordering fail to generalize. Five of five
  held-out-source shams trigger a false positive, and overall pairwise ranking is below chance.
- No loss ablation was run, so individual-loss usefulness is not claimed. The deterministic
  baseline comparison is complete and shows that a TCN is not justified for penetration, seam,
  or target-speed mismatch on this experiment.
- The learned representation and best/worst, false-positive, false-negative, sham, held-out, and
  ranking-inversion examples are persisted under `artifacts/fixed_rig_tcn_experiment_v6`.

## Next three tasks

1. Collect the small `phase17_opt_in_inspection_pilot` with the original rater ID. Keep all old
   collectors paused and do not mutate frozen stimuli, camera protocols, questions or logs.
2. Authenticate direct new labels and produce the requested noticeability/overlap report:
   raw response distribution, clean/sham false alarms, intended-obvious misses, repeats,
   repeated-view/inspected disagreement and viewing/speed usage. Keep the earlier spontaneous
   target's threshold estimates and legacy calibration separate from repeated-view results.
   Use those results to propose a focused staircase and independent fixed validation batch.
3. Stop for explicit review before any critic retraining or learned production optimization.
   Variable-rig graph/attention and normalization work remain out of scope.
