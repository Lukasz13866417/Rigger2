# Consolidated implementation plan

## 1. How to read the source specifications

This plan consolidates:

- `CODEX_IMPLEMENTATION_BRIEF_WALK_ANIMATION_V2.md`
- `CODEX_VARIABLE_RIG_MULTITASK_ADDENDUM.md`

The larger implementation brief embeds the standalone addendum verbatim. Treat the fixed-rig brief as the delivery backbone. Where learned-model or cross-rig guidance differs, the addendum takes precedence. Deterministic geometry, grading, editing, serialization, provenance, and test requirements remain unchanged.

The required delivery order is:

1. deterministic fixed-rig vertical slice;
2. fixed-rig synthetic-data pipeline and TCN critic;
3. safe editing, retrieval, orchestration, and integrations;
4. canonical variable-rig foundation and held-out-rig evaluation;
5. auxiliary tasks, human preferences, and hard-negative mining.

Do not build empty modules for later stages. Add a module only when its stage starts and ship working, tested behavior at every stage.

## 2. Product scope and non-goals

The first useful product accepts humanoid walking motion, retargets it to one known rig, extracts a gait cycle, grades it, repairs common faults, and exports a reproducible result. Initial operating assumptions are flat ground, straight locomotion, one to three gait cycles, 60 fps internally, offline grading, and neutral walking before additional styles.

The initial product is not:

- an end-to-end text-to-motion generator;
- an LLM that writes joint transforms frame by frame;
- a rendered-video critic;
- a dynamics simulator;
- a universal creature-rig model;
- a single opaque naturalness score.

Variable-rig work initially covers humanoids with different proportions, spine counts, toe bones, helper/twist joints, names, and local axes. Non-humanoid rigs remain out of scope.

## 3. Architectural decisions that apply from day one

1. Use one canonical internal convention: right-handed, Y-up, +Z-forward, metres, seconds, radians, active `[w, x, y, z]` quaternions, and root translation separate from root rotation.
2. Keep import/export coordinate transforms at adapter boundaries and persist the original-to-canonical transform.
3. Preserve metric and body-normalized features; never normalize away quantities needed for speed, clearance, or contact diagnostics.
4. Preserve original assets and derive immutable, content-addressed motion versions with complete operation history.
5. Assign a canonical source-motion ID and dataset split before windowing, mirroring, retargeting, topology augmentation, corruption, or label joins.
6. Keep style separate from quality. Unusual intended styles are not defects.
7. Treat deterministic operators and constrained candidate search as the primary repair path. Learned repair remains conservative and rig-aware.
8. Never compare raw local rotations across rigs without coordinate anchoring.
9. Never encode a missing modality as an ordinary numeric zero; pair it with validity and confidence.
10. Retain per-joint representations for localization and use deterministic functional-part pooling for fixed-count frame/clip representations.

## 4. Release sequence

| Release | Outcome | Included phases |
|---|---|---|
| R0 — deterministic vertical slice | Import, inspect, grade, corrupt, repair, and export one walk without ML | 0–4 |
| R1 — fixed-rig learned critic | Generate leak-free supervision, train/evaluate the baseline critic, merge learned and deterministic reports | 5–7 |
| R2 — useful fixed-rig product | Search safe edits, retrieve/style motions, run the orchestration loop, integrate Blender and HTTP tools | 8–10 |
| R3 — variable-rig humanoid critic | Canonical graph contracts, multiple rigs, invariant batching, graph/part encoder, cross-rig consistency | 11–14 |
| R4 — multi-task improvement loop | Add auxiliary datasets one at a time, calibrate with preferences, mine agent hard negatives | 15–17 |

## 5. Phase plan

### Phase 0 — repository bootstrap

Deliver:

- Python 3.11+ package using `pyproject.toml` and the `motionlab` name;
- `motionlab` and `python -m motionlab` entry points;
- validated configuration, structured logging, central numerical tolerances, and deterministic seed utilities;
- pytest, Ruff, mypy, coverage, and CPU CI setup;
- the initial README, changelog, implementation status, and coordinate/data-format docs;
- a synthetic humanoid fixture generator, without broad placeholder modules.

Gate:

```bash
python -m motionlab --help
ruff check .
mypy motionlab
pytest
```

All commands pass in a clean CPU environment.

### Phase 1 — rotation math, fixed-rig types, FK, and NPZ

Deliver:

- quaternion normalization, inverse, multiplication, vector rotation, log/exp, SLERP, sign unwrapping, shortest-arc difference, matrix conversion, geodesic distance, and 6D conversions in NumPy and Torch where needed;
- validated `Skeleton`, `MarkerSpec`, and `MotionClip` types;
- explicit rest/bind-orientation semantics;
- vectorized NumPy and Torch forward kinematics;
- versioned, pickle-free NPZ serialization with JSON metadata and stable content hashes;
- rig YAML parsing with semantic roles, marker offsets, coordinate metadata, basis corrections, and optional joint limits.

Gate:

- identity, inverse, composition, near-zero, near-180-degree, sign, and random rotation round trips pass;
- exact two- and three-link FK tests pass;
- NumPy and Torch FK agree within named tolerances;
- NPZ round trips preserve arrays and metadata;
- malformed trees, non-finite arrays, invalid quaternions, and bad role/marker references fail clearly.

### Phase 2 — BVH ingestion, resampling, canonicalization, and preview

Deliver:

- recursive BVH hierarchy/channel parser supporting per-joint Euler orders, root translation, frame timing, and end sites;
- configurable source axis, handedness, and unit conversion into the canonical frame;
- quaternion-SLERP resampling to 60 fps and trajectory interpolation using timestamps;
- canonical facing/origin transforms while retaining the global root trajectory;
- world-space virtual heel/toe marker evaluation;
- NPZ conversion, validation, inspection, and a minimal 3D/debug preview path;
- a tiny handwritten BVH fixture with known expected poses.

Gate:

- fixture FK positions and frame times are exact within tolerance;
- a user BVH converts, reloads, validates, and previews;
- source-to-canonical transform metadata is present and round trips are tested;
- no axis swap or scale conversion occurs inside core math.

### Phase 3 — contacts, gait structure, deterministic grading, and reports

Deliver:

- heel/toe contact probabilities and hard contacts using plane-relative height, tangential speed, hysteresis, duration cleanup, and rig-scaled thresholds;
- heel strikes, toe-offs, stance/swing intervals, continuous phase, and confidence-aware fallbacks;
- cycle extraction and root-motion/in-place loop representations;
- deterministic metrics for sliding, ground penetration/floating, speed/heading, cadence/stride, quaternion smoothness/pop/jitter, joint limits when defined, and loop seams;
- physical-unit `MetricResult`, diagnostic-event merging, compact JSON reports, and dense NPZ artifacts;
- plots showing skeleton, markers, ground, contacts, root trajectory, and metric curves.

Gate:

- the synthetic walk produces plausible alternating contacts and phase;
- stationary planted feet report approximately zero slip;
- known slip, penetration, pulse, speed, and seam defects are measured and localized correctly;
- missing contact/phase/joint-limit evidence lowers confidence or marks a metric unavailable;
- reports retain category scores and physical measurements instead of hiding them behind one scalar.

### Phase 4 — fixed-rig retargeting, primary repairs, and deterministic demo

Deliver:

- rest-relative rotation transfer with explicit basis corrections;
- body-scale/root-trajectory transfer and unmapped-joint fallback to rest;
- two-bone IK plus a bounded CCD fallback where justified;
- contact-aware retarget refinement;
- foot-slide corruption;
- foot locking and loop closure using tangent-space corrections and smooth blend windows;
- a complete deterministic demo that imports, canonicalizes, retargets, extracts a cycle, grades, corrupts, repairs, re-grades, and exports clean/corrupt/repaired artifacts plus history.

Gate:

- identity retargeting preserves motion within tolerance;
- synthetic foot locking reduces stance slip by at least 90%, without knee flips or new penetration beyond tolerance;
- loop closure reduces the weighted seam metric by at least 80%, without violating configured contact/ground tolerances;
- the demo works without a model checkpoint and emits NPZ motions, JSON reports, plots/previews, and provenance.

R0 is complete only when all Phase 0–4 checks pass together from a clean checkout.

### Phase 5 — source-data preparation, lineage, and clean filtering

Deliver:

- 100STYLE forward-walk discovery and metadata extraction, beginning with Neutral;
- canonical source identity containing original dataset/path, performer/session when known, and underlying source references;
- persisted split registry assigned before all derivations;
- take-held-out train/validation/test splits and actor-held-out support when identity exists;
- clean-motion filters using deterministic metrics and `clean_confidence` rather than assuming every mocap take is valid;
- idempotent and resumable preparation manifests with content hashes.

Gate:

- no windows or derivatives of one source take cross split boundaries;
- rerunning preparation with the same inputs produces the same IDs and manifests;
- rejected and low-confidence takes have explicit reasons;
- dataset summary includes take, style, split, duration, and quality distributions.

### Phase 6 — corruption catalog and supervised dataset

Status: complete for the fixed-rig pipeline. The catalog, held-out mechanisms, dense supervision,
bounded composition, leakage-audited features, deterministic windows, train-only normalization,
content-addressed manifests, exact-state resume, CLI, and validation gate are implemented. Hard
samples now target five observed physical-severity bins through bounded search, undergo collateral
screening, share counterfactual groups, and include matched sham/consistency targets. The remaining
real-source pass-rate study depends on a locally supplied production BVH from Phase 5.

Deliver:

- deterministic seeded corruptions for foot slide, penetration/floating, joint pop, jitter, limb phase mismatch, time warp/cadence mismatch, speed/stride mismatch, loop seam, pelvis curve, foot clearance, joint-limit/twist, and later style incoherence;
- mostly single-defect samples first, with bounded multi-defect composition;
- at least two mechanisms for major defect classes and a held-out mechanism for evaluation;
- three distinct labels: exact intervention mask, measured symptom mask, and confidence-weighted semantic responsibility target;
- measured physical severity, family-specific near-threshold/subtle/moderate/clear/severe bins,
  and only unambiguous observed-metric ranking pairs;
- matched no-symptom controls, explicit quality-equivalence validity, single-defect collateral
  policies, semantic gait-event targeting, counterfactual groups, and cross-mechanism per-defect
  consistency targets;
- fixed 160-frame/32-stride training windows by default, training-only normalization statistics, JSONL manifests, and compressed NPZ samples/shards;
- resumable, idempotent `generate-dataset` with a complete summary.

Gate:

- every corruptor is seed-deterministic, produces normalized finite rotations, and preserves declared unaffected regions within tolerance;
- the intended metric measurably worsens for an agreed pass rate; ineffective samples are rejected;
- no file path, dataset ID, corruption ID, or severity parameter enters neural features;
- the held-out corruptor split is truly absent from training;
- all derived samples retain canonical source identity and split lineage.
- summaries expose severity-bin population, collateral rejection, source/gait/side coverage,
  mechanism coverage, and sham-control counts without implying production generalization.

### Phase 7 — fixed-rig TCN critic

Deliver configuration `A. fixed_rig_flat_tcn` behind the common critic interface:

- per-joint features from local 6D rotation, root/facing-relative position and velocity, and SO(3) angular velocity;
- global root velocity/yaw rate/height, marker contacts, phase, and requested conditions;
- shared joint MLP, semantic-joint/mean/max frame aggregation, and a 6–8 block bidirectional dilated TCN;
- frame, joint-time, clip-defect, ranking, and contact heads; style and repair heads remain disabled until their stages;
- losses normalized by valid elements, ranking pairs primarily from the same source, and weak rather than falsely causal joint blame;
- CPU smoke training, CUDA mixed precision, checkpoints containing config/normalization/code version, resume, early stopping, and JSONL/TensorBoard metrics;
- deterministic-plus-learned inference report generation.
- shortcut-probe evaluation slices for held-out source clips, unseen mechanisms of known defects,
  matched shams, and subtle-vs-subtle pairs, plus a cheap linear mechanism-ID probe trained on the
  frozen shared representation;

Gate:

- each enabled head overfits 8–32 samples; pairwise training accuracy exceeds 95% on the debug fixture;
- a CPU smoke train/evaluate/resume cycle succeeds;
- random valid inputs have finite outputs and gradients;
- evaluation reports per-class clip AUROC/AUPRC, frame/event localization, joint/part localization where applicable, ranking accuracy, severity correlation, calibration, and clean-input false positives;
- held-out corruption mechanisms are evaluated separately;
- unseen-mechanism defect detection/localization is the primary synthetic-generalization result;
  high mechanism-ID probe accuracy paired with weak mechanism holdout is reported as corruptor-
  fingerprint learning rather than model success;
- the baseline checkpoint and exact evaluation manifest are reproducible.

R1 is complete when the fixed-rig critic adds useful localization or ranking on held-out corruptions without regressing deterministic report correctness.

### Phase 8 — repair suite and constrained candidate optimization

Status note (2026-09-04): the bounded first learned experiment is implemented ahead of the full
operator suite as a 4-to-8-control cubic B-spline BIPOP-CMA-ES red team. It optimizes selected
per-family severity heads, persists full generation traces, and separates broad red-team search
from hard-constrained production mode. The 24-run matrix produced six adversarial failures and no
production repair. The broader deliverables below remain planned and are not implied complete.

Deliver:

- quaternion tangent-space joint smoothing;
- limb phase/retiming with monotonic time maps and SLERP;
- speed adjustment modes, pelvis phase-curve edits, foot-clearance editing, and generalized loop closure;
- optional small operator-specific learned residuals, applied in target-rig coordinates and scaled conservatively;
- grid/random/Latin-hypercube parameter search with immutable candidate artifacts and full traces;
- lexicographic acceptance: reject hard violations, reject protected-metric regressions, improve the target metric, improve critic rank when reliable, then minimize edit magnitude;
- dry-run support, required-role declarations, affected scopes, and before/after reports for every operator.

Gate:

- each operator has a synthetic fault it improves and a clean/no-op test;
- repairs do not materially alter protected frames, contacts, requested speed, or style beyond configured tolerances;
- learned repair makes near-zero changes on clean smoke examples and is rejected when it worsens hard constraints;
- a repeated search with the same inputs reproduces every candidate and decision.

### Phase 9 — retrieval, style handling, and full orchestration

Deliver:

- a clean cycle catalog with style, speed, cadence, stride, width, asymmetry, seam quality, and confidence;
- transparent metadata-based top-N retrieval before embedding retrieval;
- request schemas for target speed, weighted style labels, avoid-list, duration/cadence, symmetry, and loop mode;
- phase-aligned style/body-region mixing only on the same target skeleton, followed by repair and re-grading;
- an immutable, content-addressed `MotionService` for import, retrieve, retarget, grade, edit, optimize, loop, export, and history;
- an orchestration policy that edits one high-confidence issue at a time and stops on convergence, budget, or absence of a safe improvement.

Gate:

- a request retrieves multiple candidates and compares them after target-rig retargeting;
- at least one safe edit iteration executes with declared protected properties;
- final motion, compact report, dense artifacts, preview, and replayable history are exported;
- the LLM-facing API never requires the model to manipulate dense transforms directly.

### Phase 10 — Blender bridge and HTTP tool server

Deliver:

- separate Blender scripts to export an active armature, export a sampled evaluated action, and import a canonical motion into a new baked action;
- documented Blender-to-canonical transforms and role-map handling;
- FastAPI service with the same schemas as the Python service, stable errors, OpenAPI, workspace isolation, and path-traversal protection;
- optional dependencies that do not affect the core package.

Gate:

- Blender rig/action round trips are documented and manually verified on the target asset;
- imported actions preserve root-motion/in-place semantics and source actions remain untouched;
- API schema/error/security tests pass;
- the full core test suite passes without Blender or server extras installed.

R2 is the first product release. It must satisfy the brief’s “useful first release” workflow end to end.

### Phase 11 — canonical graph contracts and invariance harness

Begin variable-rig work only after the R1 fixed-rig baseline is reproducible. Deliver:

- immutable validated `SkeletonGraph` with hierarchy, rest/bind/anchor distinction, coordinate metadata, semantic roles/confidence, joint kinds, critic/repair masks, and optional anchor geometry;
- validated `MotionGraph` with timestamps, root trajectory, optional local/global rotations/positions, environment/intent, modality validity, and complete provenance;
- compatibility adapters from fixed-rig `Skeleton`/`MotionClip` and a refactored importer that does not change R0/R1 demo outputs;
- masked variable-`T`/variable-`J` batching and bucketed loading;
- the mandatory graph, padding, batch-composition, coordinate, scale, rotation, temporal, import/export, and label-mask invariance test harness.

Gate:

- cycles, multiple roots without a documented super-root, bad indices, non-finite values, invalid scaling, and unknown silently assumed coordinates are rejected or explicitly flagged;
- padded extreme values cannot affect valid outputs;
- sample output is invariant to batch companions;
- canonicalization and legacy-adapter regression tests preserve fixed-rig results;
- derivatives recover constant physical motion and are stable across 30/60/120 fps resampling.

### Phase 12 — rig semantics, topology augmentation, and paired data

Deliver:

- deterministic semantic mapping in priority order: user YAML, normalized aliases, hierarchy/geometry rules, optional offline proposal, then `UNKNOWN` with confidence;
- explicit side and functional-part assignments plus name dropout/corruption tests;
- exact identity-helper insertion and ignored marker-leaf insertion;
- at least two real humanoid target rigs with reliable sparse anatomical anchors;
- cross-rig pair records containing source motion, target rig, mapping/retargeter versions, quality, confidence, and time alignment;
- safe proportion variants created through contact/joint-limit-aware retargeting; reject weak pairs.

Gate:

- identity-helper insertion preserves downstream FK exactly within tolerance;
- arbitrary dynamic-joint deletion is not used as an “exact” augmentation;
- all variants of a source motion remain in one split;
- mirror/name-dropout/unknown-role behavior is tested;
- low-confidence rig mapping lowers report confidence and prevents aggressive automated optimization.

### Phase 13 — variable-rig encoder and deterministic part hierarchy

Deliver configurations `B. variable_j_local_graph_tcn` and `C. variable_j_graph_tcn_parts`:

- cached static features and separate 64-D semantic/64-D rig-specific projections;
- typed parent-to-child/child-to-parent graph message passing;
- a universal dynamic stream using facing-relative positions, velocities, metric root motion, heights, markers, and contacts;
- an optional trusted rotation stream using 6D rotations and SO(3) derivatives with validity/confidence;
- raw, low-pass trend, and high-frequency dynamic channels using physical cutoff units;
- shared modality-aware joint MLP, concatenation fusion, six residual graph/TCN blocks, and persistent `[B,T,J,D]` tokens;
- deterministic masked pooling into roughly eight functional parts, part communication, whole-body/frame, and clip representations;
- no learned part queries in the initial implementation.

Gate:

- joint permutation equivariance, padding invariance, batch-composition invariance, mask safety, all-invalid-part handling, yaw/translation/scale behavior, local-axis reparameterization, quaternion-sign, and cyclic-shift tests pass;
- dense losses are normalized per sample, part, and semantic joint rather than by raw joint count;
- fingers, face, controllers, and irrelevant accessories do not dominate the walking critic;
- raw jitter remains observable despite providing filtered trends.

### Phase 14 — hierarchical heads, cross-rig consistency, and adoption gate

Deliver:

- joint-time, part-time, frame, clip-rank, contact, phase, and validity/OOD heads; style and repair activate only with valid labels and rig-aware inputs;
- overlapping-window inference with smooth blending and circular padding only for verified loops;
- prediction-level, functional-part contrastive, and anchor-joint consistency losses for aligned same-motion/different-rig pairs;
- semantic and rig-specific latent projections; do not force complete latent equality;
- hard negatives covering different motion/same rig, same style-speed/different phase, and same pose/different dynamics;
- OOD/confidence reporting for rig mapping, coordinates, rotation validity, contacts, ground, skeleton, motion, and each learned head;
- a shared evaluation harness for configurations A/B/C.

Gate:

- fixed-rig head parity is established before cross-rig training;
- a held-out-rig test contains no retargeted, corrupted, or augmented derivative of its source motions in training;
- local-axis reparameterizations preserve FK and invariant critic outputs;
- the variable model improves relevant held-out-rig metrics without unacceptable regression on the known target rig;
- A/B/C ablations are recorded. Configuration C is adopted only if it clears this gate.

R3 is complete only after held-out-rig, held-out-corruptor, and full invariance evaluations pass.

### Phase 15 — multi-task infrastructure and optional pretraining

Deliver configuration `D. variable_j_graph_tcn_parts_multitask` infrastructure before adding many tasks:

- dataset/task-specific batch scheduling and label masks;
- balanced sampling by dataset, task, skeleton, defect, severity, and style/action class;
- per-task raw/weighted loss, sampling rate, validation metric, shared-gradient norm, and periodic gradient-cosine logs;
- explicit fixed loss weights first; GradNorm, PCGrad, uncertainty weighting, or task-specific adapters only after measured conflict;
- optional structured-mask pretraining over whole joints/limbs/time spans and motion targets, with FK and bone-length constraints.

Gate:

- unavailable labels produce exactly zero gradients for their heads;
- task diagnostics are finite and attributable;
- no task is dominated by dataset size or joint count;
- pretraining is retained only if an ablation improves held-out defect/critic metrics.

### Phase 16 — auxiliary datasets, one at a time

Add exactly one adapter/task at a time, choosing the first target from contact/load or action labels:

- 100STYLE for style and locomotion only;
- UnderPressure for synchronized contact/load, with timing tests;
- BABEL for temporal action intervals;
- AMASS for broader motion/pretraining;
- HumanML3D for later semantic/text alignment through the canonical graph path.

Before adding AMASS, BABEL, or HumanML3D, resolve overlapping source identity through underlying AMASS paths. Audit dataset-ID predictability from the shared latent and sampling-rate fingerprints. Keep the new task only when an ablation improves its intended held-out metric without unacceptable regression elsewhere.

### Phase 17 — human preference calibration and active hard negatives

Deliver:

- an A/B/indistinguishable labeling workflow comparing motions for the same request and recording reason/confidence;
- hard pairs from current agent outputs rather than trivial clean-versus-catastrophic comparisons;
- preference tuning primarily in a high-level adapter/ranking head, with lower layers frozen or slowed and reliable synthetic/contact tasks replayed;
- agent-generated failures, real retargeter failures, and counterfactual repair outcomes as new hard negatives;
- refreshed calibration and regression evaluation before deployment.

Gate:

- preference tuning improves held-out human agreement;
- deterministic contact/anatomy and dense localization metrics remain inside regression budgets;
- low-confidence/OOD motions cannot be optimized aggressively against the preference score;
- every deployed model retains configuration, data lineage, normalization, code version, and evaluation manifest.

## 6. Required ablation ladder

All learned critics must share one report and inference interface:

```text
A. fixed_rig_flat_tcn
B. variable_j_local_graph_tcn
C. variable_j_graph_tcn_parts
D. variable_j_graph_tcn_parts_multitask
```

Each promotion requires comparison on:

- the known target rig;
- held-out source takes;
- held-out corruption mechanisms;
- at least one held-out humanoid rig for B–D;
- clean-input degradation and calibration;
- runtime/memory and long-window consistency.

A more general model does not replace a simpler model merely because it is newer.

## 7. Cross-cutting quality gates

At the end of every phase:

1. run formatting/linting, mypy for the implemented core, unit/integration tests, and a relevant CLI smoke command;
2. update `IMPLEMENTATION_STATUS.md` with implemented behavior, commands, results, tolerances, unsupported cases, assumptions, ablations, and the next three tasks;
3. update user docs, data schemas, and changelog when interfaces or persisted formats change;
4. keep generated artifacts and model checkpoints versioned by content/config, not silently overwritten;
5. stop progression if the repository or prior end-to-end demo is broken.

Release blockers at every applicable stage:

- coordinate conversions are implicit or spread through core code;
- raw local rotations are compared across rigs without anchoring;
- missing data or padded joints can affect valid predictions;
- source lineage is lost or related derivatives cross splits;
- corruption metadata leaks into model input;
- high-joint-count rigs dominate dense losses;
- style is used as a quality target;
- part queries are introduced without utilization/collapse tests;
- a position-only repair path claims to recover unobservable twist;
- automated edits can accept hard-constraint or protected-metric regressions;
- the variable model is adopted without the fixed-rig ablation.

## 8. Dependency path

```text
bootstrap
  -> rotation/data/FK
  -> BVH/canonicalization/preview
  -> contacts/cycles/metrics
  -> retarget + deterministic repair + demo
  -> source preparation + stable splits
  -> corruptions + dataset
  -> fixed-rig TCN critic
  -> repair search + retrieval/orchestration
  -> Blender/HTTP product integration
  -> canonical graph + invariance harness
  -> rig semantics + paired multi-rig data
  -> variable graph/TCN/part encoder
  -> cross-rig consistency + adoption ablation
  -> multi-task infrastructure
  -> one auxiliary dataset at a time
  -> preference calibration + active hard negatives
```

Parallel work is safe only inside a phase when it shares already-frozen contracts. Examples include independent metric implementations after `MetricResult` is fixed, independent corruptors after corruption/provenance schemas are fixed, or Blender and HTTP integrations after `MotionService` schemas are stable.

## 9. Immediate execution backlog

The deterministic foundation, Phase 6 dataset, fixed-rig critic hardening, CMA failure
decomposition, adaptive parameterization, corrected window-speed semantics, production CMA, local
spectral jitter representation, autonomous deterministic closed loop, and the first fixed-rig
perceptual-quality gate are complete. The perceptual residual scores 41.7% on held-out-source
directional pairs, so optimization is correctly stopped. The next backlog is:

1. Complete the prepared 73-pair human audit. It contains all held-out source/style pairs, every
   perturbation family, six equivalence controls, both admissible adversarial pairs, and the five
   confident absolute-score misrankings first.
2. Analyze human/synthetic agreement by family, apply the resulting CERTAIN versus HUMAN_REQUIRED
   policy, then run matched frozen and perceptually fine-tuned relative-comparator ablations.
3. Invoke anchor-relative trust-region CMA only if a comparator clears 65% held-out-source
   accuracy plus equivalence and exact swap-consistency gates. Keep variable-rig graph/attention
   paused; the absolute residual critic remains an explicit ablation.
