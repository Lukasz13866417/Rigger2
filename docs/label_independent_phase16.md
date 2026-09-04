# Phase 16 label-independent preparation

## Scope and collection boundary

Phase 16 prepares work that does not depend on the pending Phase 15 perceptual labels. The Phase
15 mannequin pilot remains the active collection protocol. Its stimuli, render implementation,
questions, scales, randomization, playback rules, and persisted observation schema were not
changed. The new candidate pool and every synthetic fixture live under a separate Phase 16
artifact root and are not part of the active playlist.

No current human observations were fitted, summarized, or used to select motions during this
milestone. No perceptual critic was retrained, no perceptual CMA run was started, and no
graph/attention critic was implemented.

## Frozen active protocol

`artifacts/phase15_mannequin_pilot/protocol_freeze.json` is the content-addressed freeze for the
active pilot. It records:

- protocol ID
  `human-evaluation-protocol:sha256:26cd8c93717bc69a80489c5a4afb8e015f50209567162686723eff715e73413a`;
- freeze ID
  `protocol-freeze:sha256:06fbc1f189aecc5d1d9714329414c64015d91f87e31f238c3b78e7e02f248018`;
- immutable snapshot ID
  `active-pilot-snapshot:sha256:9a840a3481a8d9593d3e358c0c04886c460d75aa034e2912fcb999a1ea532122`;
- mannequin/render identity and implementation hashes;
- orthographic camera, 60 Hz/two-cycle playback, first-pass acknowledgement, view, zoom, speed,
  orbit, and overlay settings;
- the exact naturalness/style/pair questions, anchored 1–7 scales, confidence, reason, interval,
  and difficulty semantics;
- the anchor set and cross-session hidden-repeat policy;
- scheduler version, seed, exact schedule ID, and adaptive-comparison contract; and
- `motionlab.subjective_observation.v2` plus served-playlist/inspection-telemetry schemas.

The freeze hashes 133 immutable active-pilot files, including 129 NPZ motion assets. The raw
observation and served-playlist logs are handled differently: their byte prefixes at freeze time
are immutable, while valid suffix appends remain allowed during collection. Validation fails on
asset drift, prefix edits or truncation, schema drift, or an implementation change without a new
registered protocol/render identity. Future rendering or UI changes therefore require a new
protocol version and new hashes.

## Separate unlabeled candidate pool

The v2 render-validated pool is in
`artifacts/phase16_label_independent/unlabeled_candidate_pool`. Its 625 records comprise 560
`UNLABELED` future-human candidates and 65 `CALIBRATION_ONLY` ladder stimuli. Every record has a
null subjective preference and `critic_training_eligible=false`; upstream pair directions are not
imported. All 625 motions are renderable under the frozen mannequin protocol and reference its
freeze/render hashes.

| Candidate group | Retained |
| --- | ---: |
| Clean 100STYLE windows | 22 |
| Hard-feasible subtle variants | 150 |
| Objective equivalence controls | 20 |
| Over-smoothed or rigid variants | 120 |
| Coordination or phase variants | 100 |
| Pelvis or torso relationship variants | 90 |
| Phase-aligned style-incoherence variants | 40 |
| Existing adversarial optimizer outputs | 18 |
| Calibration-only severity ladders | 65 |
| **Total** | **625** |

The pool spans 12 source clips. Deterministic screening marks 588 retained candidates feasible;
the summary identifies 230 mechanically valid stress candidates specifically across the
coordination/phase, pelvis/torso, and style-coherence categories. All 18 Phase 9/10 adversarial CMA
outputs are preserved with their original corpus lineage: six original GroupNorm exploits and 12
later LayerNorm red-team additions. Their optimizer seeds, selected families, source runs, run
configurations, trajectories, optimization reports, and spline coefficients are copied into 72
hash-validated, content-addressed evidence files. Candidate IDs and evidence-bundle IDs bind each
row to its corpus record, normalization, source/final motion hashes, optimizer history, linked
coefficient artifact, and final control-point setting. No Phase 16 perceptual CMA was started because
the active-collection stop condition takes precedence. The generator rejected 201 inputs or new
attempts rather than silently relaxing constraints: 164 upstream candidates and 37 newly generated
variants. Two of the 24 input clean windows were not renderable as frozen two-cycle stimuli,
leaving the 22 retained clean windows.

Each candidate persists content-addressed motion and phase-reference paths, motion/file/visual
hashes, source lineage, origin identifier, mechanism/family, parameter or severity metadata,
deterministic feasibility evidence, frozen-protocol renderability, and explicit future-query and
quality-evaluation eligibility.

## Human-analysis readiness

`motionlab.perceptual.subjective_analysis.v3` now has a complete file-producing path for the real
append-only observations once collection is declared sufficient. It exports:

- ordinal latent item quality with approximate 95% intervals using the full source/variant
  covariance (explicitly marked as L-BFGS/Laplace approximations), source baselines and
  source-relative variant effects, per-session threshold calibration, and per-rater consistency;
- a joint Davidson-style four-outcome Bradley–Terry model for direct comparisons, preserving tie
  and `not_sure` propensities;
- a separate direct-pair-versus-ordinal report whose ordinal fit explicitly excludes the pair rows
  being evaluated;
- within-rater hidden-repeat reliability by measurement cohort, matched same-anchor cross-session
  drift (separate from composition-sensitive pooled session summaries), category utilization,
  response-time/replay/inspection behavior, judgment difficulty, A/B order checks, fatigue,
  transitivity, and family/style/source breakdowns;
- human/synthetic agreement by perturbation mechanism while keeping the two supervision sources
  separate; and
- a ranked list of stimuli needing additional judgments.

For the current protocol, the production command requires the canonical freeze-recorded manifest,
observation log, and served-playlist paths. Before fitting, it authenticates every completed row
against its exact served event and manifest trial, including stimulus order, source/variant/render
lineage, family/style, population/cohort/training eligibility, pair metadata, viewing settings,
repeat/anchor fields, response semantics, and inspection telemetry. Duplicate, unknown, malformed,
unserved, or inconsistent rows fail closed; analysis output must be fresh and outside the active
pilot tree. Deterministic dummy observations test the pipeline and its leakage boundary only.
Their fitted numbers are not perceptual findings, and no dummy-derived conclusion is promoted
here. The analysis command continues to set
`critic_retraining_started=false` and blocks retraining until the human pilot report is inspected.

## Active-query scheduling

The Phase 16 scheduler produces both `RATE_SINGLE(stimulus_id)` and
`COMPARE(stimulus_a, stimulus_b)`. It scores posterior uncertainty and interval overlap, repeat
disagreement, source/style/family undercoverage, critic–human disagreement, confident but
unconfirmed critic predictions, adversarial outputs, comparator impact, and required repeat or
anchor coverage. Critic values are prioritization inputs only, never labels.

Only explicit within-rater hidden-repeat groups contribute repeat-disagreement priority; ordinary
cross-rater disagreement does not masquerade as repeat evidence. Undercoverage uses all prior
sessions, while source/style/family balancing remains session-local. Selection enforces minimum
repeat spacing, hard per-stimulus and per-pair judgment caps (including projected batch use), and
per-source, per-style, per-family, and current-session balance. Pair construction uses bounded
global/source/family nearest-neighbor lists rather than enumerating all pairs. On the deterministic
dummy fixture, 558 of 625 pool records are eligible, 3,026 bounded pair candidates are considered,
and the capped output contains 12 single ratings plus eight comparisons. Preserved adversarial
outputs remain eligible for perceptual querying even when they fail deterministic production
constraints. This schedule is synthetic readiness evidence only and is not automatically enqueued
into the active Phase 15 pilot.

## Post-label forensics

`build-post-label-forensics` creates a fresh, standalone analysis bundle for one stimulus or one
exact pair only after every observed rater has completed the full blinded manifest contract and
the frozen canonical observation log contains an authenticated completed response for that exact
request. This prevents source/variant diagnostics from unblinding a later hidden repeat or
adaptive judgment. It refuses incomplete, unrated, forged/unlinked, or mismatched requests and
refuses to write inside the active pilot directory. The
viewer synchronizes an analysis-only articulated motion view and optional skeleton overlay with a
variant/source toggle, root and foot trajectories, contact evidence, pelvis height/orientation,
joint angular velocity and jerk, deterministic metrics, and optional temporal critic outputs.

The active blinded evaluator does not import or expose this viewer or its diagnostic payload.

## Variable-rig data infrastructure only

`motionlab.rigging.graphs` adds immutable `SkeletonGraph`, `MotionGraph`, and padded
`VariableRigBatch` contracts without defining a learned model. The representation includes
canonical node/edge order, semantic role/part/side mappings, explicit helper/controller/end-site
masks, coordinate/basis provenance, separate universal and rig-coordinate static/dynamic streams,
and authoritative frame/joint/node/edge masks.

Tests cover joint permutation, zero padding, identity-helper insertion, declared local-axis
reparameterization, importer unit/world-axis canonicalization, and mixed joint/frame counts. These
are data-contract guarantees, not evidence for starting a graph or attention architecture. The
detailed field layouts are in `docs/variable_rig_data_contract.md`.

## External and internal comparator readiness

The MotionCritic compatibility check correctly returns `deferred_incompatible_representation`.
The current representation is a 23-joint MotionLabHuman rig with 160-frame local quaternions;
the expected path is a verified 24-joint SMPL mapping with 60-frame local axis-angle rotations and
root XYZ. No verified mapping, required SMPL assets, or validated 60-frame conversion is present.
No checkpoint was loaded, no mapping was improvised, and no human label was adapted. The prepared
agreement path remains evaluation-only if those representation blockers are resolved later.

The comparator machinery is prepared but did not train. Relative-experiment v3 snapshots and
hashes only train/validation IDs and targets before selection. Held-out labels, audit policy,
targets, features, and metrics are excluded from and cannot influence the validation-loss selection
decision; they are applied or evaluated only afterward. Native relative outputs include forward
and swapped probabilities plus explicit objective-equivalence provenance. Human ties never become
equivalence controls; only the exact legacy world-origin construction receives an explicit,
provenance-recorded migration.

`evaluate-comparator-readiness` remains a synthetic/precomputed-input checker. The production
post-label wrapper, `evaluate-post-label-comparator-ablations`, re-authenticates the canonical
frozen observations, verifies pilot completion and a completed subjective-analysis v3 report,
aligns direct human A/B outcomes to model pairs by motion-content hash, excludes `not_sure`, and
evaluates `absolute_q`, frozen-relative, and fine-tuned-relative scores on identical observation
targets. It reports held-out-source/mechanism, direct-human, objective-equivalence, adversarial,
swap-consistency, and Wilson metrics. It also requires a completed relative-experiment v3 artifact;
the existing Phase 13 v1 artifact is intentionally not treated as sufficient. Producing a new v3
training artifact remains behind the separate audited-supervision prerequisite and a post-pilot
decision; Phase 15 completion alone does not authorize training.

## Deterministic repair stress test

The additional 100STYLE stress run preserves the production portfolio rule: use a successful
specialized operator first, otherwise use deterministic CMA-ES. A generic CMA baseline is also
measured for every executed case. When a specialist passes, production selection is frozen before
that comparison and the CMA result is explicitly non-selectable; when it fails, the fallback run
doubles as the head-to-head baseline. Clean reference motion is withheld until all optimization is
complete and used only for evaluation, and no learned/perceptual objective is permitted. Every
method record retains target metric before/after, collateral and hard-constraint results,
production and diagnostic decisions, evaluations, runtime, selected artifacts, clean-only
distances, execution purpose, selection eligibility, and route evidence.

The completed artifact executes 14 cases drawn from seven styles not used by the Phase 11
production benchmark: `Akimbo`, `ArmsBySide`, `ArmsFolded`, `Elated`, `HandsInPockets`,
`Heavyset`, and `Proud`. Twelve cases have feasible clean endpoints and full production semantics;
11 succeed, for a production success rate of 91.7%.

| Family | Executed | Production eligible | Production success | Selected route |
| --- | ---: | ---: | ---: | --- |
| Foot slide | 2 | 2 | 2 | Foot-lock/IK; CMA comparison is evaluation-only |
| Floating contact | 2 | 2 | 2 | CMA fallback after contact-height alignment failed |
| Ground penetration | 2 | 2 | 2 | Contact-height alignment; CMA comparison is evaluation-only |
| Joint pop | 2 | 2 | 2 | Local SO(3) smoothing; CMA comparison is evaluation-only |
| Joint jitter | 2 | 2 | 2 | Local SO(3) smoothing; CMA comparison is evaluation-only |
| Speed inconsistency | 2 | 2 | 1 | CMA fallback after direct rescale failed its full gate |
| Loop seam | 2 | 0 | diagnostic only | One specialized result, one CMA fallback |
| Joint limit | 0 | 0 | not applicable | Skipped: no trustworthy authored-limit cases |

Nine successful specialized routes short-circuit the production fallback and remain selected. Each
still has a separately marked, non-selectable CMA comparison; the other five CMA runs are both the
production fallback and the baseline. Across all 14 head-to-head target objectives, the specialist
wins seven, generic CMA wins five, and two tie. These objective-only wins do not override feasibility
or portfolio routing: all 14 portfolio-selected outputs are unchanged from the earlier run. The
loop cases are not promoted to production claims because ordinary source windows lack trustworthy
cyclic-task metadata. Joint-limit coverage remains an explicit skip because the Phase 8/100STYLE
fixed-rig data contains no authored trustworthy-limit cases.

The production route accounts for 1,172 evaluations and 31.47 seconds: all specialist attempts plus
the five required CMA fallbacks. The nine non-selectable comparison runs add 2,209 evaluations and
44.72 seconds of evaluation-only overhead, for 3,381 evaluations (3,191 CMA and 190 specialized)
and 76.19 seconds overall. Speed uses the low-dimensional `SpeedCadenceBlock`, but no standalone
cadence-repair success is claimed because these cases have no explicit cadence task target. The
report SHA-256 is
`815163ed7739390efda2845ba56ac35b722ac6b294b97a2cffaf30fe4cfb1ff9`.

Final per-case evidence is in
`artifacts/phase16_label_independent/deterministic_repair_stress/deterministic_repair_stress.json`.

## Verification

The 76 focused protocol-freeze, candidate/forensics, active-query, subjective-evidence,
comparator/MotionCritic, variable-rig, and repair-stress tests pass in 13.33 seconds. The complete
repository suite passes 226 tests in 107.20 seconds; Ruff formatting/checks pass repository-wide
and mypy passes all 117 package source files.

## Commands

```bash
uv run motionlab validate-active-evaluation-protocol \
  artifacts/phase15_mannequin_pilot/protocol_freeze.json

uv run motionlab generate-unlabeled-perceptual-candidates \
  artifacts/phase8_critic_hardening/dataset \
  --pair-dataset artifacts/phase12_perceptual_quality/dataset \
  --adversarial-manifest artifacts/phase10_adaptive_optimization/adversarial_quality_corpus/manifest.json \
  --calibration-manifest artifacts/phase15_mannequin_pilot/calibration_bank/calibration_bank.json \
  --protocol-freeze artifacts/phase15_mannequin_pilot/protocol_freeze.json \
  --output <fresh-unlabeled-candidate-pool-directory>

uv run motionlab schedule-active-human-queries \
  artifacts/phase16_label_independent/unlabeled_candidate_pool/candidates.jsonl \
  --observations <append-only-observations.jsonl> \
  --session-id <session-id> \
  --output <fresh-active-query-report.json>

# Run only after the blinded pilot is complete for every observed rater.
uv run motionlab build-post-label-forensics \
  artifacts/phase15_mannequin_pilot/pilot_manifest.json \
  --protocol-freeze artifacts/phase15_mannequin_pilot/protocol_freeze.json \
  --observations artifacts/phase15_mannequin_pilot/raw_observations.jsonl \
  --stimulus-id <already-rated-stimulus-id> \
  --output <fresh-forensics-directory-outside-active-pilot>

# Run only after the user declares collection sufficient and the pilot completion gate passes.
uv run motionlab analyze-subjective-pilot \
  artifacts/phase12_perceptual_quality/dataset \
  --pilot artifacts/phase15_mannequin_pilot/pilot_manifest.json \
  --observations artifacts/phase15_mannequin_pilot/raw_observations.jsonl \
  --protocol-freeze artifacts/phase15_mannequin_pilot/protocol_freeze.json \
  --served-playlist artifacts/phase15_mannequin_pilot/served_playlist.jsonl \
  --output <fresh-human-analysis-directory-outside-active-pilot>

# Synthetic/precomputed readiness check only; this is not the authenticated real-label path.
uv run motionlab evaluate-comparator-readiness \
  <synthetic-evaluation-pairs.json-or-jsonl> \
  --predictions <synthetic-three-variant-predictions.json> \
  --evidence-origin synthetic_fixture \
  --output <fresh-comparator-readiness.json>

# After pilot completion, a v3 human report, and a separately authorized v3 relative experiment.
uv run motionlab evaluate-post-label-comparator-ablations \
  artifacts/phase12_perceptual_quality/dataset \
  --pilot artifacts/phase15_mannequin_pilot/pilot_manifest.json \
  --observations artifacts/phase15_mannequin_pilot/raw_observations.jsonl \
  --protocol-freeze artifacts/phase15_mannequin_pilot/protocol_freeze.json \
  --served-playlist artifacts/phase15_mannequin_pilot/served_playlist.jsonl \
  --pilot-report <completed-subjective-analysis-v3/pilot_report.json> \
  --absolute-evaluation <absolute-q-evaluation.json> \
  --relative-experiment <completed-relative-experiment-v3/relative_experiment.json> \
  --output <fresh-post-label-comparator-evaluation-directory>

uv run motionlab evaluate-motioncritic-baseline \
  artifacts/phase16_label_independent/motioncritic/current_rig_contract.json \
  --output <motioncritic-assessment.json>

# Evaluation-only agreement path, usable only after its representation assessment is compatible.
uv run motionlab evaluate-motioncritic-baseline \
  <verified-compatible-motioncritic-rig-contract.json> \
  --human-pairs <authenticated-human-pairs.json-or-jsonl> \
  --predictions <frozen-motioncritic-predictions.json-or-jsonl> \
  --output <fresh-motioncritic-human-agreement.json>

uv run motionlab run-deterministic-repair-stress \
  artifacts/phase8_critic_hardening/dataset \
  --prior-production-report artifacts/phase11_deterministic_closed_loop/production_cma/production_benchmark.json \
  --output <fresh-deterministic-repair-stress-directory>
```

## Stop condition and next decision

Continue the frozen Phase 15 collection without introducing Phase 16 candidates into it. Once the
user declares that enough observations have been collected, validate the freeze and produce the
human-reliability/pilot report first. Only an explicit review of that report may authorize critic
retraining, perceptual CMA, or a graph/attention model milestone.
