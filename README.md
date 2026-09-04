# MotionLab

MotionLab is a deterministic-first Python toolkit for importing, inspecting, grading, and
repairing humanoid walking animation. The initial implementation targets one known humanoid rig
and grows toward a variable-rig learned critic only after the fixed-rig pipeline is verified.

The project is in early implementation. See `IMPLEMENTATION_STATUS.md` for completed behavior and
`IMPLEMENTATION_PLAN.md` for the staged roadmap.

## Development setup

Python 3.11 or newer is required.

```bash
uv sync --extra dev
uv run motionlab --help
uv run ruff check .
uv run mypy motionlab
uv run pytest
```

Downloaded datasets, generated experiment artifacts/checkpoints, and human-evaluation logs are
kept local and excluded from Git. See `data/100style/README.md` for real-data setup and
`docs/calibrated_subjective_pilot.md` for the labeling workflow. Paths under `artifacts/` in the
implementation reports describe local research runs, not files included in a fresh clone.

## Current commands

```bash
motionlab version
motionlab check-config configs/default.yaml
motionlab inspect-bvh walk.bvh --json
motionlab convert-bvh walk.bvh --output walk.npz --target-fps 60 \
  --source-take-id take-001 --split train --clean-confidence 0.95
motionlab make-synthetic output/synthetic_walk.npz
motionlab validate-motion output/synthetic_walk.npz
motionlab preview output/synthetic_walk.npz --output output/frame.png
motionlab metrics output/synthetic_walk.npz --output output/report.json --target-speed-mps 1.2
motionlab retarget source.npz target_reference.npz --output output/retargeted.npz
motionlab corrupt output/synthetic_walk.npz --type foot_slide --output output/corrupted.npz
motionlab corruption-catalog
motionlab corrupt lineaged_clean.npz --output output/corrupted.npz --artifact-dir output/example
motionlab generate-dataset lineaged_clean.npz --output output/dataset
motionlab validate-dataset output/dataset
motionlab audit-100style data/100style --output output/real_audit
motionlab train-fixed-rig-critic output/real_audit/dataset --output output/critic
motionlab evaluate-fixed-rig-critic output/real_audit/dataset \
  --checkpoint output/critic/best.pt --output output/evaluation
motionlab optimize-critic-cma output/real_audit/dataset --sample-id sample:sha256:... \
  --checkpoint output/critic/best.pt --output output/cma --family foot_slide --mode red_team
motionlab validate-cma-inverse --output output/cma_inverse
motionlab run-adaptive-oracle-diagnostics output/real_audit/dataset \
  --experiment output/cma_experiment/experiment.json \
  --previous output/oracle_representability/representability.json \
  --failure-matrix output/failure_classification_matrix.json --output output/adaptive
motionlab run-production-cma-benchmark output/real_audit/dataset \
  --output output/production_cma
motionlab run-jitter-spectral-audit output/real_audit/dataset \
  --output output/jitter_spectral
motionlab run-autonomous-deterministic-repair output/real_audit/dataset \
  --output output/autonomous_repair
motionlab generate-perceptual-pairs output/real_audit/dataset \
  --adversarial-manifest output/adversarial_quality_corpus/manifest.json \
  --output output/perceptual_pairs
motionlab train-perceptual-residual output/perceptual_pairs \
  --source-dataset output/real_audit/dataset --backbone output/critic/best.pt \
  --output output/perceptual_critic  # add --human-labels output/human_labels.jsonl after review
motionlab label-perceptual-pairs output/perceptual_pairs \
  --labels output/human_labels.jsonl --evaluation output/perceptual_critic/evaluation.json
motionlab run-perceptual-cma output/perceptual_pairs \
  --source-dataset output/real_audit/dataset \
  --checkpoint output/perceptual_critic/perceptual_residual.pt \
  --evaluation output/perceptual_critic/evaluation.json --output output/perceptual_cma
motionlab prepare-perceptual-human-audit output/perceptual_pairs \
  --absolute-evaluation output/perceptual_critic/evaluation.json \
  --absolute-cma-report output/perceptual_cma/perceptual_cma.json --output output/human_audit
motionlab label-perceptual-pairs output/perceptual_pairs --labels output/human_labels.jsonl \
  --evaluation output/perceptual_critic/evaluation.json \
  --audit-queue output/human_audit/human_audit_queue.jsonl
motionlab analyze-perceptual-human-audit output/perceptual_pairs \
  --audit-queue output/human_audit/human_audit_queue.jsonl \
  --labels output/human_labels.jsonl --output output/human_audit/agreement.json
motionlab run-relative-comparator-experiment output/perceptual_pairs \
  --source-dataset output/real_audit/dataset --backbone output/critic/best.pt \
  --human-labels output/human_labels.jsonl --human-audit output/human_audit/agreement.json \
  --absolute-evaluation output/perceptual_critic/evaluation.json --output output/relative
motionlab repair output/corrupted.npz --operator lock_foot --side left --output output/repaired.npz
motionlab run-demo --workspace demo_workspace
```

Preview rendering requires `uv sync --extra visualization`.

The `metrics` command writes a compact JSON summary and a sibling `*.dense.npz` artifact. The JSON
contains the relative artifact path, format version, motion ID linkage, and SHA-256 checksum.

Corruptions expose separate exact intervention, observed symptom, and confidence-weighted
functional responsibility labels. Training artifacts retain measured hard-negative evidence,
split lineage, and privileged source contacts in a pickle-free, checksummed bundle; privileged and
metadata-only fields are explicitly excluded from neural inputs. See
`docs/corruption_catalog.md`.

The critic-hardening rerun expands the real gate to 12 qualifying 100STYLE source takes, adds
heading-equivalent controls and balanced task sampling, uses family-specific rather than global
synthetic ranking, and forensically audits every sham. The first learned repair experiment uses
two-stage BIPOP-CMA-ES over cubic B-spline residuals with strictly separated red-team and
production modes. See `docs/critic_hardening_cma_experiment.md`.

The follow-up failure decomposition fits exact clean residuals into the current spline space,
measures exact repair-path monotonicity, compares GroupNorm with per-frame LayerNorm, preserves
all red-team exploits, and separates deterministic production targets from learned perceptual
terms. It finds that LayerNorm fixes repair ordering and window consistency but worsens sham and
off-path exploit resistance, so graph/attention work remains paused. See
`docs/cma_failure_decomposition.md`.

The adaptive optimization follow-up validates CMA-ES on exact feasible inverse problems through
100 dimensions, introduces residual-driven coarse/local/fine spline spaces and defect-specific
parameter blocks, reclassifies the speed feasibility failure, and expands the family-neutral
adversarial-quality corpus. CMA-ES is retained; graph/attention remains paused. See
`docs/adaptive_optimization_diagnostics.md`.

The deterministic closed-loop milestone gives every synthetic benchmark a feasible clean
endpoint, runs production CMA-ES on slide/floating/pop cases, adds local spectral and compact
smoothing parameterizations for jitter, and completes task-directed repair without giving the
optimizer a clean target. CMA-ES is retained, while graph/attention and normalization work remain
paused. See `docs/deterministic_closed_loop.md`.

The fixed-rig perceptual-quality experiment adds hard-feasible subtle-motion pairs, explicit
CERTAIN/HUMAN_LABELED/UNORDERED supervision, a frozen-GroupNorm residual preference head, a local
A/B/equal labeling UI, and a gate-controlled trust-region CMA path. The first model fails the
held-out-source gate at 41.7%, so perceptual CMA is deliberately not run and variable-rig work
remains paused. See `docs/fixed_rig_perceptual_quality.md`.

The next fixed-rig experiment replaces global quality with an exactly swap-consistent three-way
relative comparator and matched frozen/fine-tuned TCN ablations. A 73-pair human audit is prepared;
because it has not yet been labeled, the ablations and anchor-relative CMA remain correctly gated.
See `docs/relative_comparator_experiment.md`.

The Phase 6 dataset builder resamples clean inputs to 60 fps, creates deterministic 160-frame
windows at stride 32, searches for five family-specific observed-severity bins, screens collateral
defects, emits matched foot-slide shams and cross-mechanism consistency targets, isolates held-out
mechanisms, and fits normalization from training samples only. Sources must have complete lineage,
a split, and sufficient `clean_confidence`; reruns with the same generation state reuse identical
content-addressed samples. Its summary reports bin population, rejection causes, source/gait/side
coverage, sham counts, and explicitly does not infer production generalization. See
`docs/data_format.md`.

Forward-compatible coordinate, modality, semantic, ground, and lineage metadata is available
through the lossless `motionlab.core.adapt_motion_clip` adapter. Assign source/split identity once
with `assign_source_lineage` before deriving windows or corruptions. See
`docs/metadata_contract.md`.

All core data uses right-handed coordinates, Y up, +Z forward, metres, seconds, radians, and
active quaternions ordered `[w, x, y, z]`.
