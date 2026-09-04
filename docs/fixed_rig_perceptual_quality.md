# Fixed-rig perceptual-quality milestone

## Decision

The first residual perceptual critic is **not ready to guide optimization**. It reaches 100% on
the held-out perturbation mechanism but only 41.7% on directional pairs from held-out source
clips, below the declared 65% gate. Perceptual CMA-ES was therefore not invoked. The next research
priority is better perceptual supervision and source-generalizing critic evaluation—not a
variable-rig graph model or controller/ranker.

The deterministic mechanical repair stack remains frozen except for concrete bugs. The current
GroupNorm fixed-rig critic remains the backbone baseline, LayerNorm remains an experimental
artifact, and learned hard-defect heads remain diagnostic signals rather than production
objectives.

## Production acceptance and method selection

`ProductionAcceptance` records each raw deterministic metric together with its required band,
threshold, satisfied flag, and excess. Candidate selection is lexicographic:

1. reject candidates that violate hard production invariants;
2. prefer candidates satisfying every required deterministic band;
3. among feasible candidates, maximize the perceptual residual score;
4. break ties with collateral change and edit magnitude.

No extra reward is assigned for driving an already-satisfied mechanical metric toward zero. The
preserved family portfolios include specialized deterministic repair, semantic parameter-block
CMA, and generic adaptive SO(3) CMA fallback. Selection is based on evaluated outputs rather than
method identity.

## Hard-feasible pair data

The generator produced 265 pairs for the current fixed 23-joint 100STYLE rig. Every included side
of every pair satisfies the same deterministic production acceptance contract. One generated
candidate and two source windows were excluded rather than weakening a threshold.

| Supervision/evidence | Count |
|---|---:|
| CERTAIN | 156 |
| HUMAN_LABELED | 0 |
| UNORDERED | 109 |
| Equivalent controls | 22 |
| Preserved adversarial pairs that are hard-feasible on both sides | 2 |
| Preserved adversarial pairs retained as red-team-only | 16 |

CERTAIN labels are limited to strong construction rules such as excessive rigidification,
excessive smoothing, exact equivalence, and the already-reviewed adversarial source preference.
Subtle phase, timing, style, and weight-transfer proxies remain UNORDERED. They are not silently
treated as “source is better.” The two admissible historical exploits carry
`adversarial_origin=true`; the remaining sixteen stay preserved but cannot train a quality ranker
whose domain is hard-feasible motion.

## Residual preference model

The model reuses the current fixed-rig GroupNorm TCN only as a frozen representation backbone. A
separate residual head produces `q(x)`, trained with pairwise logistic loss on `q(a)-q(b)` and an
explicit approximately-equal target. Naturalness, coordination, and rigidity/smoothing auxiliary
heads are active only where reason tags support them. Style and weight-transfer heads are omitted
because no trustworthy labels exist yet.

The held-out evaluation is source-separated and mechanism-separated. The selected head is refit
on training plus validation pairs for the selected epoch count; test data is never used for
selection or tuning.

| Evaluation | Directional pairs | Accuracy | ECE |
|---|---:|---:|---:|
| Held-out source | 24 | 41.7% | 0.584 |
| Held-out mechanism | 12 | 100.0% | 0.125 |
| Adversarial, hard-feasible only | 2 | 100.0% | 0.000 |
| Combined independent directional sets | 38 | 63.2% | — |

Equivalent-control false preference is 0/22 at the declared `|p(A)-0.5| <= 0.1` band. There are
29 held-out UNORDERED subtle pairs, but no accuracy is invented for them; they require human
labels. Human agreement is likewise unavailable because the initial label count is zero.

The worst failure is highly confident: on one held-out excessive-smoothing pair the model assigns
only `1.22e-6` probability to the declared A-better direction. The CMA report records five such
held-out critic misrankings for prioritized inspection.

## Optimization gate and trust region

The gate requires at least 65% directional accuracy on both held-out-source and
held-out-mechanism sets, plus no more than 25% false preference on equivalence controls. The
held-out-source condition fails, so the perceptual CMA command emits a structured not-run report:

- CMA invoked: no;
- perceptual CMA success rate: not estimable;
- exploit rate: not estimable;
- successful refinements: none;
- representative critic misrankings: five queued for human labeling.

The implemented gated path is ready for a later qualified critic: it uses three seeds, bounded
local upper-body SO(3) edits, three optimize/evaluate/recenter rounds, deterministic feasibility
at every round, and human-review status on outputs. It cannot bypass the evaluation gate.

## Human labeling

`label-perceptual-pairs` serves only on localhost. It animates A and B side by side and records
A-better, B-better, or approximately-equal plus optional reason tags. Labels append to a separate
JSONL file as `HUMAN_LABELED`; the immutable generated manifest is not rewritten. Queue priority
combines model uncertainty, critic disagreement, UNORDERED status, adversarial origin, and
held-out mechanisms. A later training run can ingest the append-only file with `--human-labels`;
the latest judgment for each pair overlays its training/evaluation label while retaining the
generated category and preference as provenance.

## Reproduction

```bash
motionlab generate-perceptual-pairs artifacts/phase8_critic_hardening/dataset \
  --adversarial-manifest artifacts/phase10_adaptive_optimization/adversarial_quality_corpus/manifest.json \
  --output artifacts/phase12_perceptual_quality/dataset
motionlab train-perceptual-residual artifacts/phase12_perceptual_quality/dataset \
  --source-dataset artifacts/phase8_critic_hardening/dataset \
  --backbone artifacts/phase8_critic_hardening/critic/best.pt \
  --output artifacts/phase12_perceptual_quality/critic
# After collecting judgments, retrain in a new output directory with:
#   --human-labels artifacts/phase12_perceptual_quality/human_labels.jsonl
motionlab run-perceptual-cma artifacts/phase12_perceptual_quality/dataset \
  --source-dataset artifacts/phase8_critic_hardening/dataset \
  --checkpoint artifacts/phase12_perceptual_quality/critic/perceptual_residual.pt \
  --evaluation artifacts/phase12_perceptual_quality/critic/evaluation.json \
  --output artifacts/phase12_perceptual_quality/perceptual_cma
motionlab label-perceptual-pairs artifacts/phase12_perceptual_quality/dataset \
  --labels artifacts/phase12_perceptual_quality/human_labels.jsonl \
  --evaluation artifacts/phase12_perceptual_quality/critic/evaluation.json
```

The consolidated machine-readable decision is
`artifacts/phase12_perceptual_quality/milestone.json`.
