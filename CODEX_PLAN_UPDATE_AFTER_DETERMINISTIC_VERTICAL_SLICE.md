# Plan update after the deterministic vertical slice

Read this as a delta to the original implementation brief and to the current `IMPLEMENTATION_STATUS.md`. Do not discard or rewrite the completed deterministic pipeline unless a new invariant test exposes a correctness problem.

## 1. Assessment of the current implementation

The completed work is still valid and is the intended foundation. In particular, preserve:

- quaternion/SO(3) math and physical-time derivatives;
- `Skeleton` / `MotionClip`, FK, NPZ, BVH ingestion, axis/unit canonicalization, and resampling;
- contact, gait-event, phase, cycle, and deterministic metric machinery;
- retargeting, two-bone IK, foot-slide corruption, foot lock, loop closure, and the end-to-end demo;
- all current tests, CLI behavior, and numerical tolerances.

Do not jump directly from this milestone to a universal multi-dataset model. Keep a fixed-rig correctness baseline throughout the project.

## 2. Immediate deterministic hardening

Complete the currently listed deterministic gaps, with the following priority:

1. cadence, stride-length, stride-time, step-width, and left/right asymmetry metrics;
2. configured joint-limit metrics with explicit per-rig validity/confidence;
3. compact JSON summaries plus dense NPZ artifacts, so model-training reports do not put large arrays in JSON;
4. at least one non-identity retarget test and one real BVH smoke test, preferably a 100STYLE forward-walk clip supplied through a local path;
5. report separate values for IK target residual and actual post-repair sole/marker slip.

Report plots are useful but should not delay the corruption dataset and fixed-rig critic.

The current relaxed contact-speed threshold is acceptable only for the synthetic demo. For generated learning data:

- retain the clean/source contact interval as a label with `contact_source=synthetic_or_source_truth`;
- separately compute the contact estimate that would actually be available at inference;
- never feed privileged source contact labels to the critic as ordinary input unless the same information is available at deployment;
- store contact and ground confidence explicitly rather than changing thresholds silently based on the known corruption.

Before treating foot lock as a production repair operator, add joint-limit checking and protection of a simultaneously planted opposite foot. This is not a blocker for training a critic, because the original clean clip—not the deterministic repair—is the clean target.

## 3. Add forward-compatible metadata now, but postpone the full variable-rig model

Extend the current structures or add lossless adapters so every clip can later expose:

- explicit unit, handedness, up/forward-axis, and original-to-canonical transforms;
- rest/bind/pose-anchor distinctions where known;
- `rotation_valid`, `position_valid`, `contact_valid`, and confidence masks;
- deform/helper/controller/end-site flags;
- semantic role, side, functional part, and confidence, with `UNKNOWN` allowed;
- ground-plane source/confidence;
- immutable provenance: source dataset, source clip/take ID, importer version, retargeter version, corruption lineage, and split lineage.

Do this without breaking the existing `Skeleton`/`MotionClip` API or changing current demo numbers. A wrapper or adapter to future `SkeletonGraph` / `MotionGraph` structures is preferable to a large rewrite.

Do not treat plain BVH identity rest rotations as trusted anatomical local-axis metadata. Keep them valid for the current fixed-rig/BVH path, but expose low/unknown axis confidence for cross-rig use.

## 4. Revised next model sequence

Maintain four configurations behind a common training/inference/report interface:

```text
A. fixed_rig_flat_tcn
B. variable_j_local_graph_tcn
C. variable_j_graph_tcn_parts
D. variable_j_graph_tcn_parts_multitask
```

Implement and evaluate them in that order. Never replace A with a more complicated model without an ablation on the known target rig and held-out rigs.

### Model A: fixed-rig baseline comes first

After the first corruption dataset exists, implement the simple fixed-rig critic:

```text
motion [B,T,J,F]
 -> shared per-joint MLP or fixed-rig flattening baseline
 -> bidirectional dilated temporal TCN
 -> dense defect/time head
 -> part-level defect head
 -> clip ranking/quality head
```

Its purpose is to prove that the supervision works before debugging variable topology. Include a tiny-overfit test and a CPU smoke train.

### Models B/C: revised variable-rig encoder

Only after Model A works, implement:

1. a static skeleton encoder, run once per rig and cached;
2. a per-frame dynamic joint encoder;
3. fusion into joint tokens `H0[B,T,J,D]`;
4. repeated spatial and temporal reasoning while retaining the joint dimension;
5. fixed semantic anatomical-part pooling/attention and part-to-joint feedback;
6. joint-, part-, frame-, and clip-level task heads.

The intended repeated block is approximately:

```text
joint tokens H
 -> local typed skeleton message passing                 # spatial
 -> shared per-joint bidirectional dilated TCN          # temporal
 -> joint-to-fixed-part masked pooling/attention
 -> part-to-part/global attention
 -> broadcast part context back to member joints
 -> pointwise MLP
 -> repeat
```

Part/global communication need not occur after every graph/TCN pair; begin with every two blocks.

## 5. Static and dynamic feature separation

The static encoder must use only rig/rest information and be stable across all animations of that rig. It should emit conceptually separate projections:

```text
static_semantic[j]  # anatomical/functional role
static_rig[j]       # morphology, exact axes, helper layout
```

The dynamic encoder receives frame-specific state. Use two streams:

```text
universal kinematic stream:
    root/facing-relative global positions
    metric and body-normalized velocities
    root speed/yaw rate
    marker heights and ground-relative motion
    optional accelerations

rig-coordinate stream:
    local rotations
    SO(3) angular velocities
    twist/joint-coordinate information
    joint-limit coordinates
    axis/rotation validity and confidence
```

Canonicalize units and world axes in preprocessing, but do not assume raw local rotations are directly comparable across rigs. Compute angular velocity with the SO(3) logarithm, not by subtracting Euler angles, quaternions, or 6D vectors.

Fuse with a debuggable concatenation MLP first. Do not start with elaborate FiLM/gating.

## 6. Fixed anatomical outputs, not `J / K` arbitrary groups

For humanoids, map a variable number of rig joints into a fixed semantic vocabulary. Never divide joints into equal-size buckets.

Start with a small fixed set such as:

```text
root/pelvis
trunk/spine
head/neck
left arm
right arm
left leg
right leg
other/accessory
```

Optionally split each leg into proximal leg and foot once contact diagnostics need it. The schema should permit a later finer canonical vocabulary of approximately 15–20 anatomical segments.

For every frame:

```text
H[t,j]  # variable number of precise rig-joint tokens
P[t,k]  # fixed number of anatomical-part tokens
B[t]    # one whole-body token per frame
C       # one whole-clip token
```

Keep all levels; pooling must not destroy `H`.

Initial part pooling should use deterministic semantic masks and masked mean or masked attention. Learned free global queries are deferred until utilization/collapse tests exist. An absent optional part must produce a defined masked token plus validity bit, never NaN.

`B[t]` may be a whole-body query over `P[t,*]`. `C` may be a clip query or masked temporal pooling over `B`. These are readouts/working-memory tokens for frame- and clip-level heads, not replacements for detailed joint/part tokens.

## 7. Revised corruption-data contract

The perturbation idea is now the primary source of dense critic supervision. Build it before multi-dataset learning.

Every corruption result must contain at least:

```text
corrupted_motion
clean_motion
corruption_family
corruption_mechanism
intervention_mask[t,j,channel]
symptom_mask[t,part,defect]
responsibility_target[t,part,defect]
responsibility_confidence
measured_metrics_before
measured_metrics_after
measured_severity
preference_confidence
source_motion_id
corruption_seed
split_lineage_id
```

The three masks have different meanings:

- `intervention`: exact channels changed by the generator;
- `symptom`: where the observable defect manifests;
- `responsibility`: parts an editor may reasonably manipulate/report.

Do not train fine joint-level “causal blame” as exact truth. Strongly supervise defect type, time interval, and functional part; treat exact joint blame as weak localization unless counterfactual evidence exists.

### Hard and soft perturbations

Automatically assert `clean > corrupted` only for hard negatives whose measured defect clearly worsens, such as:

- stance-foot sliding;
- ground penetration/floating;
- loop discontinuity;
- joint-limit violation;
- sharp pop/jitter;
- explicit root/leg speed inconsistency.

Do not automatically label mild arm-swing, pelvis-bounce, stride-width, torso-lean, or style changes as worse; they may improve the clip or represent an intended style. Use them only when the measured/perceptual order is unambiguous or later acquire human pairwise labels.

Generate ordinal supervision from measured outcomes, not requested corruption parameters:

```text
clean > mild > medium > severe
```

only when the post-generation metrics verify that order.

For each important defect family, implement multiple mechanisms and hold out at least one mechanism from training. For foot slide, examples include root drift, leg/IK trajectory edits, imperfect lock blending, and root/leg retiming mismatch. This is required to prevent corruption-fingerprint learning.

Start with single-defect examples, then add controlled multi-defect compositions.

## 8. Dataset and split rules

Split by original source take before creating windows, mirrors, retargets, topology variants, or corruptions. Every descendant shares one immutable split-lineage ID.

The neural input must never include:

- source dataset ID;
- filename/path;
- corruption mechanism/type;
- requested severity parameter;
- split ID.

These remain metadata only.

Filter presumed-clean mocap and store `clean_confidence`; do not equate “mocap” with perfect quality.

Before claiming generalization for one defect, evaluate on:

- held-out source clips;
- a held-out corruption mechanism;
- later, actual retargeter/agent failures.

## 9. Cross-rig training rules

Existing multi-task datasets do not by themselves provide diverse rig topology. Create explicit same-motion/different-rig pairs by:

- exact identity-helper insertion;
- safe marker-leaf additions;
- controlled spine/helper variants;
- at least two genuinely different humanoid rigs;
- proportion changes only through validated contact-aware retargeting.

Do not enforce equality of raw rotations or complete joint embeddings across rigs. Use, in order:

1. prediction consistency;
2. fixed-part semantic contrastive consistency;
3. known anchor-joint semantic contrastive consistency;
4. avoid full-latent MSE except exact controlled augmentations.

Align only semantic projections. Preserve morphology/local-axis information in rig-specific projections.

## 10. Multi-task datasets come after the critic baseline

After Models A–C and corruption training work, add one auxiliary task at a time:

```text
100STYLE       -> style / locomotion supervision
UnderPressure  -> contact / load supervision
BABEL          -> frame-level action intervals
AMASS          -> masked motion/self-supervised pretraining
HumanML3D      -> motion-text semantics
MotionPercept  -> human pairwise preference A > B
```

Use dataset/task-specific batches and explicit label masks. A missing label must cause exactly zero loss/gradient for that head. Log raw losses, weighted losses, shared-backbone gradient norms, and occasional inter-task gradient cosine. Begin with staged training and fixed loss weights; add GradNorm/PCGrad only after measured interference.

Human preference fine-tuning must not erase reliable low-level contact/defect behavior. Use lower shared-layer learning rates, replay, or an upper task adapter.

## 11. Required tests before adopting the variable-rig model

Add these incrementally as the corresponding feature exists:

- joint permutation equivariance;
- padding and random-padded-value invariance;
- batch-composition invariance;
- identity-helper FK/semantic invariance;
- local-axis reparameterization invariance;
- world translation/yaw and uniform-scale behavior;
- quaternion-sign and rotation-wrap invariance;
- 30/60/120 fps diagnostic consistency;
- cyclic shift equivariance for verified loops;
- overlapping-window inference stability;
- zero gradient for unavailable task labels;
- held-out corruptor evaluation;
- held-out skeleton/source-lineage evaluation;
- source-dataset probe on shared latents;
- part-query utilization/collapse tests if learned queries are introduced.

## 12. Revised execution order from the current status

Use this immediate order:

1. Finish deterministic metric breadth, compact/dense report split, and one non-identity/real-data validation path.
2. Add forward-compatible coordinate, modality, confidence, provenance, and split-lineage metadata without changing current outputs.
3. Implement the corruption-result schema and an initial hard-negative catalog with postcondition validation and stable splits.
4. Train/evaluate `fixed_rig_flat_tcn` with tiny-overfit and smoke tests.
5. Add canonical graph adapters, semantic mapping, variable-`J` batching, and invariance tests.
6. Implement the static/dynamic/two-stream joint encoder and `variable_j_local_graph_tcn`.
7. Add fixed anatomical part tokens inside the repeated spatial-temporal backbone and compare against Model B.
8. Create cross-rig pairs and consistency losses.
9. Add auxiliary datasets one at a time with ablations.
10. Add human preference and agent-generated hard negatives only after the base critic is reliable.

Update `IMPLEMENTATION_STATUS.md` after each milestone. Include exact ablations comparing A/B/C/D; do not describe the variable-rig architecture as superior until measurements support it.
