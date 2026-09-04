# Research-backed architecture addendum: variable-rig, multi-task motion encoder

**Status:** This addendum extends the fixed-rig walking-animation implementation brief. It does **not** cancel the instruction to build the deterministic fixed-rig vertical slice first. It defines interfaces that should be future-proofed now and the architecture to implement only after the fixed-rig data pipeline, metrics, corruption generator, and baseline critic work.

**Precedence:** Where this addendum conflicts with the learned-model sections of the original brief, this addendum takes precedence. The deterministic geometry, grading, editing, serialization, provenance, and testing requirements in the original brief remain in force.

---

# 32. Research verdict

The proposed high-level pipeline is technically sound:

```text
static rig information
    -> static per-joint encoding

frame-specific kinematics
    -> dynamic per-joint encoding

static + dynamic fusion
    -> spatiotemporal skeleton-aware backbone

shared contextual representation
    -> task-specific heads
```

Recent topology-agnostic motion systems independently use nearly this decomposition. In particular, SATA explicitly separates static skeleton features from dynamic motion features, then interleaves local graph processing, global spatial processing, and temporal processing. AnyTop, SAME, HuMoT, SAMoR, and Skeleton-Aware Networks provide additional evidence that cross-skeleton shared representations are feasible.

However, a naive implementation will fail for reasons that are easy to miss:

1. **Local rotations are rig-coordinate-dependent.** Two rigs can perform the same physical motion while having different local rotation matrices because bind poses, pre/post rotations, and bone-roll axes differ.
2. **Joint positions do not determine twist.** A position-only universal representation cannot recover unique animation-ready rotations.
3. **Static geometry alone does not reliably identify anatomy.** Left/right symmetry, arbitrary names, helper bones, controllers, and inconsistent rest poses create ambiguity.
4. **Variable joint count is the easy part.** Padding and masking solve tensor shapes; semantic correspondence and coordinate conventions are the real difficulty.
5. **Most proposed datasets do not actually add topology diversity.** AMASS, HumanML3D, and BABEL largely use a standardized human skeleton. They add motions and labels, not arbitrary rigs.
6. **Naive global pooling destroys localization.** A critic needs per-joint tokens for defect localization, but a topology-independent clip representation benefits from a fixed number of functional-part tokens.
7. **Naive learned part queries collapse.** Without anchoring, every query can attend to the entire body and become redundant.
8. **Synthetic corruption masks are not automatically causal blame labels.** The channel edited to create an error may differ from the body part where the error appears.
9. **Multi-task losses can fight each other.** Contact, style, reconstruction, preference, and repair gradients can conflict or differ greatly in scale.
10. **Dataset overlap can silently invalidate evaluation.** HumanML3D and BABEL are derived from or annotate AMASS material, so split provenance must be tracked globally.

The recommended design is therefore a **dual-stream, hierarchical encoder**:

- a topology-tolerant position/trajectory stream used for shared cross-rig semantics;
- an optional rig-coordinate rotation stream used where rotations are trustworthy;
- persistent per-joint tokens for localized diagnostics;
- fixed-count functional-part tokens for frame/clip tasks;
- a rig-aware decoder or operator system for actual rotation repair.

---

# 33. Scope and staging

## 33.1 Do not jump directly to arbitrary creatures

The first generalization target is:

> arbitrary **humanoid** rigs with different proportions, spine counts, toe bones, twist/helper joints, naming conventions, and local axes.

Do not initially target dogs, birds, spiders, dragons, and mechanical rigs. The data, functional correspondences, contacts, and valid diagnostics are substantially different.

## 33.2 Preserve the fixed-rig milestone

Implement in this order:

1. deterministic one-rig pipeline;
2. one-rig TCN critic and synthetic defects;
3. canonical graph schema and variable-`J` batching tests;
4. multiple humanoid rigs and reliable semantic mapping;
5. shared variable-rig encoder;
6. cross-rig consistency training;
7. auxiliary datasets and multi-task training;
8. human-preference calibration and agent hard negatives.

A fixed-rig baseline is essential. It provides a correctness oracle and reveals whether later generalization hurts the practical product.

## 33.3 Required ablation hierarchy

Maintain these model configurations behind one interface:

```text
A. fixed_rig_flat_tcn
B. variable_j_local_graph_tcn
C. variable_j_graph_tcn_parts
D. variable_j_graph_tcn_parts_multitask
```

Never replace a simpler working model without measuring whether the more general model improves held-out-rig performance without damaging the known target rig.

---

# 34. Canonical data contract

Do not feed native BVH, SMPL arrays, HumanML3D 263-D vectors, or UnderPressure arrays directly into the shared model. Every dataset adapter must emit the same explicit canonical structure.

## 34.1 `SkeletonGraph`

Implement a validated immutable structure approximately like:

```python
@dataclass(frozen=True)
class SkeletonGraph:
    skeleton_id: str

    # Required graph structure.
    parent: NDArray[np.int32]  # [J], root parent = -1
    joint_name: tuple[str, ...]  # length J
    rest_local_offset_m: NDArray[np.float32]  # [J,3]
    rest_global_position_m: NDArray[np.float32]  # [J,3]

    # Coordinate-system information.
    rest_local_rotation: NDArray[np.float32] | None  # [J,3,3]
    pre_rotation: NDArray[np.float32] | None  # [J,3,3]
    post_rotation: NDArray[np.float32] | None  # [J,3,3]
    up_axis: Literal["X", "Y", "Z"]
    forward_axis: str
    handedness: Literal["right", "left"]
    unit_scale_to_m: float

    # Static semantics and confidence.
    semantic_role: NDArray[np.int16]  # [J], UNKNOWN allowed
    functional_part: NDArray[np.int8]  # [J], UNKNOWN allowed
    side: NDArray[np.int8]  # LEFT/RIGHT/CENTER/UNKNOWN
    role_confidence: NDArray[np.float32]  # [J]
    axis_confidence: NDArray[np.float32]  # [J]

    # Bone type and model masks.
    is_deform_joint: NDArray[np.bool_]
    is_helper_joint: NDArray[np.bool_]
    is_controller: NDArray[np.bool_]
    is_end_site: NDArray[np.bool_]
    critic_joint_mask: NDArray[np.bool_]
    repair_joint_mask: NDArray[np.bool_]

    # Optional reference geometry.
    pose_anchor_position_m: NDArray[np.float32] | None  # [J,3]
    pose_anchor_kind: str | None
    pose_anchor_confidence: float

    metadata: Mapping[str, JSONValue]
```

Validation must reject or explicitly flag:

- cycles in the parent graph;
- multiple roots unless a synthetic super-root is inserted;
- parent indices outside range;
- NaN/Inf values;
- zero-length deform bones where not explicitly permitted;
- non-uniform or negative object scaling not baked into the skeleton;
- duplicate stable IDs;
- controller objects mixed with deform joints;
- unit/axis metadata that is unknown but silently assumed.

## 34.2 Separate bind pose, rest pose, and reference pose

These terms must not be conflated:

- **bind pose:** transform state used by skinning;
- **rest pose:** zero-animation local transform convention;
- **canonical anatomical pose:** normalized T/A-like geometry, if available;
- **pose anchor:** any trusted pose used to describe target geometry or rotation coordinates;
- **first motion frame:** dynamic data and only a fallback anchor.

Store `pose_anchor_kind` and confidence. A heterogeneous imported asset may have an unreliable bind/rest pose. For the known game rig, prefer its explicit bind/rest information. For low-quality external data, a selected neutral or first frame can be a fallback, but it must not be represented as universally canonical.

## 34.3 `MotionGraph`

Implement approximately:

```python
@dataclass
class MotionGraph:
    skeleton_id: str
    fps: float
    timestamps_s: NDArray[np.float64]  # [T]

    root_translation_world_m: NDArray[np.float32]  # [T,3]
    local_rotation_matrix: NDArray[np.float32] | None  # [T,J,3,3]

    # FK-derived fields may be cached but must be reproducible.
    global_position_world_m: NDArray[np.float32] | None  # [T,J,3]
    global_rotation_world: NDArray[np.float32] | None  # [T,J,3,3]

    # Per-frame environment and intent.
    ground_plane: NDArray[np.float32] | None  # [T,4] or [1,4]
    target_speed_mps: float | None
    target_heading_world: NDArray[np.float32] | None
    loop_kind: str

    # Modality validity.
    rotation_valid: NDArray[np.bool_]  # [J] or [T,J]
    position_valid: NDArray[np.bool_]
    contact_valid: NDArray[np.bool_] | None

    provenance: Provenance
```

Do not encode “missing” as a numeric zero without an accompanying validity mask.

## 34.4 Canonical coordinate frame

Every importer must convert to:

```text
right-handed coordinates
Y up
+Z forward in the canonical facing frame
metres
seconds
radians
```

Store the original-to-canonical transform and prove round-trip consistency in tests.

For each frame, derive a yaw-only facing transform `C_t` and expose both:

- world-space quantities;
- root/facing-relative quantities.

Do not throw away root trajectory. Root-relative coordinates are useful for semantics, but speed, heading, path matching, and loop displacement require a separate global stream.

## 34.5 Preserve metric and normalized scales

Cross-rig semantics benefit from height-normalized coordinates, but physical diagnostics require metres and seconds. Provide both:

```text
position_root_frame_m
position_root_frame_body_normalized
velocity_root_frame_mps
velocity_root_frame_body_lengths_per_s
body_height_m
leg_length_m
arm_span_m
```

Never normalize away body size and then claim to assess physical speed or clearance in SI units.

---

# 35. Static skeleton encoder

## 35.1 Purpose

The static encoder answers:

> What structural and semantic role does each node have on this rig, and what morphology/local-coordinate information belongs to it?

It runs once per skeleton and can be cached.

## 35.2 Inputs

Per-joint static numeric inputs should include:

```text
root-relative rest/anchor position, normalized and metric
parent-relative rest offset
bone length, metric and normalized
skeleton depth and reverse depth
node degree and child count
subtree size
end-effector flag
helper/deform/controller flags
side encoding and confidence
semantic-role embedding and confidence
functional-part embedding
axis/rest-pose confidence
```

Use typed directed edges:

```text
parent -> child
child -> parent
optional sibling relation
optional same-chain relation
```

Edge features may include normalized rest offset, length ratio, and hop/chain metadata.

## 35.3 Geometry alone is insufficient

Do not expect a GNN to infer all anatomy from topology and rest positions. A left elbow and right elbow are symmetric; knees and elbows can have similar local chain patterns; helper and twist bones can resemble true articulations; arbitrary asset names can be misleading. AnyTop reports residual left/right confusion, and recent systems rely on joint-name or semantic information in addition to graph geometry.

Implement a deterministic `RigSemanticMapper` with this priority:

1. explicit user YAML mapping;
2. recognized naming aliases;
3. hierarchy and geometric rules;
4. optional offline LLM proposal;
5. unresolved `UNKNOWN` with low confidence.

The system must never silently reinterpret a low-confidence unknown joint as a knee or foot.

For an unknown humanoid rig, require or strongly request sparse anchors for at least:

```text
pelvis/root
head
left/right shoulder
left/right elbow
left/right wrist
left/right hip
left/right knee
left/right ankle
left/right toe or foot
```

Sparse anchors are much less work than a complete manual remap and prevent catastrophic semantic errors.

## 35.4 Name robustness

Names are useful but brittle. During training:

- normalize case, separators, namespaces, digits, prefixes, and common language variants;
- map aliases such as `thigh_l`, `LeftUpLeg`, and `mixamorig:LeftUpLeg`;
- randomly drop or corrupt names on a configurable fraction of joints;
- preserve explicit side tokens separately;
- audit performance with all names removed.

Do not rely on a large text encoder in the first implementation. A learned categorical role/alias embedding is easier to debug. Text embeddings can be added later.

## 35.5 Output factorization

Do not force one static vector to be both rig-invariant and losslessly rig-specific. Emit two projections:

```text
static_semantic[j]   # anatomy / functional role
static_rig[j]        # exact morphology / local axes / helper layout
```

The semantic projection can be aligned across rigs. The rig projection remains available to repair/decoding modules.

---

# 36. Dynamic feature construction and encoding

## 36.1 Two dynamic streams

Use two conceptually separate streams.

### A. Universal kinematic stream

This stream should remain meaningful across rigs even when local rotation conventions differ:

```text
root/facing-relative global joint position
root/facing-relative global linear velocity
optional acceleration
height above estimated ground
distance/velocity of configured foot markers
root linear velocity in SI units
root yaw rate
root height
contact estimates and validity
```

This is the principal stream for shared cross-rig critic semantics.

### B. Rig-coordinate rotation stream

Use where local rotations are valid and coordinate metadata is trusted:

```text
local rotation in 6D neural representation
SO(3) angular velocity
optional global/root-frame angular velocity
joint-limit coordinates
rotation-axis confidence
```

The cross-rig model must know whether this stream is valid. Do not align raw local rotations between unrelated rigs.

## 36.2 Why the split is necessary

Local rotation matrices are defined relative to a rig’s rest pose and local axes. Equivalent physical motion can therefore have different local values. Conversely, positions alone leave bone-axis twist unobservable. The universal critic can rely heavily on positions/velocities, but rotation-sensitive diagnostics and animation repair need an anchored rotation stream or a target-rig-aware decoder.

This implies:

```text
universal semantic critic != universal lossless rotation decoder
```

A shared encoder can support both, but the rotation repair path must receive target-rig coordinate information.

## 36.3 Exact derivative rules

Never finite-difference Euler angles, quaternion components, or 6D rotation components.

For local rotation matrices `R_t`, calculate angular velocity with the SO(3) logarithm:

```text
omega_t = Log(R_t^T R_{t+1}) / dt
```

or a central variant:

```text
omega_t = Log(R_{t-1}^T R_{t+1}) / (2 dt)
```

Express and document which frame the vector uses. Add tests for:

- constant angular velocity;
- 359° to 1° wraparound;
- quaternion sign flips;
- near-180° relative rotations;
- temporal resampling.

If quaternions are used internally, enforce sign continuity before interpolation. Use SLERP for resampling and convert to 6D only for neural input/output. Use geodesic rotation loss after reconstructing an orthonormal matrix.

## 36.4 Raw and filtered dynamics

Do not smooth all derivatives before feeding the critic, because jitter is itself a target defect. Expose:

```text
raw velocity / angular velocity
low-pass trend
high-frequency residual = raw - trend
```

Filtering parameters must be in physical frequency units, not “number of frames,” and must use `dt`.

## 36.5 Dynamic encoder

Apply one shared MLP to every valid `(frame, joint)` token:

```text
dynamic_raw [B,T,J,Fd]
    -> modality-aware MLP
D [B,T,J,Dd]
```

Concatenate explicit availability/confidence bits. Missing rotation must not look like identity rotation.

---

# 37. Fusion and spatiotemporal backbone

## 37.1 Fusion

Broadcast static features over time and begin with a debuggable fusion:

```python
h0 = fusion_mlp(
    concat(
        dynamic_embedding,
        static_semantic,
        static_rig,
        modality_mask,
        confidence_features,
    )
)
```

A FiLM/gated modulation layer can be added later. Do not begin with an elaborate modulation mechanism until concatenation is validated.

## 37.2 Keep per-joint tokens alive

The central representation must remain:

```text
H [B,T,J,D]
```

through the contextual backbone. Do not flatten joints permanently and do not globally pool before dense diagnostic heads.

## 37.3 Recommended block

Use several residual blocks that combine:

1. **local typed graph message passing** for parent/child constraints;
2. **temporal mixing** for each joint, initially a shared dilated TCN;
3. **functional-part/global communication** for nonlocal coordination;
4. feed-forward MLP and LayerNorm.

A practical block is:

```text
H
 -> local graph attention/message passing
 -> residual + LayerNorm
 -> temporal depthwise/shared dilated Conv1D over T for each joint
 -> residual + LayerNorm
 -> part-token bridge or occasional global spatial attention
 -> residual + LayerNorm
 -> pointwise MLP
 -> residual + LayerNorm
```

The temporal TCN is applied by reshaping valid joints to approximately `[B*J,D,T]`, sharing weights across joints while conditioning tokens on static role embeddings.

## 37.4 Local graph is not enough

A tree-local GNN needs many layers to connect a wrist to the contralateral ankle. Walking quality depends on long-range coordination. Add one of:

- all-joint spatial attention every few blocks; or
- functional-part tokens that communicate globally and broadcast back.

For humanoids with fewer than roughly 80 active joints, occasional all-joint attention is computationally reasonable. For larger rigs, use bucketing and part-token communication.

## 37.5 Functional-part tokens

Use a fixed set of approximately 8–12 parts, for example:

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

For a walking-specific model, splitting each leg into proximal leg and foot can be useful.

Initially use deterministic masks from the semantic mapper:

```text
P[b,t,k] = masked_attention_pool(H[b,t,:,:], part == k)
```

Then apply part-to-part attention. Preserve `H` for joint localization and use `P` for body/frame/clip tasks.

Only later consider learned part queries. Naive learned queries can collapse into nearly identical global pools. If learned pooling is implemented, add:

- attention-map supervision from known functional groups;
- entropy/diversity regularization;
- query orthogonality diagnostics;
- per-query utilization logs;
- name dropout so the model cannot depend only on labels.

## 37.6 Hierarchical outputs

Produce three representation levels:

```text
H[t,j]  = joint at a frame
P[t,k]  = functional part at a frame
B[t]    = whole body at a frame
C       = whole clip
```

Use them as follows:

| Task | Representation |
|---|---|
| defect localization | `H` and `P` |
| repair proposal | `H` plus rig stream |
| foot contact/load | relevant `H`/markers |
| gait phase | `P` or `B` |
| action interval | `B` |
| style | `C`, optionally reference-conditioned |
| human preference | `C` plus deterministic metrics |
| text-motion matching | `C` |

---

# 38. Variable joint and frame counts

## 38.1 Batching

Use padded tensors and explicit masks for the humanoid phase:

```text
joint tensor: [B,T_max,J_max,F]
frame_mask:   [B,T_max]
joint_mask:   [B,J_max]
token_mask:   [B,T_max,J_max]
```

Bucket batches by similar `T` and `J` to limit wasted work. A graph-batching library is optional, not required.

## 38.2 Masking invariants

Implement these rules everywhere:

- padded attention logits become `-inf` before softmax;
- padded tokens are re-zeroed after residual branches;
- masked mean divides by valid count with a safe clamp;
- masked max handles all-invalid groups explicitly;
- losses are zero for invalid labels and normalized by valid count;
- no BatchNorm statistics may mix padded and valid joints;
- use per-token LayerNorm or carefully masked normalization;
- a sample’s output must not change when unrelated larger samples are added to its batch.

## 38.3 Do not let high-joint-count rigs dominate

A rig with fingers and 20 twist bones must not contribute five times the critic loss of a compact rig. Normalize dense losses by:

1. sample;
2. functional part;
3. valid semantic joint count inside the part.

Exclude fingers, face, control objects, and irrelevant accessories from the initial walking critic. Keep them in provenance and optionally in a passive context stream.

---

# 39. Task heads and what each dataset can actually teach

A dataset need not provide every label. Each batch activates only the heads for which labels are valid.

## 39.1 Required heads

```text
joint_time_defect_head      [B,T,J,K]
part_time_defect_head       [B,T,P,K]
frame_defect_head           [B,T,K]
clip_quality_rank_head      [B,1]
contact_head                [B,T,M]       # configured foot markers/feet
style_head                  [B,S] or embedding
phase_head                  [B,T,2]
repair_head                 [B,T,J,R]     # rig-aware, optional
input_validity_ood_head     metadata/score
```

## 39.2 Dataset reality

### 100STYLE

Provides high-quality BVH locomotion on one 28-bone skeleton at 60 fps, with style and movement type from the file organization. It is valuable for:

- walking motion prior;
- style supervision;
- corruption source clips;
- cycle/contact experiments.

It does **not** provide topology diversity or quality labels.

### AMASS

Provides broad motion and subject variation in a standardized SMPL-family representation. It is valuable for:

- self-supervised motion pretraining;
- human motion variety;
- body-shape variation.

It does not by itself teach arbitrary joint counts.

### HumanML3D

Provides text annotations over processed motion derived largely from AMASS/KIT sources. It is useful for semantic/text heads. Its native 263-D vectors must not bypass the canonical graph importer if the goal is one shared graph encoder.

### BABEL

Provides sequence and temporal action annotations that reference AMASS motion files. It is useful for action/transition heads. It overlaps AMASS provenance and must share global splits with any corresponding AMASS/HumanML3D material.

### UnderPressure

Provides synchronized motion and pressure-insole information for contact/vertical-force learning. Its skeleton and sensor timing require a dedicated adapter and synchronization tests.

### Human preference data

Provides pairwise or ordinal clip judgments. It trains the global quality/ranking head, not precise joint blame.

### Procedural corruption data

Provides the dense labels needed by the critic, but label semantics must be designed carefully as described below.

## 39.3 Multi-dataset does not equal multi-rig

To train variable topology, explicitly create or acquire topology variation:

- retarget the same source clips to several known humanoid rigs;
- insert identity helper joints into chains;
- add/remove marker-only leaves;
- vary spine subdivision;
- use body-proportion variants and IK retargeting;
- optionally add carefully licensed heterogeneous BVH/FBX assets.

Do not assume that mixing 100STYLE, AMASS, HumanML3D, BABEL, and UnderPressure produces a topology-agnostic model. Most of those sources standardize motion rather than preserve original diverse rig structures.

---

# 40. Topology augmentation and paired cross-rig data

## 40.1 Exact identity-helper insertion

A safe synthetic topology augmentation is to split a parent-to-child offset by inserting an identity joint.

Original:

```text
P -> C with offset o
```

Augmented:

```text
P -> H with offset alpha*o
H -> C with offset (1-alpha)*o
H local rotation = identity at every frame
C retains its original local rotation
```

Under ordinary FK conventions, this preserves `C` and downstream global positions/orientations exactly. Test this numerically.

## 40.2 Unsafe joint deletion

Do not arbitrarily delete a dynamically rotating degree-2 joint and assume the motion is unchanged. Merging such a joint generally requires transform baking and may not preserve a constant rest offset. Restrict exact deletion to:

- identity/static helper joints;
- marker-only leaves;
- transformations with an explicitly verified exact bake.

## 40.3 Proportion variation

Uniformly scaling the whole skeleton and root trajectory is easy and exact. Nonuniform limb scaling is not. For changed femur/tibia/arm proportions:

1. define target morphology;
2. retarget end-effector trajectories with IK/optimization;
3. enforce contacts and joint limits;
4. calculate a retarget quality report;
5. reject low-quality pairs.

Do not use bad retargeted pairs as “same semantic motion” positives.

## 40.4 Pair provenance

Every cross-rig pair must store:

```text
source_motion_id
target_skeleton_id
retargeter_version
mapping_version
retarget_quality_metrics
pair_confidence
time_alignment_id
```

All variants of one source clip must remain in the same dataset split.

---

# 41. Cross-rig consistency objectives

## 41.1 Do not equate full embeddings

Do not enforce:

```text
H_rig_A == H_rig_B
```

The latent must retain morphology, local-axis, and repair information. Instead expose:

```text
z_semantic = semantic_projection(H or P)
z_rig      = rig_projection(H)
```

Align only `z_semantic`.

## 41.2 Safer consistency hierarchy

Use, from safest to strongest:

1. **prediction consistency:** same action, phase, contact state, style, and diagnostic outputs for equivalent motions;
2. **part-level contrastive consistency:** corresponding functional parts are close;
3. **anchor-joint contrastive consistency:** known semantic anchors are close;
4. **full latent MSE:** generally avoid except in controlled exact augmentations.

## 41.3 Contrastive objective

For source motion `m` represented on rigs `a` and `b`, use positives:

```text
same motion, different rig
```

and hard negatives:

```text
different motion, same rig
same style/speed but different phase
same pose but different dynamics
```

A triplet form is:

```text
max(0, d(z_a, z_b) - d(z_a, z_negative) + margin)
```

InfoNCE is also acceptable. Always combine contrastive loss with reconstruction/prediction tasks to prevent collapse.

## 41.4 One-to-many correspondences

Twist/helper chains have no one-to-one counterpart on compact rigs. Align:

- functional-part aggregates;
- end-effector trajectories;
- chain endpoint orientation where available;
- prediction outputs;

rather than requiring every helper node to match another node.

## 41.5 Time alignment

Cross-rig representation losses are valid only for aligned physical time. Generate paired variants from the same source timeline or retain an explicit warp map. Do not contrast frame `t` with frame `t` after independent trimming or speed changes.

---

# 42. Synthetic critic labels: separate intervention, symptom, and responsibility

Procedural corruption gives exact information about what the program changed, but not necessarily exact causal blame.

Store three masks:

```text
intervention_mask[t,j,channel]
    exact variables modified by the corruption operator

symptom_mask[t,part,defect]
    where deterministic measurements detect the resulting artifact

responsibility_target[t,part,defect]
    semantic attribution used for diagnostics, with confidence
```

Example: foot sliding can be created by modifying root translation. The intervention node is the root, the visible symptom is the planted foot, and the eventual repair may involve root, pelvis, and leg-chain rotations. Calling any one of those the single “ground-truth bad joint” is misleading.

For the first learned critic:

- supervise **defect type + time + functional part** strongly;
- treat fine per-joint blame as soft/weak supervision;
- evaluate per-joint heatmaps as localization, not causal explanation;
- validate responsibility with counterfactual edits where possible.

## 42.1 Severity ordering

Do not rank corruptions solely by the parameter passed to the corruptor. A larger parameter can be less visible if contact duration or camera-independent geometry differs. After corruption:

1. run deterministic metrics;
2. verify the intended symptom increased;
3. derive measured severity in physical units;
4. only create ordinal pairs when the measured order is unambiguous.

## 42.2 Multiple corruption mechanisms

For every defect class, implement at least two mechanisms before claiming generalization. Examples for foot slide:

- root drift during stance;
- imperfect foot-lock target blending;
- ankle/leg edits that move the planted sole;
- retiming root and leg motion inconsistently.

Hold out at least one mechanism for testing.

## 42.3 Repair targets are nonunique

The original clean clip is one valid repair target, not the only valid repair. A universal repair MSE can over-smooth or erase style. Initially:

- train operator-specific residual repair;
- corrupt one concept at a time;
- add contact/style/root-speed preservation losses;
- predict small tangent-space corrections;
- use deterministic operators and candidate search as the primary repair path.

---

# 43. Self-supervised pretraining

A plain autoencoder can learn an easy identity map. Randomly masking isolated pose coordinates is also too easy because adjacent frames are highly redundant.

Use structured masking:

```text
whole joints over temporal spans
whole limbs/functional parts
contiguous frame spans
motion derivatives
contact intervals
```

Prefer targets that represent motion rather than only static pose:

```text
future/past displacement
linear velocity
SO(3) angular velocity
contact state
gait phase
masked trajectory segment
```

Include FK consistency and bone-length constraints. Keep a held-out corruption/defect evaluation to verify that pretraining actually helps the critic rather than merely reconstruction.

---

# 44. Multi-task training without negative transfer

## 44.1 Task scheduler

Use dataset/task-specific batches rather than forcing every sample into one giant label tensor:

```python
batch = task_scheduler.next_batch()
features = encoder(batch.motion_graph)
losses = heads[batch.task].compute_losses(features, batch.labels)
```

Every loss returns both a scalar and diagnostics.

Balance sampling by:

```text
dataset
task
skeleton/source rig
defect class
severity bucket
style/action class
```

Do not sample proportional only to frame count. Large datasets would drown specialized tasks.

## 44.2 Start with staged training

Recommended curriculum:

```text
Stage A: fixed-rig deterministic pipeline and baseline
Stage B: static role mapper/encoder tests
Stage C: self-supervised motion encoder
Stage D: one-rig synthetic critic
Stage E: variable-rig paired/augmented training
Stage F: style/action/contact auxiliary heads
Stage G: human preference calibration
Stage H: agent-generated hard negatives and active learning
```

Do not activate every loss in the first run.

## 44.3 Loss scaling and gradient diagnostics

At minimum log per task:

```text
raw loss
weighted loss
shared-backbone gradient norm
pairwise gradient cosine similarity, sampled periodically
validation metric
sampling frequency
```

Begin with explicit fixed weights chosen so no task dominates by raw numerical scale. If conflicts remain, evaluate:

- uncertainty-based loss weighting;
- GradNorm for gradient-magnitude balancing;
- PCGrad for conflicting gradients;
- partially task-specific upper layers/adapters.

Do not blindly combine multiple automatic weighting methods. First measure the problem.

## 44.4 Protect low-level tasks during preference tuning

Human preference labels are noisy and clip-level. They must not erase reliable contact or defect representations. During preference fine-tuning, use one or more of:

- freeze static encoder and lower backbone;
- lower learning rate for shared layers;
- replay synthetic/contact batches;
- use a separate high-level adapter/head;
- constrain changes with knowledge distillation from the pre-tuned model.

---

# 45. Dataset normalization, provenance, and leakage

## 45.1 Global source identity

Create a canonical source ID before windowing, mirroring, retargeting, corruption, or caption/action attachment:

```text
canonical_source_motion_id
original_dataset
original_path_or_sequence_id
performer_id if known
capture_session_id if known
underlying_amass_path if applicable
```

All derivatives inherit this ID.

## 45.2 Split before derivation

Assign train/validation/test by canonical source before:

- overlapping windows;
- cycle extraction;
- mirroring;
- retargeting;
- topology augmentation;
- corruption;
- style/text/action label joins.

HumanML3D/BABEL/AMASS overlap must be resolved through underlying AMASS paths, not local filenames.

## 45.3 Frame rate and derivative fingerprints

100STYLE is 60 fps; HumanML3D commonly uses 20 fps; other sources vary. Standardize timestamps and calculate all derivatives with actual `dt`. To reduce dataset-identification shortcuts:

- resample through one tested pipeline;
- use consistent units and filters;
- randomize sampling rate modestly during training while preserving physical time;
- audit whether a linear probe can recover dataset ID from the shared embedding.

High dataset-ID accuracy is evidence that the encoder may be learning capture/export fingerprints.

## 45.4 Style is not quality

Styles such as robot, limp, drunk, old, crouched, or exaggerated may intentionally violate a neutral-walk prior. Never label them “bad” merely because they look unusual. Quality must be conditioned on intended style/task.

---

# 46. Ground contact and foot geometry

The ankle joint is not the sole of the foot. A robust walking critic needs explicit markers:

```text
left heel
left toe
right heel
right toe
```

For each rig, obtain markers through this priority:

1. explicit rig configuration;
2. known foot/toe roles and calibrated virtual offsets;
3. mesh/skinning geometry analysis later;
4. low-confidence fallback.

Estimate the ground plane per clip or accept it as scene input. Do not assume world `Y=0` after arbitrary imports.

Contact labels and metrics must store:

```text
marker identity
contact confidence
contact source: pressure / manual / model / heuristic
ground-plane confidence
```

A low-confidence contact estimate must lower critic confidence rather than silently become truth.

---

# 47. Inference on long and cyclic sequences

Train on fixed windows for batching, but grade arbitrary lengths with overlapping sliding windows and weighted blending. Window-edge artifacts can otherwise resemble animation defects or create trajectory drift.

Rules:

- use circular temporal padding only for verified loops;
- use ordinary masked padding for nonloops;
- blend logits and embeddings with a smooth overlap window;
- retain exact frame timestamps;
- test cyclic time-shift equivariance for loop clips;
- compare whole-clip score stability across window offsets.

---

# 48. Confidence and out-of-distribution reporting

A variable-rig critic must be able to say that an input is poorly understood.

Every report should include:

```text
rig_mapping_confidence
coordinate_system_confidence
rotation_stream_valid_fraction
contact_confidence
ground_confidence
skeleton_OOD_score
motion_OOD_score
head-specific uncertainty where available
```

Low-confidence inputs should still receive deterministic metrics where possible, but learned naturalness/style outputs must be marked uncertain.

Do not let the LLM agent optimize aggressively against a low-confidence critic.

---

# 49. Required invariance and correctness tests

These tests are mandatory before multi-task training.

## 49.1 Graph/tensor correctness

1. **Joint permutation equivariance:** reorder joint arrays and graph indices; outputs reorder identically.
2. **Padding invariance:** append masked joints; valid outputs remain unchanged.
3. **Batch-composition invariance:** evaluate a sample alone and beside a larger rig; outputs match within tolerance.
4. **Mask safety:** padded values may contain extreme random numbers without affecting valid output.
5. **No-NaN all-invalid part:** absent optional parts produce a defined masked output and validity flag.

## 49.2 Skeleton augmentation

6. **Identity-helper insertion:** split an edge with an identity helper; FK and semantic part/clip outputs remain nearly unchanged.
7. **Marker-leaf insertion:** add ignored marker leaves; critic outputs remain stable.
8. **Name dropout:** recognized rigs remain usable when aliases/names are partly removed.
9. **Left/right mirror:** mirror skeleton and motion with role labels swapped; predictions swap appropriately.

## 49.3 Coordinate/rotation correctness

10. **World translation invariance:** root-relative diagnostics stay unchanged.
11. **World yaw equivariance/invariance:** canonical semantic outputs stay unchanged while world heading transforms correctly.
12. **Uniform scale:** normalized semantics stay stable; metric quantities scale correctly.
13. **Local-axis reparameterization:** conjugate local coordinate frames and adjust rotations; FK positions and invariant critic outputs stay stable.
14. **Quaternion sign:** `q` and `-q` produce identical features.
15. **Rotation wrap:** angular velocity remains small across equivalent angle wraparound.
16. **Round-trip import/export:** known-rig FK positions and local rotations survive within tolerance.

## 49.4 Temporal correctness

17. **Resampling invariance:** the same physical motion sampled at 30/60/120 fps gives nearly identical physical diagnostics.
18. **Cyclic shift:** rotating a verified loop in time rotates dense outputs and leaves clip outputs stable.
19. **Sliding-window agreement:** long-sequence output is stable across window start offsets.
20. **Derivative unit test:** constant linear/angular velocity is recovered in SI units.

## 49.5 Learning sanity

21. **Tiny-batch overfit:** each head can overfit a tiny labeled set.
22. **Label-mask test:** unavailable labels produce exactly zero gradient for that head.
23. **Task-gradient logs:** gradient norms/cosines are finite and correctly attributed.
24. **No metadata leakage:** corruption type, file path, dataset ID, or severity parameter is not included in neural input.
25. **Held-out corruptor:** test a defect mechanism unseen during training.
26. **Held-out skeleton:** no source clip derivative from that skeleton may appear in training.
27. **Dataset-ID probe:** measure source-dataset predictability from the latent.
28. **Part-query utilization:** if learned queries are used, all queries receive nontrivial distinct attention.

---

# 50. Recommended module layout

Add these modules only when their stage begins:

```text
motionlab/
  core/
    skeleton_graph.py
    motion_graph.py
    coordinate_contract.py
    provenance.py

  rigging/
    semantic_roles.py
    semantic_mapper.py
    rig_validation.py
    topology_augmentation.py
    coordinate_anchor.py

  features/
    static_features.py
    dynamic_features.py
    so3_derivatives.py
    normalization.py
    modality_masks.py

  models/
    static_graph_encoder.py
    dynamic_joint_encoder.py
    fusion.py
    local_graph_block.py
    temporal_tcn.py
    part_pooling.py
    spatiotemporal_encoder.py
    task_heads.py
    rig_aware_repair_decoder.py

  multitask/
    task_registry.py
    task_scheduler.py
    loss_registry.py
    gradient_diagnostics.py
    weighting.py

  datasets/
    canonical_adapter.py
    source_identity.py
    split_registry.py
    adapters/
      style100.py
      amass.py
      humanml3d.py
      babel.py
      underpressure.py

  evaluation/
    invariance_suite.py
    cross_rig.py
    latent_audits.py
```

Do not add empty placeholder modules. Introduce them stage by stage with tests.

---

# 51. Concrete architecture defaults

For the first variable-rig experiment:

```text
scope: humanoid only
active joints: deform body joints, excluding fingers/face/controllers
static embedding: 64 semantic + 64 rig-specific
raw dynamic embedding: 96
fused token width: 192 or 256
blocks: 6
local graph layers per block: 1
TCN kernel: 3 or 5
dilations: 1,2,4,8,16,32
global/part communication: every 2 blocks
functional parts: 8 initially
learned part queries: disabled initially
normalization: LayerNorm
rotation neural representation: 6D
cross-rig shared signal: root-normalized positions + velocities
rotation stream: optional with explicit validity/confidence
clip window: 128–192 frames at a declared fps
inference: overlapping windows
```

Use role-conditioned shared temporal weights rather than one independent TCN per named joint.

---

# 52. Revised implementation assignment for Codex

After the original deterministic vertical slice and fixed-rig baseline work, implement the variable-rig foundation in this exact order:

1. Define and validate `SkeletonGraph`, `MotionGraph`, coordinate metadata, modality masks, and provenance.
2. Refactor the current fixed-rig importer to emit those structures without changing existing demo results.
3. Implement joint permutation, padding, batch-composition, coordinate, and temporal invariance tests.
4. Implement deterministic semantic mapping with aliases plus YAML overrides and confidence values.
5. Implement identity-helper insertion augmentation and prove exact FK preservation.
6. Implement static numeric feature extraction and a small typed-edge GNN.
7. Implement universal position/velocity dynamic features and the optional rotation stream with SO(3) derivatives.
8. Implement masked variable-`J` batching and a shared dynamic joint MLP.
9. Implement concatenation fusion, local graph blocks, and per-joint dilated temporal TCN blocks while retaining `[B,T,J,D]`.
10. Implement deterministic functional-part pooling; do not use unsupervised learned queries yet.
11. Port fixed-rig critic heads to joint/part/frame/clip levels and verify parity on the original rig.
12. Create same-motion/different-topology positive pairs through exact helper insertion and at least two real humanoid rigs.
13. Add prediction-level and part-level contrastive consistency losses; do not use full-latent MSE.
14. Add task scheduler, label masks, loss/gradient logging, and only then add 100STYLE style and corruption tasks.
15. Add one auxiliary dataset at a time, beginning with contact or action labels, and require an ablation showing that it improves a relevant held-out metric.

At every stage update `IMPLEMENTATION_STATUS.md` with:

```text
implemented behavior
commands run
test results
numerical tolerances
known unsupported rig cases
new assumptions
ablation results
next three tasks
```

The following are release blockers:

- local rotations are compared across rigs without coordinate anchoring;
- missing features are encoded as ordinary zeros without masks;
- padded joints influence valid output;
- split lineage is lost after retargeting/corruption;
- all learned part queries attend globally or collapse;
- dense losses are dominated by rigs with more helper/finger joints;
- style labels are treated as quality labels;
- the repair head is claimed to recover twist from positions alone;
- a held-out “skeleton” test contains derivatives of the same source motion in training;
- the variable-rig model is adopted without comparison to the fixed-rig baseline.

---

# 53. Research references informing this addendum

Use these as design references, not code to copy blindly:

- **SATA: Semantic-Aware Motion Encoding for Topology-Agnostic Character Animation**, arXiv:2605.27055. Static/dynamic node decomposition, semantic conditioning, local/global spatial processing, temporal processing, and sliding-window inference.
- **SAMoR: Motion Modelling for Articulated Objects of Any Skeleton and Topology**, arXiv:2607.02148. Variable-`J` joint encoder, functional-part tokens, query-collapse failure mode, joint-name dropout, and the warning that local rotations are not comparable across arbitrary rigs.
- **AnyTop: Character Animation Diffusion with Any Topology**, arXiv:2502.17327. Rest-pose, topology, relation/distance, joint-description conditioning, skeleton-balanced training, and left/right ambiguity.
- **SAME: Skeleton-Agnostic Motion Embedding for Character Animation**, SIGGRAPH Asia 2023. Skeleton-agnostic graph autoencoder and the practical need for careful T-pose/facing preprocessing and retargeted training pairs.
- **HuMoT: Human Motion Representation using Topology-Agnostic Transformers**, arXiv:2305.18897. Variable topology/morphology encoding and the motivation for combining large generic motion data with smaller specialized labeled datasets.
- **Skeleton-Aware Networks for Deep Motion Retargeting**, arXiv:2005.05732. Skeletal convolution/pooling over homeomorphic skeletons and the limits of that assumption.
- **MoCapAnything V2: End-to-End Motion Capture for Arbitrary Skeletons**, arXiv:2604.28130. Pose-to-rotation ambiguity and the need for a reference pose-rotation coordinate anchor.
- **Masked Motion Predictors are Strong 3D Action Representation Learners**, arXiv:2308.07092. Motion-target masking rather than trivial raw-pose reconstruction.
- **On the Continuity of Rotation Representations in Neural Networks**, arXiv:1812.07035. Continuous 5D/6D rotation representations for neural learning.
- **GradNorm**, arXiv:1711.02257; **PCGrad**, arXiv:2001.06782; and uncertainty-weighted multi-task learning, arXiv:1705.07115. Multi-task weighting and gradient-conflict tools.

