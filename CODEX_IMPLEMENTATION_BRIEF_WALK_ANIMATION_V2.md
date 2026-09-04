# Codex implementation brief: LLM-orchestrated humanoid walking-animation system

> **Research revision (September 2026):** This document now has a variable-rig/multi-task addendum at the end. Build the deterministic fixed-rig vertical slice first, but use the addendum's canonical data contracts and warnings when extending the learned model. In particular, never compare raw local rotations across rigs without coordinate anchoring.


## 0. Your role

Implement this repository. Do not only produce architecture notes or pseudocode. Build a runnable, tested Python project that can:

1. ingest humanoid walking motion, initially from BVH files such as 100STYLE;
2. normalize and optionally retarget it to one fixed target humanoid rig;
3. extract contacts, gait phase, and candidate loop intervals;
4. calculate deterministic animation-quality diagnostics;
5. manufacture supervised training data by procedurally corrupting clean walks;
6. train a temporal neural critic that localizes defects and ranks candidate motions;
7. repair common defects with deterministic motion operators and an optional learned residual repair head;
8. expose grading and editing operations through a Python API, CLI, and eventually an HTTP tool server for an LLM agent;
9. export the final motion back to the target rig, with a Blender bridge as the first DCC integration.

The practical goal is not a research benchmark. The goal is to produce the best possible walking loop on a known rig through retrieval, retargeting, constrained editing, grading, and iterative candidate search. Prefer reliable engineering over novelty.

Work in small end-to-end increments. Keep the repository runnable after every increment. When something cannot yet be implemented robustly, provide a conservative fallback and expose the limitation in metadata rather than silently pretending it works.

---

# 1. Product goal and design principles

## 1.1 Product goal

Given:

- a fixed humanoid target rig;
- a desired speed;
- either a style label, style mixture, or reference walk;
- optional constraints such as `loop=true`, `in_place=true`, heading, and stride preferences;

produce:

- a high-quality cyclic walking animation on that rig;
- a machine-readable diagnostic report;
- a provenance/history record of each edit;
- exported animation data suitable for Blender or a game engine.

The initial version only needs to support:

- one known target skeleton at a time;
- flat ground;
- straight locomotion;
- one to three gait cycles;
- offline grading, so bidirectional temporal context is allowed;
- human motion;
- 60 fps internally;
- neutral walking first, followed by a small set of styles.

## 1.2 Core architecture

Use this division of labor:

- **motion retrieval** supplies a strong real-mocap starting point;
- **deterministic metrics** detect measurable geometric and temporal failures;
- **learned critic** estimates residual naturalness, coordination, style consistency, and localized defect likelihood;
- **motion operators / optimizers** make numerical edits;
- **LLM agent** chooses semantic operations and protected properties, but does not directly emit hundreds of joint transforms.

The central loop is:

```text
request
  -> retrieve clean reference motion
  -> retarget to target rig
  -> normalize / make loop / match speed
  -> grade
  -> choose one edit operation
  -> search its numerical parameters
  -> hard-filter invalid candidates
  -> rank remaining candidates
  -> accept only a safe improvement
  -> repeat
  -> export
```

## 1.3 Do not build the wrong project

Do not make the first version into:

- an end-to-end text-to-motion generator;
- a rendered-video quality model;
- a universal arbitrary-topology skeleton model;
- a physically exact dynamics simulator;
- a single scalar “naturalness” classifier;
- an LLM that writes raw joint rotations frame by frame;
- a system trained only to distinguish “mocap dataset A” from “generated dataset B.”

The first useful product is a **fixed-rig, hybrid grader and editor**.

---

# 2. Technology and engineering requirements

## 2.1 Language and libraries

Use Python 3.11 or newer.

Core runtime dependencies should be limited to approximately:

- `numpy`
- `scipy`
- `torch`
- `pydantic`
- `PyYAML`
- `typer`
- `rich`
- `platformdirs`

Optional extras:

- `matplotlib` for debug visualization;
- `fastapi` and `uvicorn` for the agent tool server;
- `pandas` and `pyarrow` for dataset/evaluation manifests;
- `tensorboard` for training logs.

Development dependencies:

- `pytest`
- `pytest-cov`
- `ruff`
- `mypy`

Do not make Blender a runtime dependency of the core package. Blender integration must live in separate scripts that run inside Blender’s Python interpreter.

## 2.2 Packaging

Use `pyproject.toml`. The package name in this specification is `motionlab`; use that name unless the repository already has a clearly better name.

Expose a CLI command:

```bash
motionlab --help
```

Also make all commands callable as:

```bash
python -m motionlab ...
```

## 2.3 Code quality

Required:

- type annotations for public functions;
- docstrings for public classes/functions;
- explicit array shape documentation;
- assertions or validation at module boundaries;
- deterministic seeds;
- no hidden global coordinate-system conversions;
- unit tests for quaternion math, FK, BVH parsing, contacts, metrics, corruptions, and repairs;
- integration smoke tests that do not require downloading any external dataset;
- structured logging rather than scattered `print` statements;
- configuration objects validated with Pydantic;
- numerical tolerances named in one central module.

Avoid premature micro-optimization. Correctness and debuggability come first. Vectorize frame/joint operations where straightforward.

## 2.4 Working protocol

Maintain these files:

- `README.md`: user-facing setup and examples;
- `IMPLEMENTATION_STATUS.md`: completed stages, current limitations, exact next tasks;
- `CHANGELOG.md`: meaningful changes;
- `docs/coordinate_conventions.md`;
- `docs/data_format.md`;
- `docs/agent_api.md`.

At the end of each implementation stage:

1. run formatting/linting;
2. run tests;
3. run a smoke CLI command;
4. update `IMPLEMENTATION_STATUS.md`;
5. do not move on while the repository is broken.

---

# 3. Repository layout

Create approximately this structure:

```text
.
├── pyproject.toml
├── README.md
├── IMPLEMENTATION_STATUS.md
├── CHANGELOG.md
├── configs/
│   ├── default.yaml
│   ├── training/
│   │   ├── smoke.yaml
│   │   └── critic_tcn.yaml
│   ├── datasets/
│   │   └── 100style.yaml
│   └── rigs/
│       ├── canonical_humanoid.yaml
│       └── example_target.yaml
├── docs/
│   ├── coordinate_conventions.md
│   ├── data_format.md
│   ├── corruption_catalog.md
│   ├── metrics.md
│   └── agent_api.md
├── motionlab/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config.py
│   ├── constants.py
│   ├── logging.py
│   ├── math/
│   │   ├── quaternion.py
│   │   ├── rotation6d.py
│   │   ├── interpolation.py
│   │   └── finite_difference.py
│   ├── motion/
│   │   ├── skeleton.py
│   │   ├── clip.py
│   │   ├── markers.py
│   │   ├── validate.py
│   │   └── transform.py
│   ├── io/
│   │   ├── bvh.py
│   │   ├── npz.py
│   │   ├── rig_spec.py
│   │   └── manifest.py
│   ├── kinematics/
│   │   ├── fk.py
│   │   ├── ik_two_bone.py
│   │   ├── ik_ccd.py
│   │   └── retarget.py
│   ├── processing/
│   │   ├── resample.py
│   │   ├── canonicalize.py
│   │   ├── contacts.py
│   │   ├── phase.py
│   │   ├── cycles.py
│   │   ├── looping.py
│   │   └── features.py
│   ├── metrics/
│   │   ├── base.py
│   │   ├── foot_sliding.py
│   │   ├── ground.py
│   │   ├── smoothness.py
│   │   ├── joint_limits.py
│   │   ├── speed.py
│   │   ├── cadence.py
│   │   ├── loop_seam.py
│   │   └── aggregate.py
│   ├── corruptions/
│   │   ├── base.py
│   │   ├── foot_slide.py
│   │   ├── ground_offset.py
│   │   ├── joint_pulse.py
│   │   ├── jitter.py
│   │   ├── limb_phase.py
│   │   ├── time_warp.py
│   │   ├── loop_seam.py
│   │   ├── pelvis_curve.py
│   │   ├── style_mix.py
│   │   └── compose.py
│   ├── dataset/
│   │   ├── prepare_100style.py
│   │   ├── windowing.py
│   │   ├── generation.py
│   │   ├── split.py
│   │   ├── normalization.py
│   │   └── torch_dataset.py
│   ├── models/
│   │   ├── temporal_blocks.py
│   │   ├── critic_tcn.py
│   │   ├── heads.py
│   │   └── checkpoint.py
│   ├── training/
│   │   ├── losses.py
│   │   ├── trainer.py
│   │   ├── evaluate.py
│   │   └── metrics.py
│   ├── repair/
│   │   ├── base.py
│   │   ├── foot_lock.py
│   │   ├── smooth_joint.py
│   │   ├── retime.py
│   │   ├── adjust_speed.py
│   │   ├── close_loop.py
│   │   ├── learned_residual.py
│   │   ├── objective.py
│   │   └── search.py
│   ├── retrieval/
│   │   ├── catalog.py
│   │   ├── features.py
│   │   └── nearest.py
│   ├── agent/
│   │   ├── schemas.py
│   │   ├── service.py
│   │   ├── tool_server.py
│   │   └── history.py
│   └── visualization/
│       ├── skeleton_plot.py
│       ├── report_plot.py
│       └── export_preview.py
├── blender_bridge/
│   ├── README.md
│   ├── export_rig.py
│   ├── export_action.py
│   └── import_action.py
├── scripts/
│   ├── make_synthetic_fixture.py
│   ├── prepare_100style.py
│   ├── train_critic.py
│   ├── evaluate_critic.py
│   └── run_demo_pipeline.py
└── tests/
    ├── fixtures/
    ├── test_quaternion.py
    ├── test_rotation6d.py
    ├── test_fk.py
    ├── test_bvh.py
    ├── test_resample.py
    ├── test_contacts.py
    ├── test_phase.py
    ├── test_looping.py
    ├── test_metrics.py
    ├── test_corruptions.py
    ├── test_retarget.py
    ├── test_model_shapes.py
    ├── test_repair.py
    └── test_cli_smoke.py
```

This structure may be simplified slightly, but do not collapse unrelated responsibilities into one large file.

---

# 4. Coordinate system and core data model

## 4.1 Canonical conventions

Use exactly one canonical representation inside `motionlab`:

- right-handed coordinates;
- `+Y` is up;
- `+Z` is character forward in canonical facing;
- `+X` is character right;
- distances are meters;
- time is seconds;
- angles are radians;
- quaternions use `[w, x, y, z]` ordering;
- quaternions represent active rotations;
- quaternion multiplication order must be documented and tested;
- local joint rotation is relative to the parent joint;
- root translation is stored separately from root local rotation;
- arrays are `float32` unless a calculation explicitly benefits from `float64`;
- the last frame of a cyclic clip is not a duplicate of the first frame.

All import/export adapters are responsible for converting to/from these conventions. Never spread axis swaps through the core code.

## 4.2 `Skeleton`

Implement an immutable or carefully validated skeleton object with at least:

```python
@dataclass(frozen=True)
class Skeleton:
    joint_names: tuple[str, ...]  # length J
    parents: NDArray[np.int32]  # [J], root parent = -1
    rest_offsets_m: NDArray[np.float32]  # [J, 3]
    rest_local_quat_wxyz: NDArray[np.float32]  # [J, 4]
    roles: Mapping[str, int]  # semantic role -> joint index
    markers: Mapping[str, MarkerSpec]
    joint_limits: Mapping[int, JointLimitSpec]
    metadata: Mapping[str, Any]
```

Validation must check:

- exactly one root;
- parent indices precede or at least form an acyclic tree;
- all arrays have the same joint count;
- offsets and quaternions are finite;
- quaternions are normalized;
- required humanoid roles are present when an operation needs them;
- marker joints exist;
- role mappings are one-to-one unless explicitly allowed.

Suggested semantic roles:

```text
root
pelvis
spine_1
spine_2
chest
neck
head
left_clavicle
left_shoulder
left_elbow
left_wrist
right_clavicle
right_shoulder
right_elbow
right_wrist
left_hip
left_knee
left_ankle
left_foot
left_toe
right_hip
right_knee
right_ankle
right_foot
right_toe
```

Not every rig must contain every role. Required subsets should be validated per operation.

## 4.3 Virtual markers

A marker is attached to a joint with a local offset:

```python
class MarkerSpec(BaseModel):
    joint: str | int
    local_offset_m: tuple[float, float, float]
```

Support at least:

```text
left_heel
left_toe
right_heel
right_toe
```

This is superior to treating one ankle joint as the entire foot. If explicit toe joints do not exist, the rig config must be able to define virtual heel/toe markers on the foot bone.

## 4.4 `MotionClip`

Implement:

```python
@dataclass
class MotionClip:
    skeleton: Skeleton
    local_quat_wxyz: NDArray[np.float32]  # [T, J, 4]
    root_translation_m: NDArray[np.float32]  # [T, 3]
    fps: float
    metadata: dict[str, Any]
```

Useful optional cached/derived arrays should not be persisted as mutable truth inside the object unless cache invalidation is reliable. Prefer pure functions or an explicit cache object.

Required properties/methods:

- `num_frames`
- `num_joints`
- `duration_s`
- `copy()` / immutable update helper
- `slice_frames(start, stop)`
- `validate()`
- `to(device/dtype)` only if a torch counterpart is introduced
- stable content hash for caching/provenance

## 4.5 Forward kinematics

Implement vectorized FK for NumPy and Torch.

For each frame and joint:

```text
global_rotation[root] = local_rotation[root]
global_position[root] = root_translation

global_rotation[j] = global_rotation[parent] * local_rotation[j]
global_position[j] = global_position[parent]
                   + global_rotation[parent] * rest_offset[j]
```

If `rest_local_quat_wxyz` is non-identity, define clearly whether stored animation quaternions are absolute local orientations or deltas from rest. Use one convention consistently.

Recommended convention:

```text
stored local_quat = absolute local orientation, including bind/rest orientation
```

Therefore, a rest-pose clip stores `rest_local_quat_wxyz` at every frame.

Write explicit tests with a two-link and three-link chain where exact expected positions are known.

---

# 5. Rotation mathematics

Implement and thoroughly test:

- quaternion normalization;
- conjugate and inverse;
- quaternion multiplication;
- quaternion-vector rotation;
- axis-angle to quaternion;
- quaternion to axis-angle / logarithm map;
- exponential map;
- matrix conversion;
- continuous quaternion sign unwrapping over time;
- SLERP;
- shortest-arc difference;
- 6D rotation representation conversion;
- geodesic angular distance;
- batch-safe finite differences on rotations.

Define repair deltas using the local tangent convention:

```text
q_repaired = q_bad * exp(delta_axis_angle)
```

Therefore the supervised target delta is:

```text
delta_target = log(inv(q_bad) * q_clean)
```

Clamp or otherwise stabilize near-zero and near-pi cases. Unit tests must cover:

- identity;
- inverse consistency;
- composition order;
- random round trips;
- sign-equivalent quaternions;
- 180-degree edge cases;
- gradients for Torch implementations where applicable.

---

# 6. BVH ingestion and internal serialization

## 6.1 BVH parser

Implement a real BVH parser rather than assuming one rotation order.

It must:

- parse hierarchy recursively;
- preserve per-joint channel order;
- support common Euler orders;
- support root translation channels;
- parse frame time and frame count;
- convert degrees to canonical quaternions;
- preserve end sites as optional metadata or virtual markers;
- detect malformed/incomplete files with useful errors;
- expose source axis/unit conversion options in config;
- optionally write BVH for debugging/export.

Create a tiny hand-written BVH fixture in `tests/fixtures/` with known poses and positions.

## 6.2 NPZ format

Define a stable internal `.npz` format with versioning. At minimum:

```text
format_version
joint_names
parents
rest_offsets_m
rest_local_quat_wxyz
local_quat_wxyz
root_translation_m
fps
metadata_json
roles_json
markers_json
joint_limits_json
```

Do not use Python pickle inside NPZ. JSON-serialize metadata.

## 6.3 Rig specification YAML

Support a rig YAML approximately like:

```yaml
name: example_humanoid
coordinate_system:
  handedness: right
  up: Y
  forward: Z
  unit_scale_to_meters: 1.0

joints:
  root: Hips
  pelvis: Hips
  spine_1: Spine
  chest: Spine2
  head: Head
  left_hip: LeftUpLeg
  left_knee: LeftLeg
  left_ankle: LeftFoot
  left_foot: LeftFoot
  left_toe: LeftToeBase
  right_hip: RightUpLeg
  right_knee: RightLeg
  right_ankle: RightFoot
  right_foot: RightFoot
  right_toe: RightToeBase
  left_shoulder: LeftArm
  left_elbow: LeftForeArm
  left_wrist: LeftHand
  right_shoulder: RightArm
  right_elbow: RightForeArm
  right_wrist: RightHand

markers:
  left_heel:
    joint: LeftFoot
    local_offset_m: [0.0, -0.04, -0.08]
  left_toe:
    joint: LeftFoot
    local_offset_m: [0.0, -0.04, 0.16]
  right_heel:
    joint: RightFoot
    local_offset_m: [0.0, -0.04, -0.08]
  right_toe:
    joint: RightFoot
    local_offset_m: [0.0, -0.04, 0.16]

retarget:
  source_role_map: {}
  basis_correction_quat_wxyz: {}

joint_limits:
  LeftLeg:
    representation: swing_twist
    swing_x_deg: [-5, 155]
    swing_z_deg: [-15, 15]
    twist_deg: [-20, 20]
```

The exact schema can improve, but keep semantic roles separate from concrete bone names.

---

# 7. Data preparation from 100STYLE or another BVH collection

## 7.1 Dataset path handling

Do not scrape or download datasets automatically. Accept a user-provided source directory, for example:

```bash
motionlab prepare-100style \
  --source /data/raw/100STYLE \
  --output /data/processed/100style \
  --rig configs/rigs/canonical_humanoid.yaml
```

The command must:

- recursively discover `.bvh` files;
- parse filename metadata conservatively;
- retain original relative path;
- allow include/exclude regexes;
- initially default to forward walking clips only;
- create a JSONL or Parquet manifest;
- log rejected files and exact reasons;
- never overwrite processed data unless `--force` is set.

## 7.2 Metadata extraction

Infer where possible:

- style label;
- action class such as walk/run/idle;
- direction such as forward/backward/strafe;
- source take identifier;
- source actor identifier if known.

Do not hard-fail when filenames differ. Store `unknown` and allow an override CSV/YAML.

## 7.3 Resampling

Internally resample to 60 fps.

Use:

- SLERP for rotations;
- cubic or linear interpolation for root translation, with a configurable default;
- no duplicate terminal frame for cycles;
- optional low-pass filtering only when explicitly configured.

Test round-trip behavior and duration preservation.

## 7.4 Canonical facing and origin

Canonicalize each clip:

- estimate initial facing from pelvis/chest or left-right hip axis plus forward convention;
- rotate the clip so mean initial heading is `+Z`;
- move initial root horizontal position to `(0, 0)` in XZ;
- preserve vertical root position;
- store the removed world transform in metadata so it can be restored if needed.

Do not remove root motion unless creating an explicit in-place representation.

## 7.5 Quality filtering of presumed-clean data

Mocap is not automatically perfect. Before treating a clip as a clean positive:

- validate finite values and quaternion norms;
- reject implausible bone-length changes caused by parsing errors;
- calculate foot penetration and sliding heuristics;
- detect severe one-frame angular spikes;
- detect missing or implausibly large root jumps;
- flag rather than necessarily reject moderate problems;
- support a manual `accepted/rejected` override manifest.

Store a `clean_confidence` score and reasons. Initially train only on high-confidence clips.

---

# 8. Retargeting to one fixed target rig

## 8.1 Retargeting scope

Implement a practical fixed-rig retargeter. It need not solve arbitrary creature retargeting.

Inputs:

- source `MotionClip` and source `Skeleton`;
- target `Skeleton`;
- role mapping;
- optional per-joint basis correction quaternions;
- target ground plane;
- retarget config.

Output:

- target-rig `MotionClip`;
- diagnostics including unmapped joints, scale factor, IK residuals, and contact preservation error.

## 8.2 Rotation transfer

Use rest-relative local rotation deltas.

For each mapped source/target joint pair:

```text
delta_src = inv(source_rest_local) * source_anim_local

delta_in_target_basis = basis_correction
                      * delta_src
                      * inv(basis_correction)

target_anim_local = target_rest_local * delta_in_target_basis
```

Provide identity basis correction by default. Support config overrides because real DCC rigs often use different bone-local axes.

Unmapped target joints should remain at rest or inherit a configurable fraction of the nearest mapped ancestor’s delta. Default to rest.

## 8.3 Scale and root motion

Estimate scale using robust leg-length ratios, preferably median of:

```text
pelvis -> knee
knee -> ankle
ankle -> toe/foot marker
```

Fallback to total skeleton height.

Scale root translation by this ratio. Preserve horizontal heading and vertical oscillation. After retargeting, adjust mean root height so feet contact the ground without persistent penetration.

## 8.4 Contact-aware IK refinement

After raw rotation transfer:

1. identify contact intervals on the source or transferred target;
2. derive target heel/toe world-space anchors;
3. lock foot position during stance using two-bone leg IK;
4. preserve ankle/foot orientation when possible;
5. if the target is unreachable, allow limited pelvis/root compensation;
6. blend IK weights in/out around contact boundaries;
7. report residual target error and any unreachable frames.

First implement analytic two-bone IK for hip-knee-ankle. Use a stable pole direction estimated from the original knee position. Add a generic CCD fallback only after the analytic solver is tested.

## 8.5 Identity test

Retargeting a clip from a skeleton to the exact same skeleton with identity mappings must reproduce the original within numerical tolerance.

Acceptance target:

- mean joint-position RMS error below `1e-5 m`;
- mean rotation geodesic error below `1e-5 rad`.

---

# 9. Foot contacts, gait phase, and cycles

## 9.1 Contact representation

Represent four channels:

```text
left_heel
left_toe
right_heel
right_toe
```

Store both:

- hard boolean contacts `[T, 4]`;
- soft confidence weights `[T, 4]`.

## 9.2 Initial heuristic contact detector

Use marker world height and tangential speed relative to the ground plane.

A starting rule:

```text
contact_candidate = height < height_threshold
                 and tangential_speed < speed_threshold
```

Thresholds must scale with leg length and be configurable. Suggested initial ranges:

- height threshold: roughly 2–4 cm for an adult-sized rig;
- tangential speed threshold: roughly 0.10–0.20 m/s;
- minimum contact duration: 3 frames at 60 fps;
- hysteresis: separate enter/exit thresholds;
- temporal cleanup: short-gap closing and short-island removal.

Do not bake these numbers as universal truths. Put them in config and record them in output metadata.

## 9.3 Gait events

Extract:

- left heel strike;
- right heel strike;
- toe-off when detectable;
- stance intervals;
- swing intervals.

Use rising/falling edges of cleaned contact signals. If heel contacts are unreliable, fall back to local minima of foot tangential speed and height.

## 9.4 Continuous phase

Define gait phase so that, for normal alternating gait:

```text
phase = 0       at left heel strike
phase = pi      at right heel strike
phase = 2*pi    at next left heel strike
```

Interpolate monotonically between events. Store as both scalar angle and stable two-channel form:

```text
[sin(phase), cos(phase)]
```

Flag clips where reliable alternating phase cannot be inferred.

## 9.5 Cycle extraction

Provide utilities to extract:

- one complete gait cycle;
- two complete cycles;
- fixed-length windows aligned to a heel strike;
- arbitrary windows for non-cyclic training.

For loop candidates, score possible start/end event pairs by pose, velocity, acceleration, and contact-state compatibility. Prefer matching left-strike to left-strike or right-strike to right-strike.

---

# 10. Loop handling

Support two explicit loop modes.

## 10.1 Root-motion loop

The pose and derivatives should be periodic after accounting for expected net root displacement and heading change over the cycle.

Store root-motion delta separately:

```text
cycle_translation_delta_m
cycle_yaw_delta_rad
```

When calculating seam error, compare endpoint states after removing this expected trajectory delta.

## 10.2 In-place loop

Remove mean horizontal root progression and optionally yaw progression while preserving local body motion. Store the removed trajectory so a game engine can reapply motion procedurally.

## 10.3 Loop closure operator

Implement a repair that distributes endpoint mismatch over a configurable seam window in quaternion tangent space and root translation space.

Requirements:

- preserve contact anchors as much as possible;
- do not simply crossfade feet through the floor;
- distribute correction smoothly, with zero or low derivative at blend boundaries;
- optionally solve a small constrained least-squares problem;
- report before/after seam metrics.

Acceptance test on synthetic mismatched loops:

- reduce weighted seam metric by at least 80%;
- do not increase maximum foot penetration beyond a configured tolerance;
- do not increase stance foot sliding by more than a configured tolerance.

---

# 11. Deterministic diagnostic metrics

All metrics must expose:

```python
class MetricResult(BaseModel):
    name: str
    clip_value: float | None
    frame_values: list[float] | None
    joint_frame_values: list | None
    events: list[DiagnosticEvent]
    units: str | None
    confidence: float
    metadata: dict[str, Any]
```

Do not force unrelated metrics into the same arbitrary 0–1 scale too early. Preserve physical units and later provide a calibrated normalized severity.

## 11.1 Foot sliding

For each foot marker during contact, measure tangential world displacement and speed relative to the ground plane.

At minimum report:

- integrated slip distance per stance interval in centimeters;
- mean tangential speed during stance;
- maximum tangential speed during stance;
- framewise weighted slip severity;
- side and interval.

Use soft contact confidence as a weight.

Do not count intentional toe pivoting identically to whole-foot translation. Prefer heel/toe marker analysis and provide separate marker traces.

## 11.2 Ground penetration and floating

For every foot marker report:

- penetration depth below ground;
- duration of penetration;
- contact frames whose marker is implausibly high;
- swing frames with insufficient clearance.

Support a general plane:

```text
n dot x + d = 0
```

not only `y=0`, although flat ground is the MVP.

## 11.3 Joint limits

When a rig provides limits, calculate violations using a stable swing-twist or specified Euler representation. Report:

- joint;
- frames;
- angular excess;
- maximum and integrated violation.

When no limit is provided, mark metric unavailable rather than inventing one.

## 11.4 Smoothness, pops, and jitter

Use quaternion geodesic finite differences to calculate approximate:

- angular velocity;
- angular acceleration;
- angular jerk.

Normalize by frame time. Use robust statistics such as 95th/99th percentiles and local outlier scores rather than only global maxima.

Distinguish:

- an isolated pulse/pop;
- high-frequency jitter over an interval;
- legitimate high acceleration at foot impact.

Condition thresholds by joint group and gait phase where practical.

## 11.5 Speed and heading

Report:

- average root horizontal speed;
- median speed;
- framewise speed;
- target-speed absolute and relative error;
- heading deviation from desired path;
- unintended lateral drift.

## 11.6 Cadence and stride

From gait events report:

- steps per minute;
- left/right stance duration;
- left/right swing duration;
- stride length;
- step width;
- left/right asymmetry;
- target mismatch when supplied.

## 11.7 Loop seam

Compare first and last frame neighborhoods, accounting for permitted root-motion delta.

Include:

- local joint pose geodesic mismatch;
- root-relative joint-position mismatch;
- angular velocity mismatch;
- angular acceleration mismatch;
- root velocity mismatch;
- contact-state mismatch;
- gait phase mismatch.

## 11.8 Optional approximate balance metric

Only enable if approximate body-segment masses or a simple mass model is provided. Estimate COM and support polygon. Label this as approximate kinematic balance, not rigorous dynamics.

## 11.9 Aggregation

Produce a report with separate categories:

```text
contact_quality
anatomical_validity
smoothness
coordination
speed_adherence
loop_quality
style_match
learned_naturalness
```

Do not hide category values behind one score. An overall score may be provided for ranking, but retain all components.

Initial overall deterministic score can be a configurable monotone transform such as:

```text
quality = exp(-sum(weight_k * normalized_severity_k))
```

Weights must live in config and be recorded in the report.

---

# 12. Procedural corruption system

## 12.1 Purpose

Create supervision from clean mocap without requiring frame-by-frame human ratings.

Each corruption must produce:

```python
@dataclass
class CorruptionResult:
    corrupted: MotionClip
    defect_type: str
    affected_frames: NDArray[np.bool_]  # [T]
    affected_joints: NDArray[np.bool_]  # [T, J] or [J]
    severity_parameter: float
    parameters: dict[str, Any]
    repair_target: MotionClip  # usually the original clean clip
    seed: int
```

Every corruption must be deterministic for a supplied seed.

## 12.2 Severity supervision

Do not assign invented absolute quality labels such as “0.73 naturalness.”

For one clean clip, generate an ordered chain:

```text
clean > mild > medium > severe
```

Store the actual physical corruption parameter and train ranking relationships. Dense masks supervise localization.

## 12.3 Multiple implementations per defect

To prevent the critic from recognizing one procedural fingerprint, implement at least two mechanisms for major defects where practical.

Example:

```text
foot slide A: move the planted foot target and solve IK
foot slide B: perturb root motion without corresponding stance compensation
```

Train on one or both; reserve an implementation variant for held-out evaluation.

## 12.4 Required corruption catalog

### A. Foot slide

Choose a stance interval and create tangential foot displacement.

Variants:

1. perturb the stance foot world target with a smooth drift curve and solve leg IK;
2. perturb root translation during stance without compensating the foot;
3. slightly desynchronize foot joint channels from root motion.

Preserve smooth boundaries. Record slip direction and centimeters.

### B. Ground penetration / floating

Variants:

- lower or raise root over an interval;
- modify only one foot target during stance/swing;
- scale vertical foot trajectory around swing apex.

Record maximum depth/height change.

### C. Joint pop

Apply a compact smooth angular pulse to one joint:

```text
zero at interval boundaries
maximum at center
configurable axis and angle
```

Use quaternion tangent-space edits. Include knee, ankle, hip, shoulder, elbow, spine, and head groups with sensible probability weights.

### D. High-frequency jitter

Generate temporally filtered random tangent rotations over selected joints and intervals. Support:

- white-ish frame noise;
- band-limited jitter;
- correlated multi-joint jitter.

Avoid quaternion renormalization artifacts.

### E. Limb phase mismatch

Circularly or locally shift arm motion relative to lower-body gait phase. Use quaternion interpolation. Support left/right independently and bilateral shifts.

### F. Time warp / cadence inconsistency

Apply a monotonic temporal warp to pose channels while keeping root trajectory unchanged, or vice versa. Include local acceleration/deceleration within a cycle.

### G. Speed/stride mismatch

Scale root progression without adjusting stride, or adjust leg motion without root progression. This creates plausible-looking but task-inconsistent motion.

### H. Loop seam mismatch

Construct a loop from phase-mismatched endpoints, perturb endpoint rotations, or introduce root velocity mismatch. Ensure exact masks include seam neighborhoods, not only frame 0.

### I. Pelvis-curve corruption

Scale, suppress, exaggerate, or phase-shift pelvis vertical motion, roll, yaw, or lateral sway.

### J. Foot-clearance corruption

Lower swing-foot trajectory or alter toe orientation during swing, preferably through IK/curve editing.

### K. Joint twist / anatomical corruption

Move joints toward or beyond configured limits. Only use rigs with corresponding limit definitions. Record angular excess.

### L. Style incoherence

Phase-align two clean clips on the same skeleton and replace/blend body regions, for example:

```text
legs from Old
pelvis from Neutral
arms from Proud
head from Robot
```

Use masks for body groups and smooth spatial blending across the hierarchy. Keep this corruption separate from physical corruption labels.

## 12.5 Corruption composition

Support composition of two or more defects, but begin training mostly with single defects.

Composition rules:

- preserve a list of constituent defects;
- union temporal/joint masks per defect channel;
- cap extreme parameter combinations;
- reject generated samples that become numerically invalid;
- calculate deterministic metrics after corruption and verify that the intended defect actually changed in the expected direction;
- discard ineffective corruptions.

---

# 13. Dataset format and generation

## 13.1 Window format

A processed training sample should include approximately:

```python
{
    "local_rot6d": float32[T, J, 6],
    "root_translation": float32[T, 3],
    "root_velocity": float32[T, 3],
    "root_angular_vel": float32[T, 3],
    "joint_position_rel": float32[T, J, 3],
    "joint_velocity_rel": float32[T, J, 3],
    "joint_angular_vel": float32[T, J, 3],
    "contact_soft": float32[T, 4],
    "phase_sincos": float32[T, 2],
    "target_speed": float32[1],
    "style_id": int64,
    "loop_mode": int64,
    "defect_mask": float32[T, K],
    "joint_defect_mask": float32[T, J, K],
    "defect_present": float32[K],
    "clean_local_quat": float32[T, J, 4],
    "clean_root_translation": float32[T, 3],
    "source_clip_id": str,
    "source_window_start": int,
    "corruption_ids": list[str],
    "corruption_seed": int,
    "severity_parameters": dict,
}
```

The exact persisted representation may use arrays plus JSON metadata, but retain this information.

## 13.2 Feature coordinates

Use both:

- local joint rotations;
- root-relative, facing-relative joint positions/velocities.

For root-relative positions:

1. subtract root position;
2. remove current or reference root yaw so forward direction is canonical;
3. preserve height;
4. document whether orientation is per-frame-facing or clip-facing.

Recommended initial choice: remove per-frame root yaw for joint-relative features, while separately supplying root yaw velocity and global trajectory features.

## 13.3 Window length

Default:

- `T = 160` frames at 60 fps;
- stride `32` frames;
- prefer windows containing at least one complete gait cycle;
- phase-aligned windows for cyclic training;
- fixed length for the first model.

Make these configurable.

## 13.4 Splitting

Prevent leakage.

All windows and corruptions derived from one original source take must belong to exactly one split.

Split by stable hash of source take ID, not by random window. Where actor IDs exist, support actor-held-out splits.

Persist `splits.json` and never silently regenerate it with a different seed.

Required evaluation splits:

- standard take-held-out validation/test;
- held-out corruption implementation;
- held-out style when possible;
- later, held-out source dataset or actor.

## 13.5 Normalization

Fit feature means/scales on training data only. Store normalization statistics in a versioned file next to model checkpoints.

For rotations represented as 6D, do not standardize in a way that destroys geometry without testing. It is acceptable to leave normalized 6D vectors near their natural range and standardize translational/velocity channels separately.

## 13.6 Storage

For the MVP use:

- one compressed NPZ per sample or modest shard;
- JSONL manifest;
- content hashes;
- no pickle.

If file-count overhead becomes a problem, add optional tar/WebDataset-style shards later without changing logical sample fields.

## 13.7 Generation command

Implement:

```bash
motionlab generate-dataset \
  --clean-manifest /data/processed/100style/manifest.jsonl \
  --output /data/datasets/walk_critic_v1 \
  --config configs/datasets/100style.yaml \
  --seed 1234
```

The command must be resumable, idempotent, and produce a summary:

```text
clean windows
corrupted windows per type
rejected corruptions and reasons
split counts
style counts
severity distributions
metric deltas before/after corruption
```

---

# 14. Learned critic model

## 14.1 MVP architecture

Implement a temporal convolutional critic first. Do not begin with a large Transformer.

Input:

```text
[B, T, J, F_joint]
[B, T, F_global]
[B, F_condition]
```

Suggested per-joint features:

```text
local rotation 6D                  6
root-relative joint position       3
root-relative joint velocity       3
local/global angular velocity      3
------------------------------------
total approximately               15
```

Suggested global frame features:

```text
root linear velocity
root angular velocity
root height
four contact confidences
phase sin/cos
target-speed error or target speed
```

## 14.2 Spatial encoding

Use a shared MLP over joints:

```text
joint_features [B,T,J,F]
  -> shared MLP
joint_tokens [B,T,J,Dj]
```

Then build a frame token from:

- mean-pooled joint token;
- max-pooled joint token;
- selected semantic joint tokens such as pelvis, feet, knees, wrists, head;
- global frame features;
- condition embedding.

Project to temporal hidden dimension around `256`.

Keep the per-joint tokens for the joint-blame and repair heads.

## 14.3 Temporal encoder

Use 6–8 residual dilated 1D-convolution blocks with approximately:

```text
kernel size: 3 or 5
dilations: 1, 2, 4, 8, 16, 32, optionally 64
hidden size: 256
activation: GELU
dropout: 0.1
normalization: GroupNorm or LayerNorm
```

The grader is offline, so symmetric context is allowed. For cyclic samples support circular padding; otherwise use reflection/replication padding and a valid-frame mask.

## 14.4 Heads

Implement:

1. **frame defect logits**: `[B,T,K]`;
2. **joint-time defect logits**: `[B,T,J,K]`;
3. **clip defect/severity logits**: `[B,K]`;
4. **overall ranking score**: `[B,1]`;
5. **contact auxiliary head**: `[B,T,4]`;
6. **style classification or embedding head**: optional initially, required in style stage;
7. **repair delta head**: `[B,T,J,3]` tangent axis-angle;
8. **optional uncertainty/log-variance head** later.

The joint-time head should combine each joint token with temporal context rather than merely reshaping a global prediction.

## 14.5 Losses

Implement configurable weighted losses:

### Frame localization

Binary cross-entropy or focal BCE on `[T,K]`, with per-class imbalance weights.

### Joint-time localization

BCE on `[T,J,K]`, optionally downweighted for defects that do not have meaningful joint attribution.

### Clip defect presence

BCE on `[K]`.

### Pairwise ranking

For ordered pairs `a > b`:

```text
L_rank = softplus(-(score_a - score_b))
```

Generate pairs from the same clean source whenever possible to avoid style/content confounds.

### Repair rotation loss

Supervise tangent deltas and reconstructed rotations:

```text
q_pred = q_bad * exp(delta_pred)
```

Use:

- Smooth L1 on tangent delta;
- quaternion geodesic loss between `q_pred` and clean rotations;
- root-translation repair loss if a root residual head is added;
- FK joint-position loss;
- velocity/smoothness loss;
- contact-anchor preservation loss during clean stance.

On clean inputs, target delta is zero. Penalize unnecessary changes strongly.

### Contact auxiliary loss

BCE against heuristic or pressure-derived contact labels.

### Style loss

Cross-entropy for known style IDs or contrastive loss for reference-style embeddings. Do not mix style mismatch into the physical defect labels.

## 14.6 Suggested initial loss weights

Provide sensible defaults but make them configurable, approximately:

```yaml
loss_weights:
  frame_defect: 1.0
  joint_defect: 0.5
  clip_defect: 0.5
  ranking: 0.5
  repair_tangent: 1.0
  repair_geodesic: 1.0
  repair_fk: 0.5
  repair_velocity: 0.2
  clean_identity: 1.0
  contact_aux: 0.2
  style: 0.2
```

Do not claim these are optimal.

## 14.7 Training implementation

Required:

- mixed precision when CUDA is available;
- CPU smoke mode;
- deterministic seeds where possible;
- gradient clipping;
- train/validation metrics after every epoch;
- early stopping option;
- best and last checkpoints;
- configuration snapshot in checkpoint;
- normalization statistics in checkpoint bundle;
- Git commit hash if available;
- TensorBoard or JSONL logs;
- resume support;
- no dataset split regeneration during resume.

CLI:

```bash
motionlab train-critic \
  --dataset /data/datasets/walk_critic_v1 \
  --config configs/training/critic_tcn.yaml \
  --output runs/critic_v1
```

## 14.8 Evaluation metrics

Report per defect and macro/micro averages:

- clip AUROC and AUPRC;
- framewise AUPRC;
- temporal event IoU/F1 with tolerance windows;
- joint-time IoU/F1;
- pairwise ranking accuracy;
- Spearman correlation with corruption severity parameter;
- repair rotation error;
- repair FK position error;
- deterministic metric improvement after repair;
- degradation on clean inputs;
- calibration curves when enough data exists.

Do not use accuracy alone on imbalanced masks.

## 14.9 Mandatory tiny-overfit test

Create a test/debug command that trains on 8–32 samples and verifies the model can overfit them. This catches data/label/model wiring errors.

Target for the debug fixture, not for real generalization:

- greater than 95% pairwise training accuracy;
- very high defect classification score;
- substantial reduction in reconstruction loss.

---

# 15. Inference report

Expose a stable Pydantic/JSON report.

Example:

```json
{
  "motion_id": "sha256:...",
  "model_version": "critic_v1",
  "rig": "example_humanoid",
  "fps": 60.0,
  "num_frames": 160,
  "conditions": {
    "target_speed_mps": 1.25,
    "style": "Neutral",
    "loop_mode": "root_motion"
  },
  "measured": {
    "average_speed_mps": 1.18,
    "cadence_spm": 109.0,
    "left_stride_m": 1.21,
    "right_stride_m": 1.19
  },
  "scores": {
    "overall": 0.76,
    "contact_quality": 0.54,
    "anatomical_validity": 0.95,
    "smoothness": 0.87,
    "coordination": 0.79,
    "speed_adherence": 0.81,
    "loop_quality": 0.91,
    "style_match": 0.84,
    "learned_naturalness": 0.73
  },
  "events": [
    {
      "id": "evt_0001",
      "type": "left_foot_slide",
      "source": "deterministic+learned",
      "frames": [43, 69],
      "time_seconds": [0.7167, 1.15],
      "joints": ["LeftFoot", "LeftLeg", "LeftUpLeg"],
      "severity": 0.82,
      "confidence": 0.94,
      "evidence": {
        "stance_slip_cm": 4.1,
        "mean_tangential_speed_mps": 0.13
      },
      "suggested_operators": ["lock_foot"]
    }
  ],
  "hard_constraint_violations": [],
  "warnings": [],
  "artifacts": {
    "dense_scores_npz": "...",
    "preview": "..."
  }
}
```

Keep dense `[T,K]` and `[T,J,K]` arrays in an artifact file rather than bloating the JSON supplied to the LLM.

Merge adjacent frame detections into events using configurable thresholding, hysteresis, and minimum duration.

---

# 16. Repair operators

Every repair operator must:

- declare required semantic roles;
- accept an immutable input motion/version;
- return a new motion version;
- report exact parameters and affected frames/joints;
- calculate before/after deterministic metrics;
- support a dry-run mode;
- avoid modifying unrelated frames/joints where possible;
- be deterministic for the same input and parameters.

## 16.1 Foot locking

Inputs:

```text
side
contact interval
anchor mode: first/median/optimized
lock strength
blend-in frames
blend-out frames
pelvis compensation limit
preserve foot orientation boolean
```

Algorithm:

1. choose a stable heel/toe or foot transform anchor during stance;
2. create smooth lock weights;
3. solve hip-knee-ankle IK per frame toward the anchor;
4. optionally distribute unreachable residual into pelvis translation/rotation within limits;
5. restore or preserve ankle/toe orientation;
6. smooth transitions;
7. reject if knee flips or joint limits are severely violated.

Synthetic acceptance test:

- reduce stance slip by at least 90%;
- endpoint pose changes outside the repair window below tolerance;
- no new ground penetration above tolerance.

## 16.2 Joint smoothing

Perform smoothing in quaternion tangent space, not componentwise quaternion averaging.

Inputs:

```text
joint or body group
frame interval
cutoff/window
strength
preserve endpoints
preserve contacts
```

Use Savitzky-Golay or another documented local smoother. Reconstruct normalized quaternions and ensure temporal sign continuity.

## 16.3 Limb retiming / phase adjustment

Retiming must use a monotonic time map and quaternion interpolation.

Support:

- one limb/body group;
- global cycle phase shift;
- local time warp;
- blend at interval boundaries;
- phase-aligned wrapping for cyclic clips.

## 16.4 Speed adjustment

Support two modes:

1. change root trajectory speed while preserving pose timing;
2. jointly retime pose and root motion to change cadence.

A better final operation may combine stride scaling, root progression, and cadence search. Preserve contacts with IK afterward.

## 16.5 Loop closure

Use the operator described in the loop section. Preserve root-motion delta semantics.

## 16.6 Pelvis curve edit

Support additive or multiplicative edits to pelvis:

- vertical translation;
- lateral translation;
- roll;
- yaw;
- pitch.

Parameterize edits by gait phase where possible, not only frame number.

## 16.7 Learned residual repair

Apply the model’s tangent repair delta with a conservative scale `alpha` in `[0,1]`.

Never blindly accept it. After applying:

- enforce quaternion normalization;
- run joint-limit checks;
- run contact/ground metrics;
- compare protected metrics;
- reject or reduce `alpha` if it causes regressions.

## 16.8 Candidate parameter search

Implement simple search first:

- fixed grid for 1–2 parameters;
- random/Latin hypercube for modest dimensions;
- optional cross-entropy method later.

The LLM chooses:

```text
operator
frame/body scope
properties to preserve
rough parameter range
```

The numerical search chooses exact values.

Use a lexicographic or constrained objective:

1. reject hard constraint violations;
2. reject protected-metric regressions beyond tolerance;
3. minimize target deterministic defect;
4. maximize learned ranking score;
5. minimize total edit magnitude.

Store every candidate and score in an optimization trace.

---

# 17. Retrieval and style handling

## 17.1 Retrieval first

The initial generator is retrieval plus editing, not synthesis from zero.

Create a catalog of clean gait-cycle candidates with:

- style label;
- speed;
- cadence;
- stride length;
- step width;
- asymmetry;
- loop seam score;
- clean-confidence score;
- optional learned style embedding.

## 17.2 Retrieval query

Support:

```python
class MotionRequest(BaseModel):
    target_speed_mps: float
    style_weights: dict[str, float] = {"Neutral": 1.0}
    avoid_styles: list[str] = []
    loop_mode: Literal["root_motion", "in_place"] = "root_motion"
    desired_duration_s: float | None = None
    desired_cadence_spm: float | None = None
    symmetry_preference: float | None = None
```

Initial distance can be a transparent weighted metric over metadata. Later add embedding distance.

Return the top `N` candidates, not only one, so downstream grading and editing can select the best retargeted result.

## 17.3 Style labels first, text later

For the first style-capable version use style IDs and weighted mixtures. The LLM converts a user request into style weights.

Example:

```json
{
  "style_weights": {
    "Rushed": 0.55,
    "Old": 0.25,
    "Neutral": 0.20
  },
  "avoid_styles": ["LimpLeft", "LimpRight"],
  "target_speed_mps": 1.45
}
```

Do not train free-form text conditioning until the rig-space pipeline is already strong.

## 17.4 Style mixing

Only mix motions after:

- same target skeleton;
- normalized scale;
- phase alignment;
- compatible speed/cadence or explicit retiming;
- compatible root trajectory.

Allow body-region masks and hierarchical feathering. Re-run contact repair and grading afterward.

---

# 18. Agent-facing API

## 18.1 Core Python service

Implement a service class with methods approximately:

```python
class MotionService:
    def import_motion(...): ...
    def retrieve(...): ...
    def retarget(...): ...
    def grade(...): ...
    def apply_operator(...): ...
    def optimize_operator(...): ...
    def make_loop(...): ...
    def export_motion(...): ...
    def get_history(...): ...
```

Use immutable content-addressed motion versions. Every operation returns a new `motion_id`.

## 18.2 Required agent tools

Expose JSON schemas for:

### `retrieve_reference`

Inputs:

```text
target speed
style weights
loop mode
number of candidates
```

Outputs candidate IDs plus metadata.

### `grade_motion`

Inputs:

```text
motion ID
condition
optional model checkpoint
```

Outputs compact report plus artifact paths.

### `apply_edit`

Inputs:

```text
motion ID
operator name
parameters
protected metrics
```

Outputs new motion ID and metric deltas.

### `optimize_edit`

Inputs:

```text
motion ID
operator
parameter ranges
target event ID
protected metrics
candidate budget
```

Outputs best accepted candidate and trace.

### `make_loop`

Inputs loop mode and constraints.

### `export_motion`

Outputs NPZ/BVH or prepares Blender import artifact.

## 18.3 LLM policy guidance

Document a recommended orchestration policy:

1. grade current motion;
2. select the highest-severity high-confidence actionable event;
3. choose the narrowest operator that addresses it;
4. declare protected properties;
5. search parameters;
6. accept only if target metric improves and protected metrics stay within tolerance;
7. re-grade;
8. stop on convergence, budget, or lack of safe improvement.

The LLM must not receive or manipulate dense joint arrays directly unless debugging.

## 18.4 HTTP server

After the core works, expose a FastAPI server:

```bash
motionlab serve --workspace ./workspace --port 8765
```

Include OpenAPI schemas, request validation, stable error codes, and local filesystem sandboxing. Do not allow arbitrary path traversal.

---

# 19. Blender bridge

## 19.1 Scope

Implement scripts for Blender as a separate integration layer.

Target workflows:

1. export active armature rig definition to canonical YAML/NPZ;
2. export an active action sampled at fixed FPS to canonical NPZ;
3. import a canonical target-rig motion as a baked Blender action.

## 19.2 `export_rig.py`

Run inside Blender. It should:

- identify the selected/active armature;
- export bone hierarchy, rest transforms, names, and lengths;
- apply an explicit Blender-to-canonical coordinate transform;
- load a user role-map YAML rather than relying only on name heuristics;
- optionally export marker offsets defined by empties or config;
- write a rig YAML/NPZ accepted by `motionlab`.

## 19.3 `export_action.py`

It should:

- sample every frame at requested FPS;
- read evaluated pose matrices;
- convert to canonical local quaternions and root translation;
- preserve action/frame metadata;
- avoid dependence on F-curve interpolation assumptions by sampling the evaluated scene.

## 19.4 `import_action.py`

It should:

- validate that motion skeleton matches the target rig mapping;
- convert canonical transforms back to Blender coordinates;
- create a new action rather than destroying the original;
- insert baked keyframes at every frame initially;
- set interpolation to linear for exact playback or document an alternative;
- support root-motion and in-place variants;
- store motion ID/report path in custom action properties.

## 19.5 Blender CLI examples

Document commands similar to:

```bash
blender --background character.blend \
  --python blender_bridge/export_rig.py -- \
  --output exported_rig.yaml \
  --role-map configs/rigs/example_target.yaml
```

and:

```bash
blender --background character.blend \
  --python blender_bridge/import_action.py -- \
  --motion final_motion.npz \
  --action-name AI_Walk_Final \
  --output character_with_walk.blend
```

---

# 20. Visualization and debugging

Implement a minimal skeleton viewer/debug exporter that can:

- render a 3D stick figure animation;
- show ground plane and heel/toe markers;
- color or annotate contact intervals;
- overlay root trajectory;
- plot selected metric curves;
- render before/after side-by-side previews when practical;
- save MP4/GIF if ffmpeg/Pillow is available;
- fall back to frame PNGs or interactive display if not.

Provide a command:

```bash
motionlab preview motion.npz --report report.json --output preview.mp4
```

Debug views are essential for verifying coordinate and contact logic.

---

# 21. Provenance and experiment history

Every produced motion version must have a provenance record:

```json
{
  "motion_id": "sha256:...",
  "parent_motion_id": "sha256:...",
  "operation": "lock_foot",
  "parameters": {},
  "created_at": "...",
  "source_assets": [],
  "metric_before": {},
  "metric_after": {},
  "model_checkpoint": null,
  "code_version": "..."
}
```

Keep original assets immutable. Store derived motions in a workspace organized by content hash. Make agent runs reproducible from a history file.

---

# 22. Human preference data, later stage

Do not block the MVP on human labels.

After synthetic training works, add a small A/B labeling tool. It should present two animations for the same request and ask:

```text
Which is better?
A / B / indistinguishable

Primary reason:
contact / smoothness / anatomy / coordination / style / looping / other
```

Store:

- request/condition;
- motion IDs;
- randomized left/right ordering;
- label;
- reason;
- annotator ID;
- timestamp;
- optional confidence.

Use hard pairs produced by the current agent, not many trivial clean-vs-catastrophic pairs. Add a preference fine-tuning stage for the ranking head only after enough labels exist.

---

# 23. Testing requirements

## 23.1 Unit tests

At minimum:

### Rotation tests

- random quaternion round trips;
- multiplication order;
- vector rotation;
- SLERP endpoints/midpoint;
- sign continuity;
- log/exp inverse;
- 6D conversion.

### FK tests

- exact two-link chain;
- batch dimensions;
- NumPy/Torch agreement;
- rest pose;
- root translation.

### BVH tests

- hierarchy parsing;
- channel order;
- non-default Euler order;
- frame time;
- expected FK positions.

### Contact tests

- stationary planted foot detected;
- moving raised foot not detected;
- hysteresis removes one-frame chatter;
- marker offsets transformed correctly.

### Phase tests

- alternating synthetic contacts yield expected phase;
- missing events produce warning/confidence reduction.

### Metrics tests

- zero slip for a stationary contact;
- known slip distance measured correctly;
- known penetration depth;
- smooth sequence has low jerk;
- injected pulse localizes;
- perfect loop has near-zero seam;
- mismatched loop has larger seam.

### Corruption tests

- deterministic under seed;
- affected masks cover actual edit;
- intended metric worsens;
- unaffected frames remain approximately unchanged;
- quaternions remain normalized.

### Retarget tests

- identity retarget;
- scale transfer;
- basis correction;
- unmapped joints remain at rest;
- contact IK reduces target error.

### Model tests

- output shapes;
- no NaNs on random valid inputs;
- backward pass;
- masked padding behavior;
- cyclic padding behavior.

### Repair tests

- foot lock reduces synthetic slip;
- loop closure reduces seam;
- smoothing reduces injected jitter;
- protected frames remain stable;
- invalid candidate rejection works.

## 23.2 Integration tests

Create a tiny synthetic humanoid fixture entirely in code. It should generate a crude periodic walk sufficient to run:

```text
create clip
serialize
load
extract contacts
calculate metrics
corrupt
generate a tiny dataset
train one or two debug steps
grade
repair
export report
```

This must run in CI without external data.

## 23.3 CI

Set up a basic CI workflow that runs:

```text
ruff check
mypy for core modules, with reasonable strictness
pytest
```

Keep GPU tests optional. All core smoke tests must pass on CPU.

---

# 24. CLI specification

Implement commands approximately like:

```bash
motionlab inspect-bvh file.bvh
motionlab convert-bvh file.bvh --output file.npz
motionlab validate-motion file.npz
motionlab preview file.npz --output preview.mp4

motionlab prepare-100style --source ... --output ... --rig ...
motionlab extract-cycles --manifest ... --output ...
motionlab retarget --motion ... --target-rig ... --output ...

motionlab metrics motion.npz --output report.json
motionlab corrupt motion.npz --type foot_slide --severity 0.5 --output bad.npz
motionlab generate-dataset --clean-manifest ... --output ... --config ...

motionlab train-critic --dataset ... --config ... --output ...
motionlab evaluate-critic --checkpoint ... --dataset ... --split test
motionlab grade motion.npz --checkpoint ... --condition request.json --output report.json

motionlab repair motion.npz --operator lock_foot --event evt_0001 --output repaired.npz
motionlab optimize-repair motion.npz --report report.json --event evt_0001 --budget 24

motionlab retrieve --catalog ... --request request.json --top-k 8
motionlab run-agent-loop --request request.json --workspace ./workspace
motionlab serve --workspace ./workspace --port 8765
```

Commands should return nonzero exit codes on failures and emit machine-readable JSON with `--json`.

---

# 25. Configuration examples

Create a default training config similar to:

```yaml
seed: 1234

motion:
  fps: 60
  window_frames: 160
  stride_frames: 32
  loop_padding: circular

features:
  local_rot6d: true
  joint_position_relative: true
  joint_velocity_relative: true
  joint_angular_velocity: true
  root_velocity: true
  root_angular_velocity: true
  contacts: true
  phase: true

model:
  joint_hidden: 32
  temporal_hidden: 256
  temporal_blocks: 7
  kernel_size: 3
  dilations: [1, 2, 4, 8, 16, 32, 64]
  dropout: 0.1
  num_defects: 12
  repair_head: true
  style_head: false

training:
  batch_size: 64
  epochs: 100
  learning_rate: 0.0003
  weight_decay: 0.0001
  gradient_clip_norm: 1.0
  mixed_precision: true
  num_workers: 4
  early_stopping_patience: 12

loss_weights:
  frame_defect: 1.0
  joint_defect: 0.5
  clip_defect: 0.5
  ranking: 0.5
  repair_tangent: 1.0
  repair_geodesic: 1.0
  repair_fk: 0.5
  repair_velocity: 0.2
  clean_identity: 1.0
  contact_aux: 0.2
  style: 0.0
```

Also create a very small `smoke.yaml` that runs on CPU in seconds.

---

# 26. Implementation stages and acceptance criteria

Implement in this order. Do not attempt all advanced features simultaneously.

## Stage 0 — repository bootstrap

Deliver:

- package layout;
- config/logging/CLI skeleton;
- lint/test setup;
- synthetic fixture generator;
- docs for coordinate conventions.

Acceptance:

```bash
python -m motionlab --help
pytest
ruff check .
```

all succeed.

## Stage 1 — rotations, skeleton, motion clip, FK, NPZ

Deliver fully tested math and data structures.

Acceptance:

- random math round trips pass;
- FK exact-chain tests pass;
- NPZ round trip reproduces arrays/metadata.

## Stage 2 — BVH import, resampling, canonicalization, preview

Deliver a real BVH parser and visual debug path.

Acceptance:

- test BVH produces expected joint positions;
- a user-supplied BVH can be converted and previewed;
- no hidden axis conversions.

## Stage 3 — contacts, phase, cycles, deterministic metrics

Deliver useful grading before ML.

Acceptance:

- synthetic walk yields plausible alternating contacts;
- injected foot slide and joint pulse are localized by metrics;
- report JSON is generated.

## Stage 4 — target-rig retargeting and basic repairs

Deliver:

- rest-relative retarget;
- scale transfer;
- two-bone IK;
- foot lock;
- loop closure.

Acceptance:

- identity retarget tolerance met;
- synthetic foot lock reduces slip by at least 90%;
- loop closure reduces seam by at least 80% without violating ground tolerance.

## Stage 5 — corruption generation and dataset pipeline

Deliver corruption catalog, manifests, stable splits, and sample loader.

Acceptance:

- every corruption is deterministic;
- intended deterministic metric worsens for at least a high percentage of generated samples;
- ineffective samples are rejected;
- no source-take leakage across splits;
- dataset summary report exists.

## Stage 6 — TCN critic and training

Deliver model, losses, trainer, evaluation, checkpoints.

Acceptance:

- tiny-overfit test succeeds;
- smoke training works on CPU;
- real training can resume;
- inference report merges deterministic and learned diagnostics.

## Stage 7 — learned repair and candidate optimization

Deliver residual repair application, safe acceptance checks, parameter search, and traces.

Acceptance:

- learned repair never modifies clean inputs substantially in the smoke test;
- candidate search can improve a target metric while respecting protected metrics;
- every accepted edit is reproducible.

## Stage 8 — retrieval, style labels, and full orchestration

Deliver catalog/retrieval, style-weight request schema, and iterative loop.

Acceptance:

- request retrieves multiple candidates;
- candidates are retargeted/graded;
- at least one safe edit iteration is executed;
- final motion, report, preview, and history are exported.

## Stage 9 — Blender bridge and HTTP server

Deliver DCC import/export and agent tool server.

Acceptance:

- Blender rig/action round trip documented and manually testable;
- server validates schemas and prevents path traversal;
- core package remains usable without Blender/FastAPI extras.

## Stage 10 — human A/B feedback

Optional after the main product is already useful.

---

# 27. End-to-end demo

Create a command or script that demonstrates the entire thin slice:

```bash
python scripts/run_demo_pipeline.py \
  --source-bvh path/to/Neutral_FW.bvh \
  --target-rig configs/rigs/example_target.yaml \
  --workspace demo_workspace
```

It should:

1. parse the BVH;
2. canonicalize/resample;
3. retarget;
4. detect contacts and extract a cycle;
5. calculate baseline diagnostics;
6. deliberately create one foot-slide defect;
7. grade the corrupted motion;
8. repair it with foot locking;
9. re-grade it;
10. export clean/corrupted/repaired NPZ files, reports, plots, and a provenance history.

The demo must work even before the learned critic exists, using deterministic metrics. Once a checkpoint is provided, it should add learned diagnostics automatically.

---

# 28. Practical defaults for the first real experiment

Use these defaults unless real data indicates a reason to adjust:

```text
source data:
    100STYLE forward-walking BVHs

first style:
    Neutral

next styles:
    Proud
    Old
    Rushed
    Robot
    LimpLeft
    LimpRight

rig:
    one fixed humanoid target rig

internal FPS:
    60

training window:
    160 frames

training stride:
    32 frames

first defect classes:
    foot_slide
    ground_penetration
    floating_contact
    joint_pop
    jitter
    arm_phase_mismatch
    cadence_mismatch
    speed_stride_mismatch
    loop_seam
    pelvis_curve_error
    foot_clearance
    joint_limit_violation

model:
    shared joint MLP + dilated bidirectional TCN

editing:
    IK, retiming, quaternion-curve smoothing, root trajectory adjustment

agent role:
    choose candidate/reference, target issue, operator, scope, and protected properties

optimizer role:
    choose exact numerical parameters
```

Style incoherence can be added as a separate class once multiple styles are processed reliably.

---

# 29. Failure modes to actively guard against

## 29.1 Dataset shortcut learning

Do not train on “all mocap clean, all generated motion bad.” Use corruptions of the same source clips, same rig, same serialization, and same rendering-independent representation.

## 29.2 Corruption fingerprint learning

Use multiple corruption implementations, held-out implementations, actual retargeter failures, and later agent-generated failures.

## 29.3 Reward hacking

Examples:

- freezing feet reduces sliding but ruins locomotion;
- excessive smoothing removes jitter but destroys impacts;
- shrinking root motion matches a loop metric but violates requested speed;
- unnatural crouching avoids leg reach problems;
- deleting arm motion avoids phase mismatch.

Prevent this with separate metrics, hard task constraints, protected properties, edit magnitude penalties, candidate filtering, and human A/B checks.

## 29.4 Bad positive data

Filter and inspect clean mocap. Keep `clean_confidence`; do not assume all source files are flawless.

## 29.5 Coordinate bugs mistaken for ML failure

Build visualizers and exact math tests before training. Store coordinate metadata and import transforms.

## 29.6 Incorrect quaternion interpolation

Never linearly interpolate quaternion components without normalization and a reason. Use SLERP or tangent-space methods.

## 29.7 Data leakage

Never split overlapping windows independently. Split by original take before window generation or via a stable source ID.

## 29.8 Overclaiming physical plausibility

Without masses, inertias, forces, and friction, label outputs as kinematic/contact/anatomical plausibility. Do not claim rigorous dynamic feasibility.

---

# 30. Definition of a useful first release

The first release is useful when a user can:

1. point the tool at a directory containing forward walking BVHs;
2. import/export their target humanoid rig through a documented rig file or Blender bridge;
3. retrieve and retarget a neutral walk;
4. create a clean one-cycle root-motion or in-place loop;
5. receive precise diagnostics such as:
   - “left foot slides 3.2 cm during frames 41–65”;
   - “loop angular-velocity mismatch is concentrated in pelvis and left shoulder”;
   - “average speed is 8% below target”;
6. automatically apply foot lock, speed adjustment, smoothing, or loop closure;
7. compare before/after metrics and preview animations;
8. generate synthetic clean/corrupt training pairs;
9. train a small critic that localizes the known defects;
10. run one or more safe, logged, LLM-selectable edit iterations.

A polished universal naturalness model is not required for the first release. A robust deterministic pipeline plus procedurally supervised critic is the intended route.

---

# 31. Immediate first implementation assignment

Begin now with an end-to-end deterministic thin slice, not the neural network.

Implement, in order:

1. `pyproject.toml`, package/CLI/test scaffolding;
2. quaternion and 6D rotation utilities;
3. `Skeleton` and `MotionClip` validation;
4. NumPy FK;
5. versioned NPZ serialization;
6. minimal but real BVH parser;
7. resampling to 60 fps;
8. marker world positions;
9. heuristic foot contacts;
10. foot-sliding, ground, speed, smoothness, and loop-seam metrics;
11. synthetic humanoid/walk fixture;
12. foot-slide corruption;
13. foot-lock repair with two-bone IK;
14. a demo command producing clean, corrupted, and repaired reports.

Only after this thin slice passes tests, implement 100STYLE preparation, the full corruption catalog, and the TCN critic.

Do not leave broad placeholder files. It is better to fully implement this first vertical slice than to create empty modules for every later feature.

At the end of the first assignment, provide:

- the updated repository;
- exact commands run;
- test/lint results;
- generated demo artifacts;
- a concise list of known limitations;
- the next three implementation tasks.


---

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

