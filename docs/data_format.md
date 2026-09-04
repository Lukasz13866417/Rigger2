# Internal data format

The first fixed-rig representation consists of a validated `Skeleton` and immutable `MotionClip`.
Persisted clips use a versioned, pickle-free compressed NPZ container.

Required NPZ fields:

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

JSON fields must contain only JSON-compatible values. Derived FK arrays are reproducible caches,
not persisted mutable truth. The later `SkeletonGraph`/`MotionGraph` migration will occur behind
explicit compatibility adapters after the fixed-rig critic baseline is reproducible.

The first such adapter is now implemented for forward metadata. It losslessly wraps the existing
objects and exposes coordinate/pose-reference contracts, anatomical semantics and flags,
modality/confidence masks, normalized ground data, and immutable source/split/operation lineage.
See `metadata_contract.md`. It intentionally does not introduce the later variable-topology
`SkeletonGraph`/`MotionGraph` tensors before the fixed-rig critic baseline.

BVH import stores the declared per-joint channel order, end-site offsets, source coordinate
description, and the source-to-canonical 3-by-3 matrix in skeleton metadata. Non-root translation
channels are rejected because the fixed-rig representation cannot preserve them losslessly.

## Deterministic report artifacts

Diagnostic persistence separates an LLM/API-friendly compact JSON summary from large numeric
arrays. The dense artifact is a pickle-free compressed NPZ with format
`motionlab.metric_arrays.v1` and these required fields:

```text
format_version
motion_id
manifest_json
metric_*_frame_values
metric_*_joint_frame_values
```

Only arrays present in a report are emitted. `manifest_json` maps each metric field to its exact
array key, shape, and dtype. The compact JSON stores the artifact's relative path and SHA-256, and
loading can verify both checksum and motion identity. This prevents dense training signals from
silently drifting away from their human-readable summary.

## Fixed-rig corruption dataset

`generate-dataset` accepts one or more clean, lineaged `MotionClip` NPZ sources on the same
skeleton. Sources must declare a take-level split and `clean_confidence` at or above the configured
threshold. Inputs are resampled to 60 fps when necessary, then windowed at 160 frames with a
32-frame stride by default. Window and source identity remain metadata and never enter neural
features.

The output is content-addressed and resumable:

```text
dataset/
  generation_state.json
  splits.json
  manifest.jsonl
  preferences.jsonl
  consistency.jsonl
  normalization.json
  normalization.npz
  summary.json
  samples/
    train/*.npz
    validation/*.npz
    test/*.npz
    heldout_corruptor/*.npz
```

Only directories required by the selected sources and mechanisms are created. A rerun with the
same state byte-compares and reuses existing samples. A changed source list, source motion ID, or
generation configuration is rejected in the same output directory instead of silently mixing
datasets.

Each `motionlab.fixed_rig_sample.v4` NPZ contains scalar `format_version`, `sample_id`,
`metadata_json`, and `declaration_json` fields plus these audited neural inputs:

| Field | Shape | Meaning |
| --- | --- | --- |
| `local_rot6d` | `[T,J,6]` | local rotations in continuous 6D form |
| `root_translation` | `[T,3]` | horizontal-origin-relative root position |
| `root_velocity` | `[T,3]` | root linear velocity |
| `root_angular_velocity` | `[T,3]` | root angular velocity |
| `joint_position_rel` | `[T,J,3]` | yaw-removed, root-relative horizontal position with world height |
| `joint_velocity_rel` | `[T,J,3]` | velocity of that facing-relative representation |
| `joint_angular_velocity` | `[T,J,3]` | local angular velocity |
| `contact_soft`, `contact_valid` | `[T,4]` | inference-available marker contacts and validity |
| `phase_sincos`, `phase_valid` | `[T,2]`, `[T]` | cyclic phase and validity |
| `target_speed`, `style_id`, `loop_mode` | `[1]` each | requested conditions |
| `joint_functional_part` | `[J]` | fixed functional-part index |

Supervision is stored separately and returned by `TrainingSample.targets`:

```text
intervention_mask[T,J,6]
symptom_mask[T,8,13]
responsibility_target[T,8,13]
responsibility_confidence[T,8,13]
defect_mask[T,13]
joint_defect_mask[T,J,13]
defect_present[13]
preference_valid[1]
preference_clean_better[1]
is_clean[1]
no_target_defect[1]
approximately_equal_quality[1]
quality_equivalence_valid[1]
clean_local_quat[T,J,4]
clean_root_translation[T,3]
```

Source dataset/clip/take/path, corruption family/mechanism/seed, requested severity, split and
split-lineage IDs, counterfactual group, gait target, and sample role are forbidden as sample
arrays. They remain in JSON metadata for auditing. Privileged source-reference contacts are not
written as neural inputs.

Every source window has a `counterfactual_group_id`, distinct from its take-level
`split_lineage_id`, shared by its clean sample, all corruption mechanisms/severity bins, and sham
controls. `consistency.jsonl` declares only justified targets: per-defect severity equality between
different mechanisms in the same observed bin, clean/sham approximate-quality equality, and
corrupted/matched-sham defect contrasts. It explicitly does not request whole-motion latent
alignment.

Heading-rotated no-defect samples use `sample_role=equivalent_control` and
`catalog_partition=equivalent`; their transform parameters remain metadata-only. Normalization is
fit on exactly the source `train` samples whose catalog partition is `clean`, `equivalent`,
`train`, or `sham`; held-out corruptors and validation/test sources are excluded. Mean/scale
statistics cover only translation, velocity, angular-velocity, joint-position, and target-speed
inputs. Rotation 6D, contact, phase, categorical conditions, and every target remain unnormalized.
The JSON manifest checksums the pickle-free normalization NPZ and records the exact training sample
IDs.

`validate-dataset` verifies all declared checksums, sample IDs and paths, take-level split lineage,
preference/consistency references, counterfactual-group lineage, observed-bin membership, sham
targets, held-out-mechanism isolation, counts, and the exact normalization fitting set.
