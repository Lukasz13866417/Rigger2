# Variable-rig data contract

Status: data infrastructure only. This contract does not introduce a variable-joint critic,
message passing, attention, training, or a production objective.

`motionlab.rigging.graphs` exposes two immutable graph records:

- `SkeletonGraph` stores permutation-canonical node order, source/canonical index maps, directed
  hierarchy edges, collapsed anatomical edges, semantic roles, canonical body parts/sides,
  helper/controller/end-site masks, and static features.
- `MotionGraph` stores per-frame features on that skeleton, graph-level root/time features, and
  explicit validity/confidence arrays.

`batch_motion_graphs` zero-pads arbitrary frame and joint counts. Its `joint_mask`, `frame_mask`,
`node_frame_mask`, `edge_mask`, and `anatomical_edge_mask` are authoritative. Padded edge indices
use `-1`; padded feature and confidence values are guaranteed to be zero.

## Coordinate streams

The contracts deliberately keep two coordinate streams separate:

- Universal features use canonical right-handed, Y-up, Z-forward metric world coordinates.
  Joint positions are root-relative; linear and angular velocities remain canonical-world
  vectors. These features do not depend on a joint's authored local axes.
- Rig-coordinate features use parent-local coordinates normalized into a declared reference
  basis. Per-joint basis metadata follows
  `G_reference = G_declared * declared_to_reference_quat_wxyz`. If this metadata is unavailable,
  the declared basis is retained and its existing axis-confidence value remains explicit.

Every `CoordinateBasisMetadata` also retains the source unit/axis provenance that produced the
already-canonical `MotionClip`. Graph construction never applies that importer transform twice.

Static and dynamic layouts are versioned independently through:

- `STATIC_UNIVERSAL_CONTRACT`
- `STATIC_RIG_CONTRACT`
- `DYNAMIC_UNIVERSAL_CONTRACT`
- `DYNAMIC_RIG_CONTRACT`
- `GLOBAL_DYNAMIC_UNIVERSAL_CONTRACT`

Each field declares its width, units, and coordinate space and can be addressed with
`contract.field_slice(name)`.

## Helpers and anatomical topology

All source joints remain auditable graph nodes. Helpers, controllers, and end sites are retained
but excluded through `model_joint_mask` unless source metadata explicitly supplies another valid
critic mask. `anatomical_parent` and `anatomical_edge_index` bypass excluded nodes, so inserting an
identity helper does not change the active anatomical graph or active motion features. Unknown
helper/twist joints inherit the nearest unambiguous canonical anatomical part; limb/center side is
then derived from that part.

## Guaranteed invariances

The deterministic test suite covers:

- source joint-storage permutation;
- frame/joint padding;
- identity helper insertion;
- declared local-axis reparameterization when basis metadata is supplied;
- importer unit and world-axis canonicalization;
- mixed joint and frame counts in one batch.

These are representation invariances, not evidence that a graph/attention architecture is
needed. That model decision remains blocked on the active human-label experiment.
