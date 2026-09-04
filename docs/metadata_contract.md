# Forward-compatible metadata contract

The fixed-rig `Skeleton` and `MotionClip` remain the source of truth. `adapt_motion_clip` wraps a
clip without modifying it and exposes the metadata needed by later canonical graph and critic
stages. Calling `adapter.to_motion_clip()` returns a detached clip with the exact original content
hash.

```python
from motionlab.core import adapt_motion_clip, assign_source_lineage

source = assign_source_lineage(
    clip,
    source_dataset="100STYLE",
    source_clip_id="BR",
    source_take_id="BR_forward_001",
    split="train",
    clean_confidence=0.9,
)
adapter = adapt_motion_clip(source, contacts=estimated_contacts)
```

## Coordinate and pose-reference metadata

`CoordinateContract` records the original handedness, signed up/forward axes, unit-to-metre scale,
and orthonormal source-to-canonical matrix. It validates that the matrix agrees with the declared
axes and supplies tested forward/inverse vector conversions. Its output contract is always
right-handed, Y up, +Z forward, metres.

Already-internal clips explicitly use `motionlab_internal_contract`; this is not a silent axis
assumption. BVH clips retain their importer transform. Plain BVH rest rotations are labeled
`bvh_zero_channel_rest` at low confidence because they are not trusted anatomical joint axes.

Rest pose, bind pose, and pose anchor are separate fields. Optional bind rotations and anchor
positions are shape-checked and immutable. Absence has zero confidence rather than a numeric
placeholder.

## Joint semantics and masks

Every joint exposes:

- a primary semantic role and all aliases, with `UNKNOWN` supported;
- role and local-axis confidence;
- side: unknown, center, left, or right;
- one of eight initial anatomical parts plus unknown;
- deform, helper, controller, and end-site flags;
- explicit critic and repair masks.

The initial parts are root/pelvis, trunk/spine, head/neck, left/right arm, left/right leg, and
other/accessory. They come from semantic roles, never equal-sized joint buckets. Controllers
cannot also be deform or active critic joints.

## Modality and ground metadata

Rotation and position validity/confidence have shape `[T,J]`. Contact validity/confidence, when
available, have shape `[T,M]` with marker names and a source. Invalid entries are required to have
zero confidence. BVH joints with no rotation channels are marked rotation-invalid rather than
treating identity quaternions as observed motion.

Ground planes may be clip-wide `[1,4]` or per-frame `[T,4]`; normals are unit length. Validity,
confidence, and source remain separate. If no ground is available, validity and confidence are
zero and the source is `unavailable`.

## Source and split lineage

Call `assign_source_lineage` once, before windows, mirrors, retargets, or corruptions. It stores:

- source dataset, clip ID, and take ID;
- a stable split-lineage ID derived from dataset plus original take unless supplied explicitly;
- optional train/validation/test assignment and clean confidence;
- importer and retargeter versions;
- typed, content-addressed operation steps and corruption lineage.

Reassignment is rejected. Resampling, canonicalization, slicing, corruption, foot lock, loop
closure, and retargeting append immutable steps while preserving the original split lineage. BVH
file import assigns lineage automatically and uses a source-file SHA-256 as its default take ID.
Legacy clips without embedded lineage remain supported and are explicitly returned with
`lineage_complete=False`.

Provenance, filenames, dataset identity, split, and corruption mechanism are metadata only. The
adapter keeps them structurally separate from modality tensors and marks that they must not enter
ordinary neural inputs.
