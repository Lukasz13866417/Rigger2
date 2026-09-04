# Coordinate conventions

All data inside `motionlab` uses one explicit convention:

- right-handed coordinates;
- +Y is up, +Z is character forward, and +X is character right;
- distances are metres, timestamps are seconds, and angles are radians;
- quaternions are active rotations in `[w, x, y, z]` order;
- `multiply(q_left, q_right)` applies `q_right` first and then `q_left` to a column vector;
- local joint orientation is relative to the parent joint;
- stored local animation orientation is absolute and includes the rest orientation;
- root translation is stored separately from root local orientation;
- the final frame of a cyclic clip is not a duplicate of the first frame.

Importers and exporters own all axis, handedness, unit, and rotation-order conversions. Core math
must not contain format-specific axis swaps. Adapters must preserve enough metadata to reproduce
the original-to-canonical transform.

## Forward kinematics

For root joint `r` and non-root joint `j` with parent `p`:

```text
global_rotation[r] = local_rotation[r]
global_position[r] = root_translation

global_rotation[j] = global_rotation[p] * local_rotation[j]
global_position[j] = global_position[p]
                   + global_rotation[p] * rest_offset[j]
```

## Rotation repair convention

Local tangent-space repair deltas are right-multiplied:

```text
q_repaired = q_bad * exp(delta_axis_angle)
delta_target = log(inv(q_bad) * q_clean)
```

