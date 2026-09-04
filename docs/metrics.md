# Deterministic metrics

Deterministic reports preserve measurements in physical units and localize actionable intervals.
The current grading slice includes:

- confidence-weighted heel/toe slip distance per contact interval in centimetres;
- maximum marker penetration and floating-contact height in centimetres;
- plane-tangential root speed and optional target mismatch in metres per second;
- cadence, same-side stride time and length, step width, and bilateral asymmetry;
- configured Euler-XYZ or local-Y swing/twist joint-limit excess and integrated violation, with
  explicit per-joint validity, source, local-axis confidence, and rig coverage;
- per-joint SO(3) angular jerk and robust pop/jitter events in radians per second cubed;
- a transparent loop-seam combination retaining pose, position, root-velocity, angular-velocity,
  and contact-state components.

Contact evidence uses four explicit virtual markers and a normalized plane `n dot x + d = 0`.
Height/speed thresholds, hysteresis, temporal cleanup, leg-length scaling, source, and confidence are
kept separate from metric values. No overall “naturalness” score is calibrated at this stage.

Generate a report with:

```bash
motionlab metrics motion.npz --output report.json --target-speed-mps 1.2
```

This writes `report.json` without dense frame arrays and `report.dense.npz` with those arrays. Use
`--dense-output PATH` to select another artifact path. Each compact metric records the dense keys
it owns; the report records the artifact checksum and motion ID. Dense gait arrays use NaN for
frames that are not measurement events.

Cadence is the mean instantaneous rate between consecutive unambiguous heel strikes. Stride time
and stride length use consecutive same-foot strikes. Length is measured along the plane-projected
root travel direction, while step width is measured on its lateral axis. Spatial gait values are
unavailable when the clip has no stable travel direction. Left/right asymmetry is the absolute
side difference divided by the bilateral mean.

Joint limits are evaluated relative to the stored rest local rotation. `euler_xyz` uses intrinsic
XYZ; `swing_twist` uses local Y as the twist axis. Missing, disabled, incomplete, or zero-confidence
limits are reported as unavailable rather than replaced with invented defaults. Plain BVH local
axes receive low confidence unless richer metadata explicitly supplies it.

Frame intervals in events are half-open: `[start, stop)`.
