# Adaptive optimization diagnostics

This milestone isolates optimizer adequacy from representation and critic failures, replaces the
fixed oracle repair basis with a residual-driven multiresolution basis, and keeps graph/attention
and optimizer replacement out of scope. The compact decision artifact is
`artifacts/phase10_adaptive_optimization/stop_report.json`.

## Required outcomes

### 1. Guaranteed-representable CMA-ES recovery

Each trial constructs `x_bad = F(x_clean, z_corrupt)` with the same cubic SO(3) spline residual
used by CMA. The clean motion, corrupted motion, and exact inverse `-z_corrupt` all pass the
unchanged production constraints. CMA minimizes exact motion distance, without a critic.

| Active dimensions | Successful trials | Median evaluations to success | Worst final objective | Worst rotation RMS |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 3/3 | 461 | 9.38e-7 | 0.0194 deg |
| 25 | 3/3 | 1,282 | 9.83e-7 | 0.0199 deg |
| 50 | 3/3 | 2,568 | 9.92e-7 | 0.0200 deg |
| 100 | 3/3 | 6,347 | 9.77e-7 | 0.0198 deg |

Overall recovery is 12/12. Every trial records the exact optimum vector, recovered vector,
evaluation trajectory, final distances, and clean/corrupt/optimum/final constraint checks. This
supports retaining CMA-ES; evaluation cost increases with dimension, but no material recovery
failure appears through 100 dimensions.

### 2. Multiresolution oracle representability

The active subspace is the smallest scalar joint-axis/root-component set capturing at least 95%
of normalized exact clean-residual energy. A shortest 95%-energy time interval receives a
four-frame halo. The hierarchy then adds coarse whole-clip splines, medium local splines over the
remaining error, and fine per-channel local splines. Joint axes may therefore have different
intervals and control counts.

| Family | Fixed 8 representable | Coarse / medium / fine parameters | Fine energy remaining | Fine representable |
| --- | --- | ---: | ---: | --- |
| floating contact | yes | 4 / 12 / 110 | 1.30e-12 | yes |
| foot slide | no | 4 / 12 / 79 | 4.30e-12 | yes |
| joint jitter | no | 36 / 108 / 684 | 2.54e-2 | no |
| joint pop | no | 4 / 12 / 27 | 5.12e-11 | yes |

Representability improves from 1/4 to 3/4. Jitter remains a genuine high-bandwidth residual: its
fine basis cuts remaining energy to 2.54%, but the derivative-sensitive target remains 1,026.7
rad/s^3 versus 437.1 for clean, so it is not mislabeled as an exact repair.

Production localization uses authored deterministic intervention data when available, otherwise
deterministic metric events, then optional critic symptom probabilities. Time receives an
eight-frame context halo and joints receive one parent/child ring. Unrelated parts are excluded by
default; a conservative full-clip fallback is explicit.

### 3. Deterministic production rerun

No production CMA run was executed. This is the intended hard gate, not a missing experiment:
zero of four fine oracle candidates satisfies all production constraints. Learned hard-defect
heads were not used as production objectives. The rerun artifact records zero eligible, zero
executed, and zero successful cases as `blocked_by_oracle_feasibility_gate`.

### 4. Remaining production failures

The three prior `PRODUCTION CONSTRAINT` runs are three seeds of one floating-contact case. The
exact clean target fails `target_speed`, the sampled bad-to-clean path never enters the feasible
set, and the best multiresolution oracle projection also fails `target_speed`. This is now
classified as one `inconsistent_target_constraints` case, rather than an optimizer failure.

The root cause is that the 160-frame window retains a full-take speed command that the clean
window itself misses. Parameterization-induced infeasibility, path-only infeasibility, and genuine
task incompatibility each have zero identified prior cases in this audit.

### 5. Adversarial-quality corpus

Twelve LayerNorm red-team exploits were appended to the six prior GroupNorm exploits. All 18 are
family-neutral `source > exploit` pairs. Each entry links the source/exploit motion tensors,
complete critic outputs, deterministic metrics, CMA trajectory, six-frame manual overlay label,
and source/checkpoint/dataset lineage. The 12 additions share the visual label
`perceptually_worse_global_pose_distortion`; no new critic optimization was run in this milestone.

## Parameterization implementation

`MotionParameterBlock` is the common numerical interface. CMA can directly optimize one block or
a `CompositeMotionParameterization`:

- contact/IK: target offset, foot-lock strength and blends, pelvis limit, and root compensation;
- speed/cadence: local/global temporal warp, displacement scale, stride, and cadence;
- loop seam: phase/cut, pose closure, and root-velocity closure;
- generic pose: smooth root and SO(3) residual splines;
- adaptive pose: independently localized multiresolution scalar patches.

The reusable block driver persists motions, named parameter vectors and bounds, feasibility
rejections, and per-generation history.

## Normalization artifact policy

Both checkpoints remain experimental artifacts. GroupNorm is more robust but has worse locality
and repair-path monotonicity. LayerNorm has exact aligned-window consistency and a monotonic repair
path, but sham false positives rise from 0.132 to 0.500 and bizarre red-team outcomes rise from
6/12 to 12/12. LayerNorm is not promoted and no additional normalization training cycle was run.

Deferred comparisons are per-frame RMSNorm, a normalization-free residual block, and LayerNorm
with adversarial/sham hardening.

## Reproduction

```bash
motionlab validate-cma-inverse \
  --output artifacts/phase10_adaptive_optimization/cma_representable_inverse

motionlab run-adaptive-oracle-diagnostics \
  artifacts/phase8_critic_hardening/dataset \
  --experiment artifacts/phase8_critic_hardening/cma_experiment/experiment.json \
  --previous artifacts/phase9_cma_diagnostics/oracle_representability/representability.json \
  --failure-matrix artifacts/phase9_cma_diagnostics/failure_classification_matrix.json \
  --output artifacts/phase10_adaptive_optimization
```

## Decision

Retain CMA-ES. Fix the window-level target-speed command contract and improve the jitter residual
space or target metric before judging deterministic production repair again. Graph/attention and
variable-rig work remain paused.
