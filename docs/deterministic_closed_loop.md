# Deterministic production repair and autonomous closed loop

This milestone corrects benchmark target semantics, runs production-constrained CMA-ES on the
three representable corruption families, adds a jitter-specific frequency representation, and
evaluates task-directed repair without exposing a clean motion to the repair system. CMA-ES
remains the optimizer. Variable-rig, graph, and attention work remains paused.

## Target-speed semantics and eligibility

Synthetic benchmark metadata now distinguishes:

- `task_target_speed`: an externally requested speed, or `null`;
- `clean_window_reference_speed`: speed measured on the exact clean benchmark window;
- `parent_clip_speed`: the inherited full-clip command, retained only as provenance;
- `target_speed`, `target_speed_tolerance`, and `target_speed_source`: the actual constraint.

Without an external task-speed requirement, `target_speed` is the measured clean-window speed.
With an external target, a clean window outside the declared tolerance is ineligible. Every
eligible pair is checked against every production constraint, and the invariant that an eligible
pair has a feasible clean endpoint is asserted in code and tests.

The audit covers all train-partition hard-corruption records in the three production families:

| Family | Candidates | Legacy eligible | Corrected eligible | Newly eligible |
|---|---:|---:|---:|---:|
| foot slide | 114 | 23 | 114 | 91 |
| floating contact | 119 | 24 | 119 | 95 |
| joint pop | 120 | 25 | 120 | 95 |
| **Total** | **353** | **72** | **353** | **281** |

No production tolerance was relaxed.

## Production CMA-ES

The benchmark uses two distinct source clips per family and three seeds per case: 18 runs total.
Optimization receives no clean target. The objective is the localized or clip-level deterministic
target metric, and deterministic production invariants plus collateral budgets are hard gates.
Clean distance and oracle projection are computed only by the evaluator.

| Family | Success | Median evaluations | Median runtime | Target before → after | Clean distance |
|---|---:|---:|---:|---:|---:|
| foot slide | 6/6 | 92 | 1.36 s | 5.848 → 1.866 cm | 0.0379 m root RMS |
| floating contact | 6/6 | 138.5 | 2.62 s | 4.792 → 0.000 cm | 0.0574 m root RMS |
| joint pop | 6/6 | 1,202 | 19.17 s | 10,605 → 5,350 rad/s³ | 3.924° rotation RMS |

Success requires no hard-constraint or collateral-gate rejection and at least 0.2 cm target
reduction for contact families or 100 rad/s³ for pop. It is an improvement criterion, not a claim
that the residual defect has fallen below a universal perceptual threshold.

The foot and floating blocks alter root translation only, so their clean rotation distances are
approximately `1e-6` degrees. Median collateral deltas are fully persisted. Notable values are a
`+0.0323 m/s` foot-repair target-speed error, a `-1.485 cm` floating-run slide change, and a
`+0.0780` pop-run loop-seam change; all accepted candidates remain inside hard budgets.

The adaptive clean-oracle target medians are 1.553 cm for foot slide, 4.688 cm for floating
contact, and 105.49 rad/s³ for joint pop. The floating result is intentionally informative: the
clean reference itself contains about 4.69 cm of the clip-wide metric while CMA can drive the
metric to zero. Thus “production success” means constraint-valid deterministic defect reduction,
not clean-motion recovery or perceptual optimality.

The existing foot-lock/two-bone-IK repair has a 2.021 cm median and beats generic CMA in 4/6
paired runs; generic CMA has a lower pooled median but wins only 2/6 individual comparisons. The
local deterministic smoothing operator reaches a 58.18 rad/s³ median on pop, far better than the
budgeted generic CMA result. Specialized operators remain preferred when applicable.

One broad foot-search population initially produced almost no feasible samples. The hardened
policy keeps CMA-ES and uses a smaller-sigma retry only when flat fitness occurs, spending only
the remaining evaluation budget.

## Jitter-specific representation

`LocalSpectralJitterBlock` represents local tangent corrections with an orthonormal DCT basis and
composes them through SO(3). It does not increase generic spline density. On two moderate
white-noise jitter cases, the exact support is frames 29–130 and the same three torso/shoulder
joints:

- 95% oracle energy needs 507 and 514 spectral coefficients and leaves 0.0618° and 0.0484° clean
  rotation RMS;
- the full 918-mode local spectral basis reconstructs both endpoints to about `1e-6` degrees;
- white noise is genuinely broadband, so the spectral oracle is representable but not
  low-dimensional.

`JitterSmoothingBlock` exposes seven production parameters: cutoff frequency, smoothing strength,
blend-in/out, and one strength per affected joint. Its oracle grid reduces localized jerk p95 from
6,342 to 1,049 rad/s³ and from 4,963 to 192 rad/s³, with 0.149° and 0.092° clean rotation RMS.
The clean reference is used only to evaluate/select this representability audit, not as a proposed
production objective.

## Autonomous deterministic repair

The end-to-end harness supplies a corrupted animation and task requirements declaring the primary
deterministic objective. The system maps that objective to a permitted block, localizes it from
inference-available diagnostics, and runs constrained CMA-ES. Synthetic family labels,
generation parameters, clean lineage, and postconditions are removed from the system input. The
clean motion is not in the optimizer function contract and is consulted only by the evaluator.

Across two clips per family at seed 7101, block activation, diagnostic-only recommendation, hard
constraint satisfaction, and repair success are all 6/6. Median results are:

| Family | Target reduction | Evaluations | Runtime | Clean distance |
|---|---:|---:|---:|---:|
| foot slide | 3.783 cm | 92 | 1.36 s | 0.0289 m root RMS |
| floating contact | 2.788 cm | 203.5 | 3.09 s | 0.0553 m root RMS |
| joint pop | 5,028 rad/s³ | 1,202 | 19.42 s | 4.378° rotation RMS |

This is a successful first deterministic closed loop, within a deliberately narrow scope:
fixed-rig synthetic moderate defects, measurable cases, known deterministic task objectives, and
two source clips per family. It does not establish naturalness or perceptual quality.

## Reproduction

```bash
motionlab run-production-cma-benchmark artifacts/phase8_critic_hardening/dataset \
  --output artifacts/phase11_deterministic_closed_loop/production_cma
motionlab run-jitter-spectral-audit artifacts/phase8_critic_hardening/dataset \
  --output artifacts/phase11_deterministic_closed_loop/jitter_spectral
motionlab run-autonomous-deterministic-repair artifacts/phase8_critic_hardening/dataset \
  --output artifacts/phase11_deterministic_closed_loop/autonomous_repair
```

Machine-readable reports are under `artifacts/phase11_deterministic_closed_loop`.

## Stop decision

The current GroupNorm critic remains the baseline, LayerNorm remains an experimental artifact,
all adversarial-quality pairs remain preserved, and learned hard-defect heads remain diagnostic
or training signals rather than production objectives. No normalization, variable-rig, graph, or
attention work was started.

The next research decision is whether to build a learned perceptual objective for naturalness,
whole-body coordination, believable weight transfer, subtle anatomical plausibility, style
coherence, and robotic-but-numerically-valid motion. That decision is intentionally deferred.
