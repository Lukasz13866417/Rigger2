# Critic hardening and spline CMA-ES experiment

This report records the bounded follow-up to the first fixed-rig TCN. It expands clean controls,
replaces global synthetic ranking with family-specific ranking, audits every sham, and runs the
first learned optimization experiment. It does not add graph or attention layers.

## Expanded real-data gate

Fourteen official 100STYLE forward-walking BVHs were checked using the unchanged cleanliness
thresholds. Twelve qualified: seven train, two validation, and three test takes. `GracefulArms`
and `LookUp` were retained as downloaded evidence but rejected because their measured ground
penetration exceeded the fixed 2 cm limit. They do not appear in the dataset.

Two independent 160-frame windows were selected per qualified take. Each window produces its
unaltered clean sample and two no-defect heading controls at -35 and +35 degrees. The resulting
version-7 dataset contains:

| Item | Count |
| --- | ---: |
| distinct qualified source clips | 12 |
| source windows / clean samples | 24 / 24 |
| heading-equivalent controls | 48 |
| matched foot-slide shams | 114 |
| hard corruptions | 1,388 |
| all samples | 1,574 |
| train / validation / known test / held-out corruptor | 595 / 168 / 256 / 555 |

All five observed-severity bins are populated for every requested mechanism. All accepted hard
samples remain inside their requested bin, all shams lack the target symptom, held-out corruptors
remain absent from training, and all manifests/checksums validate. Normalization uses only the
595 training records.

Training sampling is deterministic and balanced: half of an epoch is drawn from clean,
equivalent, and sham controls, and half from positive samples balanced across defect families.
Sampling with replacement prevents the persisted synthetic prevalence from becoming the task
prior.

## Family-specific ranking and fixed TCN rerun

The architecture remains the 128-channel fixed-rig flat TCN. Its former scalar ranking head is
replaced with a seven-component nonnegative family-ranking vector. Ranking loss is masked to the
active family; inactive families receive no ordering gradient. There is no learned global quality
score. The separate clip-severity vector remains available for the optimization red team.

Training stopped at epoch 17 and selected epoch 9. Total train/validation loss moved from
2.289/2.435 at epoch 1 to 0.757/0.881 at the selected epoch. Validation then worsened to 1.701 by
epoch 17, so the existing early-stop rule correctly retained epoch 9.

Primary-mechanism clip detection on unseen source takes:

| Family | Known AUROC / AP | Held-out mechanism AUROC / AP | Known family-rank rho | Held-out family-rank rho |
| --- | ---: | ---: | ---: | ---: |
| floating | 0.996 / 0.985 | n/a | 0.901 | n/a |
| foot slide | 0.888 / 0.513 | 0.918 / 0.773 | 0.737 | 0.752 |
| penetration | 1.000 / 1.000 | n/a | 0.901 | n/a |
| jitter | 1.000 / 1.000 | 1.000 / 1.000 | 0.861 | 0.831 |
| pop | 1.000 / 1.000 | 1.000 / 1.000 | -0.961 | -0.978 |
| loop seam | 0.976 / 0.909 | 0.866 / 0.440 | 0.934 | 0.140 |
| speed mismatch | 1.000 / 1.000 | 0.948 / 0.882 | -0.014 | -0.899 |

Pairwise within-family ordering over all 1,210 justified edges is 0.636 overall: floating 0.870,
foot slide 0.831, penetration 0.917, jitter 0.729, pop 0.092, loop seam 0.914, and speed mismatch
0.450. The model is therefore a strong detector for several families but the pop and speed
ordering heads are not suitable optimization objectives.

Known subtle-bin recall is 1.0 for floating, penetration, jitter, pop, and speed, but 0.333 for
foot slide and loop seam. Held-out subtle recall is 0.333 foot slide, 1.0 jitter, 1.0 pop, 0.0
loop seam, and 0.833 speed. High pop detection with reversed ordering is explicitly not treated
as severity success.

The 30-sample memorization stress gate reached 0.947 temporal F1 and 0.996 mean within-family rank
rho, but only 0.933 exact clip match because two near-threshold loop/speed samples cross-fired on
each other's head. Extending to 350 epochs did not fix it. This failure is retained as evidence;
the model and gate thresholds were not tuned around it.

## Sham forensics

All 114 shams across train, validation, and test are audited. Each audit directory contains
reconstructed clean and sham NPZ clips, a six-frame overlay, target probability/severity/rank,
all other learned heads, deterministic before/after/delta metrics, and root, local-rotation, and
FK-joint RMS distances.

The target foot-slide head fires on 15/114 shams (0.132). Any head fires on 33/114 (0.289).
Thirteen target positives have no material deterministic alternate artifact and are classified as
target-head shortcuts. Two have material angular-smoothness degradation and are kept distinct as
other-artifact evidence. Many shortcut shams are numerical round trips with sub-micrometre pose
distance, showing that the remaining false positives are primarily source/window priors rather
than the sham edit itself.

## BIPOP-CMA-ES implementation

The optimizer uses a low-dimensional cubic B-spline residual, never a per-frame/per-joint search.
It starts with four control points and least-squares lifts the same curve to eight for refinement.
Selected pelvis-chain, leg, spine, shoulder, and elbow joints use
`R_new = R_old * Exp(delta_omega)`; separate splines control root XYZ translation and yaw.

The objective is a weighted selection of individual learned family severity heads. A global rank
does not exist and deterministic metrics are not part of the red-team objective. The production
mode rejects candidates outside target-speed, penetration, trusted authored-limit, declared-loop,
task-displacement, absurd-motion, or inference bounds.

Every run persists original/final clips, every generation's best clip, CMA mean/sigma/covariance,
spline coefficients and channel layout, all learned predictions, deterministic metrics,
distances, run parameters, rejection counts, and an overlay preview.

## Multi-seed results

Floating was included because the learned detector beat its deterministic metric baseline on
both AUROC and AP. The matrix covers foot slide, jitter, pop, and floating, three seeds, and both
modes: 24 runs total.

| Outcome | Runs |
| --- | ---: |
| category 1: genuine measured repair | 0 |
| category 2: score exploit without measured gain | 0 |
| category 3: bizarre/adversarial solution | 6 |
| category 4: no meaningful or feasible improvement | 18 |

All foot-slide red-team seeds reduced learned severity from 0.0572 to 0.00035-0.00059. One
representative also reduced measured stance slip from 3.95 to 1.75 cm, but increased penetration
from 1.02 to 12.80 cm, distorted gait, and is correctly category 3. All floating red-team seeds
reduced learned severity from 0.228 to 0.0093-0.0137; the representative collapsed contact/gait
events and changed the body pose, so these are also category 3. These six outputs are explicitly
eligible as adversarial training data.

Jitter and pop learned severity began at approximately 4.9e-5 and 1.5e-6 despite strong defect
probability, leaving no useful optimization range; their red-team runs are category 4. All 12
production runs returned no feasible improvement. The representative windows already differed
from the full-take speed command by 0.135-0.163 m/s, and the short bounded search found no
candidate that simultaneously entered the hard 0.05 m/s target-speed band and satisfied the
remaining constraints. Invalid candidates were rejected rather than reported as repairs.

## What the experiment discovered

The first learned optimization attempt did not produce a production repair. It did produce useful
failure data:

- the clip detector and clip-severity head can disagree radically, especially for jitter/pop;
- low-frequency coordinated pose/root changes can suppress foot/floating scores while creating
  penetration, gait-event collapse, and implausible support;
- window-level task-speed supervision must be aligned with the production command before the
  production feasible set is useful;
- support/contact validity, gait-event continuity, collateral-family severity, and OOD confidence
  are missing critic dimensions;
- new corruptions should include smooth whole-body spline residuals, contact-event collapse,
  score-suppressing penetration trades, and task-speed-preserving adversarial edits.

The result is a successful red team and a failed production repair milestone. Do not proceed to
graph/attention architecture work on the strength of these outputs.

## Artifacts

- `artifacts/phase8_critic_hardening/real_data_audit.json`
- `artifacts/phase8_critic_hardening/dataset/summary.json`
- `artifacts/phase8_critic_hardening/critic/best.pt`
- `artifacts/phase8_critic_hardening/evaluation/evaluation.json`
- `artifacts/phase8_critic_hardening/evaluation/sham_forensics/sham_audit.json`
- `artifacts/phase8_critic_hardening/cma_experiment/experiment.json`
