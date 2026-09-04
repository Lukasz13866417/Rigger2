# CMA failure decomposition and normalization ablation

This diagnostic follows the first fixed-rig CMA experiment without adding graph or attention
layers. It separates spline representability, optimizer eligibility, critic landscape, and
production feasibility, then repeats the critic with per-frame LayerNorm.

## Oracle representability

The target metric is the exact metric used by each corruption postcondition, not a generic
clip-wide proxy. Foot and ground measurements retain the clean contact schedule and ground plane;
pop and jitter measurements retain the original target joints and derivative-support interval.
The exact rotation residual is `Log(R_bad^-1 R_clean)` and root residual is `p_clean - p_bad`.
Each exposed channel is fit with bounded least squares using the same scale and coefficient bounds
as CMA.

| Family | 4-control result | 8-control result | Residual-driven localized result | Current-space class |
| --- | --- | --- | --- | --- |
| floating | root RMS 12.39 mm; 75.8% target gap | 6.27 mm; 101.0% target gap | 17 controls, 0.287 mm; 99.97% | production constraint |
| foot slide | root RMS 7.80 mm; 9.5% target gap | 3.63 mm; 47.9% target gap | 17 controls, 0.167 mm; 95.8% | search space |
| joint jitter | rotation RMS 0.218 deg; 0.001% target gap | 0.217 deg; 0.001% target gap | 76 controls, 0.0268 deg; 92.6% | search space |
| joint pop | rotation RMS 0.0914 deg; no target gain | 0.0907 deg; no target gain | 15 controls, numerical pose error; 100% | search space |

The global pose RMS can look small while the defect remains: derivative-based jitter/pop metrics
are sensitive to precisely the frame-scale residual that an eight-control cubic spline cannot
represent. Localized knots were therefore added only from exact residual support. Jitter required
frame-scale knots over its measured interval; pop required repeated knots around its one-frame
pulse. These enlarged fits diagnose required bandwidth and are not silently adopted as a new CMA
space.

No expanded-joint test was required. The largest rotation RMS outside exposed joints was only
`3.8e-7` degrees across all joint-frames. The apparently nonzero energy *fraction* on root-only
corruptions is quaternion round-trip noise, and the report retains both the fraction and absolute
energy so it cannot be mistaken for a missing anatomical channel.

Every oracle candidate was rejected by the unchanged production target-speed constraint. The
selected 160-frame windows already miss the full-take command by 0.135--0.163 m/s, above the hard
0.05 m/s limit. Floating is representable with eight controls but not production-valid; the other
three fail the current-space oracle before the constraint is considered.

## Oracle-objective CMA eligibility

The specified oracle-objective rerun applies only when a valid repair exists in the current space
under the same production constraints. There are zero such cases, so running CMA with a relaxed
constraint would answer a different question and was not done. Optimizer adequacy is therefore
not identifiable from this batch; no case is classified `OPTIMIZER`.

## Repair-path monotonicity

Each pair was evaluated at alpha `0.00, 0.05, ..., 1.00` using the exact SO(3) exponential path
and root interpolation. Negative alpha/severity Spearman is the desired direction.

| Family | GroupNorm rho | GroupNorm adjacent | inversions | LayerNorm rho | LayerNorm adjacent | inversions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| floating | -0.995 | 90% | 2 | -1.000 | 100% | 0 |
| foot slide | +1.000 | 0% | 20 | -1.000 | 100% | 0 |
| joint jitter | +1.000 | 0% | 20 | -1.000 | 100% | 0 |
| joint pop | +0.995 | 10% | 18 | -1.000 | 100% | 0 |

The GroupNorm foot, jitter, and pop severity landscapes point away from known clean repairs. This
is direct critic-landscape evidence independent of CMA. LayerNorm completely fixes these four
one-dimensional repair paths.

## GroupNorm versus per-frame LayerNorm

Only normalization changed. Width 128, dilations 1/2/4/8/16, losses, data, optimizer, seed 2027,
batch size, and early-stop rule are identical. GroupNorm selected epoch 9; LayerNorm selected
epoch 23.

| Aggregate metric | GroupNorm | LayerNorm |
| --- | ---: | ---: |
| known-source AUROC / AP | 0.980 / 0.915 | 0.982 / 0.934 |
| known temporal / part F1 | 0.736 / 0.578 | 0.784 / 0.654 |
| held-out-mechanism AUROC / AP | 0.947 / 0.819 | 0.926 / 0.797 |
| held-out temporal / part F1 | 0.427 / 0.227 | 0.386 / 0.265 |
| known severity / rank rho | -0.332 / 0.480 | 0.687 / 0.648 |
| held-out severity / rank rho | -0.224 / -0.031 | 0.599 / 0.499 |
| all justified pair accuracy | 0.636 | 0.841 |
| sham target-head false-positive rate | 0.132 | 0.500 |
| mean repair-path adjacent accuracy | 0.250 | 1.000 |
| maximum aligned interior-window difference | 0.003873 | 0.000000 |
| bizarre score-reducing red-team runs | 6/12 | 12/12 |

The overlap test slices two 160-frame windows, offset by 16 frames, from one shared full-motion
feature tensor. It compares the 20 globally identical frames farther than the 62-frame receptive
radius from both boundaries. This isolates model window dependence from feature recomputation.
LayerNorm is bit-identical; GroupNorm is not.

LayerNorm improves repair-path monotonicity, severity/ranking, and window consistency, but it does
not improve exploit resistance. All twelve repeated red-team runs find bizarre low-score outputs,
including jitter/pop cases that GroupNorm's collapsed severity heads could not optimize. It also
raises sham target false positives to 50%. LayerNorm is therefore not adopted as a production
critic from this ablation.

## Preserved adversarial examples

All six original category-3 candidates are packaged as `source > exploit` adversarial-quality
pairs without assigning an existing defect-family label. Every record includes source and
optimized motion, full frame/part/clip critic tensors, deterministic metrics, spline coefficients,
CMA generation trajectory, overlay inspection, and source-run lineage.

Visual overlay and geometric inspection consistently find strange root compensation, whole-body
coordination/anatomy distortion, and unusual amplitude scaling. There is no freezing or
high-frequency cluster in these smooth-spline examples. The cluster labels remain descriptive
perceptual concepts, not synthetic defect ground truth.

## Production objective policy

Production mode now defaults to directly measured, scale-normalized family targets and rejects an
attempt to use learned hard-defect severity heads as the production objective. Red-team mode keeps
those learned heads specifically to expose critic failures. Foot slide, trusted-contact
penetration/floating, loop seam, target speed, authored joint limits, and appropriate deterministic
smoothness remain deterministic objectives or hard constraints. Future learned production terms
are reserved for residual naturalness, coordination, style coherence, anatomy beyond explicit
limits, and plausible weight transfer.

## Final classification matrix

| Family | red-team seeds | production seeds | primary reason |
| --- | --- | --- | --- |
| foot slide | 3 x `SEARCH SPACE` | 3 x `SEARCH SPACE` | eight controls recover only 47.9% of exact target gap |
| joint jitter | 3 x `SEARCH SPACE` | 3 x `SEARCH SPACE` | frame-scale residual is outside spline bandwidth |
| joint pop | 3 x `SEARCH SPACE` | 3 x `SEARCH SPACE` | one-frame pulse is outside spline bandwidth |
| floating | 3 x `CRITIC LANDSCAPE` | 3 x `PRODUCTION CONSTRAINT` | representable; red mode exploits critic, production oracle fails speed gate |

Totals across the 24 failed original runs are 18 `SEARCH SPACE`, 3 `CRITIC LANDSCAPE`, 3
`PRODUCTION CONSTRAINT`, 0 `OPTIMIZER`, and 0 `MIXED/UNKNOWN`. Jitter/pop/foot also have secondary
critic-landscape evidence, but the requested precedence assigns a failed current-space oracle to
`SEARCH SPACE`.

Graph/attention work remains stopped. The next decision should address spline bandwidth by defect,
window-aligned production commands, sham robustness, and off-path adversarial-quality supervision;
then rerun oracle-objective CMA on a nonempty feasible set.

## Artifacts

- `artifacts/phase9_cma_diagnostics/oracle_representability/representability.json`
- `artifacts/phase9_cma_diagnostics/oracle_objective_cma/eligibility.json`
- `artifacts/phase9_cma_diagnostics/group_norm_repair_path/repair_path_monotonicity.json`
- `artifacts/phase9_cma_diagnostics/layer_norm_repair_path/repair_path_monotonicity.json`
- `artifacts/phase9_cma_diagnostics/normalization_ablation.json`
- `artifacts/phase9_cma_diagnostics/failure_classification_matrix.json`
- `artifacts/phase9_cma_diagnostics/adversarial_exploits/manifest.json`
- `artifacts/phase9_cma_diagnostics/layer_norm_red_team/red_team_matrix.json`
- `artifacts/phase9_cma_diagnostics/production_objective_policy.json`
