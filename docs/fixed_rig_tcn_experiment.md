# Fixed-rig flat-TCN experiment

> Historical first-run report. The expanded hardening rerun and initial learned optimization
> experiment are documented in `docs/critic_hardening_cma_experiment.md`.

This is the deliberately small first learned-critic experiment. It stops after the fixed-rig
baseline: no graph/attention model and no repair search were implemented.

## Real-motion gate

The audit uses four official 100STYLE forward-walk BVHs at 60 fps: `ArmsBySide_FW` and
`Neutral_FW` for training, `Old_FW` for validation, and `Proud_FW` for test. One deterministic
160-frame window is used from each take. All four takes passed fixed cleanliness checks, with
qualification confidence from 0.406 to 0.574 and bilateral heel-strike/toe-off events present.
Joint-limit excess is omitted because plain BVH local axes do not provide trustworthy authored
joint limits for this rig.

The final v6 dataset contains 255 samples:

| Dimension | Counts |
| --- | --- |
| Role | 4 clean, 233 hard corruptions, 18 shams |
| Data split | 80 train, 39 validation, 41 known-mechanism test, 95 held-out-corruptor |
| Family | 4 clean, 56 foot slide, 20 penetration, 20 floating, 37 loop seam, 40 pop, 40 jitter, 38 speed mismatch |

All five measured-severity bins are populated globally for all 12 requested mechanism keys. All
accepted samples fall inside their requested observed-severity bins. The audit contains 47 left
and 49 right contact-target records, and held-out mechanism keys are absent from training.
Eighteen shams were accepted and all lack the target foot-slide symptom.

Of 240 hard-corruption attempts, 233 were accepted. The seven explicit rejects were two
`foot_slide/root_drift` collateral-defect failures, three
`loop_seam/root_velocity_mismatch` per-source bin misses, and two
`speed_inconsistency/leg_pose_amplitude_scale` per-source bin misses. Thresholds were not changed.
The loop end-rotation search floor was reduced from 0.04 to 0.0005 rad so it could generate the
existing near-threshold bin; this changed the generator range, not the acceptance definition.

The expected-speed feature is fixed from the qualified clean source. Exact deterministic metric
summaries are persisted as metadata for the two trivial baselines and are explicitly forbidden
from the TCN's neural-input arrays.

## Model and reproducibility

`FixedRigFlatTCN` flattens 23 joints with 15 features per joint and appends 21 global features. A
1x1 input projection feeds five noncausal, same-length residual Conv1D blocks with 128 channels,
kernel size 3, and dilations 1, 2, 4, 8, and 16. It has no temporal downsampling, attention, or
graph layers. Heads emit frame logits/severity `[B,T,K]`, anatomical-part logits `[B,T,P,K]`,
clip logits/severity, a ranking score, and a 128-dimensional clip representation.

The tiny gate deliberately includes clean and sham controls plus near-threshold and severe
examples across all seven modeled families. At 250 epochs it reached:

- clip exact match: 1.000;
- temporal localization F1: 0.914;
- severity-order Spearman correlation from the dedicated ranking head: 0.920;
- loss: 3.759 to 0.104.

The exact-resume test compares uninterrupted two-epoch training with one epoch plus checkpoint
resume. Histories and every model tensor match bit-for-bit. Checkpoints persist model, optimizer,
next epoch, best state, all RNG states, immutable configs, and manifest/normalization hashes.

The selected actual run uses AdamW at 0.001 and early stopping. Training ran 26 epochs; epoch 18
was selected (train loss 0.420, validation loss 8.217). Epoch 1 was 3.234/11.628 and epoch 26 was
0.224/9.337. The widening train/validation gap is material evidence of overfitting/source shift.
A post-tiny 0.0005 learning-rate check had worse best validation loss (8.998), so it was not
selected; test metrics did not drive this choice.

## Primary-mechanism results

The main tables count a family as positive only when it is the sample's primary mechanism.
Samples carrying that family only as a collateral label are excluded from that family's binary
comparison. The machine-readable report also retains a separate all-labeled-symptoms view.

Known mechanism, unseen `Proud_FW` source:

| Family | AUROC | AP | Temporal F1 | Part F1 | Severity rho | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| floating | 1.000 | 1.000 | 0.057 | 0.046 | -0.900 | 0.087 |
| foot slide | 0.761 | 0.287 | 0.000 | 0.040 | 0.564 | 0.317 |
| penetration | 0.733 | 0.519 | 0.126 | 0.056 | 1.000 | 0.360 |
| jitter | 0.967 | 0.760 | 0.007 | 0.000 | -1.000 | 0.112 |
| pop | 0.833 | 0.644 | 0.000 | 0.000 | 0.900 | 0.065 |
| loop seam | 0.578 | 0.327 | 0.281 | 0.002 | -1.000 | 0.099 |
| speed mismatch | 1.000 | 1.000 | 0.433 | 0.327 | 0.900 | 0.376 |

Held-out primary mechanism on `Proud_FW`:

| Family | Held-out mechanism | AUROC | AP | Temporal F1 | Part F1 | Severity rho | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| foot slide | hip rotation drift | 0.800 | 0.630 | 0.000 | 0.078 | -0.900 | 0.130 |
| jitter | band-limited correlated | 1.000 | 1.000 | 0.109 | 0.007 | -1.000 | 0.174 |
| pop | smooth rotation pulse | 0.800 | 0.728 | 0.103 | 0.002 | -0.100 | 0.143 |
| loop seam | root-velocity mismatch | 0.700 | 0.282 | 0.000 | 0.000 | -1.000 | 0.092 |
| speed mismatch | leg-pose amplitude scale | 0.863 | 0.819 | 0.000 | 0.000 | 0.700 | 0.266 |

Penetration and floating have no second mechanism in this restricted experiment, so no held-out
primary-mechanism result is claimed for them.

At the fixed 0.5 threshold, known-source recall is 1.0 in every bin only for foot slide and speed
mismatch. Penetration detects clear/severe only; floating, jitter, pop, and loop seam have zero
recall in all or nearly all bins. For held-out mechanisms, foot slide has recall 1.0 in all five
bins, jitter detects severe only, and pop/loop/speed have zero recall. This can coexist with useful
AUROC because the outputs are badly calibrated.

Pairwise severity ranking accuracy is 0.400 over 55 pairs: floating 0.20, foot slide 0.20,
penetration 1.00, jitter 0.00, pop 0.50, loop seam 0.20, and speed mismatch 0.80. Small
cross-mechanism predicted-severity differences are not treated as success because several
severity heads are nearly collapsed or reversed.

All five test shams caused at least one false positive, for a sham false-positive rate of 1.000.
The linear mechanism-ID probe scored 0.086 against 0.143 chance on 35 known-mechanism test
samples (69 probe-training samples). It does not expose a linearly decodable mechanism shortcut,
but it also does not rescue the localization, ranking, calibration, or sham failures.

## Deterministic baselines

AUROC/AP on primary known mechanisms:

| Family | Relevant metric | Metric alone | Summary MLP | TCN |
| --- | --- | ---: | ---: | ---: |
| floating | floating-contact cm | 0.733 / 0.531 | 0.878 / 0.716 | 1.000 / 1.000 |
| foot slide | stance-slip cm | 0.600 / 0.651 | 0.581 / 0.153 | 0.761 / 0.287 |
| penetration | penetration cm | 1.000 / 1.000 | 1.000 / 1.000 | 0.733 / 0.519 |
| jitter | angular jerk p99 | 0.772 / 0.717 | 0.806 / 0.592 | 0.967 / 0.760 |
| pop | angular jerk p99 | 0.722 / 0.296 | 0.472 / 0.128 | 0.833 / 0.644 |
| loop seam | weighted seam | 0.861 / 0.354 | 0.458 / 0.105 | 0.578 / 0.327 |
| speed mismatch | target-speed error | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |

The TCN clearly adds clip-discrimination value for floating, jitter, and pop. Foot slide is mixed:
better AUROC but much worse AP than the deterministic metric. It adds no value for penetration or
speed mismatch and is worse than the seam metric for loop seam.

## Decision after the fixed baseline

1. **Does synthetic supervision work?** Partly. The TCN can memorize the tiny set and obtains
   useful cross-mechanism clip AUROC for foot slide, jitter, pop, and speed mismatch. It is not yet
   a usable critic because localization, ranking, calibration, and shams fail.
2. **Which families generalize across mechanisms?** Jitter is strongest for clip recognition;
   foot slide, pop, and speed mismatch show promising discrimination. None has convincing
   held-out temporal-and-part localization. Loop seam is weak.
3. **Which look like mechanism-fingerprint classifiers?** None is proven by the linear probe,
   which is at/below chance. Loop seam is weak on both known and held-out mechanisms rather than a
   clear fingerprint case. More sources are needed before drawing a stronger conclusion.
4. **Do shams fool the model?** Yes: 5/5 test shams trigger at least one defect.
5. **Does performance collapse on subtle examples?** Yes. Overall pairwise ranking is 0.40, and
   most thresholded near/subtle recalls are zero despite acceptable AUROC in some families.
6. **Does the TCN add value beyond deterministic metrics?** Yes for floating, jitter, and pop;
   mixed for foot slide; no for penetration, seam, or target-speed mismatch.
7. **Which labels/losses were useful?** The clip labels and direct ranking head are learnable on
   the tiny gate. On held-out data the clip objective retains discrimination, while frame/part and
   severity objectives mostly do not generalize. Exact causal attribution is not trained. Loss
   usefulness cannot be isolated without an ablation, so no stronger causal claim is made.
8. **Which corruptors should be redesigned or removed?** Redesign the contact/ground preservation
   of held-out leg-pose speed corruption, improve per-source near-bin reach for loop root velocity,
   and replace/expand the reversible-roundtrip sham family so shams cover mechanisms and sources.
   Keep reporting and rejecting severe root-drift collateral failures. No corruptor is removed on
   this four-window result alone.

The decision is to stop here. Do not build the variable-rig graph/attention critic or Phase 8
repair search until the sham, source-diversity, localization, calibration, and severity-ranking
failures are addressed and the fixed baseline is rerun.

## Artifacts

- `artifacts/fixed_rig_tcn_experiment_v6/real_audit/real_data_audit.json`
- `artifacts/fixed_rig_tcn_experiment_v6/real_audit/dataset/audit.jsonl`
- `artifacts/fixed_rig_tcn_experiment_v6/tiny_overfit_final/tiny_overfit.json`
- `artifacts/fixed_rig_tcn_experiment_v6/training/metrics.jsonl`
- `artifacts/fixed_rig_tcn_experiment_v6/training/best.pt`
- `artifacts/fixed_rig_tcn_experiment_v6/evaluation_v2/evaluation.json`
- `artifacts/fixed_rig_tcn_experiment_v6/evaluation_v2/representations.npz`
- `artifacts/fixed_rig_tcn_experiment_v6/evaluation_v2/failure_examples.json`
