# Corruption supervision and catalog

The corruption pipeline produces immutable `CorruptionResult` objects. Requested generator
strength is retained as metadata, but never treated as observed defect severity. Each candidate
hard negative is measured after generation and rejected by default unless its family-specific
metric worsens by the declared minimum.

## Observed-severity targeting

Hard samples are generated into five fixed labels—`near_threshold`, `subtle`, `moderate`, `clear`,
and `severe`—whose numeric bounds belong to a particular family and physical metric. The factory
uses deterministic bounded parameter search, remeasures each candidate, and accepts it only when
the resulting metric delta is inside the requested bin. The selected corruptor parameter remains
audit metadata; it is not the persisted symptom severity and is never a neural input.

Ordinal preferences are emitted only when all five independently targeted bins are populated and
their measured outcomes pass `verify_ordinal_chain`. Soft/perceptual edits retain explicit
non-ordinal parameter presets and never acquire an automatic clean-is-better target.

## Dense supervision contract

Every result stores:

- clean and corrupted motions plus source motion ID, seed, and split-lineage ID;
- corruption family and concrete mechanism;
- `intervention_mask[T,J,6]`, the exact generated root-translation/local-rotation channel edits;
- `symptom_mask[T,8,13]`, the measured time/functional-part defect locations;
- `responsibility_target[T,8,13]` and a separate confidence tensor for reasonable editor actions;
- before/after physical metrics, measured delta, postcondition, and preference confidence;
- requested severity and generator parameters as metadata only.

The functional parts are root/pelvis, trunk/spine, head/neck, left/right arm, left/right leg, and
other/accessory. The defect vocabulary is foot slide, ground penetration, floating contact, loop
seam, joint limit, joint pop, joint jitter, speed inconsistency, limb phase mismatch, cadence
inconsistency, pelvis curve, foot clearance, and style incoherence.

The tensors are intentionally different. Root-drift foot slide, for example, has a root
translation intervention, a planted-leg symptom, and editing responsibility on both the leg and
root/pelvis. Exact fine-joint causal blame is not asserted.

## Contact supervision

`ContactSupervision` keeps source-reference contacts separate from estimates available at
inference on the clean and corrupted clips. When callers provide authored or synthetic source
contacts, the source is recorded as `provided_source_truth:*`. Otherwise the clean motion's
detector output is labeled `clean_inference_proxy:*`; it is not presented as authored truth.
Contact confidence and ground confidence remain separate.

Source-reference contacts are privileged label fields. Artifact manifests list them in
`privileged_dense_fields` and `neural_input_exclusions`. Dataset features re-detect contact on the
motion being encoded and never expose the privileged reference to the model.

Walking corruptions persist a clean-source `gait_target` with half-open frame bounds, side,
stance/swing or loop-seam context, heel-strike/toe-off references, double support where present,
and covered phase octants. This metadata is privileged supervision/audit context. The encoded
motion's contact and phase features are still recomputed independently.

## Single-defect collateral validation

Every automatic hard candidate persists a policy declaring its primary metric, required primary
delta, collateral metrics, allowed deltas, and whether an excess must be rejected or additionally
labeled. The current common screen covers foot sliding, penetration, floating contact, global
angular smoothness, loop seam, authored joint limits, and root-speed change. A disallowed excess is
rejected; an unavoidable declared secondary symptom, such as slide caused by root-only speed
scaling, is added to the dense symptom/responsibility targets and the sample is marked non-single.
No significant detected secondary defect is silently saved under only its primary label.

## Matched sham controls

For every accepted root-drift foot-slide severity, the factory attempts a matched control with the
same root trajectory edit and contact-aware two-bone IK compensation. It is saved only if the root
intervention still matches and added trusted-stance slip is at most 0.2 cm. Sham samples have zero
defect masks and explicit `no_target_defect=true`, `approximately_equal_quality=true` targets, plus
a link to their matched corrupted sample. They share the source window's counterfactual group.

## Versioned mechanism catalog

| Family | Mechanism | Dataset role | Preference | Measured postcondition |
| --- | --- | --- | --- | --- |
| foot slide | root drift | train | hard | stance slip increases |
| foot slide | hip rotation drift | held-out | hard | stance slip increases |
| ground penetration | root vertical offset | train | hard | maximum penetration increases |
| floating contact | root vertical offset | train | hard | maximum contact height increases |
| joint pop | local rotation pulse | train | hard | localized peak angular jerk increases |
| joint pop | smooth rotation pulse | held-out | hard | localized peak angular jerk increases |
| joint jitter | white tangent noise | train | hard | localized angular-jerk p95 increases |
| joint jitter | band-limited correlated | held-out | hard | localized angular-jerk p95 increases |
| limb phase mismatch | unilateral circular shift | train | hard | source-relative arm phase deviation increases |
| limb phase mismatch | bilateral circular shift | held-out | hard | source-relative arm phase deviation increases |
| cadence inconsistency | pose-stream monotonic warp | train | hard | source-relative pose timing deviation increases |
| cadence inconsistency | root-stream monotonic warp | held-out | hard | source-relative root timing deviation increases |
| speed inconsistency | root progression scale | train | hard | target-speed error increases |
| speed inconsistency | leg pose amplitude scale | held-out | hard | source-relative stride-pose deviation increases |
| loop seam | end rotation offset | train | hard | weighted seam error increases |
| loop seam | root velocity mismatch | held-out | hard | weighted seam error increases |
| pelvis curve | vertical amplitude scale | train | soft | source-relative vertical curve deviation increases |
| foot clearance | swing-ankle IK lowering | train | hard | source-relative clearance loss increases |
| joint limit | authored limit excess | train | hard | configured joint-limit excess increases |
| style incoherence | aligned region blend | train/manual donor | soft | source-relative region pose deviation increases |

Held-out mechanisms are written only to `samples/heldout_corruptor`; dataset validation rejects
them if they appear in training. Pelvis and style edits are soft examples: they expose localization
targets but do not automatically claim that the clean motion is preferable. Style blending needs a
phase-aligned donor and is therefore part of the callable catalog, not automatic one-source
generation. Joint-limit generation similarly rejects rigs with no valid authored limits.

`verify_ordinal_chain` accepts a near-threshold-to-severe sequence only when all results share one source,
family, mechanism, baseline, and metric, all are validated hard negatives, and measured values are
strictly increasing. The dataset builder emits only adjacent
clean/near-threshold/subtle/moderate/clear/severe preference edges from such a verified chain.

## Bounded composition

`compose_corruptions` accepts two or three training-partition hard-negative operators. It unions
their intervention and symptom masks, combines responsibility targets conservatively, records all
constituents, and remeasures every constituent metric on the final clip. The composition is
rejected if a later operator erases an earlier defect. Nested composites, held-out mechanisms, and
soft corruptions are rejected.

## Corruption artifact format

`save_corruption_artifact(directory, result)` writes:

```text
manifest.json
clean_motion.npz
corrupted_motion.npz
labels.npz
```

All NPZ files load with `allow_pickle=False`. The manifest declares every dense array's shape and
dtype, SHA-256 checksums, motion-ID links, source split lineage, and neural-input exclusions. By
default, writing fails if lineage is incomplete or the measured postcondition is not a hard
negative. Pass `require_hard_negative=False` only when intentionally persisting a declared soft
example.

The CLI can inspect the catalog and emit an individual artifact:

```bash
motionlab corruption-catalog
motionlab corrupt clean.npz --output corrupted.npz --artifact-dir example/
```

For fixed-window sample generation, normalization, manifests, and validation, see
`data_format.md`.
