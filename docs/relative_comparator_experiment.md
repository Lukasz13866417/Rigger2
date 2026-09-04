# Fixed-rig relative-comparator experiment

## Current decision

The relative-comparator implementation is ready, but the experiment has not been trained. The
required human audit contains zero of 73 judgments, so training either ablation would reuse
unverified synthetic orderings and violate the supervision gate. No perceptual CMA-ES evaluation
has been invoked, and graph/attention remains stopped.

The earlier absolute residual critic remains preserved as the `absolute_q` ablation with 41.7%
directional held-out-source accuracy.

## Human-audit queue

The audit queue contains all 71 derivatives of the three completely held-out source/styles plus
both hard-feasible historical adversarial pairs. The five most confident absolute-critic
misrankings occupy the first five positions. This produces 73 requested judgments—five mandatory
misrankings plus 68 additional pairs.

| Coverage | Count |
|---|---:|
| HandsInPockets held-out style | 24 |
| Heavyset held-out style | 24 |
| Proud held-out style | 23 |
| Hard-feasible adversarial | 2 |
| Subtle | 29 |
| Medium | 18 |
| Clear | 20 |
| Exact-equivalence controls | 6 |

Every perturbation family is represented. Family counts are six asymmetric-timing, six exact
origin-equivalence, twelve rigidification, twelve smoothing, six style-inconsistency, two
preserved exploits, twelve torso counter-rotation, twelve upper-body phase, and five
weight-transfer proxy pairs.

Human labels remain append-only and now include an optional 1–5 confidence value. Analysis reports
human/synthetic agreement and a 95% Wilson interval per family. Until the full audit is complete,
all directional automatic labels are suspended as `HUMAN_REQUIRED`; exact world-origin
equivalence remains `CERTAIN`. After completion, a directional family needs at least 70% measured
human agreement to regain automatic `CERTAIN` supervision. Subjective UNORDERED families continue
to require human judgments.

Current agreement by family is not estimable because no human judgments exist. Null values are
persisted rather than treating the old CERTAIN label as human truth.

## Comparator

Both variants use the existing fixed-rig GroupNorm TCN as the shared temporal encoder and expose
its `[batch, time, hidden]` representation. For aligned encodings `EA` and `EB`, one shared ordered
head receives:

```text
[EA, EB, EB-EA, abs(EB-EA), EA*EB]
```

It evaluates both `(A,B)` and `(B,A)` and antisymmetrizes their directional evidence. A separate
symmetric head receives the mean, absolute difference, and product. The final softmax order is:

```text
P(A better), P(B better), P(approximately equal)
```

This construction guarantees that swapping inputs exchanges A/B probabilities while leaving the
equal probability unchanged. The explicit swap test passes at `1e-7` tolerance.

The matched variants are:

- `relative_comparator_frozen`: freeze every existing hard-defect TCN parameter;
- `relative_comparator_finetuned`: fine-tune only the TCN input projection and temporal residual
  blocks, using `2e-5` versus `2e-3` for the comparison head.

Both start from the same checkpoint and seed. Normalization is unchanged. Evaluation is already
implemented separately for held-out source, held-out mechanism, human labels, audited synthetic
CERTAIN labels, equivalence controls, adversarial pairs, swap consistency, perturbation family,
human confidence, and human reason tag. Each accuracy includes a 95% Wilson interval. No aggregate
score hides those categories.

Frozen-versus-fine-tuned and held-out-source results are currently `not_run`, because the audit is
a hard prerequisite—not because either implementation failed to execute.

## MotionCritic baseline assessment

The public MotionCritic checkpoint expects 60-frame tensors with 24 SMPL local axis-angle joints
plus root XYZ. The current data has 23 fixed-rig joints with different rest frames. A defensible
comparison would require an explicit retarget to the exact SMPL hierarchy/rest orientations and
the separately downloaded SMPL body assets; passing renamed or zero-filled joints would corrupt
motion semantics. The upstream environment also targets an older Python/PyTorch stack.

The baseline is therefore deferred until human-labeled pairs exist and the necessary licensed
SMPL assets/mapping are explicitly introduced. Its pairwise accuracy is currently null. This does
not affect the higher-priority relative-comparator experiment.

Upstream references: <https://github.com/ou524u/MotionCritic> and
<https://motioncritic.github.io/>.

## Anchor-relative CMA

The gated optimizer does not require an absolute `q(x)`. Within each trust-region round it freezes
the current animation as anchor and minimizes the negative logit of
`P(candidate better than anchor)`, subject to the existing deterministic feasibility checks. An
accepted candidate becomes the next anchor. The code makes no global transitivity claim.

An optional generation-local Bradley–Terry fit is recorded but remains disabled until an actual
qualified run shows anchor-only comparisons to be too coarse. The current structured CMA report
has `cma_es_invoked=false` because the human-audit/comparator gate has not passed.

## Continue the audit

```bash
motionlab label-perceptual-pairs artifacts/phase12_perceptual_quality/dataset \
  --labels artifacts/phase13_relative_comparator/human_audit/human_labels.jsonl \
  --evaluation artifacts/phase12_perceptual_quality/critic/evaluation.json \
  --audit-queue artifacts/phase13_relative_comparator/human_audit/human_audit_queue.jsonl
```

After labeling, rerun `analyze-perceptual-human-audit`, then
`run-relative-comparator-experiment`. Only a comparator reaching at least 65% held-out-source
accuracy, with its confidence interval reported and equivalence/swap gates intact, can enable
`run-relative-perceptual-cma`.
