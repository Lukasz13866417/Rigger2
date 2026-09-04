# Noticeability protocol: first review stop

## Current UI: inspection opens only on request

The active pilot is now `artifacts/phase17_opt_in_inspection_pilot/pilot_manifest.json`, at the
same localhost URL. After a NO/MAYBE/YES answer is saved, the animation and inspection panel
remain hidden. Choose **Next animation** to continue immediately, or **Inspect (optional)** to
open replay, camera, speed, localization and the optional inspected judgment. Inspection time
starts at that explicit click, not while deciding whether to inspect. Continuing directly
leaves inspection missing; it does not fabricate a NO response or an empty inspection event.

The three pre-answer plays and 0.85x–1x speeds are unchanged. The primary target remains
`repeated_view_notice`; this is a separately frozen **inspection UI revision**, not a relabeling
of the primary construct. The wrapper implementation and rendered HTML identities are embedded
in the frozen manifest. All previous collectors and their source files remain intact. One
saved primary answer, its inspection amendment and eight served/playback events were preserved
at this cutover. Completed trials from all prior continuations are skipped without copying their
observations into the new log.

Verification: 283 repository tests pass, including six new opt-in browser cases under both
camera versions. Ruff, formatting and mypy (136 package files) pass. The live page returns
HTTP 200 with the exact frozen HTML; prior evidence and copied motion hashes match.

```bash
uv run motionlab prepare-opt-in-inspection \
  artifacts/phase17_repeated_view_pilot/pilot_manifest.json artifacts/phase17_opt_in_inspection_pilot
uv run motionlab freeze-noticeability-pilot \
  artifacts/phase17_opt_in_inspection_pilot/pilot_manifest.json
uv run motionlab serve-noticeability-pilot \
  artifacts/phase17_opt_in_inspection_pilot/pilot_manifest.json
```

## Active update: up to three viewings

The user-requested repeated-view protocol is `motionlab.noticeability.repeated_view.v1`, first served from
`artifacts/phase17_repeated_view_pilot/pilot_manifest.json` at the same localhost URL. Reload the
page and use rater ID `123`.

- Before answering, watch **one to three times**. Answer after the first complete play or click
  **Watch again** for up to two additional plays.
- Select **0.85x, 0.90x, 0.95x or 1x** before each play. Speed selection is disabled during a
  play; 0.5x is not allowed before the primary answer. Camera inspection remains post-answer.
- The server enforces the replay limit and the full wall time at the selected speed. Each
  pre-answer viewing stores its index, speed, content duration, wall duration and timestamps.
- Answers are saved as **`repeated_view_notice`**, with `spontaneous_notice=null`. They are not
  mixed into single-view spontaneous calibration, training exports, or spontaneous-label proxies.
  Analysis and exports explicitly identify this new target. Training remains stopped.
- The original single-view pilot and its frozen collector source files remain unchanged. It was
  paused with **zero saved answers and three serving/playback events**. Its exposed-but-unanswered
  trial can be rated in the new protocol and is flagged as prior exposure. Previously saved
  answers, if any, are skipped using scheduling-only continuation history, never copied as new labels.
- The 68-trial selection, blinding, original render hashes and overlap delay are unchanged.
  Deep optional inspection after saving remains separate; its replay count does not include the
  pre-answer viewings.

The new manifest's `primary_viewing_policy` governs pre-answer playback; the retained historical
`viewing_settings` describe the shared renderer/camera and post-answer inspection. Matching
animation/render hashes do not imply matching speed or presentation conditions. Those conditions
are given by the new protocol and its per-view records.

```bash
uv run motionlab prepare-repeated-view-noticeability \
  artifacts/phase17_noticeability_pilot/pilot_manifest.json artifacts/phase17_repeated_view_pilot
uv run motionlab freeze-noticeability-pilot artifacts/phase17_repeated_view_pilot/pilot_manifest.json
uv run motionlab serve-noticeability-pilot artifacts/phase17_repeated_view_pilot/pilot_manifest.json
```

The remaining sections describe the original single-view milestone and its preserved evidence.

Historical repeated-view verification: 277 tests passed, including 27 noticeability tests;
Ruff and mypy (135 package files) passed. That now-paused collector returned HTTP 200. Original
evidence hashes matched, and its initial analysis/export contained zero human labels.

The primary future learned target is `P(spontaneous_notice = YES)` during one normal-speed
presentation, not an OK-to-good quality ranking. This implementation stops at the human pilot.
It does not train a critic, enable learned production optimization, or start graph/attention work.

## Original single-view pilot (preserved)

Open `http://127.0.0.1:8765` and use the same rater ID, `123`. Reload an old browser tab to get the
new interface. The prepared manifest is
`artifacts/phase17_noticeability_pilot/pilot_manifest.json`.

There are **68 scheduled trials**: 24 exact-render legacy overlaps, 36 ladder samples, four new
clean/sham controls, and four hidden repeats. The new ladders use Neutral and Proud source clips,
six physical strengths each, for arm/leg phase mismatch, rigidity, and torso counter-rotation.
Proud and Neutral retain articulated arm swing; selecting only hands-fixed styles could make
an entire arm-phase ladder uninformative. These are exploratory construction choices, not
claims that a human can or cannot see a given level.

The 24 old presentations are not immediately adjacent to their replacements. A conservative
24-hour delay after the last legacy submission unlocks those trials on **September 5, 2026,
23:16:56 Europe/Warsaw**. The 44 new/control/repeat presentations are available first. Overlap is
restricted to the original rater and marked as prior exposure; it is not a naive-viewer dataset.
Collection can be resumed. An interrupted/exposed but unanswered trial is skipped, not silently
replayed as a first viewing. Consequently, the final answered count can be below 68.

The broader original anchors and preserved exploits remain in the overlap. Across 64 assets,
populations are 48 `NOTICEABILITY_THRESHOLD`, 11 `CLEAN_SHAM_CONTROL`, three `CALIBRATION_ONLY`,
and two `ADVERSARIAL_CRITIC`. `DEPLOYMENT_CANDIDATE` is supported but none is fabricated for this
pilot. Adversarial source aliases are resolved to their original intended walking style before
display; the UI does not reveal exploit/family/severity/old score/population.

## What the person does

1. Read the intended style/goal, then press **Watch once at 1x**.
2. Watch one complete, uninterrupted presentation at the default camera. No replay, pause,
   alternate view, speed, zoom, orbit, pan, or skeleton controls are available yet.
3. The animation is hidden and the person answers:
   “During that normal viewing, did you notice anything that looked unintentionally wrong or
   unnatural?” Choose NO, MAYBE / UNSURE, or YES.
4. That answer is immediately appended and cannot be edited. Only the successful server
   acknowledgement unlocks optional inspection. A second API submission is rejected.
5. Optionally replay, change speed/camera, and answer the separate inspectability question.
   Tags, body part, interval and confidence are optional. YES never requires a diagnosis.
6. **Save optional inspection and next** works without any optional judgment or localization.

Tab blur/visibility loss or a substantial initial playback stall invalidates that presentation.
No human label is invented. Reloading after a saved primary answer does not ask it again; an
optional inspection amendment may remain missing. The service binds only to loopback, rejects
cross-origin writes, serializes submissions, and permits only one live browser session per rater.

Both original mannequin camera versions remain distinct. The original renderer functions,
960-by-640 backing canvas, viewport geometry, projection, two-cycle window, mirror convention,
default camera and render hashes are preserved. Old v1 overlaps retain their old inspection
camera limits; new v2 samples retain wheel zoom, panning and free orbit. The **question/protocol
changes**, deliberately; matching animation hashes do not make the two psychological constructs
interchangeable. Browser/window variation was not fully measured in the historical protocol,
so identical backing-canvas hashes are not a claim of identical physical viewing conditions.

## Evidence and preservation

`motionlab.noticeability.v1` has two append-only event types:

- `spontaneous_notice`: DIRECT_HUMAN, frozen trial/render/source/variant/context identity,
  rater/session/trial index, first-response time, full initial-presentation duration, exposure
  flag and timestamp. Inspection, localization and replay fields are empty at this stage.
- `inspection`: a separate DIRECT_HUMAN amendment referencing exactly one primary observation.
  It contains optional `inspected_notice`, bounded replay/speed/camera telemetry, time spent at
  each speed, total inspection time, and optional diagnostic fields.

Analyses join these in memory; neither event is rewritten. Each event has a content hash and
is checked against the frozen manifest, exact assets, served/start/completed evidence and event
ordering. No repeated observation is averaged on write. Code changes to the collector invalidate
its freeze and require a separately versioned/frozen collection rather than mutating a live pilot.

The completed Phase 15 session has **30 preserved observations**, not 43: the continuation's
13 scheduling-history rows are not new observations. Its exact-byte snapshot is
`artifacts/phase17_legacy_quality_freeze/`. Original logs, manifests, timestamps, telemetry,
render hashes and questions/scales remain unchanged. Compatibility records reference their
original observation IDs and use the alias `perceptual_quality_v1`; original v3/v4 identities
remain available.

| Legacy ordinal score | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Single-animation observations | 0 | 0 | 10 | 3 | 8 | 2 | 1 |

The six remaining rows are pairwise quality judgments, never detection labels. All 24 eligible
single-animation render/rater identities are in the overlap: 13 original-camera and 11
free-camera observations. Selection retains score/source/style/family/confidence/difficulty/
disagreement/control/render strata. There are fewer than 30 eligible old singles; extra labels
are not fabricated to reach the requested approximate overlap size.

The older, separate Phase 14 skeleton archive also remains untouched: four observations under
its original render protocol and different rater. It is excluded from this exact-mannequin
overlap and is not counted as four additional completed-session judgments.

## Calibration and reversible migration

The regularized cumulative-logit model maps an old score to probabilities for NO/MAYBE/YES.
Its score slope is constrained nonnegative for NO (nonpositive for YES), with learned ordered
thresholds. There are no hardcoded score cutoffs. Fits are separated by render version, evaluated
by leaving out an entire original source, and compared with a training-only smoothed-prior
baseline. Reports retain confusion matrices, log loss, multiclass Brier, category curves,
source/style/family breakdowns and source-cluster bootstrap uncertainty.

Weak proxies require, **within the matching render stratum**, at least 30 distinct overlap
observations, four sources, three examples of every response class, and greater than 5%
held-source improvement over the prior baseline in both log loss and Brier. A rating category
also needs its own support. This conservative initial policy means the current 24-item overlap
can assess the relationship but cannot authorize proxy training by itself. Failure to pass this
gate is not proof that old quality judgments contain no useful information.

Derived proxy files contain exact source observation ID, migration version/hash, three
probabilities, uncertainty, source ordinal score, `derived=true`, `WEAK_PROXY` and lower loss
weight. They never replace raw rows. Recomputing the same model is idempotent; a changed model
requires a new output artifact. Deleting only a derived proxy file is reversible. Pairwise/style
ratings cannot enter this conversion. For matching render/rater identity, DIRECT_HUMAN always
outranks a proxy. The first export is deliberately direct-only, even if proxies later qualify.

## Analysis and next-stage model contracts

The human report contains raw response counts/rates and intervals, clean/sham false alarms,
intended-obvious NO misses with MAYBE separate, repeat agreement, spontaneous/inspected
transitions, exposure strata, and per-family summaries. Intended-obvious controls are hypotheses
to check, not human ground truth. The modest latent model uses stimulus detectability and shared
ordinal thresholds; a regularized session effect is included only with enough cross-session
repeat support. Its approximate uncertainty is explicitly marked exploratory.

Psychometric threshold/JND estimates are withheld without repeated, bracketing data. The
binomial psychometric curve estimates the event YES versus non-YES; raw MAYBE remains a distinct
response, never a direct NO label. A single six-rung screen is not sufficient for a confident
threshold. Optional interleaved staircases balance family/source tracks, move down after YES,
up after NO, and hold after MAYBE; every fifth trial is a clean/sham control. They select only
frozen permitted candidates and skip interrupted trials without labels. A separate fixed
validation design is generated only after supported threshold estimates exist. That plan still
requires rendering a fresh fixed batch; it is not silently inserted into an active playlist.

The new `FixedRigNoticeability` keeps the GroupNorm TCN and concatenates explicit declared
style/goal context to its pooled clip representation before a 64-unit hidden layer and three
logits. `p_notice_spontaneous` is the YES probability. Human supervision is clip/event-level;
the loss rejects per-frame target replication. Exports retain the exact displayed cycle window,
mirror setting and physical measurements separately from human responses. Source-group splits
must keep all repetitions/renders/derived variants together. Context vocabulary must be fixed
and supplied identically at training and inference; unseen-style generalization is not claimed.

Detection utility functions provide YES-versus-NO AUROC/AP (MAYBE excluded explicitly),
three-class Brier/log loss, calibration bins, and operating-point tradeoffs. Held-out validation
is required for candidate tau selection, with explicit false-alarm/recall requirements; no
default 0.5 is supplied. Reviewed production behavior is deterministic feasibility **and**
`p_notice <= tau`, with penalty `max(0, p_notice - tau)`. A separate aggressive red-team
objective does not change production behavior. Prior adversarial examples, absolute-q and
relative-comparator ablations, GroupNorm baseline and the LayerNorm experimental artifact are
all retained. No learned objective is activated by this implementation.

## Commands

```bash
uv run motionlab freeze-human-protocol artifacts/phase17_legacy_quality_freeze \
  artifacts/phase15_mannequin_pilot/pilot_manifest.json \
  artifacts/phase15_free_camera_pilot/pilot_manifest.json
uv run motionlab build-noticeability-threshold-pilot \
  artifacts/phase17_legacy_quality_freeze/legacy_registry.json artifacts/phase17_noticeability_pilot
uv run motionlab freeze-noticeability-pilot artifacts/phase17_noticeability_pilot/pilot_manifest.json
uv run motionlab serve-noticeability-pilot artifacts/phase17_noticeability_pilot/pilot_manifest.json

# After human collection; use fresh versioned output filenames for subsequent analyses.
uv run motionlab analyze-noticeability \
  artifacts/phase17_noticeability_pilot/pilot_manifest.json artifacts/phase17_analysis_after_labels.json
uv run motionlab fit-legacy-noticeability-calibration \
  artifacts/phase17_legacy_quality_freeze/legacy_registry.json \
  artifacts/phase17_noticeability_pilot/pilot_manifest.json artifacts/phase17_calibration_after_labels.json
uv run motionlab migrate-legacy-labels \
  artifacts/phase17_legacy_quality_freeze/legacy_registry.json \
  artifacts/phase17_calibration_after_labels.json artifacts/phase17_proxies_after_labels.json
uv run motionlab validate-label-migration \
  artifacts/phase17_legacy_quality_freeze/legacy_registry.json \
  artifacts/phase17_calibration_after_labels.json artifacts/phase17_proxies_after_labels.json
uv run motionlab export-noticeability-training \
  artifacts/phase17_noticeability_pilot/pilot_manifest.json artifacts/phase17_direct_export.json

# Optional separately reviewed follow-up, not part of the active fixed screen.
uv run motionlab build-noticeability-staircase-pilot \
  artifacts/phase17_noticeability_pilot/pilot_manifest.json artifacts/phase17_staircases --steps 30
uv run motionlab freeze-noticeability-pilot artifacts/phase17_staircases/pilot_manifest.json
uv run motionlab build-noticeability-validation-plan \
  artifacts/phase17_staircases/pilot_manifest.json artifacts/phase17_fixed_validation_plan.json
```

`build-noticeability-overlap` prepares an overlap-only pilot if needed. Builders refuse occupied
output directories; existing frozen pilots should be served/resumed, not rebuilt. The
`train-noticeability-critic` and `evaluate-noticeability-critic` commands are explicit stop guards
at this milestone, **not completed training/evaluation orchestration**. Their future implementation
requires review of the direct-human pilot and an approved split/evaluation plan. No checkpoint,
held-out-source/mechanism critic result or deployed operating threshold exists yet.

## Required first-stop report

The pre-collection analysis/calibration/migration artifacts report honestly:

- Preserved completed-session observations: 30; old score distribution as above.
- Selected overlap: 24; completed noticeability re-labels: 0.
- Calibration curve/usefulness: awaiting direct overlap evidence; weak proxies: 0.
- Clean/sham false alarms, obvious misses, repeat reliability, thresholds/JND, NO/MAYBE/YES
  rates and spontaneous-versus-inspected disagreement: **not estimable before human collection**.
- Recommended next step: complete the small screen, review false alarms and per-family
  crossings, then choose a focused staircase/fixed-validation follow-up. Stop before retraining.

Browser regression tests run on isolated generated fixtures with both original camera versions.
Dummy statistical tests verify mechanics, not perceptual quality or real model accuracy.

Verification: 266 repository tests pass, including 16 noticeability cases; repository-wide Ruff
and mypy (131 package files) pass. The live loopback endpoint returns HTTP 200, all 69 frozen
pilot assets validate, and both the completed-session snapshots and the separate Phase 14 raw
checksum remain unchanged. The machine-readable initial report is
`artifacts/phase17_noticeability_precollection/first_stop_report.json`.
