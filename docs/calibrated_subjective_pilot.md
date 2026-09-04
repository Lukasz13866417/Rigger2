# Revised mannequin subjective pilot

## Status and collection boundary

The Phase 14 skeleton-rendered pilot is paused and must not receive more labels. Its four existing
observations are too few to support a renderer comparison, critic training, or a scientific claim
about perceptual measurement. They remain immutable legacy evidence:

| Preserved Phase 14 file | Rows | SHA-256 |
| --- | ---: | --- |
| `pilot_manifest.json` | - | `09828d73de51d79fe4eac029b8279938e711753a799d4afd28a712c64001c2fd` |
| `raw_observations.jsonl` | 4 | `3008ca90bcad10d7ddc84c0fc6fd8549be171155fa52a02ae18a4a0ee26dc997` |
| `served_playlist.jsonl` | 5 | `834e0aea86f54487ec969221dc7c6793343416565793761ee62ffa5cc87edd06` |

`artifacts/phase14_subjective_pilot/PILOT_PAUSED.json` records this boundary. The label server
checks for that sentinel at startup and on every request. A refreshed, read-only analysis of all
four legacy observations is stored under
`artifacts/phase15_mannequin_pilot/legacy_phase14_baseline_analysis`; the older Phase 14 analysis
directory is deliberately not rewritten.

The Phase 15 mannequin pilot has entered human collection. Its zero-observation analysis is the
original pre-collection snapshot, not the live collection status. Critic retraining remains
blocked; no interim conclusions are drawn from the current labels.

The user-requested camera change is a separately versioned continuation at
`artifacts/phase15_free_camera_pilot`. Original Phase 15 manifests, motions, logs, and freeze remain
unchanged. The original v3 collection implementation is preserved in `subjective_v3.py`.

The continuation uses collection protocol v4 and mannequin rendering protocol v2. Submitted
progress is authenticated and mapped into a frozen, scheduling-only history; it is never copied
into the new observation log as if those motions had been rated with the new controls. Resume
with the same participant ID and session. An unsubmitted trial starts again; submitted trials
are skipped. Existing raters who reached scored session-1 trials need not repeat the tutorial.
Analyze evidence separately by camera protocol; repeats spanning the change are not exact-render
reliability measurements. Combining completion or reliability across protocols requires an
explicit protocol-aware analysis, not concatenation of the logs.

At cutover, 13 submitted scores and 21 served events were retained in the old pilot; all 129 motion
files were copied with identical hashes. The old server is paused via `PILOT_PAUSED.json`, and the
free-camera continuation is served at `http://127.0.0.1:8765`.

## Renderer and inspection protocol

The versioned `motionlab.fixed_mannequin_canvas.v1` renderer replaces bare line segments with a
neutral articulated mannequin. It uses oriented anatomical solids, fixed neutral materials and
lighting, directional feet, hands, and head, a projected ground grid, and global joint rotations.
The old skeleton is available only as an optional diagnostic overlay. The fixed settings hash is
`render:sha256:cea7579f211ffb88143ee439f1de9fbdca3af6e4394d9b86a9dd78c69cec1f4b`;
the legacy renderer and its exact hash remain supported for old observations.

Every scored trial first requires one complete playback at 1x using the default three-quarter
camera and 1.0 zoom. The server validates that first pass. This lock never applies to calibration
tutorials: their severity, speed, camera, zoom, replay, and Continue controls are available
immediately. After a scored first pass the rater may:

- choose 0.25x, 0.5x, 1x, 1.5x, or 2x playback;
- select front, side, or three-quarter views and orbit within the bounded pitch range;
- zoom from 0.65x to 1.8x, reset the view, or enable the skeleton overlay;
- replay and inspect before committing an answer.

The observation stores first-pass acknowledgement, playback-rate dwell and changes, replay count,
camera presets and orbit motion, zoom changes and range, overlay use, A/B toggles, and total
inspection time. Once a rating is confirmed, playback and all inspection controls freeze while the
rater answers the separate post-rating difficulty question.

Same-viewport A/B trials share the exact camera, zoom, speed, playback time, and phase across both
versions. Toggling never resets the view or advances one version independently. Public payloads
remain blinded: defect family, mechanism, population, expected quality, anchor status, and training
eligibility stay server-side.

### Free-camera continuation

The v2 renderer retains the same geometry, motions, lighting, scales, and standardized first pass.
After that first pass (immediately in tutorials):

- Scroll over the viewport to zoom, or use the zoom buttons. The range is now 0.1x–5x.
- Left-drag to orbit freely around the character; vertical rotation extends to ±89 degrees.
- Shift-left-drag, middle-drag, or right-drag to pan in screen space.
- Reset camera restores the original framing, including pan, zoom, rotation, and overlay.

Pan and zoom remain shared between A and B. Wheel gestures are coalesced in inspection telemetry;
camera state includes bounded, finite horizontal/vertical pan offsets. The first-pass and
post-rating locks also cover these new controls.

## Calibration bank and purpose separation

The independent calibration bank contains 13 explicit five-rung ladders (clean, mild, medium,
strong, severe), for 65 stimuli total:

- measured deterministic families: foot slide, floating contact, joint jitter, joint pop, stride
  pose amplitude, and ground penetration;
- perceptual or preset families: upper-body phase mismatch, excessive rigidification, excessive
  smoothing, torso counter-rotation mismatch, asymmetric limb timing, reduced pelvis bounce, and
  loop seam.

Every non-clean rung changes rendered tracked positions or global rotations; every clean rung has
exactly zero render delta. The bank is therefore a representability check, not a source of critic
labels. The pilot shows three complete ladders as a disclosed, unscored tutorial before collection.
Tutorial content is disjoint from scored content by stimulus ID, source window, visual-content hash,
and rendered-content hash.

All pilot material has one of three explicit populations:

- `CALIBRATION_ONLY`: disclosed tutorials; never scored and never training-eligible;
- `HARD_FEASIBLE_PERCEPTUAL`: hard, constraint-feasible scored stimuli; the only population that
  can become critic-training evidence;
- `EVALUATION_ANCHOR`: broad/easy measurement anchors; scored for reliability and scale coverage,
  but never training-eligible.

The analysis filter fails closed: an observation is exportable only when it explicitly declares
training eligibility and every referenced stimulus is `HARD_FEASIBLE_PERCEPTUAL`. Legacy,
unclassified, calibration, and anchor observations remain excluded even if they contain a rating.

## Phase 15 schedule

Seed 9502 prepares 40 unique scored stimuli: 32 hard-feasible perceptual items and eight evaluation
anchors. Three tutorial ladders add 15 calibration-only assets, for 55 pilot assets total. The two
sessions each contain 24 scheduled scored presentations.

Four hard-feasible items and four anchors have hidden cross-session repeats. The realized minimum
global index distance is 15 trials (14 intervening trials), exceeding the declared minimum of 12.
The response-adaptive pool contains 12 same-viewport toggle comparisons, split six per session and
capped at 12 total. Adaptive comparisons may draw only from the hard-feasible population and must
match source and editing goal. They prioritize close ratings, within-rater disagreement,
model-human errors, and underrepresented source/family cells; the served-order log records the
realized selection.

## Reports and interpretation

`analyze-subjective-pilot` preserves the ordinal and four-outcome pair evidence while reporting:

- rating/category use, response time, confidence, inspection telemetry, and post-rating difficulty;
- repeat agreement separately for hard-feasible items and evaluation anchors;
- anchor drift, source/family coverage, A/B order behavior, ties, and not-sure outcomes;
- exploratory family perceptibility from supported scored evidence, separate from severity-ladder
  ordering accuracy;
- the strict training-eligibility audit and excluded counts by population.

Tutorial ladders are unscored and cannot establish family perceptibility by themselves. The old and
new pilots also do not form a matched renderer-only experiment: the renderer, controls, tutorial,
and schedule all changed, and Phase 14 has only four observations. Phase 15 can evaluate whether the
new protocol yields usable measurements, but it cannot by itself support a causal claim that the
mannequin renderer improved ratings over the old skeleton UI.

Inspect `pilot_report.json`, `reliability_dashboard.json`, and
`human_synthetic_agreement.json` before any critic retraining. A completed pilot is evidence for a
training decision, not automatic permission to retrain.

## Commands

Serve the free-camera continuation:

```bash
uv run motionlab label-perceptual-pairs \
  artifacts/phase12_perceptual_quality/dataset \
  --pilot artifacts/phase15_free_camera_pilot/pilot_manifest.json \
  --observations artifacts/phase15_free_camera_pilot/raw_observations.jsonl \
  --served-playlist artifacts/phase15_free_camera_pilot/served_playlist.jsonl
```

Only after the user declares collection sufficient and the completion gate passes, generate the
reports without retraining. Do not rerun a pilot/calibration builder or change the live collection
protocol without a new explicit request. For the authorized camera cutover, stop the old server
before snapshotting progress, then prepare a fresh sibling artifact and freeze:

```bash
uv run motionlab prepare-camera-continuation \
  --pilot artifacts/phase15_mannequin_pilot/pilot_manifest.json \
  --output <fresh-free-camera-pilot-directory>
```

The command refuses to overwrite existing artifacts. The original protocol can still be
validated/analyzed independently when collection is declared sufficient:

```bash
uv run motionlab analyze-subjective-pilot \
  artifacts/phase12_perceptual_quality/dataset \
  --pilot artifacts/phase15_mannequin_pilot/pilot_manifest.json \
  --observations artifacts/phase15_mannequin_pilot/raw_observations.jsonl \
  --protocol-freeze artifacts/phase15_mannequin_pilot/protocol_freeze.json \
  --served-playlist artifacts/phase15_mannequin_pilot/served_playlist.jsonl \
  --output <fresh-human-analysis-directory-outside-active-pilot>
```

For a future, separately versioned fresh artifact, build the calibration bank first and pass it to the separate mannequin pilot
builder. `prepare-subjective-pilot` intentionally remains the frozen legacy skeleton builder and
rejects mannequin settings, preventing an old schedule from masquerading as the revised protocol.

```bash
uv run motionlab prepare-subjective-calibration-bank \
  artifacts/phase8_critic_hardening/dataset \
  --output <fresh-calibration-bank-directory> --seed <new-seed>

uv run motionlab prepare-mannequin-subjective-pilot \
  artifacts/phase12_perceptual_quality/dataset \
  --audit-queue artifacts/phase13_relative_comparator/human_audit/human_audit_queue.jsonl \
  --corruption-dataset artifacts/phase8_critic_hardening/dataset \
  --calibration-bank <fresh-calibration-bank-directory>/calibration_bank.json \
  --output <fresh-versioned-pilot-directory> --seed <new-seed> --comparisons 12
```

## Automated browser verification

The end-to-end test starts the evaluator on an unused localhost port and writes only to pytest's
temporary directory. It covers unrestricted tutorial inspection, the scored mandatory first pass,
all inspection controls,
telemetry, rating freeze, post-rating difficulty, single-stimulus and shared-state toggle paths,
blinding, append-only persistence, and session completion.

```bash
MOTIONLAB_REQUIRE_E2E_BROWSER=1 uv run pytest -q tests/test_subjective_ui.py tests/test_free_camera.py
```

It uses an installed Brave, Chrome, or Chromium executable, falling back to Playwright Chromium.
`MOTIONLAB_E2E_BROWSER` can select an explicit executable. No browser extension is required: the
human evaluator is an ordinary localhost web page.

Artifact-backed browser tests skip explicitly in a fresh Git checkout because downloaded motions
and local pilot artifacts are not published. With those artifacts present,
`MOTIONLAB_REQUIRE_E2E_BROWSER=1` makes a missing browser a failure rather than a skip.
