"""Post-judgment motion diagnostics kept outside the blinded rating application."""

# ruff: noqa: E501

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from motionlab.critic.forensics import motion_deterministic_metrics
from motionlab.dataset.features import motion_feature_arrays
from motionlab.io.npz import load_motion_npz
from motionlab.kinematics.fk import forward_kinematics_numpy, marker_world_positions
from motionlab.metrics.smoothness import smoothness_metric
from motionlab.perceptual.observation_auth import (
    resolve_frozen_evidence_paths,
    validate_current_protocol_observations,
)
from motionlab.perceptual.subjective import (
    MANNEQUIN_VIEWING_SETTINGS,
    PAIR_OUTCOMES,
    RAW_OBSERVATION_VERSION,
    _two_cycle_payload,
)
from motionlab.processing.contacts import FOOT_MARKER_NAMES, detect_foot_contacts

POST_LABEL_FORENSICS_VERSION = "motionlab.post_label_motion_forensics.v2"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _resolve_stimuli(manifest_path: Path) -> tuple[Path, dict[str, dict[str, Any]]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    raw_root = manifest.get("stimulus_directory")
    root = Path(raw_root).resolve() if raw_root else Path(manifest_path).parent.resolve()
    stimuli = {
        str(item["stimulus_id"]): item
        for item in manifest.get("stimuli", [])
        if item.get("stimulus_id")
    }
    if not stimuli:
        raise ValueError("forensics requires a subjective pilot manifest with stimuli")
    return root, stimuli


def _completed_response(row: Mapping[str, Any]) -> bool:
    task_type = row.get("task_type")
    if task_type == "pair_comparison":
        return (
            len(row.get("stimulus_ids", [])) == 2
            and row.get("rating") is None
            and row.get("pair_outcome_display_order") in PAIR_OUTCOMES
            and row.get("pair_outcome_canonical")
            in {"first_better", "second_better", "effectively_equal", "not_sure"}
        )
    rating = row.get("rating")
    return (
        task_type in {"naturalness", "style_adherence"}
        and isinstance(rating, int)
        and not isinstance(rating, bool)
        and 1 <= rating <= 7
        and row.get("pair_outcome_display_order") is None
        and row.get("pair_outcome_canonical") is None
    )


def _served_key(row: Mapping[str, Any], *, observation: bool) -> tuple[Any, ...]:
    return (
        row.get("rater_id"),
        row.get("session_id"),
        row.get("session_index"),
        row.get("trial_id"),
        row.get("trial_index"),
        row.get("playlist_seed" if observation else "seed"),
        row.get("task_type"),
        tuple(row.get("presentation_order", [])),
        row.get("schedule_reason"),
    )


def _assert_already_rated(
    requested: tuple[str, ...],
    observations: list[dict[str, Any]],
    *,
    manifest: Mapping[str, Any],
    served_playlist: list[dict[str, Any]],
) -> list[str]:
    requested_set = set(requested)
    stimulus_meta = {
        str(item["stimulus_id"]): item
        for item in manifest.get("stimuli", [])
        if isinstance(item, dict) and item.get("stimulus_id")
    }
    served = {_served_key(row, observation=False) for row in served_playlist}
    matches: list[str] = []
    for row in observations:
        observed = {str(value) for value in row.get("stimulus_ids", [])}
        if observed != requested_set:
            continue
        observation_id = row.get("observation_id")
        if (
            row.get("format_version") != RAW_OBSERVATION_VERSION
            or row.get("render_protocol_hash") != manifest.get("render_protocol_hash")
            or not isinstance(observation_id, str)
            or not observation_id.startswith("observation:sha256:")
            or not _completed_response(row)
            or _served_key(row, observation=True) not in served
            or any(value not in stimulus_meta for value in observed)
        ):
            continue
        canonical_ids = [str(value) for value in row.get("stimulus_ids", [])]
        expected_sources = list(
            dict.fromkeys(stimulus_meta[value].get("source_id") for value in canonical_ids)
        )
        if row.get("source_ids") != expected_sources:
            continue
        if row.get("variant_ids") != [
            stimulus_meta[value].get("variant_id") for value in canonical_ids
        ]:
            continue
        matches.append(observation_id)
    if not matches:
        kind = "pair" if len(requested) > 1 else "stimulus"
        raise ValueError(
            "post-label forensics refused: requested "
            f"{kind} has no valid completed observation linked to a served trial"
        )
    return matches


def _completed_raters(
    manifest: Mapping[str, Any],
    observations: list[dict[str, Any]],
) -> set[str]:
    """Return raters with every fixed trial and the declared adaptive quota complete."""
    sessions = manifest.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("forensics requires a manifest completion contract")
    expected_base = {
        str(trial["trial_id"])
        for session in sessions
        if isinstance(session, Mapping) and isinstance(session.get("trials"), list)
        for trial in session["trials"]
        if isinstance(trial, Mapping) and trial.get("trial_id") is not None
    }
    if not expected_base:
        raise ValueError("forensics requires scored trials in its completion contract")
    policy = manifest.get("adaptive_policy", {})
    expected_adaptive = (
        int(policy.get("maximum_followups_total", 0))
        if isinstance(policy, Mapping) and policy.get("enabled") is True
        else 0
    )
    observed_raters = {
        str(row["rater_id"])
        for row in observations
        if isinstance(row.get("rater_id"), str) and row.get("rater_id")
    }
    completed: set[str] = set()
    for rater_id in observed_raters:
        rows = [row for row in observations if row.get("rater_id") == rater_id]
        base = {str(row["trial_id"]) for row in rows if str(row.get("trial_id")) in expected_base}
        adaptive = {
            str(row["trial_id"])
            for row in rows
            if str(row.get("schedule_reason", "")).startswith("adaptive:")
        }
        if base == expected_base and len(adaptive) >= expected_adaptive:
            completed.add(rater_id)
    if completed != observed_raters:
        raise ValueError(
            "post-label forensics refused: the blinded pilot must be complete for every "
            "observed rater before diagnostics are unlocked"
        )
    return completed


def _validate_frozen_evidence_paths(
    manifest_path: Path,
    observations_path: Path,
    protocol_freeze_path: Path,
) -> Path:
    try:
        frozen_playlist, _ = resolve_frozen_evidence_paths(
            manifest_path,
            observations_path,
            protocol_freeze_path,
        )
    except ValueError as exc:
        message = str(exc).replace("analysis requires", "forensics requires")
        raise ValueError(message) from exc
    return frozen_playlist


def _contact_intervals(mask: np.ndarray) -> list[list[int]]:
    padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
    changes = np.flatnonzero(np.diff(padded))
    return [[int(start), int(stop)] for start, stop in changes.reshape(-1, 2)]


def _load_critic_temporal(path: Path | None, start: int, stop: int) -> dict[str, Any] | None:
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as archive:
        names = [str(value) for value in archive["defect_names"].tolist()]
        curves = {}
        for key in ("frame_probability", "frame_severity"):
            if key not in archive.files:
                continue
            value = np.asarray(archive[key], dtype=np.float64)
            if value.ndim != 2 or value.shape[1] != len(names):
                raise ValueError(f"critic temporal output {key!r} has an invalid shape")
            curves[key] = value[start:stop].round(7).tolist()
    if not curves:
        raise ValueError("critic output has no supported temporal arrays")
    return {"defect_names": names, **curves, "source": str(Path(path).resolve())}


def _motion_diagnostics(
    path: Path,
    phase_reference_path: Path,
    *,
    critic_output: Path | None,
) -> dict[str, Any]:
    clip = load_motion_npz(path)
    payload = _two_cycle_payload(
        path,
        phase_reference_path=phase_reference_path,
        viewing_settings=MANNEQUIN_VIEWING_SETTINGS,
    )
    start = int(payload["cycle_window"]["source_start_frame"])
    stop = int(payload["cycle_window"]["source_stop_frame"])
    position, global_quat = forward_kinematics_numpy(
        clip.skeleton, clip.local_quat_wxyz, clip.root_translation_m
    )
    pelvis = clip.skeleton.roles.get("pelvis", clip.skeleton.root_index)
    pelvis_euler = Rotation.from_quat(np.roll(global_quat[:, pelvis], -1, axis=-1)).as_euler(
        "xyz", degrees=True
    )
    features = motion_feature_arrays(clip)
    angular_speed = np.linalg.norm(
        np.asarray(features["joint_angular_velocity"], dtype=np.float64), axis=-1
    )
    jerk = np.asarray(smoothness_metric(clip).joint_frame_values, dtype=np.float64)

    foot_position: dict[str, list[list[float]]] = {}
    contact_payload: dict[str, Any] = {
        "available": False,
        "marker_names": [],
        "hard_intervals": {},
        "confidence": [],
    }
    try:
        markers = marker_world_positions(clip, FOOT_MARKER_NAMES)
        contacts = detect_foot_contacts(clip)
        foot_position = {
            name: np.asarray(markers[name][start:stop], dtype=np.float64).round(6).tolist()
            for name in FOOT_MARKER_NAMES
        }
        contact_payload = {
            "available": True,
            "marker_names": list(contacts.marker_names),
            "hard_intervals": {
                name: _contact_intervals(contacts.hard[start:stop, index])
                for index, name in enumerate(contacts.marker_names)
            },
            "confidence": contacts.confidence[start:stop].round(6).tolist(),
            "source": contacts.source,
        }
    except ValueError as exc:
        contact_payload["unavailable_reason"] = str(exc)

    return {
        "motion_path": str(path.resolve()),
        "motion_content_hash": clip.content_hash,
        "fps": clip.fps,
        "frame_count": stop - start,
        "timestamps_s": (np.arange(stop - start) / clip.fps).round(6).tolist(),
        "cycle_window": payload["cycle_window"],
        "viewer": {
            key: payload[key]
            for key in (
                "joint_names",
                "parents",
                "roles",
                "positions",
                "global_quat_wxyz",
                "rest_offsets_m",
                "render_protocol_hash",
                "render_hash",
            )
        },
        "root_trajectory_m": clip.root_translation_m[start:stop].round(6).tolist(),
        "foot_trajectories_m": foot_position,
        "contacts": contact_payload,
        "pelvis": {
            "joint_name": clip.skeleton.joint_names[pelvis],
            "height_m": position[start:stop, pelvis, 1].round(6).tolist(),
            "euler_xyz_degrees": pelvis_euler[start:stop].round(6).tolist(),
        },
        "joint_motion": {
            "joint_names": list(clip.skeleton.joint_names),
            "angular_speed_rad_s": angular_speed[start:stop].round(6).tolist(),
            "angular_jerk_rad_s3": jerk[start:stop].round(6).tolist(),
        },
        "deterministic_metrics": motion_deterministic_metrics(clip),
        "critic_temporal_outputs": _load_critic_temporal(critic_output, start, stop),
    }


_VIEWER_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>MotionLab post-label forensics</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif}body{margin:0;background:#12151b;color:#e8edf5}
header,main{max-width:1500px;margin:auto;padding:14px}h1{font-size:19px;margin:0 0 10px}.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
button,select,label{background:#202735;color:#edf3fb;border:1px solid #3b465b;border-radius:6px;padding:7px}
.grid{display:grid;grid-template-columns:minmax(520px,1.1fr) minmax(440px,1fr);gap:12px;margin-top:12px}.panel{background:#171c25;border:1px solid #30394a;border-radius:8px;padding:10px}
canvas{display:block;width:100%;background:#0f131a;border-radius:5px}#motion{aspect-ratio:16/10}#curve{aspect-ratio:16/7}pre{white-space:pre-wrap;max-height:260px;overflow:auto;font-size:11px}.warning{color:#ffca65}@media(max-width:1000px){.grid{grid-template-columns:1fr}}
</style></head><body><header><h1>Post-label motion forensics</h1>
<div class="warning">Analysis-only view. This bundle is unlocked only by a persisted prior judgment.</div>
<div class="controls"><button id="play">Pause</button><select id="take"></select>
<label><input id="skeleton" type="checkbox"> skeleton overlay</label>
<select id="curveKind"><option value="root">root trajectory</option><option value="feet">foot height/contact</option><option value="pelvis">pelvis height/orientation</option><option value="velocity">joint angular velocity</option><option value="jerk">joint angular jerk</option><option value="critic">critic temporal output</option></select>
<select id="joint"></select><select id="defect"></select></div></header>
<main class="grid"><section class="panel"><canvas id="motion" width="900" height="560"></canvas></section>
<section class="panel"><canvas id="curve" width="900" height="390"></canvas><pre id="metrics"></pre></section></main>
<script>const bundle=__BUNDLE__;const takes=[];
for(const item of bundle.items){takes.push({name:item.label+' — variant',data:item.variant});if(item.source)takes.push({name:item.label+' — source',data:item.source});}
const take=document.querySelector('#take'),joint=document.querySelector('#joint'),defect=document.querySelector('#defect');
takes.forEach((x,i)=>take.add(new Option(x.name,String(i))));let active=0,frame=0,playing=true,last=performance.now();
function resetSelectors(){const d=takes[active].data;joint.textContent='';d.joint_motion.joint_names.forEach((x,i)=>joint.add(new Option(x,String(i))));defect.textContent='';const c=d.critic_temporal_outputs;for(const x of(c?c.defect_names:[]))defect.add(new Option(x,x));document.querySelector('#metrics').textContent=JSON.stringify(d.deterministic_metrics,null,2);frame=0;}
take.onchange=()=>{active=Number(take.value);resetSelectors()};document.querySelector('#play').onclick=()=>{playing=!playing;document.querySelector('#play').textContent=playing?'Pause':'Play'};
function project(p,root){const yaw=.4363,pitch=-.1222,x=p[0]-root[0],y=p[1],z=p[2]-root[2],x1=Math.cos(yaw)*x-Math.sin(yaw)*z,z1=Math.sin(yaw)*x+Math.cos(yaw)*z,y1=Math.cos(pitch)*y-Math.sin(pitch)*z1;return[450+x1*285,500-y1*285]}
function motion(){const c=document.querySelector('#motion'),g=c.getContext('2d'),d=takes[active].data.viewer,pos=d.positions[frame%d.positions.length],root=pos[d.roles.pelvis??0];g.clearRect(0,0,c.width,c.height);g.strokeStyle='#293344';g.lineWidth=1;for(let y=500;y>40;y-=55){g.beginPath();g.moveTo(0,y);g.lineTo(900,y);g.stroke()}
for(let j=0;j<d.parents.length;j++){const p=d.parents[j];if(p<0)continue;const a=project(pos[p],root),b=project(pos[j],root);g.strokeStyle='#718aab';g.lineWidth=(j===d.roles.head?18:12);g.lineCap='round';g.beginPath();g.moveTo(...a);g.lineTo(...b);g.stroke()}
const head=project(pos[d.roles.head??0],root);g.fillStyle='#91a8c2';g.beginPath();g.arc(...head,17,0,Math.PI*2);g.fill();if(document.querySelector('#skeleton').checked){for(let j=0;j<d.parents.length;j++){const p=d.parents[j];if(p<0)continue;g.strokeStyle='#f0f4f8';g.lineWidth=2;g.beginPath();g.moveTo(...project(pos[p],root));g.lineTo(...project(pos[j],root));g.stroke()}}}
const palette=['#67d5ff','#ff9f66','#7ee787','#d2a8ff'];function linePlot(g,series,labels,w,h){let vals=series.flat().filter(Number.isFinite);if(!vals.length){g.fillStyle='#aaa';g.fillText('No data available',20,30);return}let lo=Math.min(...vals),hi=Math.max(...vals);if(hi-lo<1e-9){lo-=1;hi+=1}g.strokeStyle='#3a4352';g.strokeRect(55,18,w-75,h-55);series.forEach((s,k)=>{g.strokeStyle=palette[k%palette.length];g.lineWidth=2;g.beginPath();s.forEach((v,i)=>{const x=55+i/Math.max(1,s.length-1)*(w-75),y=18+(hi-v)/(hi-lo)*(h-55);i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke();g.fillStyle=palette[k%palette.length];g.fillText(labels[k],65+k*170,h-12)});const x=55+frame/Math.max(1,series[0].length-1)*(w-75);g.strokeStyle='#fff';g.beginPath();g.moveTo(x,18);g.lineTo(x,h-37);g.stroke()}
function curve(){const c=document.querySelector('#curve'),g=c.getContext('2d'),d=takes[active].data,kind=document.querySelector('#curveKind').value;g.clearRect(0,0,c.width,c.height);g.font='14px system-ui';let s=[],l=[];
if(kind==='root'){s=[d.root_trajectory_m.map(x=>x[0]),d.root_trajectory_m.map(x=>x[2])];l=['root X','root Z']}
if(kind==='feet'){for(const [name,v] of Object.entries(d.foot_trajectories_m)){s.push(v.map(x=>x[1]));l.push(name)} }
if(kind==='pelvis'){s=[d.pelvis.height_m,...[0,1,2].map(k=>d.pelvis.euler_xyz_degrees.map(x=>x[k]))];l=['height','roll°','pitch°','yaw°']}
if(kind==='velocity'){const j=Number(joint.value||0);s=[d.joint_motion.angular_speed_rad_s.map(x=>x[j])];l=[d.joint_motion.joint_names[j]]}
if(kind==='jerk'){const j=Number(joint.value||0);s=[d.joint_motion.angular_jerk_rad_s3.map(x=>x[j])];l=[d.joint_motion.joint_names[j]]}
if(kind==='critic'&&d.critic_temporal_outputs){const k=d.critic_temporal_outputs.defect_names.indexOf(defect.value);for(const name of['frame_probability','frame_severity'])if(d.critic_temporal_outputs[name]){s.push(d.critic_temporal_outputs[name].map(x=>x[k]));l.push(name)}}linePlot(g,s,l,c.width,c.height)}
function tick(now){const d=takes[active].data;if(playing&&now-last>=1000/d.fps){frame=(frame+1)%d.frame_count;last=now}motion();curve();requestAnimationFrame(tick)}resetSelectors();requestAnimationFrame(tick);
</script></body></html>"""


def build_post_label_forensics(
    manifest_path: Path,
    observations_path: Path,
    stimulus_ids: tuple[str, ...],
    output_directory: Path,
    *,
    protocol_freeze_path: Path,
    critic_outputs: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Create diagnostics only after the exact item and entire blinded pilot are complete."""
    if not 1 <= len(stimulus_ids) <= 2 or len(set(stimulus_ids)) != len(stimulus_ids):
        raise ValueError("request one unique stimulus or one unique pair")
    manifest_path = Path(manifest_path).resolve()
    observations_path = Path(observations_path).resolve()
    served_playlist_path = _validate_frozen_evidence_paths(
        manifest_path,
        observations_path,
        Path(protocol_freeze_path),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root, stimuli = _resolve_stimuli(manifest_path)
    missing = [value for value in stimulus_ids if value not in stimuli]
    if missing:
        raise ValueError(f"unknown stimulus IDs: {missing}")
    observations = _read_jsonl(observations_path)
    served_playlist = _read_jsonl(served_playlist_path)
    validate_current_protocol_observations(observations, manifest, served_playlist)
    _completed_raters(manifest, observations)
    evidence_ids = _assert_already_rated(
        stimulus_ids,
        observations,
        manifest=manifest,
        served_playlist=served_playlist,
    )
    output = Path(output_directory).resolve()
    try:
        output.relative_to(manifest_path.parent)
    except ValueError:
        pass
    else:
        raise ValueError("forensics output must remain outside the active pilot directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"forensics output must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=True)
    supplied = {} if critic_outputs is None else dict(critic_outputs)
    items: list[dict[str, Any]] = []
    for index, stimulus_id in enumerate(stimulus_ids):
        stimulus = stimuli[stimulus_id]
        motion_path = root / str(stimulus["motion"])
        source_path = root / str(stimulus.get("phase_reference_motion", stimulus["motion"]))
        variant = _motion_diagnostics(
            motion_path,
            source_path,
            critic_output=supplied.get(stimulus_id),
        )
        source = None
        if source_path.resolve() != motion_path.resolve():
            source = _motion_diagnostics(source_path, source_path, critic_output=None)
        items.append(
            {
                "stimulus_id": stimulus_id,
                "label": "A"
                if len(stimulus_ids) == 2 and index == 0
                else ("B" if len(stimulus_ids) == 2 else "stimulus"),
                "variant": variant,
                "source": source,
            }
        )
    bundle = {
        "format_version": POST_LABEL_FORENSICS_VERSION,
        "access_policy": (
            "authenticated_persisted_observation_and_completed_pilot_required; "
            "never exposed by blinded rating UI"
        ),
        "protocol_freeze": str(Path(protocol_freeze_path).resolve()),
        "manifest": str(manifest_path),
        "observations": str(observations_path),
        "served_playlist": str(served_playlist_path),
        "unlocking_observation_ids": evidence_ids,
        "items": items,
    }
    _write_json(output / "forensics.json", bundle)
    html = _VIEWER_HTML.replace(
        "__BUNDLE__", json.dumps(bundle, sort_keys=True, allow_nan=False).replace("</", "<\\/")
    )
    (output / "viewer.html").write_text(html, encoding="utf-8")
    report = {
        "format_version": POST_LABEL_FORENSICS_VERSION,
        "stimulus_ids": list(stimulus_ids),
        "unlocked_by_observation_count": len(evidence_ids),
        "forensics": str((output / "forensics.json").resolve()),
        "viewer": str((output / "viewer.html").resolve()),
        "critic_temporal_outputs_included": any(
            item["variant"]["critic_temporal_outputs"] is not None for item in items
        ),
    }
    _write_json(output / "summary.json", report)
    return report
