"""Minimal localhost pair-labeling UI with animated skeleton canvases."""

# ruff: noqa: E501 -- embedded HTML/JavaScript is kept readable as a self-contained local UI.

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from motionlab.io.npz import load_motion_npz
from motionlab.kinematics.fk import forward_kinematics_numpy
from motionlab.perceptual.dataset import load_perceptual_pairs

LABEL_FORMAT_VERSION = "motionlab.perceptual_human_label.v1"
ALLOWED_PREFERENCES = {"a_better", "b_better", "approximately_equal"}
ALLOWED_REASONS = {
    "naturalness",
    "coordination",
    "style",
    "weight transfer",
    "rigidity/smoothing",
    "other",
}

_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>MotionLab pair labeler</title>
<style>
body{font:16px system-ui;margin:20px;background:#111;color:#eee}main{max-width:1100px;margin:auto}
.views{display:grid;grid-template-columns:1fr 1fr;gap:16px}canvas{width:100%;background:#1b1b1b}
button{font-size:16px;padding:10px 18px;margin:8px}.reasons label{margin-right:14px}
#status{color:#9bd}code{color:#fdc}@media(max-width:700px){.views{grid-template-columns:1fr}}
</style></head><body><main><h1>Hard-feasible motion pair</h1><p id="status">Loading…</p>
<div class="views"><section><h2>A</h2><canvas id="a" width="520" height="420"></canvas></section>
<section><h2>B</h2><canvas id="b" width="520" height="420"></canvas></section></div>
<div class="reasons"><p>Optional reasons:</p>
<label><input type=checkbox value="naturalness"> naturalness</label>
<label><input type=checkbox value="coordination"> coordination</label>
<label><input type=checkbox value="style"> style</label>
<label><input type=checkbox value="weight transfer"> weight transfer</label>
<label><input type=checkbox value="rigidity/smoothing"> rigidity/smoothing</label>
<label><input type=checkbox value="other"> other</label></div>
<p><label>Confidence <select id="confidence"><option value="1">1 — unsure</option>
<option value="2">2</option><option value="3" selected>3 — moderate</option>
<option value="4">4</option><option value="5">5 — very sure</option></select></label></p>
<p><button onclick="label('a_better')">A better</button><button onclick="label('b_better')">B better</button>
<button onclick="label('approximately_equal')">approximately equal</button></p>
<script>
let queue=[],current=null,motion=null,frame=0;
function draw(id,data,f){const c=document.getElementById(id),x=c.getContext('2d'),p=data.positions[f];
x.clearRect(0,0,c.width,c.height);let ys=p.map(v=>v[1]),zs=p.map(v=>v[2]),mnY=Math.min(...ys),mxY=Math.max(...ys),mnZ=Math.min(...zs),mxZ=Math.max(...zs);
let s=.82*Math.min(c.width/Math.max(mxZ-mnZ,.1),c.height/Math.max(mxY-mnY,.1));
let map=v=>[c.width/2+(v[2]-(mnZ+mxZ)/2)*s,c.height*.92-(v[1]-mnY)*s];x.strokeStyle='#ddd';x.lineWidth=3;
for(let j=0;j<p.length;j++){let q=data.parents[j];if(q<0)continue;let u=map(p[j]),v=map(p[q]);x.beginPath();x.moveTo(v[0],v[1]);x.lineTo(u[0],u[1]);x.stroke()}}
async function next(){if(!queue.length){document.getElementById('status').textContent='Queue complete';return}current=queue.shift();motion=await (await fetch('/api/pair/'+encodeURIComponent(current.pair_id))).json();frame=0;
document.getElementById('status').innerHTML='<code>'+current.pair_id+'</code> · '+current.perturbation_mechanism+' · priority '+current.priority.toFixed(3)}
async function label(preference){let reasons=[...document.querySelectorAll('input:checked')].map(x=>x.value),confidence=Number(document.getElementById('confidence').value);await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pair_id:current.pair_id,preference,reason_tags:reasons,confidence})});document.querySelectorAll('input').forEach(x=>x.checked=false);next()}
async function init(){queue=await (await fetch('/api/queue')).json();next();setInterval(()=>{if(!motion)return;draw('a',motion.a,frame);draw('b',motion.b,frame);frame=(frame+1)%motion.a.positions.length},1000/30)}init();
</script></main></body></html>"""


def _existing_labels(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(json.loads(line)["pair_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def load_latest_human_labels(labels_path: Path) -> dict[str, dict[str, Any]]:
    """Load append-only labels, treating the latest record as an explicit revision."""
    if not Path(labels_path).exists():
        return {}
    labels: dict[str, dict[str, Any]] = {}
    for line in Path(labels_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        label = json.loads(line)
        pair_id = str(label.get("pair_id", ""))
        if label.get("format_version") != LABEL_FORMAT_VERSION:
            raise ValueError("unsupported human-label format")
        if label.get("preference") not in ALLOWED_PREFERENCES:
            raise ValueError("unsupported pair preference in human-label file")
        reasons = [str(value) for value in label.get("reason_tags", [])]
        if any(reason not in ALLOWED_REASONS for reason in reasons):
            raise ValueError("unsupported reason tag in human-label file")
        confidence = label.get("confidence")
        if confidence is not None and (not isinstance(confidence, int) or not 1 <= confidence <= 5):
            raise ValueError("human-label confidence must be an integer from 1 to 5")
        labels[pair_id] = {**label, "reason_tags": sorted(set(reasons))}
    return labels


def apply_human_labels(
    records: list[dict[str, Any]],
    labels_path: Path,
) -> list[dict[str, Any]]:
    """Overlay the latest validated human judgment while retaining generated provenance."""
    by_id = {str(record["pair_id"]): record for record in records}
    labels = load_latest_human_labels(labels_path)
    for pair_id in labels:
        if pair_id not in by_id:
            raise ValueError(f"human label references unknown pair: {pair_id}")

    merged: list[dict[str, Any]] = []
    for source in records:
        record = dict(source)
        label = labels.get(str(record["pair_id"]))
        if label is not None:
            record.update(
                {
                    "generated_supervision_category": source["supervision_category"],
                    "generated_preference": source["preference"],
                    "supervision_category": "HUMAN_LABELED",
                    "preference": label["preference"],
                    "reason_tags": label["reason_tags"],
                    "label_basis": "human pair judgment",
                    "human_label_timestamp_utc": label.get("timestamp_utc"),
                    "human_confidence": label.get("confidence"),
                }
            )
        merged.append(record)
    return merged


def build_labeling_queue(
    pair_dataset_directory: Path,
    *,
    evaluation_report: Path | None = None,
    labels_path: Path | None = None,
    audit_queue_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Prioritize uncertainty, disagreement proxies, adversarial cases, and new mechanisms."""
    records = load_perceptual_pairs(pair_dataset_directory)
    scores: dict[str, dict[str, Any]] = {}
    if evaluation_report is not None:
        report = json.loads(Path(evaluation_report).read_text(encoding="utf-8"))
        scores = {str(row["pair_id"]): row for row in report.get("pair_scores", [])}
    completed = set() if labels_path is None else _existing_labels(labels_path)
    queue = []
    for record in records:
        if record["pair_id"] in completed:
            continue
        predicted = scores.get(str(record["pair_id"]))
        uncertainty = 1.0 if predicted is None else 1.0 - float(predicted["confidence"])
        priority = uncertainty
        critic_disagreement = predicted is not None and predicted.get("correct") is False
        if critic_disagreement:
            priority += 1.0
        if record["supervision_category"] == "UNORDERED":
            priority += 0.5
        if record["adversarial_origin"]:
            priority += 0.75
        if record["mechanism_partition"] == "heldout":
            priority += 0.35
        queue.append(
            {
                "pair_id": record["pair_id"],
                "perturbation_mechanism": record["perturbation_mechanism"],
                "supervision_category": record["supervision_category"],
                "adversarial_origin": record["adversarial_origin"],
                "critic_disagreement": critic_disagreement,
                "priority": priority,
            }
        )
    ranked = sorted(queue, key=lambda row: (-row["priority"], row["pair_id"]))
    if audit_queue_path is None:
        return ranked
    audit_rows = [
        json.loads(line)
        for line in Path(audit_queue_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    audit_order = {str(row["pair_id"]): index for index, row in enumerate(audit_rows)}
    return sorted(
        (row for row in ranked if str(row["pair_id"]) in audit_order),
        key=lambda row: audit_order[str(row["pair_id"])],
    )


def record_human_label(
    labels_path: Path,
    *,
    pair_id: str,
    preference: str,
    reason_tags: list[str],
    confidence: int | None = None,
) -> dict[str, Any]:
    """Append one validated human label without modifying the immutable pair manifest."""
    if preference not in ALLOWED_PREFERENCES:
        raise ValueError("unsupported pair preference")
    if any(reason not in ALLOWED_REASONS for reason in reason_tags):
        raise ValueError("unsupported perceptual reason tag")
    if confidence is not None and (not isinstance(confidence, int) or not 1 <= confidence <= 5):
        raise ValueError("human-label confidence must be an integer from 1 to 5")
    record = {
        "format_version": LABEL_FORMAT_VERSION,
        "pair_id": pair_id,
        "supervision_category": "HUMAN_LABELED",
        "preference": preference,
        "reason_tags": sorted(set(reason_tags)),
        "confidence": confidence,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    path = Path(labels_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def _motion_payload(path: Path) -> dict[str, Any]:
    motion = load_motion_npz(path)
    positions, _ = forward_kinematics_numpy(
        motion.skeleton,
        motion.local_quat_wxyz,
        motion.root_translation_m,
    )
    return {
        "fps": motion.fps,
        "parents": motion.skeleton.parents.tolist(),
        "positions": np.asarray(positions, dtype=np.float32).round(5).tolist(),
    }


def serve_pair_labeler(
    pair_dataset_directory: Path,
    labels_path: Path,
    *,
    pilot_manifest: Path,
    served_playlist_path: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Serve the calibrated hybrid evaluator; the legacy split-screen UI is retired."""
    from motionlab.perceptual.subjective import serve_subjective_evaluator

    serve_subjective_evaluator(
        pair_dataset_directory,
        pilot_manifest,
        labels_path,
        served_playlist_path=served_playlist_path,
        host=host,
        port=port,
    )
