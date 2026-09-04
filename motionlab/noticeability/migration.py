"""Regularized ordinal calibration and reversible, evidence-gated weak proxies."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from motionlab.noticeability.legacy import read_registry
from motionlab.noticeability.protocol import (
    NOTICE_VALUES,
    PROTOCOL,
    authenticated_observations,
    identity,
    read_json,
    write_json,
)

MODEL_VERSION = "motionlab.legacy_noticeability_ordinal.v1"


def ordinal_probabilities(parameters: Any, scores: Any) -> np.ndarray:
    a, slope, gap = np.asarray(parameters)
    first = expit(a + slope * (np.asarray(scores) - 4))
    second = expit(a + slope * (np.asarray(scores) - 4) + gap)
    return np.stack([first, second - first, 1 - second], axis=-1)


def fit_ordinal(scores: Any, labels: Any) -> list[float]:
    x, y = np.asarray(scores, float), np.asarray(labels, int)
    if len(x) == 0 or len(x) != len(y) or not np.isin(y, [0, 1, 2]).all():
        raise ValueError("nonempty matched ordinal observations required")

    def objective(p: np.ndarray) -> float:
        probabilities = ordinal_probabilities(p, x)
        return float(
            -np.log(np.clip(probabilities[np.arange(len(y)), y], 1e-9, 1)).sum()
            + 0.5 * (p[0] ** 2 / 9 + p[1] ** 2 + (p[2] - 1) ** 2 / 4)
        )

    result = minimize(
        objective, [-0.5, 0.2, 1.0], method="L-BFGS-B", bounds=[(-12, 12), (0, 8), (0.03, 12)]
    )
    if not result.success:
        raise ValueError(f"ordinal calibration did not converge: {result.message}")
    return [float(x) for x in result.x]


def probability_metrics(labels: Any, probabilities: Any) -> dict[str, Any]:
    y, p = np.asarray(labels, int), np.asarray(probabilities, float)
    if len(y) == 0:
        return {"count": 0, "log_loss": None, "brier": None, "accuracy": None, "confusion": None}
    if (
        p.shape != (len(y), 3)
        or not np.isfinite(p).all()
        or (p < 0).any()
        or not np.allclose(p.sum(1), 1)
    ):
        raise ValueError("probabilities must be finite three-class distributions")
    confusion = np.zeros((3, 3), int)
    for actual, predicted in zip(y, p.argmax(1), strict=True):
        confusion[actual, predicted] += 1
    return {
        "count": len(y),
        "log_loss": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean()),
        "brier": float(np.square(p - np.eye(3)[y]).sum(1).mean()),
        "accuracy": float((p.argmax(1) == y).mean()),
        "confusion": confusion.tolist(),
        "class_order": list(NOTICE_VALUES),
    }


def matched_overlap(registry: dict[str, Any], manifest_path: Path) -> list[dict[str, Any]]:
    manifest = read_json(manifest_path)
    if manifest["protocol_version"] != PROTOCOL:
        raise ValueError("repeated-view judgments cannot enter spontaneous-notice calibration")
    if (
        read_json(manifest_path.parent / manifest["legacy_registry"])["registry_id"]
        != registry["registry_id"]
    ):
        raise ValueError("overlap belongs to a different legacy registry")
    old = {r["original_observation_id"]: r for r in registry["records"]}
    trials = {r["trial_id"]: r for r in manifest["trials"]}
    items = {r["stimulus_id"]: r for r in manifest["stimuli"]}
    matched = []
    for row in authenticated_observations(manifest_path):
        oid = trials[row["trial_id"]].get("original_observation_id")
        if oid is None:
            continue
        record = old[oid]
        raw = record["raw_observation"]
        if "legacy_quality_ordinal" not in record:
            raise ValueError("pairwise/style-only evidence cannot calibrate noticeability")
        if (
            row["render_hash"] != raw["render_hashes"][0]
            or row["render_protocol_hash"] != raw["render_protocol_hash"]
            or row["rater_id"] != raw["rater_id"]
        ):
            raise ValueError("protocol/render/rater mismatch prevents direct overlap calibration")
        matched.append(
            {
                "original_observation_id": oid,
                "direct_observation_id": row["observation_id"],
                "score": record["legacy_quality_ordinal"],
                "label": NOTICE_VALUES.index(row["spontaneous_notice"]),
                "source": row["source_id"],
                "family": items[row["stimulus_id"]]["family"],
                "style": items[row["stimulus_id"]]["style"],
                "render": row["render_protocol_hash"],
            }
        )
    return matched


def calibrate_matches(matches: list[dict[str, Any]], *, seed: int = 9601) -> dict[str, Any]:
    """Leave-source-out evaluation within each render version; no in-sample proxy gate."""
    strata: dict[str, Any] = {}
    rng = np.random.default_rng(seed)
    for render in sorted({m["render"] for m in matches}):
        rows = [m for m in matches if m["render"] == render]
        sources = sorted({m["source"] for m in rows})
        x = np.array([r["score"] for r in rows])
        y = np.array([r["label"] for r in rows])
        params = fit_ordinal(x, y)
        predictions: list[list[float]] = []
        baseline: list[list[float]] = []
        held: list[dict[str, Any]] = []
        if len(sources) >= 2:
            for source in sources:
                train = np.array([r["source"] != source for r in rows])
                test = ~train
                p = fit_ordinal(x[train], y[train])
                predictions.extend(ordinal_probabilities(p, x[test]).tolist())
                prior = (np.bincount(y[train], minlength=3) + 1) / (train.sum() + 3)
                baseline.extend([prior.tolist()] * int(test.sum()))
                held.extend(r for r, keep in zip(rows, test, strict=True) if keep)
        metrics = probability_metrics([r["label"] for r in held], predictions)
        base_metrics = probability_metrics([r["label"] for r in held], baseline)
        # Cluster bootstrap intervals describe sampling uncertainty, not invented human labels.
        curves = []
        if len(sources) >= 2:
            for _ in range(64):
                chosen = rng.choice(sources, len(sources), replace=True)
                sample = [r for s in chosen for r in rows if r["source"] == s]
                p = fit_ordinal([r["score"] for r in sample], [r["label"] for r in sample])
                curves.append(ordinal_probabilities(p, np.arange(1, 8)))
        curve = ordinal_probabilities(params, np.arange(1, 8))
        categories = {}
        for index in range(7):
            n = int((x == index + 1).sum())
            categories[str(index + 1)] = {
                "count": n,
                "probabilities": curve[index].tolist(),
                "uncertainty": {
                    "method": "source_cluster_bootstrap_95_percent",
                    "supported": n >= 3 and len(sources) >= 3,
                    "interval": np.quantile(
                        np.asarray(curves)[:, index], [0.025, 0.975], axis=0
                    ).tolist()
                    if curves
                    else None,
                },
            }
        enough = (
            len({r["original_observation_id"] for r in rows}) >= 30
            and len(sources) >= 4
            and min(Counter(y).get(k, 0) for k in range(3)) >= 3
        )
        useful = bool(
            enough
            and metrics["log_loss"] < base_metrics["log_loss"] * 0.95
            and metrics["brier"] < base_metrics["brier"] * 0.95
        )
        strata[render] = {
            "count": len(rows),
            "source_count": len(sources),
            "parameters": params,
            "curve": categories,
            "held_out_source": metrics,
            "held_out_baseline": base_metrics,
            "held_out_by_group": {
                key: {
                    group: probability_metrics(
                        [r["label"] for r in held if r[key] == group],
                        [p for r, p in zip(held, predictions, strict=True) if r[key] == group],
                    )
                    for group in sorted({r[key] for r in held})
                }
                for key in ("source", "style", "family")
            },
            "proxy_gate_passed": useful,
            "gate_policy": (
                "At least 30 unique overlap stimuli, 4 sources, 3 per response class, and "
                ">5% held-source improvement in both Brier and log loss over prior baseline."
            ),
            "held_out_predictions": [
                {**r, "probabilities": p} for r, p in zip(held, predictions, strict=True)
            ],
        }
    result = {
        "migration_model_version": MODEL_VERSION,
        "overlap_count": len(matches),
        "status": "awaiting_direct_human_overlap" if not matches else "evaluated",
        "render_strata": strata,
        "proxy_gate_passed": any(s["proxy_gate_passed"] for s in strata.values()),
        "raw_observations_modified": False,
        "seed": seed,
    }
    return result


def fit_calibration(registry_path: Path, manifest_path: Path, output: Path) -> dict[str, Any]:
    registry = read_registry(registry_path)
    matches = matched_overlap(registry, manifest_path)
    model = {
        **calibrate_matches(matches),
        "registry_id": registry["registry_id"],
        "overlap_evidence": matches,
        "noticeability_manifest": str(manifest_path.resolve()),
    }
    model["migration_model_hash"] = identity("notice-migration-model", model)
    write_json(output, model)
    return model


def validate_model(model: dict[str, Any], registry: dict[str, Any]) -> None:
    if (
        model["migration_model_version"] != MODEL_VERSION
        or model["registry_id"] != registry["registry_id"]
        or model["migration_model_hash"]
        != identity(
            "notice-migration-model",
            {k: v for k, v in model.items() if k != "migration_model_hash"},
        )
    ):
        raise ValueError("migration model hash/version/registry mismatch")
    current = matched_overlap(registry, Path(model["noticeability_manifest"]))
    # Model may predate later collected rows, but every fitted row must still be authenticated.
    if any(row not in current for row in model["overlap_evidence"]):
        raise ValueError("migration model evidence no longer authenticates")


def migrate(registry_path: Path, model_path: Path, output: Path) -> dict[str, Any]:
    registry, model = read_registry(registry_path), read_json(model_path)
    validate_model(model, registry)
    proxies = []
    for record in registry["records"]:
        if "legacy_quality_ordinal" not in record:
            continue
        row = record["raw_observation"]
        stratum = model["render_strata"].get(row["render_protocol_hash"])
        if not stratum or not stratum["proxy_gate_passed"]:
            continue
        category = stratum["curve"][str(record["legacy_quality_ordinal"])]
        if not category["uncertainty"]["supported"]:
            continue
        proxy = {
            "kind": "legacy_notice_proxy",
            "original_observation_id": record["original_observation_id"],
            "migration_model_version": MODEL_VERSION,
            "migration_model_hash": model["migration_model_hash"],
            "probabilities": dict(zip(NOTICE_VALUES, category["probabilities"], strict=True)),
            "uncertainty": category["uncertainty"],
            "source_old_rating": record["legacy_quality_ordinal"],
            "derived": True,
            "label_strength": "WEAK_PROXY",
            "loss_weight": 0.2,
            "render_hash": row["render_hashes"][0],
            "render_protocol_hash": row["render_protocol_hash"],
            "rater_id": row["rater_id"],
        }
        proxy["proxy_id"] = identity("notice-proxy", proxy)
        proxies.append(proxy)
    result = {
        "format_version": "motionlab.noticeability_migration.v1",
        "registry_id": registry["registry_id"],
        "migration_model_hash": model["migration_model_hash"],
        "derived_records": proxies,
        "reversible": "Delete this derived artifact only; raw sources are never edited.",
        "status": "weak_proxies_created" if proxies else "no_proxies_gate_not_met",
    }
    write_json(output, result)
    return result


def validate_migration(
    registry_path: Path, model_path: Path, migration_path: Path
) -> dict[str, Any]:
    # Idempotent recomputation refuses to overwrite a differing derived artifact.
    result = migrate(registry_path, model_path, migration_path)
    ids = [r["original_observation_id"] for r in result["derived_records"]]
    if len(set(ids)) != len(ids):
        raise ValueError("a proxy must reference exactly one unique original observation")
    return {"valid": True, "proxy_count": len(ids), "raw_data_preserved": True}


def prefer_direct(
    direct: list[dict[str, Any]], proxies: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if proxies and any(r.get("event_type") == "repeated_view_notice" for r in direct):
        raise ValueError("repeated-view evidence cannot replace spontaneous-notice proxies")
    keys = {(r["render_hash"], r["rater_id"]) for r in direct}
    return [*direct, *[p for p in proxies if (p["render_hash"], p["rater_id"]) not in keys]]
