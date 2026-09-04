"""Transparent human detection analysis; no synthetic severity is a perceptual label."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import beta, rankdata

from motionlab.noticeability.migration import probability_metrics
from motionlab.noticeability.protocol import (
    NOTICE_VALUES,
    authenticated_observations,
    read_json,
    write_json,
)


def rate_summary(values: list[str]) -> dict[str, Any]:
    counts = Counter(values)
    n = len(values)
    return {
        "count": n,
        "counts": {k: counts[k] for k in NOTICE_VALUES},
        "rates": {k: counts[k] / n if n else None for k in NOTICE_VALUES},
        "yes_interval_95": beta.ppf(
            [0.025, 0.975], counts["YES"] + 0.5, n - counts["YES"] + 0.5
        ).tolist()
        if n
        else None,
    }


def latent_detectability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Modest MAP cumulative-logit model with stimulus effects and a shared threshold gap."""
    if not rows:
        return {"status": "awaiting_human_labels", "stimuli": {}}
    keys = sorted({r["stimulus_id"] for r in rows})
    sessions = sorted({r["session_id"] for r in rows})
    ix = np.array([keys.index(r["stimulus_id"]) for r in rows])
    si = np.array([sessions.index(r["session_id"]) for r in rows])
    y = np.array([NOTICE_VALUES.index(r["spontaneous_notice"]) for r in rows])
    cross_session = sum(
        len({r["session_id"] for r in rows if r["stimulus_id"] == k}) > 1 for k in keys
    )
    use_session = len(sessions) >= 2 and cross_session >= 10 and min(Counter(si).values()) >= 20
    size = len(keys) + 1 + (len(sessions) if use_session else 0)

    def predictions(p: np.ndarray) -> np.ndarray:
        difficulty = p[ix]
        if use_session:
            effects = p[len(keys) + 1 :]
            difficulty = difficulty + effects[si] - effects.mean()
        gap = np.exp(p[len(keys)])
        low, high = expit(-gap / 2 - difficulty), expit(gap / 2 - difficulty)
        return np.stack([low, high - low, 1 - high], axis=-1)

    def objective(p: np.ndarray) -> float:
        return float(
            -np.log(np.clip(predictions(p)[np.arange(len(y)), y], 1e-9, 1)).sum()
            + 0.5 * np.square(p[: len(keys)] / 1.5).sum()
            + 0.5 * p[len(keys)] ** 2
            + (2 * np.square(p[len(keys) + 1 :]).sum() if use_session else 0)
        )

    fit = minimize(objective, np.zeros(size), method="L-BFGS-B", bounds=[(-8, 8)] * size)
    if not fit.success:
        return {"status": "latent_model_did_not_converge", "stimuli": {}}
    gap = np.exp(fit.x[len(keys)])
    variance = np.diag(np.asarray(fit.hess_inv.todense()))
    estimates = {}
    for index, key in enumerate(keys):
        d, se = fit.x[index], np.sqrt(max(variance[index], 0))
        raw = [r["spontaneous_notice"] for r in rows if r["stimulus_id"] == key]
        estimates[key] = {
            "latent_detectability": float(d),
            "p_spontaneous_yes": float(expit(d - gap / 2)),
            "approximate_conditional_interval_95": expit(
                np.array([d - 1.96 * se, d + 1.96 * se]) - gap / 2
            ).tolist(),
            "raw": rate_summary(raw),
        }
    return {
        "status": "exploratory_regularized_fit",
        "model": "shared-threshold cumulative logit; stimulus MAP effects",
        "thresholds": [-float(gap) / 2, float(gap) / 2],
        "session_effect_included": use_session,
        "uncertainty_caveat": (
            "Approximate MAP/L-BFGS curvature, conditional on shared thresholds; "
            "small-pilot intervals are not validated confidence claims."
        ),
        "stimuli": estimates,
    }


def psychometric(rows: list[dict[str, Any]], items: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        item = items[row["stimulus_id"]]
        if item.get("ladder_id"):
            groups[item["ladder_id"]].append({**row, "strength": item["physical_strength"]})
    result: dict[str, Any] = {}
    for group, observations in groups.items():
        levels = sorted({r["strength"] for r in observations})
        counts = [
            rate_summary([r["spontaneous_notice"] for r in observations if r["strength"] == level])
            for level in levels
        ]
        # Refuse a point-estimate threshold from a sparse, non-crossing initial screen.
        result[group] = {
            "levels": [
                {"physical_strength": level, **summary}
                for level, summary in zip(levels, counts, strict=True)
            ],
            "notice_threshold": None,
            "jnd": None,
            "status": "insufficient_repeated_crossing_data",
        }
        if len(levels) < 4 or len(observations) < 24 or min(c["count"] for c in counts) < 3:
            continue
        x = np.array([r["strength"] for r in observations], float)
        y = np.array([r["spontaneous_notice"] == "YES" for r in observations], float)
        span = levels[-1] - levels[0]
        if span <= 0 or not (counts[0]["rates"]["YES"] < 0.5 < counts[-1]["rates"]["YES"]):
            continue
        z = (x - levels[0]) / span
        fit = minimize(
            lambda p, z=z, y=y: float(
                np.logaddexp(0, p[0] + p[1] * z).sum() - y @ (p[0] + p[1] * z) + 0.02 * (p @ p)
            ),
            [-2.0, 4.0],
            method="L-BFGS-B",
            bounds=[(-20, 20), (0.001, 40)],
        )
        if fit.success:
            a, b = fit.x
            threshold = float(levels[0] - span * a / b)
            if levels[0] <= threshold <= levels[-1]:
                result[group].update(
                    status="exploratory_psychometric_fit",
                    notice_threshold=threshold,
                    jnd=float(span * np.log(3) / b),
                    threshold_definition=(
                        "50% YES; MAYBE remains separately reported, not silently NO"
                    ),
                    jnd_definition="half of 25-75% YES interval",
                    uncertainty="requires independent fixed validation",
                )
    return result


def analyze(manifest_path: Path, output: Path | None = None) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    items = {r["stimulus_id"]: r for r in manifest["stimuli"]}
    trials = {r["trial_id"]: r for r in manifest["trials"]}
    rows = authenticated_observations(manifest_path)
    clean = [r for r in rows if items[r["stimulus_id"]]["population"] == "CLEAN_SHAM_CONTROL"]
    obvious = [
        r for r in rows if items[r["stimulus_id"]].get("intended_range") == "intended_obvious"
    ]
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        grouped[(row["render_hash"], row["rater_id"])].append(row["spontaneous_notice"])
    pairs = [pair for group in grouped.values() for pair in combinations(group, 2)]
    inspected = [r for r in rows if r["inspected_notice"] is not None]
    transitions = [
        [
            sum(r["spontaneous_notice"] == a and r["inspected_notice"] == b for r in inspected)
            for b in NOTICE_VALUES
        ]
        for a in NOTICE_VALUES
    ]
    report = {
        "format_version": "motionlab.noticeability_analysis.v1",
        "status": "awaiting_direct_human_labels"
        if not rows
        else "pilot_analysis_only_review_before_training",
        "direct_observation_count": len(rows),
        "overlap_relabel_count": sum(
            "original_observation_id" in trials[r["trial_id"]] for r in rows
        ),
        "spontaneous": rate_summary([r["spontaneous_notice"] for r in rows]),
        "inspected": rate_summary([r["inspected_notice"] for r in inspected]),
        "clean_sham": rate_summary([r["spontaneous_notice"] for r in clean]),
        "clean_sham_false_alarm_rate": sum(r["spontaneous_notice"] == "YES" for r in clean)
        / len(clean)
        if clean
        else None,
        "intended_obvious_controls": rate_summary([r["spontaneous_notice"] for r in obvious]),
        "intended_obvious_miss_rate_no": sum(r["spontaneous_notice"] == "NO" for r in obvious)
        / len(obvious)
        if obvious
        else None,
        "obvious_control_caveat": (
            "Construction intent is not human ground truth; MAYBE is reported separately, "
            "not counted as a confident miss."
        ),
        "repeat_pairs": len(pairs),
        "repeat_agreement": sum(a == b for a, b in pairs) / len(pairs) if pairs else None,
        "spontaneous_inspected_disagreement": sum(
            r["spontaneous_notice"] != r["inspected_notice"] for r in inspected
        )
        / len(inspected)
        if inspected
        else None,
        "spontaneous_inspected_transition_counts": transitions,
        "transition_class_order": list(NOTICE_VALUES),
        "by_family": {
            f: rate_summary(
                [r["spontaneous_notice"] for r in rows if items[r["stimulus_id"]]["family"] == f]
            )
            for f in sorted({i["family"] for i in items.values()})
        },
        "prior_exposure": {
            str(prior): rate_summary(
                [r["spontaneous_notice"] for r in rows if r["prior_exposure"] == prior]
            )
            for prior in (False, True)
        },
        "latent_detectability": latent_detectability(rows),
        "psychometric": psychometric(rows, items),
        "critic_retraining_permitted": False,
        "production_tau": None,
        "next_batch": (
            "Complete the small blinded overlap/fixed screen; review clean false alarms and "
            "source effects, then interleave staircases near observed crossings and collect "
            "a separate fixed validation batch. Do not train yet."
        ),
    }
    if output:
        write_json(output, report)
    return report


def detector_metrics(
    labels: list[str], probabilities: list[list[float]], *, tau: float | None = None
) -> dict[str, Any]:
    """Three-class calibration plus explicitly YES-versus-NO binary discrimination."""
    y = np.array([NOTICE_VALUES.index(x) for x in labels])
    p = np.asarray(probabilities, float)
    result = probability_metrics(y, p)
    definite = y != 1
    actual = y[definite] == 2
    scores = p[definite, 2] if len(y) else np.array([])
    positives, negatives = int(actual.sum()), int((~actual).sum())
    auroc = ap = None
    if positives and negatives:
        ranks = rankdata(scores)
        auroc = float(
            (ranks[actual].sum() - positives * (positives + 1) / 2) / (positives * negatives)
        )
        # Group ties (not arbitrary sort order) for threshold-averaged AP.
        previous_recall, ap = 0.0, 0.0
        for cutoff in sorted(set(scores), reverse=True):
            selected = scores >= cutoff
            recall = float(actual[selected].sum() / positives)
            ap += (recall - previous_recall) * float(actual[selected].mean())
            previous_recall = recall
    result.update(
        auroc_yes_vs_no=auroc,
        ap_yes_vs_no=ap,
        maybe_excluded_from_binary=int((y == 1).sum()),
        tau=tau,
        recall_at_tau=float((scores[actual] > tau).mean())
        if tau is not None and positives
        else None,
        false_positive_at_tau=float((scores[~actual] > tau).mean())
        if tau is not None and negatives
        else None,
    )
    bins = []
    if len(y):
        for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
            mask = (p[:, 2] >= lo) & (p[:, 2] <= lo + 0.2 if lo == 0.8 else p[:, 2] < lo + 0.2)
            bins.append(
                {
                    "lower": lo,
                    "upper": lo + 0.2,
                    "count": int(mask.sum()),
                    "mean_predicted_yes": float(p[mask, 2].mean()) if mask.any() else None,
                    "observed_yes_rate": float((y[mask] == 2).mean()) if mask.any() else None,
                    "maybe_count": int((y[mask] == 1).sum()),
                }
            )
    result["yes_calibration_bins"] = bins
    result["binary_brier_yes_vs_no"] = (
        float(np.square(scores - actual).mean()) if len(scores) else None
    )
    return result
