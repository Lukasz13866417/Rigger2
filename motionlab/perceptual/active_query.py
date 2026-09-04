"""Bounded active-query scheduling for partially labeled perceptual pools.

This module only recommends queries.  It never edits a pilot manifest, a stimulus, or an
append-only observation log, and it deliberately constructs a bounded neighbor graph instead of
materializing every possible stimulus pair.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

ACTIVE_QUERY_SCHEDULE_VERSION = "motionlab.active_human_query_schedule.v1"
RATE_SINGLE = "RATE_SINGLE"
COMPARE = "COMPARE"
QueryType = Literal["RATE_SINGLE", "COMPARE"]


@dataclass(frozen=True)
class ActiveQueryConfig:
    """Controls a deterministic, bounded recommendation batch."""

    max_recommendations: int = 20
    max_rate_single: int = 12
    max_compare: int = 8
    minimum_rate_single: int = 1
    minimum_compare: int = 1
    pair_neighbor_span: int = 2
    minimum_repeat_spacing: int = 12
    max_observations_per_stimulus: int = 3
    max_observations_per_pair: int = 2
    max_recommendations_per_source: int = 4
    max_recommendations_per_style: int = 6
    max_recommendations_per_family: int = 6
    required_anchors_per_session: int = 2
    include_calibration_only: bool = False

    def validate(self) -> None:
        integer_fields = {
            "max_recommendations": self.max_recommendations,
            "max_rate_single": self.max_rate_single,
            "max_compare": self.max_compare,
            "pair_neighbor_span": self.pair_neighbor_span,
            "minimum_repeat_spacing": self.minimum_repeat_spacing,
            "max_observations_per_stimulus": self.max_observations_per_stimulus,
            "max_observations_per_pair": self.max_observations_per_pair,
            "max_recommendations_per_source": self.max_recommendations_per_source,
            "max_recommendations_per_style": self.max_recommendations_per_style,
            "max_recommendations_per_family": self.max_recommendations_per_family,
            "required_anchors_per_session": self.required_anchors_per_session,
        }
        if any(value < 0 for value in integer_fields.values()):
            raise ValueError(f"active-query limits must be nonnegative: {integer_fields}")
        if self.max_recommendations <= 0:
            raise ValueError("max_recommendations must be positive")
        if self.pair_neighbor_span <= 0:
            raise ValueError("pair_neighbor_span must be positive")
        if not 0 <= self.minimum_rate_single <= self.max_rate_single:
            raise ValueError("minimum_rate_single must not exceed max_rate_single")
        if not 0 <= self.minimum_compare <= self.max_compare:
            raise ValueError("minimum_compare must not exceed max_compare")
        if self.minimum_rate_single + self.minimum_compare > self.max_recommendations:
            raise ValueError("minimum query-type counts exceed max_recommendations")


@dataclass(frozen=True)
class _ItemSignal:
    stimulus_id: str
    source_id: str
    style: str
    family: str
    population: str
    quality: float | None
    interval: tuple[float, float] | None
    critic_score: float | None
    critic_confidence: float
    comparator_impact: float
    adversarial: bool
    anchor: bool
    repeat_target_count: int | None


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _string_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if item is not None]


def _repeat_target_count(row: dict[str, Any]) -> int | None:
    """Return the bounded total-observation target for a requested repeat."""
    raw_target = row.get("repeat_target")
    if isinstance(raw_target, int) and not isinstance(raw_target, bool):
        if raw_target < 0:
            raise ValueError("repeat_target must be nonnegative")
        return raw_target or None
    if raw_target is not None or bool(row.get("required_repeat", False)):
        # Backward-compatible boolean/non-numeric repeat requests mean one initial judgment
        # plus one repeat, rather than an unbounded exemption from the observation cap.
        return 2
    return None


def _repeat_is_due(
    item: _ItemSignal,
    observation_count: int,
    config: ActiveQueryConfig,
) -> bool:
    """Whether another repeat is due without exceeding the per-stimulus hard cap."""
    if item.repeat_target_count is None or observation_count <= 0:
        return False
    effective_target = min(
        item.repeat_target_count,
        config.max_observations_per_stimulus,
    )
    return observation_count < effective_target


def _canonical_pair(first: str, second: str) -> tuple[str, str]:
    return (first, second) if first <= second else (second, first)


def _rank_percentiles(values: dict[str, float]) -> dict[str, float]:
    """Return deterministic midrank percentiles, including for tied scores."""
    if not values:
        return {}
    ordered = sorted((value, key) for key, value in values.items())
    denominator = max(1, len(ordered) - 1)
    result: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        percentile = ((index + end - 1) / 2.0) / denominator
        for _, key in ordered[index:end]:
            result[key] = percentile
        index = end
    return result


def _prediction_map(value: dict[str, Any] | list[dict[str, Any]] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        rows = value.get("predictions")
        if isinstance(rows, list):
            return {
                str(row["stimulus_id"]): row
                for row in rows
                if isinstance(row, dict) and row.get("stimulus_id") is not None
            }
        return {str(key): item for key, item in value.items()}
    return {
        str(row["stimulus_id"]): row
        for row in value
        if isinstance(row, dict) and row.get("stimulus_id") is not None
    }


def _prediction_fields(value: Any) -> tuple[float | None, float]:
    if isinstance(value, dict):
        score = _finite_float(value.get("score", value.get("quality", value.get("critic_score"))))
        explicit_confidence = _finite_float(value.get("confidence"))
        if explicit_confidence is not None:
            return score, _clip01(explicit_confidence)
        uncertainty = _finite_float(value.get("uncertainty"))
        return score, 0.0 if uncertainty is None else 1.0 / (1.0 + max(0.0, uncertainty))
    return _finite_float(value), 0.0


def _latent_map(latent_quality: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if latent_quality is None:
        return {}
    rows = latent_quality.get("item_quality", [])
    if not isinstance(rows, list):
        return {}
    return {
        str(row["stimulus_id"]): row
        for row in rows
        if isinstance(row, dict) and row.get("stimulus_id") is not None
    }


def _item_signals(
    candidates: list[dict[str, Any]],
    latent_quality: dict[str, Any] | None,
    critic_predictions: dict[str, Any] | list[dict[str, Any]] | None,
) -> dict[str, _ItemSignal]:
    latent = _latent_map(latent_quality)
    predictions = _prediction_map(critic_predictions)
    result: dict[str, _ItemSignal] = {}
    for row in candidates:
        if row.get("stimulus_id") is None:
            raise ValueError("every active-query candidate needs a stimulus_id")
        stimulus_id = str(row["stimulus_id"])
        if stimulus_id in result:
            raise ValueError(f"duplicate active-query stimulus_id: {stimulus_id}")
        fitted = latent.get(stimulus_id, {})
        quality = _finite_float(fitted.get("latent_quality"))
        raw_interval = fitted.get("quality_95_percent_interval")
        interval = None
        if isinstance(raw_interval, (list, tuple)) and len(raw_interval) == 2:
            low, high = _finite_float(raw_interval[0]), _finite_float(raw_interval[1])
            if low is not None and high is not None and low <= high:
                interval = (low, high)
        critic_score, critic_confidence = _prediction_fields(
            predictions.get(stimulus_id, row.get("critic_prediction"))
        )
        source_id = str(row.get("source_id", "unknown"))
        family = str(row.get("family", row.get("mechanism", "unknown")))
        mechanism = str(row.get("mechanism", family)).lower()
        population = str(row.get("stimulus_population", row.get("population", "UNLABELED")))
        result[stimulus_id] = _ItemSignal(
            stimulus_id=stimulus_id,
            source_id=source_id,
            style=str(row.get("style", row.get("source_style", row.get("style_id", source_id)))),
            family=family,
            population=population,
            quality=quality,
            interval=interval,
            critic_score=critic_score,
            critic_confidence=critic_confidence,
            comparator_impact=_clip01(
                _finite_float(row.get("comparator_impact", row.get("evaluation_impact"))) or 0.0
            ),
            adversarial=bool(
                row.get("adversarial_optimizer_output", row.get("is_adversarial", False))
            )
            or "adversarial" in mechanism
            or "cma" in mechanism,
            anchor=population == "EVALUATION_ANCHOR" or bool(row.get("required_anchor", False)),
            repeat_target_count=_repeat_target_count(row),
        )
    return result


def _observation_state(
    observations: list[dict[str, Any]],
    items: dict[str, _ItemSignal],
    session_id: str,
) -> dict[str, Any]:
    item_counts: Counter[str] = Counter()
    session_item_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    last_item_position: dict[str, int] = {}
    last_pair_position: dict[tuple[str, str], int] = {}
    pair_outcomes: dict[tuple[str, str], list[str]] = defaultdict(list)
    repeat_rating_groups: dict[tuple[str, str, str], list[tuple[str, int]]] = defaultdict(list)
    repeat_pair_groups: dict[
        tuple[str, str, tuple[str, ...]],
        list[tuple[tuple[str, str], str]],
    ] = defaultdict(list)
    global_source: Counter[str] = Counter()
    global_style: Counter[str] = Counter()
    global_family: Counter[str] = Counter()
    session_source: Counter[str] = Counter()
    session_style: Counter[str] = Counter()
    session_family: Counter[str] = Counter()
    session_anchor_count = 0
    session_position = 0
    for row in observations:
        ids = [value for value in _string_values(row.get("stimulus_ids")) if value in items]
        in_session = str(row.get("session_id", "")) == session_id
        signals = [items[stimulus_id] for stimulus_id in ids]
        global_source.update({signal.source_id for signal in signals})
        global_style.update({signal.style for signal in signals})
        global_family.update({signal.family for signal in signals})
        for stimulus_id in ids:
            item_counts[stimulus_id] += 1
            if in_session:
                session_item_counts[stimulus_id] += 1
                last_item_position[stimulus_id] = session_position
        if in_session:
            session_source.update({signal.source_id for signal in signals})
            session_style.update({signal.style for signal in signals})
            session_family.update({signal.family for signal in signals})
        rating = row.get("rating")
        rater_id = str(row.get("rater_id") or "")
        task_type = str(row.get("task_type") or "")
        repeat_groups = tuple(sorted(set(_string_values(row.get("hidden_repeat_group_ids")))))
        if len(ids) == 1 and isinstance(rating, int) and not isinstance(rating, bool) and rater_id:
            for repeat_group in repeat_groups:
                repeat_rating_groups[(rater_id, task_type, repeat_group)].append((ids[0], rating))
        if len(ids) == 2 and row.get("task_type") == "pair_comparison":
            pair = _canonical_pair(ids[0], ids[1])
            pair_counts[pair] += 1
            if in_session:
                last_pair_position[pair] = session_position
            outcome = row.get("pair_outcome_canonical")
            if isinstance(outcome, str):
                # Normalize direction to the sorted pair before measuring repeat disagreement.
                if ids != list(pair) and outcome in {"first_better", "second_better"}:
                    outcome = "second_better" if outcome == "first_better" else "first_better"
                pair_outcomes[pair].append(outcome)
                if rater_id and repeat_groups:
                    repeat_pair_groups[(rater_id, task_type, repeat_groups)].append((pair, outcome))
        if in_session and any(items[value].anchor for value in ids):
            session_anchor_count += 1
        if in_session:
            session_position += 1
    item_repeat_disagreement: dict[str, float] = defaultdict(float)
    for grouped_values in repeat_rating_groups.values():
        by_item: dict[str, list[int]] = defaultdict(list)
        for stimulus_id, rating in grouped_values:
            by_item[stimulus_id].append(rating)
        for stimulus_id, values in by_item.items():
            if len(values) >= 2:
                item_repeat_disagreement[stimulus_id] = max(
                    item_repeat_disagreement[stimulus_id],
                    _repeat_disagreement(values),
                )
    pair_repeat_disagreement: dict[tuple[str, str], float] = defaultdict(float)
    for grouped_pair_values in repeat_pair_groups.values():
        by_pair: dict[tuple[str, str], list[str]] = defaultdict(list)
        for repeated_pair, pair_outcome in grouped_pair_values:
            by_pair[repeated_pair].append(pair_outcome)
        for repeated_pair, pair_values in by_pair.items():
            if len(pair_values) >= 2:
                pair_repeat_disagreement[repeated_pair] = max(
                    pair_repeat_disagreement[repeated_pair],
                    _repeat_disagreement(pair_values),
                )
    return {
        "item_counts": item_counts,
        "session_item_counts": session_item_counts,
        "pair_counts": pair_counts,
        "last_item_position": last_item_position,
        "last_pair_position": last_pair_position,
        "pair_outcomes": pair_outcomes,
        "item_repeat_disagreement": item_repeat_disagreement,
        "pair_repeat_disagreement": pair_repeat_disagreement,
        "global_source": global_source,
        "global_style": global_style,
        "global_family": global_family,
        "session_source": session_source,
        "session_style": session_style,
        "session_family": session_family,
        "session_anchor_count": session_anchor_count,
        "session_observation_count": session_position,
    }


def _repeat_disagreement(values: list[Any]) -> float:
    if len(values) < 2:
        return 0.0
    counts = Counter(values)
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return _clip01((max(values) - min(values)) / 6.0)
    return 1.0 - max(counts.values()) / len(values)


def _bounded_pair_candidates(
    items: dict[str, _ItemSignal],
    candidate_rows: dict[str, dict[str, Any]],
    span: int,
) -> set[tuple[str, str]]:
    """Build at most a small constant number of neighbor edges per item."""
    result: set[tuple[str, str]] = set()

    def ordering(values: list[_ItemSignal]) -> list[_ItemSignal]:
        return sorted(
            values,
            key=lambda item: (
                item.quality is None,
                item.quality if item.quality is not None else 0.0,
                item.critic_score is None,
                item.critic_score if item.critic_score is not None else 0.0,
                item.stimulus_id,
            ),
        )

    def connect(values: list[_ItemSignal]) -> None:
        ordered = ordering(values)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 : index + 1 + span]:
                if left.stimulus_id != right.stimulus_id:
                    result.add(_canonical_pair(left.stimulus_id, right.stimulus_id))

    active_items = list(items.values())
    connect(active_items)
    for attribute in ("source_id", "family"):
        grouped: dict[str, list[_ItemSignal]] = defaultdict(list)
        for item in active_items:
            grouped[str(getattr(item, attribute))].append(item)
        for values in grouped.values():
            connect(values)
    for stimulus_id, row in candidate_rows.items():
        explicit = _string_values(
            row.get("comparison_candidate_ids", row.get("paired_stimulus_ids"))
        )
        raw_pair_impacts = row.get("comparator_pair_impacts", {})
        ranked_impact_ids: list[str] = []
        if isinstance(raw_pair_impacts, dict):
            ranked_impacts = [
                (str(other_id), _finite_float(value) or 0.0)
                for other_id, value in raw_pair_impacts.items()
            ]
            ranked_impact_ids = [
                other_id
                for other_id, _ in sorted(
                    ranked_impacts,
                    key=lambda pair: (-pair[1], pair[0]),
                )
            ][:span]
        ordered_explicit = list(dict.fromkeys(ranked_impact_ids + sorted(set(explicit))))
        for other_id in ordered_explicit[:span]:
            if other_id in items and other_id != stimulus_id:
                result.add(_canonical_pair(stimulus_id, other_id))
    return result


def _interval_uncertainties(items: dict[str, _ItemSignal]) -> dict[str, float]:
    widths = {
        stimulus_id: item.interval[1] - item.interval[0]
        for stimulus_id, item in items.items()
        if item.interval is not None
    }
    maximum = max(widths.values(), default=0.0)
    return {
        stimulus_id: (
            1.0
            if item.interval is None
            else 0.0
            if maximum <= 0.0
            else _clip01(widths[stimulus_id] / maximum)
        )
        for stimulus_id, item in items.items()
    }


def _coverage_score(count: int, *, explicitly_held_out: bool = False) -> float:
    return max(1.0 if explicitly_held_out and count == 0 else 0.0, 1.0 / math.sqrt(count + 1.0))


def _score_items(
    items: dict[str, _ItemSignal],
    candidate_rows: dict[str, dict[str, Any]],
    state: dict[str, Any],
    config: ActiveQueryConfig,
) -> dict[str, dict[str, float]]:
    uncertainties = _interval_uncertainties(items)
    human_ranks = _rank_percentiles(
        {
            stimulus_id: item.quality
            for stimulus_id, item in items.items()
            if item.quality is not None
        }
    )
    critic_ranks = _rank_percentiles(
        {
            stimulus_id: item.critic_score
            for stimulus_id, item in items.items()
            if item.critic_score is not None
        }
    )
    result: dict[str, dict[str, float]] = {}
    for stimulus_id, item in items.items():
        row = candidate_rows[stimulus_id]
        item_count = int(state["item_counts"][stimulus_id])
        rank_disagreement = (
            abs(human_ranks[stimulus_id] - critic_ranks[stimulus_id])
            if stimulus_id in human_ranks and stimulus_id in critic_ranks
            else 0.0
        )
        anchor_due = (
            item.anchor and state["session_anchor_count"] < config.required_anchors_per_session
        )
        repeat_due = _repeat_is_due(item, item_count, config)
        result[stimulus_id] = {
            "posterior_uncertainty": uncertainties[stimulus_id],
            "repeat_disagreement": float(state["item_repeat_disagreement"][stimulus_id]),
            "source_undercoverage": _coverage_score(
                int(state["global_source"][item.source_id]),
                explicitly_held_out=bool(row.get("held_out_source", False)),
            ),
            "style_undercoverage": _coverage_score(
                int(state["global_style"][item.style]),
                explicitly_held_out=bool(row.get("held_out_style", False)),
            ),
            "family_undercoverage": _coverage_score(
                int(state["global_family"][item.family]),
                explicitly_held_out=bool(row.get("held_out_family", False)),
            ),
            "critic_human_disagreement": rank_disagreement,
            "confident_critic_unconfirmed": (item.critic_confidence if item_count == 0 else 0.0),
            "adversarial_optimizer_output": float(item.adversarial),
            "comparator_evaluation_impact": item.comparator_impact,
            "required_repeat_or_anchor": float(repeat_due or anchor_due),
            "judgment_sparsity": 1.0 / (item_count + 1.0),
        }
    return result


_COMPONENT_WEIGHTS = {
    "posterior_uncertainty": 1.5,
    "interval_overlap": 1.3,
    "repeat_disagreement": 1.5,
    "source_undercoverage": 0.65,
    "style_undercoverage": 0.55,
    "family_undercoverage": 0.8,
    "critic_human_disagreement": 1.2,
    "confident_critic_unconfirmed": 0.9,
    "adversarial_optimizer_output": 0.9,
    "comparator_evaluation_impact": 1.1,
    "required_repeat_or_anchor": 1.8,
    "judgment_sparsity": 0.5,
    "direct_pair_evidence_gap": 0.6,
}


def _priority(components: dict[str, float]) -> float:
    return float(
        sum(
            _COMPONENT_WEIGHTS.get(name, 0.0) * _clip01(value) for name, value in components.items()
        )
    )


def _pair_overlap(left: _ItemSignal, right: _ItemSignal) -> float:
    if left.interval is not None and right.interval is not None:
        intersection = min(left.interval[1], right.interval[1]) - max(
            left.interval[0], right.interval[0]
        )
        if intersection <= 0.0:
            return 0.0
        denominator = max(
            1.0e-12,
            min(left.interval[1] - left.interval[0], right.interval[1] - right.interval[0]),
        )
        return _clip01(intersection / denominator)
    if left.quality is not None and right.quality is not None:
        return 1.0 / (1.0 + abs(left.quality - right.quality))
    return 1.0


def _pair_rank_disagreement(
    left: _ItemSignal,
    right: _ItemSignal,
    prior_outcomes: list[str],
) -> float:
    if (
        left.critic_score is None
        or right.critic_score is None
        or not prior_outcomes
        or math.isclose(left.critic_score, right.critic_score, abs_tol=1.0e-12)
    ):
        return 0.0
    decisive = [value for value in prior_outcomes if value in {"first_better", "second_better"}]
    if not decisive:
        return 0.0
    majority = Counter(decisive).most_common(1)[0][0]
    critic = "first_better" if left.critic_score > right.critic_score else "second_better"
    return float(majority != critic)


def _explicit_pair_impact(
    first_id: str,
    second_id: str,
    candidate_rows: dict[str, dict[str, Any]],
) -> float:
    values: list[float] = []
    for source_id, other_id in ((first_id, second_id), (second_id, first_id)):
        raw = candidate_rows[source_id].get("comparator_pair_impacts", {})
        if not isinstance(raw, dict):
            continue
        value = _finite_float(raw.get(other_id))
        if value is not None:
            values.append(_clip01(value))
    return max(values, default=0.0)


def _query_sources(query: dict[str, Any], items: dict[str, _ItemSignal]) -> set[str]:
    return {items[value].source_id for value in query["stimulus_ids"]}


def _query_styles(query: dict[str, Any], items: dict[str, _ItemSignal]) -> set[str]:
    return {items[value].style for value in query["stimulus_ids"]}


def _query_families(query: dict[str, Any], items: dict[str, _ItemSignal]) -> set[str]:
    return {items[value].family for value in query["stimulus_ids"]}


def _spacing_ok(
    query: dict[str, Any],
    position: int,
    latest_item: dict[str, int],
    latest_pair: dict[tuple[str, str], int],
    config: ActiveQueryConfig,
) -> bool:
    for stimulus_id in query["stimulus_ids"]:
        previous = latest_item.get(stimulus_id)
        if previous is not None and position - previous - 1 < config.minimum_repeat_spacing:
            return False
    if query["query_type"] == COMPARE:
        pair = _canonical_pair(query["stimulus_ids"][0], query["stimulus_ids"][1])
        previous = latest_pair.get(pair)
        if previous is not None and position - previous - 1 < config.minimum_repeat_spacing:
            return False
    return True


def _balanced_ok(
    query: dict[str, Any],
    items: dict[str, _ItemSignal],
    selected_sources: Counter[str],
    selected_styles: Counter[str],
    selected_families: Counter[str],
    config: ActiveQueryConfig,
) -> bool:
    return (
        all(
            selected_sources[value] < config.max_recommendations_per_source
            for value in _query_sources(query, items)
        )
        and all(
            selected_styles[value] < config.max_recommendations_per_style
            for value in _query_styles(query, items)
        )
        and all(
            selected_families[value] < config.max_recommendations_per_family
            for value in _query_families(query, items)
        )
    )


def rank_active_human_queries(
    candidates: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    *,
    session_id: str,
    latent_quality: dict[str, Any] | None = None,
    critic_predictions: dict[str, Any] | list[dict[str, Any]] | None = None,
    config: ActiveQueryConfig | None = None,
) -> dict[str, Any]:
    """Rank a bounded set of single-rating and direct-comparison recommendations.

    Candidate and observation inputs are treated as immutable values.  Calibration-only or
    explicitly ineligible candidates are excluded by default, as are non-renderable candidates.
    Deterministically infeasible candidates are excluded unless they are preserved adversarial
    optimizer outputs whose perceptual failure mode is itself the reason to rate them. Critic
    values are prioritization signals only, never labels.
    """
    selected_config = config or ActiveQueryConfig()
    selected_config.validate()
    candidate_copy = [dict(row) for row in candidates]
    all_items = _item_signals(candidate_copy, latent_quality, critic_predictions)
    candidate_rows = {str(row["stimulus_id"]): row for row in candidate_copy}
    items = {
        stimulus_id: item
        for stimulus_id, item in all_items.items()
        if (
            bool(candidate_rows[stimulus_id].get("active_query_eligible", True))
            or (selected_config.include_calibration_only and item.population == "CALIBRATION_ONLY")
        )
        and (
            candidate_rows[stimulus_id].get("deterministic_feasible") is not False
            or candidate_rows[stimulus_id].get("adversarial_optimizer_output") is True
            or (selected_config.include_calibration_only and item.population == "CALIBRATION_ONLY")
        )
        and candidate_rows[stimulus_id].get("renderable") is not False
        and (selected_config.include_calibration_only or item.population != "CALIBRATION_ONLY")
    }
    candidate_rows = {stimulus_id: candidate_rows[stimulus_id] for stimulus_id in items}
    state = _observation_state(observations, items, session_id)
    component_by_item = _score_items(items, candidate_rows, state, selected_config)

    queries: list[dict[str, Any]] = []
    for stimulus_id, item in sorted(items.items()):
        count = int(state["item_counts"][stimulus_id])
        if count >= selected_config.max_observations_per_stimulus:
            continue
        components = component_by_item[stimulus_id]
        queries.append(
            {
                "query_type": RATE_SINGLE,
                "expression": f"RATE_SINGLE({stimulus_id})",
                "stimulus_ids": [stimulus_id],
                "source_ids": [item.source_id],
                "families": [item.family],
                "priority": _priority(components),
                "priority_components": components,
                "existing_observation_count": count,
            }
        )

    pairs = _bounded_pair_candidates(items, candidate_rows, selected_config.pair_neighbor_span)
    for first_id, second_id in sorted(pairs):
        if any(
            int(state["item_counts"][stimulus_id]) >= selected_config.max_observations_per_stimulus
            for stimulus_id in (first_id, second_id)
        ):
            continue
        pair = (first_id, second_id)
        count = int(state["pair_counts"][pair])
        prior_outcomes = state["pair_outcomes"].get(pair, [])
        pair_disagreement = float(state["pair_repeat_disagreement"][pair])
        if count >= selected_config.max_observations_per_pair:
            continue
        first, second = items[first_id], items[second_id]
        left_components = component_by_item[first_id]
        right_components = component_by_item[second_id]
        components = {
            name: (left_components[name] + right_components[name]) / 2.0 for name in left_components
        }
        components["interval_overlap"] = _pair_overlap(first, second)
        components["repeat_disagreement"] = max(
            components["repeat_disagreement"], pair_disagreement
        )
        components["critic_human_disagreement"] = max(
            components["critic_human_disagreement"],
            _pair_rank_disagreement(first, second, prior_outcomes),
        )
        # A required single-item repeat/anchor must not be silently discharged by changing the
        # task into a comparison.  Only an observed pair disagreement makes the pair itself due.
        components["required_repeat_or_anchor"] = float(pair_disagreement > 0.0)
        components["direct_pair_evidence_gap"] = float(count == 0)
        components["comparator_evaluation_impact"] = max(
            components["comparator_evaluation_impact"],
            _explicit_pair_impact(first_id, second_id, candidate_rows),
        )
        queries.append(
            {
                "query_type": COMPARE,
                "expression": f"COMPARE({first_id}, {second_id})",
                "stimulus_ids": [first_id, second_id],
                "source_ids": sorted({first.source_id, second.source_id}),
                "families": sorted({first.family, second.family}),
                "priority": _priority(components),
                "priority_components": components,
                "existing_observation_count": count,
            }
        )

    queries.sort(key=lambda row: (-float(row["priority"]), str(row["expression"])))
    selected: list[dict[str, Any]] = []
    selected_type: Counter[str] = Counter()
    existing_sources: Counter[str] = Counter(state["session_source"])
    existing_styles: Counter[str] = Counter(state["session_style"])
    existing_families: Counter[str] = Counter(state["session_family"])
    balanced_sources: Counter[str] = existing_sources.copy()
    balanced_styles: Counter[str] = existing_styles.copy()
    balanced_families: Counter[str] = existing_families.copy()
    recommended_sources: Counter[str] = Counter()
    recommended_styles: Counter[str] = Counter()
    recommended_families: Counter[str] = Counter()
    existing_item_counts: Counter[str] = Counter(state["item_counts"])
    projected_item_counts: Counter[str] = existing_item_counts.copy()
    recommended_item_counts: Counter[str] = Counter()
    latest_item = dict(state["last_item_position"])
    latest_pair = dict(state["last_pair_position"])
    unused = list(queries)
    type_maximum = {
        RATE_SINGLE: selected_config.max_rate_single,
        COMPARE: selected_config.max_compare,
    }
    type_minimum = {
        RATE_SINGLE: selected_config.minimum_rate_single,
        COMPARE: selected_config.minimum_compare,
    }
    while unused and len(selected) < selected_config.max_recommendations:
        position = int(state["session_observation_count"]) + len(selected)
        remaining_slots = selected_config.max_recommendations - len(selected)
        unmet = {
            query_type
            for query_type, minimum in type_minimum.items()
            if selected_type[query_type] < minimum
            and any(row["query_type"] == query_type for row in unused)
        }
        forced_types = (
            unmet
            if remaining_slots <= sum(type_minimum[value] - selected_type[value] for value in unmet)
            else set(type_minimum)
        )
        chosen_index = None
        for index, query in enumerate(unused):
            query_type = str(query["query_type"])
            if (
                query_type not in forced_types
                or selected_type[query_type] >= type_maximum[query_type]
            ):
                continue
            if not _spacing_ok(query, position, latest_item, latest_pair, selected_config):
                continue
            if any(
                projected_item_counts[stimulus_id] >= selected_config.max_observations_per_stimulus
                for stimulus_id in query["stimulus_ids"]
            ):
                continue
            if not _balanced_ok(
                query,
                items,
                balanced_sources,
                balanced_styles,
                balanced_families,
                selected_config,
            ):
                continue
            chosen_index = index
            break
        if chosen_index is None:
            break
        query = unused.pop(chosen_index)
        query["scheduled_position"] = position
        query["priority_reasons"] = [
            name
            for name, value in sorted(
                query["priority_components"].items(),
                key=lambda pair: (-_COMPONENT_WEIGHTS.get(pair[0], 0.0) * pair[1], pair[0]),
            )
            if value > 0.0
        ]
        selected.append(query)
        selected_type[str(query["query_type"])] += 1
        for value in _query_sources(query, items):
            balanced_sources[value] += 1
            recommended_sources[value] += 1
        for value in _query_styles(query, items):
            balanced_styles[value] += 1
            recommended_styles[value] += 1
        for value in _query_families(query, items):
            balanced_families[value] += 1
            recommended_families[value] += 1
        for stimulus_id in query["stimulus_ids"]:
            projected_item_counts[stimulus_id] += 1
            recommended_item_counts[stimulus_id] += 1
            latest_item[stimulus_id] = position
        if query["query_type"] == COMPARE:
            latest_pair[_canonical_pair(query["stimulus_ids"][0], query["stimulus_ids"][1])] = (
                position
            )

    return {
        "format_version": ACTIVE_QUERY_SCHEDULE_VERSION,
        "status": "ready" if selected else "no_eligible_queries",
        "session_id": session_id,
        "recommendations": selected,
        "recommendation_counts": {
            RATE_SINGLE: selected_type[RATE_SINGLE],
            COMPARE: selected_type[COMPARE],
        },
        "candidate_audit": {
            "input_stimulus_count": len(candidates),
            "eligible_stimulus_count": len(items),
            "bounded_pair_candidate_count": len(pairs),
            "all_pairs_were_enumerated": False,
            "pair_construction": "bounded_global_source_and_family_nearest_neighbors",
            "pair_neighbor_span": selected_config.pair_neighbor_span,
        },
        "constraint_audit": {
            "minimum_repeat_spacing": selected_config.minimum_repeat_spacing,
            "source_counts": dict(sorted(balanced_sources.items())),
            "style_counts": dict(sorted(balanced_styles.items())),
            "family_counts": dict(sorted(balanced_families.items())),
            "existing_session_source_counts": dict(sorted(existing_sources.items())),
            "existing_session_style_counts": dict(sorted(existing_styles.items())),
            "existing_session_family_counts": dict(sorted(existing_families.items())),
            "global_source_counts": dict(sorted(state["global_source"].items())),
            "global_style_counts": dict(sorted(state["global_style"].items())),
            "global_family_counts": dict(sorted(state["global_family"].items())),
            "recommended_source_counts": dict(sorted(recommended_sources.items())),
            "recommended_style_counts": dict(sorted(recommended_styles.items())),
            "recommended_family_counts": dict(sorted(recommended_families.items())),
            "existing_item_counts": dict(sorted(existing_item_counts.items())),
            "recommended_item_counts": dict(sorted(recommended_item_counts.items())),
            "projected_item_counts": dict(sorted(projected_item_counts.items())),
            "max_observations_per_stimulus": selected_config.max_observations_per_stimulus,
            "max_recommendations_per_source": selected_config.max_recommendations_per_source,
            "max_recommendations_per_style": selected_config.max_recommendations_per_style,
            "max_recommendations_per_family": selected_config.max_recommendations_per_family,
            "session_balancing_used_existing_session_counts": True,
        },
        "priority_policy": {
            "component_weights": dict(sorted(_COMPONENT_WEIGHTS.items())),
            "critic_predictions_are_prioritization_only_not_labels": True,
            "dummy_or_partial_human_inputs_are_not_substantive_findings": True,
        },
        "config": asdict(selected_config),
    }


def _load_json_or_jsonl(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        if not path.is_file():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_active_query_schedule(
    candidate_pool_path: Path,
    observations_path: Path,
    output_path: Path,
    *,
    session_id: str,
    latent_quality_path: Path | None = None,
    critic_predictions_path: Path | None = None,
    config: ActiveQueryConfig | None = None,
) -> dict[str, Any]:
    """Read analysis inputs and write a recommendation artifact, leaving inputs untouched."""
    output = Path(output_path)
    protected_inputs = {
        Path(value).resolve()
        for value in (
            candidate_pool_path,
            observations_path,
            latent_quality_path,
            critic_predictions_path,
        )
        if value is not None
    }
    if output.resolve() in protected_inputs:
        raise ValueError("active-query output must not overwrite any input artifact")
    if output.exists():
        raise ValueError(f"active-query output must be fresh: {output}")
    raw_candidates = _load_json_or_jsonl(Path(candidate_pool_path))
    if isinstance(raw_candidates, dict):
        raw_candidates = raw_candidates.get("candidates", raw_candidates.get("stimuli"))
    if not isinstance(raw_candidates, list) or not all(
        isinstance(row, dict) for row in raw_candidates
    ):
        raise ValueError("candidate pool must be a JSON/JSONL list or contain candidates/stimuli")
    raw_observations = _load_json_or_jsonl(Path(observations_path))
    if not isinstance(raw_observations, list) or not all(
        isinstance(row, dict) for row in raw_observations
    ):
        raise ValueError("observations must be an append-only JSONL/JSON list")
    latent = None if latent_quality_path is None else _load_json_or_jsonl(latent_quality_path)
    if latent is not None and not isinstance(latent, dict):
        raise ValueError("latent quality artifact must be a JSON object")
    predictions = (
        None if critic_predictions_path is None else _load_json_or_jsonl(critic_predictions_path)
    )
    if predictions is not None and not isinstance(predictions, (dict, list)):
        raise ValueError("critic predictions must be a JSON object or JSONL list")
    report = rank_active_human_queries(
        raw_candidates,
        raw_observations,
        session_id=session_id,
        latent_quality=latent,
        critic_predictions=predictions,
        config=config,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
