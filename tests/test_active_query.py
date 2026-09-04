from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from motionlab.perceptual.active_query import (
    COMPARE,
    RATE_SINGLE,
    ActiveQueryConfig,
    prepare_active_query_schedule,
    rank_active_human_queries,
)


def _candidates(count: int = 16) -> list[dict[str, Any]]:
    result = []
    for index in range(count):
        result.append(
            {
                "stimulus_id": f"stimulus-{index:03d}",
                "source_id": f"source-{index % 6}",
                "style": f"style-{index % 4}",
                "family": f"family-{index % 5}",
                "stimulus_population": (
                    "EVALUATION_ANCHOR" if index == 0 else "HARD_FEASIBLE_PERCEPTUAL"
                ),
                "deterministic_feasible": True,
                "renderable": True,
                "required_anchor": index == 0,
                "required_repeat": index == 1,
                "repeat_target": 3 if index == 1 else None,
                "adversarial_optimizer_output": index == 2,
                "comparator_impact": 1.0 if index == 3 else 0.1,
                "held_out_source": index == 4,
            }
        )
    return result


def _latent(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "fit",
        "item_quality": [
            {
                "stimulus_id": row["stimulus_id"],
                "latent_quality": float(index),
                "quality_95_percent_interval": [
                    float(index) - (3.0 if index == 5 else 0.25),
                    float(index) + (3.0 if index == 5 else 0.25),
                ],
            }
            for index, row in enumerate(candidates)
        ],
    }


def _observations(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # stimulus-001 has an old, disagreeing repeat and stimulus-006 was just seen. Unknown filler
    # IDs are intentional: session position, not recognized-observation count, defines spacing.
    rows: list[dict[str, Any]] = [
        {
            "task_type": "naturalness",
            "stimulus_ids": ["stimulus-001"],
            "rating": 2,
            "session_id": "session-active",
            "rater_id": "rater-repeat",
            "hidden_repeat_group_ids": ["repeat-stimulus-001"],
        },
        {
            "task_type": "naturalness",
            "stimulus_ids": ["stimulus-001"],
            "rating": 7,
            "session_id": "session-old",
            "rater_id": "rater-repeat",
            "hidden_repeat_group_ids": ["repeat-stimulus-001"],
        },
    ]
    rows.extend(
        {
            "task_type": "naturalness",
            "stimulus_ids": [f"filler-{index}"],
            "rating": 4,
            "session_id": "session-active",
        }
        for index in range(15)
    )
    rows.append(
        {
            "task_type": "naturalness",
            "stimulus_ids": [candidates[6]["stimulus_id"]],
            "rating": 4,
            "session_id": "session-active",
        }
    )
    return rows


def test_active_query_ranking_is_bounded_balanced_and_deterministic() -> None:
    candidates = _candidates()
    candidates.append(
        {
            "stimulus_id": "calibration-excluded",
            "source_id": "calibration",
            "style": "calibration",
            "family": "foot_slide",
            "stimulus_population": "CALIBRATION_ONLY",
            "deterministic_feasible": True,
            "renderable": True,
        }
    )
    candidates.append(
        {
            "stimulus_id": "infeasible-excluded",
            "source_id": "bad",
            "style": "bad",
            "family": "bad",
            "deterministic_feasible": False,
            "renderable": True,
        }
    )
    candidates.append(
        {
            "stimulus_id": "infeasible-adversarial-eligible",
            "source_id": "adversarial-source",
            "style": "adversarial-style",
            "family": "preserved_critic_exploit",
            "deterministic_feasible": False,
            "renderable": True,
            "adversarial_optimizer_output": True,
        }
    )
    observations = _observations(candidates)
    predictions = {
        row["stimulus_id"]: {
            "score": float(len(candidates) - index),
            "confidence": 0.95,
        }
        for index, row in enumerate(candidates)
    }
    config = ActiveQueryConfig(
        max_recommendations=8,
        max_rate_single=5,
        max_compare=3,
        minimum_repeat_spacing=12,
        max_recommendations_per_source=2,
        max_recommendations_per_style=4,
        max_recommendations_per_family=4,
    )
    first = rank_active_human_queries(
        candidates,
        observations,
        session_id="session-active",
        latent_quality=_latent(candidates),
        critic_predictions=predictions,
        config=config,
    )
    second = rank_active_human_queries(
        list(reversed(candidates)),
        observations,
        session_id="session-active",
        latent_quality=_latent(candidates),
        critic_predictions=predictions,
        config=config,
    )

    assert first == second
    assert first["status"] == "ready"
    assert {row["query_type"] for row in first["recommendations"]} == {
        RATE_SINGLE,
        COMPARE,
    }
    assert len(first["recommendations"]) <= 8
    scheduled_ids = {
        stimulus_id for row in first["recommendations"] for stimulus_id in row["stimulus_ids"]
    }
    assert "calibration-excluded" not in scheduled_ids
    assert "infeasible-excluded" not in scheduled_ids
    assert "infeasible-adversarial-eligible" in scheduled_ids
    assert "stimulus-006" not in scheduled_ids
    assert first["candidate_audit"]["eligible_stimulus_count"] == len(_candidates()) + 1
    assert max(first["constraint_audit"]["source_counts"].values()) <= 2
    audit = first["candidate_audit"]
    assert audit["all_pairs_were_enumerated"] is False
    assert audit["bounded_pair_candidate_count"] <= 4 * audit["eligible_stimulus_count"] * 2
    required_repeat = next(
        row
        for row in first["recommendations"]
        if row["query_type"] == RATE_SINGLE and row["stimulus_ids"] == ["stimulus-001"]
    )
    assert required_repeat["priority_components"]["repeat_disagreement"] > 0.8
    assert required_repeat["priority_components"]["required_repeat_or_anchor"] == 1.0
    assert first["priority_policy"]["critic_predictions_are_prioritization_only_not_labels"]


def test_pair_graph_grows_linearly_not_quadratically() -> None:
    candidates = _candidates(600)
    report = rank_active_human_queries(
        candidates,
        [],
        session_id="new-session",
        config=ActiveQueryConfig(
            max_recommendations=6,
            max_rate_single=3,
            max_compare=3,
            pair_neighbor_span=2,
        ),
    )
    pair_count = report["candidate_audit"]["bounded_pair_candidate_count"]
    assert pair_count < len(candidates) * 8
    assert pair_count < len(candidates) * (len(candidates) - 1) // 2


def test_undercoverage_uses_all_prior_sessions_not_only_active_session() -> None:
    candidates = _candidates(2)
    candidates[1]["source_id"] = "never-seen-source"
    candidates[1]["style"] = "never-seen-style"
    candidates[1]["family"] = "never-seen-family"
    observations = [
        {
            "task_type": "naturalness",
            "stimulus_ids": ["stimulus-000"],
            "rating": 4,
            "session_id": "prior-session",
        }
    ]
    report = rank_active_human_queries(
        candidates,
        observations,
        session_id="active-session",
        config=ActiveQueryConfig(
            max_recommendations=2,
            max_rate_single=2,
            max_compare=0,
            minimum_rate_single=2,
            minimum_compare=0,
            required_anchors_per_session=0,
        ),
    )
    by_id = {
        row["stimulus_ids"][0]: row["priority_components"] for row in report["recommendations"]
    }

    assert by_id["stimulus-000"]["source_undercoverage"] < 1.0
    assert by_id["stimulus-001"]["source_undercoverage"] == 1.0
    assert report["constraint_audit"]["global_source_counts"] == {"source-0": 1}


def test_only_within_rater_explicit_hidden_repeats_count_as_repeat_disagreement() -> None:
    candidate = _candidates(1)[0]
    base = {
        "task_type": "naturalness",
        "stimulus_ids": ["stimulus-000"],
        "session_id": "prior-session",
        "hidden_repeat_group_ids": ["repeat-000"],
    }
    config = ActiveQueryConfig(
        max_recommendations=1,
        max_rate_single=1,
        max_compare=0,
        minimum_rate_single=1,
        minimum_compare=0,
        max_observations_per_stimulus=3,
        required_anchors_per_session=0,
    )
    cross_rater = rank_active_human_queries(
        [candidate],
        [
            {**base, "rating": 1, "rater_id": "rater-a"},
            {**base, "rating": 7, "rater_id": "rater-b"},
        ],
        session_id="active-session",
        config=config,
    )
    within_rater = rank_active_human_queries(
        [candidate],
        [
            {**base, "rating": 1, "rater_id": "rater-a"},
            {**base, "rating": 7, "rater_id": "rater-a"},
        ],
        session_id="active-session",
        config=config,
    )

    assert cross_rater["recommendations"][0]["priority_components"]["repeat_disagreement"] == 0.0
    assert within_rater["recommendations"][0]["priority_components"]["repeat_disagreement"] == 1.0


def test_pair_repeat_disagreement_requires_matching_hidden_repeat_group() -> None:
    candidates = _candidates(2)
    base = {
        "task_type": "pair_comparison",
        "stimulus_ids": ["stimulus-000", "stimulus-001"],
        "session_id": "prior-session",
        "rater_id": "rater-a",
    }
    config = ActiveQueryConfig(
        max_recommendations=1,
        max_rate_single=0,
        max_compare=1,
        minimum_rate_single=0,
        minimum_compare=1,
        max_observations_per_stimulus=3,
        max_observations_per_pair=3,
        required_anchors_per_session=0,
    )
    ordinary = rank_active_human_queries(
        candidates,
        [
            {**base, "pair_outcome_canonical": "first_better"},
            {**base, "pair_outcome_canonical": "second_better"},
        ],
        session_id="active-session",
        config=config,
    )
    repeated = rank_active_human_queries(
        candidates,
        [
            {
                **base,
                "pair_outcome_canonical": "first_better",
                "hidden_repeat_group_ids": ["repeat-a", "repeat-b"],
            },
            {
                **base,
                "pair_outcome_canonical": "second_better",
                "hidden_repeat_group_ids": ["repeat-b", "repeat-a"],
            },
        ],
        session_id="active-session",
        config=config,
    )

    assert ordinary["recommendations"][0]["priority_components"]["repeat_disagreement"] == 0.0
    assert repeated["recommendations"][0]["priority_components"]["repeat_disagreement"] == 0.5


def test_calibration_only_candidates_require_explicit_opt_in() -> None:
    candidate = {
        **_candidates(1)[0],
        "stimulus_population": "CALIBRATION_ONLY",
        "active_query_eligible": False,
        "deterministic_feasible": False,
    }
    common = dict(
        max_recommendations=1,
        max_rate_single=1,
        max_compare=0,
        minimum_rate_single=1,
        minimum_compare=0,
        required_anchors_per_session=0,
    )

    excluded = rank_active_human_queries(
        [candidate],
        [],
        session_id="active-session",
        config=ActiveQueryConfig(**common),
    )
    included = rank_active_human_queries(
        [candidate],
        [],
        session_id="active-session",
        config=ActiveQueryConfig(**common, include_calibration_only=True),
    )

    assert excluded["status"] == "no_eligible_queries"
    assert included["recommendations"][0]["stimulus_ids"] == ["stimulus-000"]


def test_boolean_required_repeat_is_satisfied_after_one_repeat() -> None:
    candidate = {
        "stimulus_id": "repeat-me",
        "source_id": "source-repeat",
        "style": "style-repeat",
        "family": "family-repeat",
        "deterministic_feasible": True,
        "renderable": True,
        "required_repeat": True,
    }
    config = ActiveQueryConfig(
        max_recommendations=1,
        max_rate_single=1,
        max_compare=0,
        minimum_rate_single=1,
        minimum_compare=0,
        minimum_repeat_spacing=0,
        max_observations_per_stimulus=3,
        required_anchors_per_session=0,
    )
    first_observation = {
        "task_type": "naturalness",
        "stimulus_ids": ["repeat-me"],
        "rating": 3,
        "session_id": "session-repeat",
    }

    due = rank_active_human_queries(
        [candidate],
        [first_observation],
        session_id="session-repeat",
        config=config,
    )
    assert due["recommendations"][0]["priority_components"][
        "required_repeat_or_anchor"
    ] == pytest.approx(1.0)

    complete = rank_active_human_queries(
        [candidate],
        [first_observation, dict(first_observation, rating=4)],
        session_id="session-repeat",
        config=config,
    )
    assert complete["recommendations"][0]["priority_components"][
        "required_repeat_or_anchor"
    ] == pytest.approx(0.0)


def test_repeat_target_cannot_bypass_hard_observation_cap() -> None:
    candidate = {
        "stimulus_id": "repeat-me",
        "source_id": "source-repeat",
        "style": "style-repeat",
        "family": "family-repeat",
        "repeat_target": 10,
    }
    observation = {
        "task_type": "naturalness",
        "stimulus_ids": ["repeat-me"],
        "rating": 3,
        "session_id": "session-repeat",
    }
    config = ActiveQueryConfig(
        max_recommendations=1,
        max_rate_single=1,
        max_compare=0,
        minimum_rate_single=1,
        minimum_compare=0,
        minimum_repeat_spacing=0,
        max_observations_per_stimulus=2,
        required_anchors_per_session=0,
    )

    due = rank_active_human_queries(
        [candidate],
        [observation],
        session_id="session-repeat",
        config=config,
    )
    assert due["recommendations"][0]["priority_components"][
        "required_repeat_or_anchor"
    ] == pytest.approx(1.0)

    capped = rank_active_human_queries(
        [candidate],
        [observation, dict(observation, rating=4)],
        session_id="session-repeat",
        config=config,
    )
    assert capped["status"] == "no_eligible_queries"
    assert capped["recommendations"] == []


def test_hard_balance_caps_include_existing_same_session_exposure() -> None:
    candidates = [
        {
            "stimulus_id": "seen",
            "source_id": "source-full",
            "style": "style-full",
            "family": "family-full",
        },
        {
            "stimulus_id": "blocked-source",
            "source_id": "source-full",
            "style": "style-new",
            "family": "family-new",
        },
        {
            "stimulus_id": "blocked-style",
            "source_id": "source-style",
            "style": "style-full",
            "family": "family-new",
        },
        {
            "stimulus_id": "blocked-family",
            "source_id": "source-family",
            "style": "style-family",
            "family": "family-full",
        },
        {
            "stimulus_id": "other-source",
            "source_id": "source-open",
            "style": "style-open",
            "family": "family-open",
        },
    ]
    observations = [
        {
            "task_type": "naturalness",
            "stimulus_ids": ["seen"],
            "rating": 4,
            "session_id": "active-session",
        }
    ]
    report = rank_active_human_queries(
        candidates,
        observations,
        session_id="active-session",
        config=ActiveQueryConfig(
            max_recommendations=1,
            max_rate_single=1,
            max_compare=0,
            minimum_rate_single=1,
            minimum_compare=0,
            minimum_repeat_spacing=0,
            max_recommendations_per_source=1,
            max_recommendations_per_style=1,
            max_recommendations_per_family=1,
            required_anchors_per_session=0,
        ),
    )

    assert report["recommendations"][0]["stimulus_ids"] == ["other-source"]
    audit = report["constraint_audit"]
    assert audit["existing_session_source_counts"] == {"source-full": 1}
    assert audit["existing_session_style_counts"] == {"style-full": 1}
    assert audit["existing_session_family_counts"] == {"family-full": 1}
    assert audit["recommended_source_counts"] == {"source-open": 1}
    assert audit["recommended_style_counts"] == {"style-open": 1}
    assert audit["recommended_family_counts"] == {"family-open": 1}
    assert audit["source_counts"] == {"source-full": 1, "source-open": 1}


def test_repeat_spacing_ignores_interleaved_observations_from_other_sessions() -> None:
    candidate = {
        "stimulus_id": "spaced-repeat",
        "source_id": "source-repeat",
        "style": "style-repeat",
        "family": "family-repeat",
        "required_repeat": True,
    }
    first = {
        "task_type": "naturalness",
        "stimulus_ids": ["spaced-repeat"],
        "rating": 3,
        "session_id": "active-session",
    }
    other_session_rows = [
        {
            "task_type": "naturalness",
            "stimulus_ids": [f"other-{index}"],
            "rating": 4,
            "session_id": "other-session",
        }
        for index in range(12)
    ]
    config = ActiveQueryConfig(
        max_recommendations=1,
        max_rate_single=1,
        max_compare=0,
        minimum_rate_single=1,
        minimum_compare=0,
        minimum_repeat_spacing=12,
        required_anchors_per_session=0,
    )

    interleaved = rank_active_human_queries(
        [candidate],
        [first, *other_session_rows],
        session_id="active-session",
        config=config,
    )
    assert interleaved["status"] == "no_eligible_queries"

    same_session_rows = [dict(row, session_id="active-session") for row in other_session_rows]
    spaced = rank_active_human_queries(
        [candidate],
        [first, *same_session_rows],
        session_id="active-session",
        config=config,
    )
    assert spaced["recommendations"][0]["stimulus_ids"] == ["spaced-repeat"]
    assert spaced["recommendations"][0]["scheduled_position"] == 13


def test_disagreeing_pair_cannot_bypass_hard_pair_observation_cap() -> None:
    candidates = [
        {
            "stimulus_id": "pair-a",
            "source_id": "source-a",
            "style": "style-a",
            "family": "family-a",
        },
        {
            "stimulus_id": "pair-b",
            "source_id": "source-b",
            "style": "style-b",
            "family": "family-b",
        },
    ]
    observations = [
        {
            "task_type": "pair_comparison",
            "stimulus_ids": ["pair-a", "pair-b"],
            "pair_outcome_canonical": outcome,
            "session_id": "active-session",
        }
        for outcome in ("first_better", "second_better")
    ]
    report = rank_active_human_queries(
        candidates,
        observations,
        session_id="active-session",
        config=ActiveQueryConfig(
            max_recommendations=1,
            max_rate_single=0,
            max_compare=1,
            minimum_rate_single=0,
            minimum_compare=1,
            minimum_repeat_spacing=0,
            max_observations_per_pair=2,
            required_anchors_per_session=0,
        ),
    )

    assert report["status"] == "no_eligible_queries"
    assert report["recommendations"] == []


def test_pair_cannot_use_a_stimulus_already_at_its_hard_cap() -> None:
    candidates = _candidates(2)
    observation = {
        "task_type": "naturalness",
        "stimulus_ids": ["stimulus-000"],
        "rating": 4,
        "session_id": "prior-session",
    }
    report = rank_active_human_queries(
        candidates,
        [observation],
        session_id="active-session",
        config=ActiveQueryConfig(
            max_recommendations=1,
            max_rate_single=0,
            max_compare=1,
            minimum_rate_single=0,
            minimum_compare=1,
            max_observations_per_stimulus=1,
            required_anchors_per_session=0,
        ),
    )

    assert report["status"] == "no_eligible_queries"
    assert report["recommendations"] == []


def test_batch_cannot_overcommit_per_stimulus_observation_cap() -> None:
    candidates = _candidates(6)
    report = rank_active_human_queries(
        candidates,
        [],
        session_id="active-session",
        config=ActiveQueryConfig(
            max_recommendations=3,
            max_rate_single=0,
            max_compare=3,
            minimum_rate_single=0,
            minimum_compare=1,
            max_observations_per_stimulus=1,
            max_recommendations_per_source=6,
            max_recommendations_per_style=6,
            max_recommendations_per_family=6,
            required_anchors_per_session=0,
        ),
    )

    recommended_counts = report["constraint_audit"]["recommended_item_counts"]
    assert recommended_counts
    assert max(recommended_counts.values()) == 1
    assert max(report["constraint_audit"]["projected_item_counts"].values()) == 1


def test_explicit_comparator_impact_adds_only_a_bounded_pair() -> None:
    candidates = _candidates(4)
    candidates[0]["comparator_pair_impacts"] = {"stimulus-003": 1.0}
    report = rank_active_human_queries(
        candidates,
        [],
        session_id="new-session",
        latent_quality=_latent(candidates),
        config=ActiveQueryConfig(
            max_recommendations=1,
            max_rate_single=0,
            max_compare=1,
            minimum_rate_single=0,
            minimum_compare=1,
            pair_neighbor_span=1,
        ),
    )
    recommendation = report["recommendations"][0]
    assert recommendation["query_type"] == COMPARE
    assert recommendation["stimulus_ids"] == ["stimulus-000", "stimulus-003"]
    assert recommendation["priority_components"]["comparator_evaluation_impact"] == 1.0


def test_schedule_file_api_does_not_mutate_inputs(tmp_path: Path) -> None:
    candidates = _candidates(10)
    pool_path = tmp_path / "candidate_pool.json"
    observations_path = tmp_path / "observations.jsonl"
    latent_path = tmp_path / "latent.json"
    output_path = tmp_path / "derived" / "active_queries.json"
    pool_path.write_text(json.dumps({"candidates": candidates}), encoding="utf-8")
    observations_path.write_text("", encoding="utf-8")
    latent_path.write_text(json.dumps(_latent(candidates)), encoding="utf-8")
    before_pool = pool_path.read_bytes()
    before_observations = observations_path.read_bytes()

    result = prepare_active_query_schedule(
        pool_path,
        observations_path,
        output_path,
        session_id="session-test",
        latent_quality_path=latent_path,
        config=ActiveQueryConfig(
            max_recommendations=4,
            max_rate_single=2,
            max_compare=2,
        ),
    )

    assert output_path.is_file()
    assert json.loads(output_path.read_text()) == result
    assert pool_path.read_bytes() == before_pool
    assert observations_path.read_bytes() == before_observations
    assert result["priority_policy"]["dummy_or_partial_human_inputs_are_not_substantive_findings"]

    with pytest.raises(ValueError, match="output must be fresh"):
        prepare_active_query_schedule(
            pool_path,
            observations_path,
            output_path,
            session_id="session-test",
        )

    with pytest.raises(ValueError, match="must not overwrite"):
        prepare_active_query_schedule(
            pool_path,
            observations_path,
            observations_path,
            session_id="session-test",
        )
    assert observations_path.read_bytes() == before_observations
