from __future__ import annotations

import numpy as np

from motionlab.optimization.benchmark_semantics import prepare_synthetic_production_benchmark
from motionlab.optimization.cmaes import CMAOptimizationConfig
from motionlab.testing.synthetic import make_synthetic_contact_walk


def test_synthetic_benchmark_uses_clean_window_speed_not_parent_target() -> None:
    base = make_synthetic_contact_walk(num_cycles=3)
    parent_speed = float(base.metadata["expected_speed_mps"])
    clean = base.slice_frames(17, 137)
    root = clean.root_translation_m.copy()
    root[30:80, 1] += 0.02
    corrupted = clean.with_updates(root_translation_m=root)
    pair = prepare_synthetic_production_benchmark(
        clean,
        corrupted,
        CMAOptimizationConfig(mode="production"),
        target_speed_tolerance=0.01,
    )
    assert pair.eligible
    assert pair.task_target_speed is None
    assert pair.parent_clip_speed == parent_speed
    assert pair.target_speed == pair.clean_window_reference_speed
    assert pair.clean.metadata["target_speed_source"] == "measured_clean_window_speed"
    assert pair.clean_endpoint_constraint_reasons == ()
    pair.assert_clean_endpoint_feasible()


def test_external_task_target_mismatch_makes_pair_ineligible_without_relaxing_constraints() -> None:
    clean = make_synthetic_contact_walk(num_cycles=2)
    corrupted = clean.copy()
    pair = prepare_synthetic_production_benchmark(
        clean,
        corrupted,
        CMAOptimizationConfig(mode="production"),
        task_target_speed=0.4,
        target_speed_tolerance=0.01,
    )
    assert not pair.eligible
    assert "clean_reference_misses_external_task_speed" in pair.ineligibility_reasons
    assert "target_speed" in pair.clean_endpoint_constraint_reasons
    assert pair.target_speed_tolerance == 0.01
    assert not np.isclose(pair.target_speed, pair.clean_window_reference_speed)
