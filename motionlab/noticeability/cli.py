"""Noticeability workflow commands, separate from the frozen legacy collector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from motionlab.noticeability import opt_in_inspection, repeated_protocol, repeated_server
from motionlab.noticeability.analysis import analyze
from motionlab.noticeability.legacy import freeze_legacy_protocols
from motionlab.noticeability.migration import fit_calibration, migrate, validate_migration
from motionlab.noticeability.pilot import build_pilot, build_staircase_pilot
from motionlab.noticeability.protocol import freeze_pilot, read_json
from motionlab.noticeability.repeated_continuation import (
    analyze_repeated,
    prepare_repeated_view_pilot,
)
from motionlab.noticeability.server import serve_pilot
from motionlab.noticeability.workflow import export_training, fixed_validation_plan


def register_commands(app: typer.Typer) -> None:
    def emit(value: object) -> None:
        typer.echo(json.dumps(value, sort_keys=True, indent=2))

    @app.command("freeze-human-protocol")
    def freeze(output: Path, pilots: Annotated[list[Path], typer.Argument()]) -> None:
        """Losslessly snapshot paused historical quality protocols and compatibility records."""
        r = freeze_legacy_protocols(pilots, output)
        emit(
            {
                k: r[k]
                for k in (
                    "registry_id",
                    "raw_observation_count",
                    "old_score_distribution",
                    "pairwise_observation_count",
                )
            }
        )

    @app.command("build-noticeability-overlap")
    def overlap(registry: Path, output: Path, count: int = 40) -> None:
        """Prepare only the blinded exact-render overlap set; freeze it before serving."""
        r = build_pilot(registry, output, overlap_count=count, include_threshold=False)
        emit(r["overlap_selection"])

    @app.command("build-noticeability-threshold-pilot")
    def build(
        registry: Path, output: Path, overlap_count: int = 40, sources: int = 2, seed: int = 9601
    ) -> None:
        """Small combined overlap + fixed exploratory threshold screen, without human labels."""
        r = build_pilot(
            registry, output, overlap_count=overlap_count, source_count=sources, seed=seed
        )
        emit(
            {
                "trial_count": len(r["trials"]),
                "stimulus_count": len(r["stimuli"]),
                "retraining": False,
            }
        )

    @app.command("freeze-noticeability-pilot")
    def freeze_new(pilot: Path) -> None:
        if read_json(pilot)["protocol_version"] == repeated_protocol.PROTOCOL:
            emit(repeated_protocol.freeze_pilot(pilot))
        else:
            emit(freeze_pilot(pilot))

    @app.command("prepare-repeated-view-noticeability")
    def repeated_view(parent: Path, output: Path) -> None:
        """Preserve a paused single-view pilot and continue with up to three mild-speed viewings."""
        r = prepare_repeated_view_pilot(parent, output)
        emit({"trials": len(r["trials"]), "protocol_version": r["protocol_version"]})

    @app.command("prepare-opt-in-inspection")
    def opt_in(parent: Path, output: Path) -> None:
        """Preserve previous progress and continue with inspection hidden until requested."""
        r = opt_in_inspection.prepare_pilot(parent, output)
        emit({"trials": len(r["trials"]), "inspection": "explicit_opt_in"})

    @app.command("build-noticeability-staircase-pilot")
    def staircase(parent: Path, output: Path, steps: int = 30, seed: int = 9621) -> None:
        """Optional interleaved staircase with clean/sham controls, separate from validation."""
        r = build_staircase_pilot(parent, output, steps=steps, seed=seed)
        emit({"scheduled_steps": r["scheduler"]["steps"], "freeze_before_serving": True})

    @app.command("serve-noticeability-pilot")
    def serve(pilot: Path, port: int = 8765) -> None:
        manifest = read_json(pilot)
        if manifest.get("inspection_presentation", {}).get("mode") == "explicit_opt_in":
            opt_in_inspection.serve_pilot(pilot, port=port)
        elif manifest["protocol_version"] == repeated_protocol.PROTOCOL:
            repeated_server.serve_pilot(pilot, port=port)
        else:
            serve_pilot(pilot, port=port)

    @app.command("fit-legacy-noticeability-calibration")
    def calibration(registry: Path, pilot: Path, output: Path) -> None:
        emit(fit_calibration(registry, pilot, output))

    @app.command("migrate-legacy-labels")
    def migrate_command(registry: Path, model: Path, output: Path) -> None:
        r = migrate(registry, model, output)
        emit({"status": r["status"], "proxy_count": len(r["derived_records"])})

    @app.command("validate-label-migration")
    def validate(registry: Path, model: Path, migration: Path) -> None:
        emit(validate_migration(registry, model, migration))

    @app.command("analyze-noticeability")
    def analysis(pilot: Path, output: Path) -> None:
        if read_json(pilot)["protocol_version"] == repeated_protocol.PROTOCOL:
            emit(analyze_repeated(pilot, output))
        else:
            emit(analyze(pilot, output))

    @app.command("export-noticeability-training")
    def export(pilot: Path, output: Path) -> None:
        r = export_training(pilot, output)
        emit({"records": len(r["records"]), "review_before_training": True})

    @app.command("build-noticeability-validation-plan")
    def validation_plan(pilot: Path, output: Path) -> None:
        emit(fixed_validation_plan(pilot, output))

    @app.command("train-noticeability-critic")
    def train() -> None:
        """Intentional first-milestone stop gate, not a training implementation."""
        typer.echo(
            "STOP: review direct-human overlap/threshold results first. The clip head is "
            "implemented, but training orchestration and production activation are "
            "deliberately deferred. No training launched."
        )
        raise typer.Exit(1)

    @app.command("evaluate-noticeability-critic")
    def evaluate() -> None:
        """First-milestone gate: there is no trained noticeability checkpoint yet."""
        typer.echo(
            "No trained noticeability critic exists. Use analyze-noticeability for human "
            "pilot results. Detector metrics and held-out tau selection utilities are "
            "available for the reviewed next stage."
        )
        raise typer.Exit(1)
