#!/usr/bin/env python3
"""Run the complete resumable Phase 2 feature-gap experiment."""

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from vlm_bench.io import write_json


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase2-feature-gap-qwen25-vl-3b"))
    parser.add_argument("--development-manifest", type=Path, default=Path("data/processed-v2/manifests/development.jsonl"))
    parser.add_argument("--data-root", type=Path, default=Path("data/processed-v2"))
    parser.add_argument("--routes", type=Path, default=Path("results/task-route-design-qwen25-vl-3b/routes.json"))
    parser.add_argument(
        "--baseline-predictions",
        type=Path,
        default=Path("results/task-route-design-qwen25-vl-3b/development-baseline/predictions.jsonl"),
    )
    parser.add_argument(
        "--identity-root",
        type=Path,
        default=Path("results/task-route-benchmark-qwen25-vl-3b"),
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    root = args.output_dir / "smoke" if args.smoke else args.output_dir
    prepared = root / "prepared"
    bridges = root / "bridge-fit"
    feature_validation = root / "feature-validation"
    repairs = root / "repair-benchmark"
    analysis = root / "analysis"
    state_path = root / "pipeline_state.json"
    root.mkdir(parents=True, exist_ok=True)
    state = {
        "mode": "smoke" if args.smoke else "full",
        "status": "running",
        "started_at": _timestamp(),
        "stages": {},
    }
    write_json(state_path, state)
    environment = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}

    common = [sys.executable]
    stages = [
        ("prepare", common + [
            "scripts/prepare_phase2.py",
            "--manifest", str(args.development_manifest),
            "--routes", str(args.routes),
            "--output-dir", str(prepared),
        ]),
        ("fit", common + [
            "scripts/fit_phase2_bridges.py",
            "--manifest", str(prepared / "calibration.jsonl"),
            "--routes", str(prepared / "routes.json"),
            "--data-root", str(args.data_root),
            "--output-dir", str(bridges),
        ]),
        ("validate_features", common + [
            "scripts/validate_phase2_bridges.py",
            "--manifest", str(prepared / "evaluation.jsonl"),
            "--routes", str(prepared / "routes.json"),
            "--bridges-dir", str(bridges),
            "--data-root", str(args.data_root),
            "--output-dir", str(feature_validation),
        ]),
        ("evaluate_answers", common + [
            "scripts/run_phase2_repair.py",
            "--manifest", str(prepared / "evaluation.jsonl"),
            "--routes", str(prepared / "routes.json"),
            "--bridges-dir", str(bridges),
            "--data-root", str(args.data_root),
            "--output-dir", str(repairs),
        ]),
    ]
    if args.smoke:
        for _, command in stages[1:]:
            command.extend(["--route", "task-attribute", "--ranks", "8"])
        stages[1][1].extend(["--limit", "8"])
        stages[2][1].extend(["--per-capability", "1"])
        stages[3][1].extend(["--limit", "2"])
    else:
        stages.append(("analyze", common + [
            "scripts/analyze_phase2.py",
            "--manifest", str(prepared / "evaluation.jsonl"),
            "--routes", str(prepared / "routes.json"),
            "--baseline-predictions", str(args.baseline_predictions),
            "--identity-root", str(args.identity_root),
            "--repair-root", str(repairs),
            "--feature-validation-root", str(feature_validation),
            "--output-dir", str(analysis),
        ]))

    try:
        for name, command in stages:
            state["stages"][name] = {"status": "running", "started_at": _timestamp(), "command": command}
            write_json(state_path, state)
            print(f"PHASE2_STAGE_START {name} {_timestamp()}", flush=True)
            subprocess.run(command, check=True, env=environment)
            state["stages"][name].update({"status": "complete", "completed_at": _timestamp()})
            write_json(state_path, state)
            print(f"PHASE2_STAGE_COMPLETE {name} {_timestamp()}", flush=True)
    except BaseException as error:
        state["status"] = "failed"
        state["failed_at"] = _timestamp()
        state["error"] = repr(error)
        write_json(state_path, state)
        raise
    state["status"] = "complete"
    state["completed_at"] = _timestamp()
    write_json(state_path, state)


if __name__ == "__main__":
    main()
