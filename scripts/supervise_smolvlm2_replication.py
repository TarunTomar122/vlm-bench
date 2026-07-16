#!/usr/bin/env python3
"""Advance the resumable SmolVLM2 replication after each artifact is complete."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from vlm_bench.io import write_json


ROOT = Path("results/robust-route-search-smolvlm2-2b")
CONFIG = Path("configs/robust_route_search_smolvlm2_2b.json")
MODEL_CONFIG = Path("configs/baseline_smolvlm2_2b.json")
PREPARED = Path("data/processed-v2/robust-route-search-smolvlm2-2b/prepared")
ABLATION = Path("results/smolvlm2-2b-single-block")
CONTROLS_CONFIG = Path("configs/robust_route_controls_smolvlm2_2b.json")
FRESH_OCR_MANIFEST = Path("data/fresh-ocr-iiit5k-v1/manifests/heldout.jsonl")
FRESH_OCR_ROOT = Path("results/fresh-ocr-iiit5k-smolvlm2-2b")
LANES = (("generic", "object", "spatial"), ("attribute", "counting", "ocr"))


def running(*needles: str) -> bool:
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, UnicodeDecodeError):
            continue
        if all(needle in command for needle in needles):
            return True
    return False


def status(path: Path) -> str:
    if not path.exists():
        return "pending"
    return json.loads(path.read_text(encoding="utf-8")).get("status", "pending")


def launch(key: str, command: list[str], state: dict) -> None:
    if running(*command[1:]):
        return
    attempts = int(state["attempts"].get(key, 0))
    if attempts >= 2:
        state["events"].append({"at": time.time(), "message": f"{key} stopped after two attempts"})
        return
    state["attempts"][key] = attempts + 1
    log = Path("logs") / f"smolvlm2-{key.replace(':', '-')}.log"
    log.parent.mkdir(exist_ok=True)
    with log.open("ab", buffering=0) as handle:
        subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    state["events"].append({"at": time.time(), "message": f"launched {key} attempt {attempts + 1}"})


def command(script: str, *args: str) -> list[str]:
    return [".venv/bin/python", script, *args]


def complete_blocks() -> bool:
    return all((ABLATION / f"block-{block:02d}" / "ablation_record.json").exists() for block in range(27))


def family_complete(family: str) -> bool:
    return status(ROOT / "families" / family / "state.json") == "complete"


def step(state: dict) -> bool:
    baseline = ROOT / "baseline" / "summary.json"
    if not baseline.exists():
        return False
    if not complete_blocks():
        launch(
            "ablation-left",
            command("scripts/run_layer_ablation.py", "--config", str(MODEL_CONFIG), "--manifest", "data/processed-v2/manifests/all.jsonl", "--data-root", "data/processed-v2", "--baseline-dir", str(ROOT / "baseline"), "--output-dir", str(ABLATION), "--blocks", "0-13", "--split", "development", "--summary-stem", "left"),
            state,
        )
        launch(
            "ablation-right",
            command("scripts/run_layer_ablation.py", "--config", str(MODEL_CONFIG), "--manifest", "data/processed-v2/manifests/all.jsonl", "--data-root", "data/processed-v2", "--baseline-dir", str(ROOT / "baseline"), "--output-dir", str(ABLATION), "--blocks", "14-26", "--split", "development", "--summary-stem", "right"),
            state,
        )
        return False
    priors = ABLATION / "sensitivity.json"
    if not priors.exists():
        launch(
            "build-priors",
            command("scripts/build_single_block_priors.py", "--baseline-dir", str(ROOT / "baseline"), "--ablation-dir", str(ABLATION), "--development-manifest", "data/processed-v2/manifests/development.jsonl", "--output-dir", str(ABLATION), "--blocks", "0-26"),
            state,
        )
        return False
    for lane in LANES:
        family = next((item for item in lane if not family_complete(item)), None)
        if family:
            launch(f"family:{family}", command("scripts/run_robust_route_search.py", "--family", family, "--search-config", str(CONFIG)), state)
    families_done = all(family_complete(family) for lane in LANES for family in lane)
    frozen = ROOT / "frozen_routes.json"
    if families_done and not frozen.exists():
        launch("finalize", command("scripts/run_robust_route_search.py", "--finalize", "--search-config", str(CONFIG)), state)
        return False
    if not frozen.exists():
        return False
    if not CONTROLS_CONFIG.exists():
        launch(
            "freeze-controls",
            command("scripts/freeze_matched_controls.py", "--model-config", str(MODEL_CONFIG), "--selection-manifest", str(PREPARED / "selection.jsonl"), "--sensitivity", str(priors), "--output", str(CONTROLS_CONFIG), "--output-dir", str(ROOT / "controls")),
            state,
        )
        return False
    if status(ROOT / "controls" / "state.json") != "complete":
        launch("controls", command("scripts/run_robust_route_controls.py", "--config", str(CONTROLS_CONFIG), "--frozen-routes", str(frozen)), state)
        return False
    analysis = ROOT / "analysis" / "analysis.json"
    if not analysis.exists():
        launch("analysis", command("scripts/analyze_robust_route_search.py", "--root", str(ROOT), "--manifest", str(PREPARED / "selection.jsonl")), state)
        return False
    if FRESH_OCR_MANIFEST.exists() and not (FRESH_OCR_ROOT / "analysis.json").exists():
        launch(
            "fresh-ocr-transfer",
            command("scripts/run_fresh_ocr_transfer.py", "--frozen-routes", str(frozen), "--manifest", str(FRESH_OCR_MANIFEST), "--output-dir", str(FRESH_OCR_ROOT)),
            state,
        )
        return False
    return FRESH_OCR_MANIFEST.exists()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=int, default=90)
    args = parser.parse_args()
    if args.interval_seconds < 30:
        raise ValueError("interval must be at least 30 seconds")
    os.environ.setdefault("PYTHONPATH", "src")
    state_path = ROOT / "supervisor-state.json"
    while True:
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"schema_version": 1, "attempts": {}, "events": []}
        done = step(state)
        state["updated_at"] = time.time()
        write_json(state_path, state)
        if done:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
