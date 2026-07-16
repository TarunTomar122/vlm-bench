#!/usr/bin/env python3
"""Evaluate frozen SmolVLM2 K6 routes on the sealed IIIT5K OCR transfer set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm_bench.benchmark import BaselineRunner, summarize
from vlm_bench.external_eval import bootstrap_advantage, paired_accuracy, prediction_map
from vlm_bench.io import read_jsonl, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-routes", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=Path("configs/baseline_smolvlm2_2b.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/fresh-ocr-iiit5k-v1/manifests/heldout.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/fresh-ocr-iiit5k-smolvlm2-2b"))
    args = parser.parse_args()
    frozen = json.loads(args.frozen_routes.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen":
        raise ValueError("Routes must be frozen before transfer evaluation")
    rows = list(read_jsonl(args.manifest))
    if not rows or {row["capability"] for row in rows} != {"ocr"}:
        raise ValueError("Expected a non-empty OCR-only sealed manifest")
    expected_ids = {str(row["id"]) for row in rows}
    generic = frozen["families"]["generic"]["budgets"]["6"]["blocks"]
    ocr = frozen["families"]["ocr"]["budgets"]["6"]["blocks"]
    conditions = {"full": [], "generic-k6": generic, "ocr-k6": ocr}
    config = json.loads(args.model_config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner = BaselineRunner(config, args.manifest, args.output_dir / "runtime", args.manifest.parent.parent)
    try:
        for name, blocks in conditions.items():
            destination = args.output_dir / name
            destination.mkdir(exist_ok=True)
            path = destination / "predictions.jsonl"
            existing = prediction_map(list(read_jsonl(path))) if path.exists() else {}
            if not set(existing) <= expected_ids:
                raise ValueError(f"Unexpected prediction IDs in {path}")
            runner.set_skip_vision_blocks(blocks)
            with path.open("a", encoding="utf-8", buffering=1) as handle:
                for index, row in enumerate((row for row in rows if str(row["id"]) not in existing), start=1):
                    prediction = runner.predict(row)
                    prediction["condition"] = name
                    prediction["skip_vision_blocks"] = blocks
                    handle.write(json.dumps(prediction, sort_keys=True) + "\n")
                    if index == 1 or index % 25 == 0:
                        print(f"[{name}] saved {index} new predictions", flush=True)
            completed = prediction_map(list(read_jsonl(path)))
            if set(completed) != expected_ids:
                raise RuntimeError(f"{name} does not cover the sealed manifest")
            write_json(destination / "summary.json", summarize([completed[str(row["id"])] for row in rows]))
    finally:
        runner.close()

    maps = {name: prediction_map(list(read_jsonl(args.output_dir / name / "predictions.jsonl"))) for name in conditions}
    ordered_ids = sorted(expected_ids)
    analysis = {
        "status": "sealed fresh OCR transfer; frozen routes only",
        "manifest_sha256": sha256_file(args.manifest),
        "frozen_routes_sha256": sha256_file(args.frozen_routes),
        "routes": {"generic-k6": generic, "ocr-k6": ocr},
        "conditions": {name: paired_accuracy(maps["full"], values, ordered_ids) for name, values in maps.items()},
        "ocr_minus_generic_k6": bootstrap_advantage(maps["ocr-k6"], maps["generic-k6"], ordered_ids, 20260716),
    }
    write_json(args.output_dir / "analysis.json", analysis)


if __name__ == "__main__":
    main()
