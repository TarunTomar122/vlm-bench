#!/usr/bin/env python3
"""Prepare image-disjoint manifests and fixed routes for Phase 2."""

import argparse
import json
from collections import Counter
from pathlib import Path

from vlm_bench.io import read_jsonl, sha256_file, write_json, write_jsonl
from vlm_bench.phase2 import select_four_block_routes, split_by_image


def _counts(rows: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(row["capability"] for row in rows).items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase2_feature_gap.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows = list(read_jsonl(args.manifest))
    routes = select_four_block_routes(json.loads(args.routes.read_text(encoding="utf-8")))
    calibration, evaluation = split_by_image(
        rows,
        int(config["calibration_per_capability"]),
        int(config["seed"]),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = args.output_dir / "calibration.jsonl"
    evaluation_path = args.output_dir / "evaluation.jsonl"
    routes_path = args.output_dir / "routes.json"
    write_jsonl(calibration_path, calibration)
    write_jsonl(evaluation_path, evaluation)
    write_json(routes_path, routes)
    write_json(args.output_dir / "prepare_summary.json", {
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.manifest),
        "calibration": {
            "examples": len(calibration),
            "capability_counts": _counts(calibration),
            "manifest_sha256": sha256_file(calibration_path),
        },
        "evaluation": {
            "examples": len(evaluation),
            "capability_counts": _counts(evaluation),
            "manifest_sha256": sha256_file(evaluation_path),
        },
        "image_overlap": len(
            {row["image_sha256"] for row in calibration}
            & {row["image_sha256"] for row in evaluation}
        ),
        "routes": [
            {
                "phase2_name": route["phase2_name"],
                "source_name": route["name"],
                "blocks": route["skip_vision_blocks"],
                "target_capability": route["target_capability"],
            }
            for route in routes
        ],
        "config": config,
    })


if __name__ == "__main__":
    main()
