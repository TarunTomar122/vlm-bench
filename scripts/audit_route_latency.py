#!/usr/bin/env python3
"""Remeasure route latency with warmup, randomized order, and bracketed baselines."""
import argparse
import gc
import json
import random
import statistics
from pathlib import Path

from vlm_bench.benchmark import BaselineRunner
from vlm_bench.io import read_jsonl, write_json


SEED = 20260714


def _balanced_rows(rows: list[dict], per_capability: int) -> list[dict]:
    grouped = {}
    for row in rows:
        grouped.setdefault(row["capability"], []).append(row)
    selected = []
    for capability in sorted(grouped):
        candidates = sorted(grouped[capability], key=lambda row: row["id"])
        if len(candidates) < per_capability:
            raise ValueError(f"Not enough {capability} examples")
        selected.extend(candidates[:per_capability])
    return selected


def _measure(config: dict, manifest: Path, data_root: Path, output_dir: Path, rows: list[dict], repeats: int) -> dict:
    runner = BaselineRunner(config, manifest, output_dir, data_root)
    try:
        for index in range(20):
            runner._predict(rows[index % len(rows)])
        per_repeat = []
        for _ in range(repeats):
            predictions = [runner._predict(row) for row in rows]
            per_repeat.append({
                "vision_encoder_median_ms": statistics.median(row["vision_encoder_ms"] for row in predictions),
                "total_median_ms": statistics.median(row["total_ms"] for row in predictions),
            })
        return {
            "examples_per_repeat": len(rows),
            "repeats": repeats,
            "per_repeat": per_repeat,
            "parameters": runner.parameter_counts,
        }
    finally:
        runner.close()
        del runner
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_qwen25_vl_3b.json"))
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--per-capability", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    base_config = json.loads(args.config.read_text(encoding="utf-8"))
    candidates = [
        candidate for candidate in json.loads(args.candidate_config.read_text(encoding="utf-8"))
        if len(candidate["skip_vision_blocks"]) == args.budget
    ]
    rng = random.Random(SEED)
    rng.shuffle(candidates)
    midpoint = len(candidates) // 2
    schedule = (
        [("baseline-start", None)]
        + [(candidate["name"], candidate) for candidate in candidates[:midpoint]]
        + [("baseline-middle", None)]
        + [(candidate["name"], candidate) for candidate in candidates[midpoint:]]
        + [("baseline-end", None)]
    )
    rows = _balanced_rows(list(read_jsonl(args.manifest)), args.per_capability)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for name, candidate in schedule:
        record_path = args.output_dir / f"{name}.json"
        if record_path.exists():
            records.append(json.loads(record_path.read_text(encoding="utf-8")))
            continue
        config = base_config if candidate is None else {
            **base_config,
            "skip_vision_blocks": candidate["skip_vision_blocks"],
        }
        measurement = _measure(
            config,
            args.manifest,
            args.data_root,
            args.output_dir / f"runtime-{name}",
            rows,
            args.repeats,
        )
        record = {
            "name": name,
            "kind": "baseline" if candidate is None else "route",
            "skip_vision_blocks": [] if candidate is None else candidate["skip_vision_blocks"],
            "assignments": [] if candidate is None else candidate.get("assignments", []),
            **measurement,
        }
        write_json(record_path, record)
        records.append(record)

    baseline_records = [record for record in records if record["kind"] == "baseline"]
    baseline_vision = statistics.median(
        repeat["vision_encoder_median_ms"]
        for record in baseline_records for repeat in record["per_repeat"]
    )
    baseline_total = statistics.median(
        repeat["total_median_ms"]
        for record in baseline_records for repeat in record["per_repeat"]
    )
    routes = []
    for record in records:
        if record["kind"] != "route":
            continue
        vision = statistics.median(repeat["vision_encoder_median_ms"] for repeat in record["per_repeat"])
        total = statistics.median(repeat["total_median_ms"] for repeat in record["per_repeat"])
        routes.append({
            **record,
            "vision_encoder_median_ms": vision,
            "total_median_ms": total,
            "vision_speedup_percent": 100 * (baseline_vision / vision - 1),
            "total_speedup_percent": 100 * (baseline_total / total - 1),
        })
    write_json(args.output_dir / "summary.json", {
        "method": "Three baseline measurements bracket randomized routes; 20 warmups per model.",
        "example_ids": [row["id"] for row in rows],
        "baseline_vision_encoder_median_ms": baseline_vision,
        "baseline_total_median_ms": baseline_total,
        "baseline_trials": baseline_records,
        "routes": routes,
    })


if __name__ == "__main__":
    main()
