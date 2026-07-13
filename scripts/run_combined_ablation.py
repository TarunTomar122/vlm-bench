#!/usr/bin/env python3
"""Evaluate progressive multi-block vision-encoder pruning candidates."""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from vlm_bench.benchmark import BaselineRunner
from vlm_bench.io import read_jsonl, write_json


def _accuracy(rows: list[dict]) -> float:
    return sum(bool(row["correct"]) for row in rows) / len(rows)


def _paired_metrics(baseline: list[dict], candidate: list[dict]) -> dict:
    lost = sum(bool(base["correct"]) and not bool(variant["correct"]) for base, variant in zip(baseline, candidate))
    recovered = sum(not bool(base["correct"]) and bool(variant["correct"]) for base, variant in zip(baseline, candidate))
    return {
        "examples": len(baseline),
        "baseline_accuracy": _accuracy(baseline),
        "candidate_accuracy": _accuracy(candidate),
        "accuracy_drop": _accuracy(baseline) - _accuracy(candidate),
        "lost_correct": lost,
        "recovered_correct": recovered,
    }


def _balanced_latency_rows(rows: list[dict], per_capability: int) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["capability"]].append(row)
    selected = []
    for capability in sorted(grouped):
        if len(grouped[capability]) < per_capability:
            raise ValueError(f"Not enough {capability} rows for latency selection")
        selected.extend(grouped[capability][:per_capability])
    return selected


def _measure_latency(runner: BaselineRunner, rows: list[dict], repeats: int) -> dict:
    per_repeat = []
    for _ in range(repeats):
        samples = [runner._predict(row) for row in rows]
        per_repeat.append({
            "vision_encoder_median_ms": statistics.median(item["vision_encoder_ms"] for item in samples),
            "total_median_ms": statistics.median(item["total_ms"] for item in samples),
        })
    return {
        "examples_per_repeat": len(rows),
        "repeats": repeats,
        "vision_encoder_median_ms": statistics.median(item["vision_encoder_median_ms"] for item in per_repeat),
        "total_median_ms": statistics.median(item["total_median_ms"] for item in per_repeat),
        "per_repeat": per_repeat,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_qwen25_vl_3b.json"))
    parser.add_argument("--candidate-config", type=Path, default=Path("configs/combined_ablation_candidates.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latency-per-capability", type=int, default=20)
    parser.add_argument("--latency-repeats", type=int, default=3)
    args = parser.parse_args()

    base_config = json.loads(args.config.read_text(encoding="utf-8"))
    candidates = json.loads(args.candidate_config.read_text(encoding="utf-8"))
    manifest_rows = list(read_jsonl(args.manifest))
    latency_rows = _balanced_latency_rows(manifest_rows, args.latency_per_capability)
    baseline_by_id = {row["id"]: row for row in read_jsonl(args.baseline_dir / "predictions.jsonl")}
    if len(baseline_by_id) != len(manifest_rows):
        raise ValueError("The baseline predictions must contain every manifest row exactly once")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for candidate in candidates:
        name = candidate["name"]
        skipped = candidate["skip_vision_blocks"]
        run_dir = args.output_dir / name
        run_config = {**base_config, "skip_vision_blocks": skipped}
        runner = BaselineRunner(run_config, args.manifest, run_dir, args.data_root)
        try:
            runner.run()
            latency = _measure_latency(runner, latency_rows, args.latency_repeats)
        finally:
            runner.close()

        prediction_rows = list(read_jsonl(run_dir / "predictions.jsonl"))
        baseline_rows = [baseline_by_id[row["id"]] for row in prediction_rows]
        overall = _paired_metrics(baseline_rows, prediction_rows)
        capabilities = {}
        for capability in sorted({row["capability"] for row in prediction_rows}):
            base = [row for row in baseline_rows if row["capability"] == capability]
            variant = [row for row in prediction_rows if row["capability"] == capability]
            capabilities[capability] = _paired_metrics(base, variant)
        result = {
            "name": name,
            "skip_vision_blocks": skipped,
            "overall": overall,
            "capabilities": capabilities,
            "latency": latency,
        }
        write_json(run_dir / "combined_ablation_record.json", result)
        results.append(result)

    write_json(args.output_dir / "summary.json", {
        "latency_example_ids": [row["id"] for row in latency_rows],
        "results": results,
    })


if __name__ == "__main__":
    main()
