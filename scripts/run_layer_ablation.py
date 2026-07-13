#!/usr/bin/env python3
"""Run one-block identity ablations and build task-by-layer sensitivity maps."""
import argparse
import csv
import gc
import json
import statistics
import subprocess
from pathlib import Path

from vlm_bench.benchmark import BaselineRunner
from vlm_bench.io import read_jsonl, write_json


def _accuracy(rows):
    return sum(bool(row["correct"]) for row in rows) / len(rows) if rows else None


def _groups(rows):
    keys = [("overall", lambda row: True), ("source", lambda row: row["source"]),
            ("capability", lambda row: row["capability"])]
    output = {}
    for kind, selector in keys:
        values = {}
        for row in rows:
            key = selector(row)
            if key is True:
                key = "overall"
            values.setdefault(key, []).append(row)
        output[kind] = {key: _accuracy(items) for key, items in sorted(values.items())}
    return output


def _latency(rows):
    values = [row["vision_encoder_ms"] for row in rows if row.get("vision_encoder_ms") is not None]
    return statistics.median(values) if values else None


def _nvidia_smi():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_qwen25_vl_3b.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--blocks", default="0-31", help="Block range, e.g. 0-31 or 0,4,8")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--suite", choices=["controlled", "external"])
    parser.add_argument("--split", choices=["development", "test"])
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if "-" in args.blocks:
        first, last = (int(value) for value in args.blocks.split("-", 1))
        blocks = list(range(first, last + 1))
    else:
        blocks = [int(value) for value in args.blocks.split(",") if value]
    baseline_rows = list(read_jsonl(args.baseline_dir / "predictions.jsonl"))
    if args.limit is not None:
        baseline_rows = baseline_rows[:args.limit]
    baseline_groups = _groups(baseline_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for block in blocks:
        run_dir = args.output_dir / f"block-{block:02d}"
        prediction_path = run_dir / "predictions.jsonl"
        run_config = {**config, "skip_vision_block": block}
        runner = BaselineRunner(run_config, args.manifest, run_dir, args.data_root)
        try:
            runner.run(limit=args.limit, suite=args.suite, split=args.split)
        finally:
            runner.close()
            del runner
            gc.collect()
        rows = list(read_jsonl(prediction_path))
        groups = _groups(rows)
        record = {
            "block": block,
            "examples": len(rows),
            "accuracy": _accuracy(rows),
            "accuracy_drop": baseline_groups["overall"].get("overall", 0) - _accuracy(rows),
            "vision_latency_median_ms": _latency(rows),
            "vision_speedup_vs_baseline": (
                _latency(baseline_rows) / _latency(rows) if _latency(rows) else None
            ),
            "groups": groups,
            "gpu": _nvidia_smi(),
        }
        records.append(record)
        write_json(run_dir / "ablation_record.json", record)
    write_json(args.output_dir / "summary.json", {"blocks": records, "baseline": baseline_groups})
    with (args.output_dir / "heatmap.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["block", "accuracy", "accuracy_drop", "vision_latency_median_ms", "vision_speedup_vs_baseline"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: record[field] for field in fields} for record in records)
    try:
        import matplotlib.pyplot as plt
        capabilities = sorted(baseline_groups["capability"])
        matrix = [[
            100 * (baseline_groups["capability"][capability] - record["groups"]["capability"].get(capability, 0))
            for record in records
        ] for capability in capabilities]
        fig, ax = plt.subplots(figsize=(max(10, len(records) * 0.45), 3.2))
        image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=-5, vmax=20)
        ax.set_xticks(range(len(records)), [str(record["block"]) for record in records])
        ax.set_yticks(range(len(capabilities)), capabilities)
        ax.set_xlabel("Skipped vision block")
        ax.set_ylabel("Capability")
        ax.set_title("Qwen2.5-VL-3B identity ablation: accuracy drop (percentage points)")
        fig.colorbar(image, ax=ax, label="Accuracy drop (pp)")
        fig.tight_layout()
        fig.savefig(args.output_dir / "capability_layer_heatmap.png", dpi=180)
        plt.close(fig)
    except ImportError:
        (args.output_dir / "heatmap.README").write_text(
            "Install matplotlib and rerun the plotting step to render the PNG.\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
