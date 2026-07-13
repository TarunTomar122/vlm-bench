#!/usr/bin/env python3
"""Create paired, capability-level analysis artifacts for a completed block sweep."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from vlm_bench.io import read_jsonl, write_json


CAPABILITIES = ("attribute", "counting", "object", "ocr", "spatial")


def _accuracy(rows: list[dict]) -> float:
    return sum(bool(row["correct"]) for row in rows) / len(rows)


def _paired_metrics(baseline: list[dict], ablated: list[dict]) -> dict:
    if len(baseline) != len(ablated):
        raise ValueError("Baseline and ablation contain different example counts")
    lost = sum(bool(base["correct"]) and not bool(variant["correct"]) for base, variant in zip(baseline, ablated))
    recovered = sum(not bool(base["correct"]) and bool(variant["correct"]) for base, variant in zip(baseline, ablated))
    return {
        "examples": len(baseline),
        "baseline_accuracy": _accuracy(baseline),
        "ablated_accuracy": _accuracy(ablated),
        "accuracy_drop": (_accuracy(baseline) - _accuracy(ablated)),
        "lost_correct": lost,
        "recovered_correct": recovered,
        "net_correct_change": recovered - lost,
    }


def _group_rows(rows: list[dict], field: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    return dict(groups)


def _render_report(analysis: dict) -> str:
    baseline = analysis["baseline"]
    rows = analysis["blocks"]
    candidates = analysis["screening_candidates"]
    lines = [
        "# One-Block Vision Encoder Ablation",
        "",
        "## Result",
        "",
        f"The unpruned baseline is {100 * baseline['accuracy']:.2f}% on {baseline['examples']} fixed examples. "
        "Each intervention replaces exactly one of Qwen2.5-VL-3B's 32 vision blocks with an identity mapping.",
        "",
        "The sweep identifies sensitivity, not deployable pruning. A candidate must still pass a multi-block "
        "ablation, repeated latency measurement, and recovery fine-tuning before it can be called safe.",
        "",
        "## Most Sensitive Blocks",
        "",
        "| Block | Accuracy drop | Lost correct | Recovered |",
        "|---:|---:|---:|---:|",
    ]
    for block in sorted(rows, key=lambda row: row["overall"]["accuracy_drop"], reverse=True)[:8]:
        overall = block["overall"]
        lines.append(
            f"| {block['block']} | {100 * overall['accuracy_drop']:.2f} pp | "
            f"{overall['lost_correct']} | {overall['recovered_correct']} |"
        )
    lines.extend([
        "",
        "## Screening Candidates",
        "",
        "Threshold: at most 1.0 percentage point overall accuracy drop and at most 2.0 points on every "
        "capability. These are candidates for the next, multi-block experiment only.",
        "",
        "| Block | Overall drop | Largest capability drop |",
        "|---:|---:|---:|",
    ])
    for block in candidates:
        lines.append(
            f"| {block['block']} | {100 * block['overall']['accuracy_drop']:.2f} pp | "
            f"{100 * block['max_capability_drop']:.2f} pp ({block['max_capability']}) |"
        )
    lines.extend([
        "",
        "## Capability Map",
        "",
        "`capability_accuracy_drop_heatmap.png` plots accuracy drop in percentage points for every "
        "capability and block. Positive means the block was useful under this intervention; negative "
        "means the ablation happened to improve this finite benchmark sample.",
        "",
        "Latency from this one-pass sweep is diagnostic only. It is not used to rank candidates because "
        "individual block timings vary at the millisecond level; the next experiment needs repeated, "
        "matched-compute multi-block latency trials.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--ablation-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline = list(read_jsonl(args.baseline_dir / "predictions.jsonl"))
    baseline_by_id = {row["id"]: row for row in baseline}
    if len(baseline_by_id) != len(baseline):
        raise ValueError("Baseline prediction IDs must be unique")

    blocks = []
    csv_rows = []
    for record_path in sorted(args.ablation_dir.glob("block-*/ablation_record.json")):
        with record_path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        predictions = list(read_jsonl(record_path.parent / "predictions.jsonl"))
        prediction_by_id = {row["id"]: row for row in predictions}
        if set(prediction_by_id) != set(baseline_by_id):
            raise ValueError(f"Prediction IDs do not match baseline for block {record['block']}")
        ordered_base = [baseline_by_id[row["id"]] for row in predictions]
        overall = _paired_metrics(ordered_base, predictions)
        capabilities = {}
        for capability in CAPABILITIES:
            base_rows = [row for row in ordered_base if row["capability"] == capability]
            variant_rows = [row for row in predictions if row["capability"] == capability]
            capabilities[capability] = _paired_metrics(base_rows, variant_rows)
        max_capability = max(capabilities, key=lambda key: capabilities[key]["accuracy_drop"])
        block = {
            "block": record["block"],
            "overall": overall,
            "capabilities": capabilities,
            "vision_latency_median_ms": record["vision_latency_median_ms"],
            "vision_speedup_vs_baseline": record["vision_speedup_vs_baseline"],
            "max_capability": max_capability,
            "max_capability_drop": capabilities[max_capability]["accuracy_drop"],
        }
        blocks.append(block)
        csv_rows.append({
            "block": block["block"],
            "overall_drop_pp": 100 * overall["accuracy_drop"],
            "lost_correct": overall["lost_correct"],
            "recovered_correct": overall["recovered_correct"],
            "vision_latency_median_ms": block["vision_latency_median_ms"],
            "vision_speedup_vs_baseline": block["vision_speedup_vs_baseline"],
            **{f"{name}_drop_pp": 100 * metrics["accuracy_drop"] for name, metrics in capabilities.items()},
        })

    blocks.sort(key=lambda row: row["block"])
    analysis = {
        "baseline": {"examples": len(baseline), "accuracy": _accuracy(baseline)},
        "blocks": blocks,
        "screening_candidates": [
            block for block in blocks
            if block["overall"]["accuracy_drop"] <= 0.01 and block["max_capability_drop"] <= 0.02
        ],
    }
    write_json(args.ablation_dir / "analysis.json", analysis)
    with (args.ablation_dir / "capability_accuracy_drops.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    (args.ablation_dir / "README.md").write_text(_render_report(analysis), encoding="utf-8")

    matrix = [[100 * block["capabilities"][capability]["accuracy_drop"] for block in blocks] for capability in CAPABILITIES]
    fig, ax = plt.subplots(figsize=(14, 3.4))
    image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=-5, vmax=25)
    ax.set_xticks(range(len(blocks)), [str(block["block"]) for block in blocks])
    ax.set_yticks(range(len(CAPABILITIES)), CAPABILITIES)
    ax.set_xlabel("Skipped vision block")
    ax.set_ylabel("Capability")
    ax.set_title("Qwen2.5-VL-3B: accuracy drop from one-block identity ablation (percentage points)")
    fig.colorbar(image, ax=ax, label="Accuracy drop (pp)")
    fig.tight_layout()
    fig.savefig(args.ablation_dir / "capability_accuracy_drop_heatmap.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
