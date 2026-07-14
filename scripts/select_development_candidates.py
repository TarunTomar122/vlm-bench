#!/usr/bin/env python3
"""Select vision-block pruning candidates strictly from development predictions."""
import argparse
import json
from pathlib import Path

from vlm_bench.io import read_jsonl, write_json


CAPABILITIES = ("attribute", "counting", "object", "ocr", "spatial")


def _paired_metrics(baseline: list[dict], variant: list[dict]) -> dict:
    if [row["id"] for row in baseline] != [row["id"] for row in variant]:
        raise ValueError("Paired prediction IDs do not match")
    lost = sum(bool(base["correct"]) and not bool(row["correct"]) for base, row in zip(baseline, variant))
    recovered = sum(not bool(base["correct"]) and bool(row["correct"]) for base, row in zip(baseline, variant))
    examples = len(baseline)
    baseline_correct = sum(bool(row["correct"]) for row in baseline)
    variant_correct = sum(bool(row["correct"]) for row in variant)
    return {
        "examples": examples,
        "baseline_accuracy": baseline_correct / examples,
        "candidate_accuracy": variant_correct / examples,
        "accuracy_drop": (baseline_correct - variant_correct) / examples,
        "lost_correct": lost,
        "recovered_correct": recovered,
        "net_correct_change": recovered - lost,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--ablation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-overall-drop-pp", type=float, default=1.0)
    parser.add_argument("--max-capability-net-loss", type=int, default=3)
    args = parser.parse_args()

    split_by_id = {row["id"]: row["split"] for row in read_jsonl(args.manifest)}
    baseline_by_id = {row["id"]: row for row in read_jsonl(args.baseline_dir / "predictions.jsonl")}
    if set(baseline_by_id) != set(split_by_id):
        raise ValueError("Baseline predictions and manifest IDs do not match")
    development_ids = {row_id for row_id, split in split_by_id.items() if split == "development"}
    baseline = sorted((baseline_by_id[row_id] for row_id in development_ids), key=lambda row: row["id"])
    selected = []
    blocks = []
    for record_path in sorted(args.ablation_dir.glob("block-*/ablation_record.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        predictions_by_id = {row["id"]: row for row in read_jsonl(record_path.parent / "predictions.jsonl")}
        predictions = [predictions_by_id[row["id"]] for row in baseline]
        overall = _paired_metrics(baseline, predictions)
        capabilities = {}
        for capability in CAPABILITIES:
            base = [row for row in baseline if row["capability"] == capability]
            variant = [predictions_by_id[row["id"]] for row in base]
            capabilities[capability] = _paired_metrics(base, variant)
        max_net_loss = max(-metrics["net_correct_change"] for metrics in capabilities.values())
        block = {
            "block": record["block"],
            "overall": overall,
            "capabilities": capabilities,
            "max_capability_net_loss": max_net_loss,
        }
        block["selected"] = (
            overall["accuracy_drop"] <= args.max_overall_drop_pp / 100
            and max_net_loss <= args.max_capability_net_loss
        )
        blocks.append(block)
        if block["selected"]:
            selected.append(record["block"])

    result = {
        "protocol": {
            "selection_split": "development",
            "evaluation_split": "test",
            "max_overall_drop_pp": args.max_overall_drop_pp,
            "max_capability_net_loss": args.max_capability_net_loss,
            "note": "Selected blocks require a new, test-only multi-block evaluation. Existing full-manifest combined results are exploratory.",
        },
        "development_examples": len(baseline),
        "selected_blocks": selected,
        "blocks": blocks,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "development_selection.json", result)
    lines = [
        "# Development-Only Candidate Selection",
        "",
        f"Selected {len(selected)} of {len(blocks)} blocks using {len(baseline)} development examples.",
        "",
        "Selection rule: overall drop <= " + f"{args.max_overall_drop_pp:.1f} pp and no capability loses more than "
        + f"{args.max_capability_net_loss} net-correct example.",
        "",
        "The test split has not informed selection. It must be used only for a newly run, fixed multi-block evaluation.",
        "",
        "## Selected Blocks",
        "",
        ", ".join(str(block) for block in selected) if selected else "None",
        "",
    ]
    (args.output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
