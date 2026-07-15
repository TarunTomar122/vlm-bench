#!/usr/bin/env python3
"""Analyze the sealed external evaluation without modifying its frozen routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlm_bench.external_eval import bootstrap_advantage, paired_accuracy, prediction_map, validate_protocol
from vlm_bench.io import read_jsonl, sha256_file, write_json


BOOTSTRAP_SEED = 20260715


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=Path("configs/external_frozen_evaluation.json"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/external-frozen-qwen25-vl-3b"))
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    manifest = list(read_jsonl(Path(protocol["manifest"])))
    expected_ids = {row["id"] for row in manifest}
    capabilities = protocol["capabilities"]
    by_capability = {
        capability: sorted(row["id"] for row in manifest if row["capability"] == capability)
        for capability in capabilities
    }
    predictions = {}
    for condition in protocol["conditions"]:
        path = args.results_dir / condition / "predictions.jsonl"
        predictions[condition] = prediction_map(list(read_jsonl(path)))
        if set(predictions[condition]) != expected_ids:
            raise ValueError(f"{condition} predictions do not exactly match the frozen manifest")

    baseline = predictions["full"]
    analysis = {
        "status": "sealed external evaluation; routes were frozen before inference",
        "protocol_sha256": sha256_file(args.protocol),
        "manifest_sha256": protocol["manifest_sha256"],
        "conditions": {},
        "task_vs_generic": {},
    }
    all_ids = sorted(expected_ids)
    for condition, values in predictions.items():
        analysis["conditions"][condition] = {
            "overall": paired_accuracy(baseline, values, all_ids),
            "capabilities": {
                capability: paired_accuracy(baseline, values, by_capability[capability])
                for capability in capabilities
            },
        }
    for index, condition in enumerate(("task-k8", "task-k4")):
        analysis["task_vs_generic"][condition] = {
            "overall": bootstrap_advantage(
                predictions[condition], predictions["generic-k8"], all_ids,
                BOOTSTRAP_SEED + index * 100,
            ),
            "capabilities": {
                capability: bootstrap_advantage(
                    predictions[condition], predictions["generic-k8"], by_capability[capability],
                    BOOTSTRAP_SEED + index * 100 + cap_index + 1,
                )
                for cap_index, capability in enumerate(capabilities)
            },
        }
    write_json(args.results_dir / "analysis.json", analysis)

    lines = [
        "# Frozen External Evaluation",
        "",
        "The 1,250 examples come from source families excluded from route selection. All routes were committed before inference. Positive task-versus-generic values mean the conditional route is more accurate.",
        "",
        "| Condition | Overall accuracy | Drop from full | Attribute drop | Counting drop | Object drop | OCR drop | Spatial drop |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in protocol["conditions"]:
        result = analysis["conditions"][condition]
        overall = result["overall"]
        drops = [result["capabilities"][capability]["accuracy_drop_pp"] for capability in capabilities]
        lines.append(
            f"| {condition} | {100 * overall['candidate_accuracy']:.2f}% | "
            f"{overall['accuracy_drop_pp']:.2f} pp | " + " | ".join(f"{drop:.2f} pp" for drop in drops) + " |"
        )
    lines.extend([
        "",
        "## Conditional Versus Generic K8",
        "",
        "| Route | Overall advantage | 95% interval | Attribute | Counting | Object | OCR | Spatial |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for condition in ("task-k8", "task-k4"):
        result = analysis["task_vs_generic"][condition]
        overall = result["overall"]
        values = [result["capabilities"][capability]["mean_pp"] for capability in capabilities]
        lines.append(
            f"| {condition} | {overall['mean_pp']:.2f} pp | "
            f"[{overall['ci95_low_pp']:.2f}, {overall['ci95_high_pp']:.2f}] | "
            + " | ".join(f"{value:.2f} pp" for value in values) + " |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Task K8 versus generic K8 is the matched-compute test. Task K8 does not win overall, and its confidence interval crosses zero.",
        "- Spatial is the only statistically clear task-K8 advantage. Counting is a statistically clear task-K8 failure.",
        "- Task K4 has the highest compressed-model accuracy, but it removes half as many blocks as generic K8 and is not a matched-compute control.",
        "- The external set is now consumed. These outcomes must not be used to alter routes or hyperparameters.",
        "",
        "Concurrent execution invalidates latency comparisons from this run. Use the existing fixed-clock latency audit for speed claims.",
        "",
    ])
    (args.results_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
