#!/usr/bin/env python3
"""Analyze Phase 2 feature repair against full and identity-route predictions."""

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from vlm_bench.io import read_jsonl, write_json
from vlm_bench.phase2 import CAPABILITIES


def _load_by_id(path: Path) -> dict[str, dict]:
    rows = list(read_jsonl(path))
    output = {row["id"]: row for row in rows}
    if len(rows) != len(output):
        raise ValueError(f"Duplicate IDs in {path}")
    return output


def _paired(base: dict[str, dict], variant: dict[str, dict], ids: list[str]) -> dict:
    if not ids:
        raise ValueError("Paired metric group is empty")
    base_correct = [int(bool(base[row_id]["correct"])) for row_id in ids]
    variant_correct = [int(bool(variant[row_id]["correct"])) for row_id in ids]
    return {
        "examples": len(ids),
        "baseline_accuracy": statistics.fmean(base_correct),
        "variant_accuracy": statistics.fmean(variant_correct),
        "accuracy_drop_pp": 100 * (statistics.fmean(base_correct) - statistics.fmean(variant_correct)),
        "lost_correct": sum(a == 1 and b == 0 for a, b in zip(base_correct, variant_correct)),
        "recovered_correct": sum(a == 0 and b == 1 for a, b in zip(base_correct, variant_correct)),
    }


def _bootstrap_advantage(
    first: dict[str, dict], second: dict[str, dict], ids: list[str], seed: int, repeats: int = 10000
) -> dict:
    differences = [int(bool(first[row_id]["correct"])) - int(bool(second[row_id]["correct"])) for row_id in ids]
    rng = random.Random(seed)
    samples = []
    for _ in range(repeats):
        samples.append(100 * statistics.fmean(rng.choice(differences) for _ in differences))
    samples.sort()
    return {
        "advantage_pp": 100 * statistics.fmean(differences),
        "ci95_pp": [samples[int(0.025 * repeats)], samples[int(0.975 * repeats)]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-config", type=Path, default=Path("configs/phase2_feature_gap.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--identity-root", type=Path, required=True)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--feature-validation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ranks")
    args = parser.parse_args()

    phase2_config = json.loads(args.phase2_config.read_text(encoding="utf-8"))
    ranks = [int(value) for value in args.ranks.split(",")] if args.ranks else phase2_config["ranks"]
    routes = json.loads(args.routes.read_text(encoding="utf-8"))
    route_by_name = {route["phase2_name"]: route for route in routes}
    manifest = list(read_jsonl(args.manifest))
    manifest_by_id = {row["id"]: row for row in manifest}
    ids = list(manifest_by_id)
    baseline_all = _load_by_id(args.baseline_predictions)
    baseline = {row_id: baseline_all[row_id] for row_id in ids}

    predictions = {}
    for route in routes:
        identity_all = _load_by_id(args.identity_root / route["name"] / "predictions.jsonl")
        predictions[(route["phase2_name"], "identity")] = {row_id: identity_all[row_id] for row_id in ids}
        for rank in ranks:
            predictions[(route["phase2_name"], rank)] = _load_by_id(
                args.repair_root / route["phase2_name"] / f"rank-{rank:03d}" / "predictions.jsonl"
            )

    feature_validation = {}
    for route in routes:
        rows = list(read_jsonl(args.feature_validation_root / f"{route['phase2_name']}.jsonl"))
        identity_values = [row["identity"]["relative_l2"] for row in rows]
        feature_validation[route["phase2_name"]] = {
            "examples": len(rows),
            "identity_relative_l2_mean": statistics.fmean(identity_values),
            "ranks": {
                str(rank): {
                    "relative_l2_mean": statistics.fmean(row["repaired"][str(rank)]["relative_l2"] for row in rows),
                    "cosine_mean": statistics.fmean(row["repaired"][str(rank)]["mean_token_cosine"] for row in rows),
                }
                for rank in ranks
            },
        }

    results = []
    for capability_index, capability in enumerate(CAPABILITIES):
        capability_ids = [row_id for row_id in ids if manifest_by_id[row_id]["capability"] == capability]
        task_name = f"task-{capability}"
        generic_name = "generic"
        task_identity = predictions[(task_name, "identity")]
        generic_identity = predictions[(generic_name, "identity")]
        item = {
            "capability": capability,
            "examples": len(capability_ids),
            "baseline_accuracy": _paired(baseline, baseline, capability_ids)["baseline_accuracy"],
            "task_route": route_by_name[task_name],
            "generic_route": route_by_name[generic_name],
            "task_identity": _paired(baseline, task_identity, capability_ids),
            "generic_identity": _paired(baseline, generic_identity, capability_ids),
            "ranks": {},
        }
        for rank in ranks:
            task_repaired = predictions[(task_name, rank)]
            generic_repaired = predictions[(generic_name, rank)]
            item["ranks"][str(rank)] = {
                "task_repaired": _paired(baseline, task_repaired, capability_ids),
                "generic_repaired": _paired(baseline, generic_repaired, capability_ids),
                "task_repair_vs_identity": _bootstrap_advantage(
                    task_repaired, task_identity, capability_ids,
                    int(phase2_config["seed"]) + capability_index * 1000 + rank,
                ),
                "task_vs_generic_repaired": _bootstrap_advantage(
                    task_repaired, generic_repaired, capability_ids,
                    int(phase2_config["seed"]) + capability_index * 1000 + rank + 500,
                ),
            }
        results.append(item)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis = {
        "status": "development-set mechanism experiment; not held-out publication evidence",
        "method": "SCP-inspired final-boundary low-rank residual calibration; not Short-LVLM SCP",
        "examples": len(ids),
        "results": results,
        "feature_validation": feature_validation,
    }
    write_json(args.output_dir / "analysis.json", analysis)

    lines = [
        "# Phase 2 Feature-Gap Repair",
        "",
        "This is a development-set mechanism experiment. The bridge is SCP-inspired but is not a reproduction of Short-LVLM SCP.",
        "",
        "## Target-Capability Results",
        "",
        "| Capability | Rank | Identity drop | Repaired drop | Repair gain | Task vs repaired generic |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        identity_drop = item["task_identity"]["accuracy_drop_pp"]
        for rank in ranks:
            rank_result = item["ranks"][str(rank)]
            repaired_drop = rank_result["task_repaired"]["accuracy_drop_pp"]
            repair_gain = rank_result["task_repair_vs_identity"]["advantage_pp"]
            task_advantage = rank_result["task_vs_generic_repaired"]["advantage_pp"]
            lines.append(
                f"| {item['capability']} | {rank} | {identity_drop:.2f} pp | {repaired_drop:.2f} pp | "
                f"{repair_gain:+.2f} pp | {task_advantage:+.2f} pp |"
            )
    lines.extend([
        "",
        "## Evaluation Feature Gap",
        "",
        "| Route | Identity relative L2 | " + " | ".join(f"Rank {rank}" for rank in ranks) + " |",
        "|---|---:" + "|---:" * len(ranks) + "|",
    ])
    for route in routes:
        gap = feature_validation[route["phase2_name"]]
        repaired = " | ".join(f"{gap['ranks'][str(rank)]['relative_l2_mean']:.4f}" for rank in ranks)
        lines.append(f"| {route['phase2_name']} | {gap['identity_relative_l2_mean']:.4f} | {repaired} |")
    lines.extend([
        "",
        "Positive repair gain means the repaired task route answered more evaluation examples correctly than its uncorrected identity route. Confidence intervals and paired counts are stored in `analysis.json`.",
    ])
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
