#!/usr/bin/env python3
"""Analyze task-specific routes against matched generic, contiguous, and random controls."""
import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from vlm_bench.io import read_jsonl, write_json


CAPABILITIES = ("attribute", "counting", "object", "ocr", "spatial")
BOOTSTRAP_SEED = 20260714


def _assignment_rows(summary: dict, audited_latency: dict[str, dict] | None = None) -> list[dict]:
    rows = []
    for result in summary["results"]:
        audit = (audited_latency or {}).get(result["name"])
        common = {
            "route": result["name"],
            "blocks": result["skip_vision_blocks"],
            "budget": len(result["skip_vision_blocks"]),
            "overall_drop_pp": 100 * result["overall"]["accuracy_drop"],
            "vision_speedup_percent": (
                audit["vision_speedup_percent"]
                if audit else result["paired_latency"]["median_vision_speedup_percent"]
            ),
            "total_speedup_percent": (
                audit["total_speedup_percent"]
                if audit else result["paired_latency"]["median_total_speedup_percent"]
            ),
            "latency_source": "fixed_clock_audit" if audit else "original_dynamic_clock",
            "removed_parameters": result["parameters"]["removed_model"],
            "removed_vision_percent": 100 * result["parameters"]["removed_vision"] / result["parameters"]["full_vision"],
            "peak_allocated_mib": result["peak_allocated_mib"],
        }
        for assignment in result["assignments"]:
            route_type = assignment["route_type"]
            capabilities = [assignment["capability"]] if assignment.get("capability") else CAPABILITIES
            for capability in capabilities:
                rows.append({
                    **common,
                    "route_type": route_type,
                    "capability": capability,
                    "capability_drop_pp": 100 * result["capabilities"][capability]["accuracy_drop"],
                    "repeat": assignment.get("repeat"),
                })
    return rows


def _bootstrap_mean(values: list[float], seed: int, iterations: int = 10_000) -> dict:
    rng = random.Random(seed)
    count = len(values)
    estimates = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(iterations)
    )
    return {
        "examples": count,
        "mean_pp": 100 * statistics.fmean(values),
        "ci95_low_pp": 100 * estimates[int(0.025 * iterations)],
        "ci95_high_pp": 100 * estimates[int(0.975 * iterations)],
    }


def _prediction_map(path: Path, capability: str) -> dict[str, float]:
    return {
        row["id"]: float(bool(row["correct"]))
        for row in read_jsonl(path)
        if row["capability"] == capability
    }


def _add_paired_intervals(
    comparisons: list[dict],
    rows: list[dict],
    predictions_root: Path,
    baseline_predictions: Path,
) -> None:
    route_by_group: dict[tuple[str, int, str], list[str]] = defaultdict(list)
    for row in rows:
        key = (row["capability"], row["budget"], row["route_type"])
        if row["route"] not in route_by_group[key]:
            route_by_group[key].append(row["route"])
    for index, comparison in enumerate(comparisons):
        capability = comparison["capability"]
        budget = comparison["budget"]
        task_route = comparison["task_route"]["route"]
        task = _prediction_map(predictions_root / task_route / "predictions.jsonl", capability)
        baseline = _prediction_map(baseline_predictions, capability)
        if set(task) != set(baseline):
            raise ValueError(f"Task and baseline IDs differ for {capability} budget {budget}")
        ids = sorted(task)
        intervals = {
            "vs_baseline": _bootstrap_mean(
                [task[row_id] - baseline[row_id] for row_id in ids],
                BOOTSTRAP_SEED + index * 10,
            )
        }
        for offset, route_type in enumerate(("generic_macro", "contiguous_macro"), start=1):
            control_route = route_by_group[(capability, budget, route_type)][0]
            control = _prediction_map(predictions_root / control_route / "predictions.jsonl", capability)
            intervals[f"vs_{route_type}"] = _bootstrap_mean(
                [task[row_id] - control[row_id] for row_id in ids],
                BOOTSTRAP_SEED + index * 10 + offset,
            )
        random_routes = route_by_group[(capability, budget, "random")]
        random_predictions = [
            _prediction_map(predictions_root / route / "predictions.jsonl", capability)
            for route in random_routes
        ]
        intervals["vs_random_mean"] = _bootstrap_mean(
            [
                task[row_id] - statistics.fmean(control[row_id] for control in random_predictions)
                for row_id in ids
            ],
            BOOTSTRAP_SEED + index * 10 + 3,
        )
        comparison["paired_accuracy_advantage"] = intervals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path)
    parser.add_argument("--baseline-predictions", type=Path)
    parser.add_argument("--latency-audit-summary", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    audited_latency = None
    if args.latency_audit_summary:
        audit = json.loads(args.latency_audit_summary.read_text(encoding="utf-8"))
        audited_latency = {route["name"]: route for route in audit["routes"]}
    rows = _assignment_rows(summary, audited_latency)

    grouped: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["capability"], row["budget"], row["route_type"])].append(row)

    comparisons = []
    for capability in CAPABILITIES:
        budgets = sorted({row["budget"] for row in rows})
        for budget in budgets:
            task_rows = grouped.get((capability, budget, "task_specific"), [])
            if not task_rows:
                continue
            task = task_rows[0]
            controls = {}
            for route_type in ("generic_macro", "contiguous_macro", "random"):
                values = grouped.get((capability, budget, route_type), [])
                if not values:
                    continue
                controls[route_type] = {
                    "runs": len(values),
                    "mean_capability_drop_pp": statistics.fmean(row["capability_drop_pp"] for row in values),
                    "mean_vision_speedup_percent": statistics.fmean(row["vision_speedup_percent"] for row in values),
                    "mean_total_speedup_percent": statistics.fmean(row["total_speedup_percent"] for row in values),
                }
            comparisons.append({
                "capability": capability,
                "budget": budget,
                "task_route": task,
                "controls": controls,
            })

    if bool(args.predictions_root) != bool(args.baseline_predictions):
        raise ValueError("Provide both --predictions-root and --baseline-predictions")
    if args.predictions_root:
        _add_paired_intervals(
            comparisons,
            rows,
            args.predictions_root,
            args.baseline_predictions,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "analysis.json", {
        "comparisons": comparisons,
        "interpretation": (
            "These are development-set route-search results. They select configurations for later "
            "frozen evaluation and are not held-out performance claims."
        ),
    })
    csv_fields = [
        "route", "route_type", "capability", "budget", "blocks", "capability_drop_pp",
        "overall_drop_pp", "vision_speedup_percent", "total_speedup_percent",
        "removed_parameters", "removed_vision_percent", "peak_allocated_mib", "repeat",
        "latency_source",
    ]
    with (args.output_dir / "route_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows({**row, "blocks": " ".join(map(str, row["blocks"]))} for row in rows)

    lines = [
        "# Task-Specific Vision Routes",
        "",
        "These are development-set route-search results, not held-out claims. Four-block latency uses the fixed-clock audit; larger budgets retain diagnostic dynamic-clock timing.",
        "",
        "| Task | Removed blocks | Task route drop | Generic drop | Contiguous drop | Random mean drop | Vision speedup | Total speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        task = item["task_route"]
        controls = item["controls"]
        value = lambda name: controls.get(name, {}).get("mean_capability_drop_pp")
        display = lambda number: "n/a" if number is None else f"{number:.2f} pp"
        lines.append(
            f"| {item['capability']} | {item['budget']} | {task['capability_drop_pp']:.2f} pp | "
            f"{display(value('generic_macro'))} | {display(value('contiguous_macro'))} | "
            f"{display(value('random'))} | {task['vision_speedup_percent']:.2f}% | "
            f"{task['total_speedup_percent']:.2f}% |"
        )
    lines.extend([
        "",
        "A task route is useful only if it preserves its target capability better than controls at comparable measured speed. Identity substitution removes the selected block parameters from the active runtime module tree, but this experiment does not yet serialize a standalone compact checkpoint.",
        "",
    ])
    if args.predictions_root:
        lines.extend([
            "## Four-Block Paired Accuracy Advantage",
            "",
            "Positive means the task-specific route is more accurate than the control. Intervals are paired 95% bootstrap intervals over examples.",
            "",
            "| Task | Versus generic | 95% interval | Versus random mean | 95% interval |",
            "|---|---:|---:|---:|---:|",
        ])
        for item in comparisons:
            if item["budget"] != 4:
                continue
            generic = item["paired_accuracy_advantage"]["vs_generic_macro"]
            random_mean = item["paired_accuracy_advantage"]["vs_random_mean"]
            lines.append(
                f"| {item['capability']} | {generic['mean_pp']:.2f} pp | "
                f"[{generic['ci95_low_pp']:.2f}, {generic['ci95_high_pp']:.2f}] | "
                f"{random_mean['mean_pp']:.2f} pp | "
                f"[{random_mean['ci95_low_pp']:.2f}, {random_mean['ci95_high_pp']:.2f}] |"
            )
        lines.append("")
    (args.output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
