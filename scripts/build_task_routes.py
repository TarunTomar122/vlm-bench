#!/usr/bin/env python3
"""Build capability-aware routes and matched controls from one-block discovery sweeps."""
import argparse
import json
import random
from pathlib import Path

from vlm_bench.io import read_jsonl, write_json, write_jsonl


CAPABILITIES = ("attribute", "counting", "object", "ocr", "spatial")
BLOCKS = tuple(range(32))
SEED = 20260714


def _load_predictions(path: Path) -> dict[str, dict]:
    rows = list(read_jsonl(path))
    output = {row["id"]: row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"Duplicate prediction IDs in {path}")
    return output


def _merge_predictions(paths: list[Path]) -> dict[str, dict]:
    merged = {}
    for path in paths:
        for row_id, row in _load_predictions(path).items():
            if row_id in merged:
                raise ValueError(f"Duplicate prediction ID across discovery sets: {row_id}")
            merged[row_id] = row
    return merged


def _accuracy(rows: list[dict]) -> float:
    return sum(bool(row["correct"]) for row in rows) / len(rows)


def _capability_drops(baseline: dict[str, dict], variant: dict[str, dict]) -> dict[str, float]:
    if set(baseline) != set(variant):
        raise ValueError("Baseline and intervention IDs differ")
    drops = {}
    for capability in CAPABILITIES:
        ids = [row_id for row_id, row in baseline.items() if row["capability"] == capability]
        base_rows = [baseline[row_id] for row_id in ids]
        variant_rows = [variant[row_id] for row_id in ids]
        drops[capability] = _accuracy(base_rows) - _accuracy(variant_rows)
    return drops


def _add_route(routes: dict[tuple[int, ...], dict], blocks: list[int], assignment: dict) -> None:
    key = tuple(sorted(blocks))
    if key not in routes:
        routes[key] = {
            "name": f"route-{len(routes):03d}-k{len(key):02d}",
            "skip_vision_blocks": list(key),
            "assignments": [],
        }
    routes[key]["assignments"].append(assignment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-baseline-dir", type=Path, required=True)
    parser.add_argument("--old-ablation-dir", type=Path, required=True)
    parser.add_argument("--color-baseline-dir", type=Path, required=True)
    parser.add_argument("--color-ablation-dir", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budgets", default="4,8,12,16")
    parser.add_argument("--random-repeats", type=int, default=3)
    args = parser.parse_args()

    budgets = [int(value) for value in args.budgets.split(",")]
    if any(budget <= 0 or budget >= len(BLOCKS) for budget in budgets):
        raise ValueError("Every route budget must be between 1 and 31")

    baseline = _merge_predictions([
        args.old_baseline_dir / "predictions.jsonl",
        args.color_baseline_dir / "predictions.jsonl",
    ])
    block_drops = {}
    for block in BLOCKS:
        variant = _merge_predictions([
            args.old_ablation_dir / f"block-{block:02d}" / "predictions.jsonl",
            args.color_ablation_dir / f"block-{block:02d}" / "predictions.jsonl",
        ])
        block_drops[block] = _capability_drops(baseline, variant)

    rankings = {
        capability: sorted(BLOCKS, key=lambda block: (block_drops[block][capability], block))
        for capability in CAPABILITIES
    }
    macro_drop = {
        block: sum(block_drops[block].values()) / len(CAPABILITIES)
        for block in BLOCKS
    }
    generic_ranking = sorted(BLOCKS, key=lambda block: (macro_drop[block], block))

    routes: dict[tuple[int, ...], dict] = {}
    for capability in CAPABILITIES:
        for budget in budgets:
            _add_route(routes, rankings[capability][:budget], {
                "route_type": "task_specific",
                "capability": capability,
                "budget": budget,
            })
    for budget in budgets:
        _add_route(routes, generic_ranking[:budget], {
            "route_type": "generic_macro",
            "capability": None,
            "budget": budget,
        })
        start = min(
            range(len(BLOCKS) - budget + 1),
            key=lambda candidate: sum(macro_drop[block] for block in range(candidate, candidate + budget)),
        )
        _add_route(routes, list(range(start, start + budget)), {
            "route_type": "contiguous_macro",
            "capability": None,
            "budget": budget,
            "start_block": start,
        })
        for repeat in range(args.random_repeats):
            rng = random.Random(SEED + budget * 100 + repeat)
            _add_route(routes, rng.sample(BLOCKS, budget), {
                "route_type": "random",
                "capability": None,
                "budget": budget,
                "repeat": repeat,
                "seed": SEED + budget * 100 + repeat,
            })

    development_rows = list(read_jsonl(args.development_manifest))
    development_ids = {row["id"] for row in development_rows}
    if not development_ids <= set(baseline):
        missing = sorted(development_ids - set(baseline))
        raise ValueError(f"Discovery baseline is missing development IDs: {missing[:5]}")
    baseline_rows = [baseline[row["id"]] for row in development_rows]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "routes.json", list(routes.values()))
    write_json(args.output_dir / "sensitivity.json", {
        "discovery_examples": len(baseline),
        "capability_counts": {
            capability: sum(row["capability"] == capability for row in baseline.values())
            for capability in CAPABILITIES
        },
        "single_block_accuracy_drops": {
            str(block): block_drops[block] for block in BLOCKS
        },
        "capability_rankings_least_to_most_sensitive": rankings,
        "generic_macro_ranking_least_to_most_sensitive": generic_ranking,
        "unique_routes": len(routes),
        "route_assignments": sum(len(route["assignments"]) for route in routes.values()),
    })
    baseline_dir = args.output_dir / "development-baseline"
    baseline_dir.mkdir(exist_ok=True)
    write_jsonl(baseline_dir / "predictions.jsonl", baseline_rows)


if __name__ == "__main__":
    main()
