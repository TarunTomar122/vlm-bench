#!/usr/bin/env python3
"""Validate Phase 2 bridge feature reconstruction on evaluation images."""

import argparse
import json
from pathlib import Path

import torch

from vlm_bench.benchmark import BaselineRunner, VisionBlockIdentity
from vlm_bench.io import read_jsonl, write_json
from vlm_bench.phase2 import LowRankResidualBridge, feature_gap_metrics


class FinalStateRunner(BaselineRunner):
    def __init__(self, config: dict, manifest: Path, output_dir: Path, data_root: Path) -> None:
        super().__init__(config, manifest, output_dir, data_root)
        self.original_blocks = list(self.model.visual.blocks)

    def capture_final(self, inputs: dict, skipped: list[int]) -> torch.Tensor:
        skipped_set = set(skipped)
        for index, original in enumerate(self.original_blocks):
            self.model.visual.blocks[index] = VisionBlockIdentity() if index in skipped_set else original
        captured = None

        def capture(_module, _args, output):
            nonlocal captured
            captured = output.detach()

        handle = self.model.visual.blocks[-1].register_forward_hook(capture)
        try:
            with torch.inference_mode():
                self.model.visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
        finally:
            handle.remove()
        if captured is None:
            raise RuntimeError("Failed to capture final vision-block state")
        return captured

    def close(self) -> None:
        for index, original in enumerate(self.original_blocks):
            self.model.visual.blocks[index] = original
        super().close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=Path("configs/baseline_qwen25_vl_3b.json"))
    parser.add_argument("--phase2-config", type=Path, default=Path("configs/phase2_feature_gap.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--bridges-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed-v2"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--route", action="append", dest="route_names")
    parser.add_argument("--ranks")
    parser.add_argument("--per-capability", type=int, default=20)
    args = parser.parse_args()

    model_config = json.loads(args.model_config.read_text(encoding="utf-8"))
    phase2_config = json.loads(args.phase2_config.read_text(encoding="utf-8"))
    routes = json.loads(args.routes.read_text(encoding="utf-8"))
    if args.route_names:
        routes = [route for route in routes if route["phase2_name"] in set(args.route_names)]
    ranks = [int(value) for value in args.ranks.split(",")] if args.ranks else phase2_config["ranks"]
    rows = []
    counts = {}
    for row in read_jsonl(args.manifest):
        count = counts.get(row["capability"], 0)
        if count < args.per_capability:
            rows.append(row)
            counts[row["capability"]] = count + 1
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runner = FinalStateRunner(model_config, args.manifest, args.output_dir / "runtime", args.data_root)
    try:
        for route_index, route in enumerate(routes, start=1):
            output_path = args.output_dir / f"{route['phase2_name']}.jsonl"
            existing = {}
            if output_path.exists():
                existing = {row["id"]: row for row in read_jsonl(output_path)}
            bridge_modules = {}
            for rank in ranks:
                state = torch.load(
                    args.bridges_dir / route["phase2_name"] / f"bridge-rank-{rank:03d}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                bridge_modules[rank] = LowRankResidualBridge(state).to(model_config["device"])
            mode = "a" if output_path.exists() else "w"
            with output_path.open(mode, encoding="utf-8", buffering=1) as handle:
                for position, row in enumerate(rows, start=1):
                    if row["id"] in existing:
                        continue
                    inputs, _, _, _ = runner._inputs(row)
                    full = runner.capture_final(inputs, [])
                    pruned = runner.capture_final(inputs, route["skip_vision_blocks"])
                    repaired = {
                        str(rank): feature_gap_metrics(full, bridge(pruned))
                        for rank, bridge in bridge_modules.items()
                    }
                    handle.write(json.dumps({
                        "id": row["id"],
                        "capability": row["capability"],
                        "route": route["phase2_name"],
                        "blocks": route["skip_vision_blocks"],
                        "identity": feature_gap_metrics(full, pruned),
                        "repaired": repaired,
                    }, sort_keys=True) + "\n")
                    if position == 1 or position % 10 == 0 or position == len(rows):
                        print(
                            f"[{route_index}/{len(routes)}] {route['phase2_name']} "
                            f"examples={position}/{len(rows)}",
                            flush=True,
                        )
                    del inputs, full, pruned, repaired
            write_json(args.output_dir / f"{route['phase2_name']}-summary.json", {
                "route": route,
                "ranks": ranks,
                "examples": len(rows),
                "capability_counts": counts,
            })
            del bridge_modules
    finally:
        runner.close()


if __name__ == "__main__":
    main()
