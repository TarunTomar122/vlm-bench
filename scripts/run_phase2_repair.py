#!/usr/bin/env python3
"""Evaluate Phase 2 low-rank feature-gap repairs with resumable predictions."""

import argparse
import json
from pathlib import Path

import torch

from vlm_bench.benchmark import BaselineRunner
from vlm_bench.io import write_json
from vlm_bench.phase2 import LowRankResidualBridge


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
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    base_config = json.loads(args.model_config.read_text(encoding="utf-8"))
    phase2_config = json.loads(args.phase2_config.read_text(encoding="utf-8"))
    routes = json.loads(args.routes.read_text(encoding="utf-8"))
    if args.route_names:
        routes = [route for route in routes if route["phase2_name"] in set(args.route_names)]
    ranks = [int(value) for value in args.ranks.split(",")] if args.ranks else phase2_config["ranks"]

    conditions = [(route, rank) for route in routes for rank in ranks]
    for index, (route, rank) in enumerate(conditions, start=1):
        condition_dir = args.output_dir / route["phase2_name"] / f"rank-{rank:03d}"
        bridge_path = args.bridges_dir / route["phase2_name"] / f"bridge-rank-{rank:03d}.pt"
        if not bridge_path.exists():
            raise FileNotFoundError(f"Missing bridge checkpoint: {bridge_path}")
        print(f"[{index}/{len(conditions)}] evaluating {route['phase2_name']} rank={rank}", flush=True)
        run_config = {**base_config, "skip_vision_blocks": route["skip_vision_blocks"]}
        runner = BaselineRunner(run_config, args.manifest, condition_dir, args.data_root)
        bridge_state = torch.load(bridge_path, map_location="cpu", weights_only=True)
        bridge = LowRankResidualBridge(bridge_state).to(run_config["device"])

        def repair(_module, _args, output):
            return bridge(output)

        handle = runner.model.visual.blocks[-1].register_forward_hook(repair)
        try:
            summary = runner.run(limit=args.limit)
            write_json(condition_dir / "repair_metadata.json", {
                "route": route,
                "rank": rank,
                "bridge_path": str(bridge_path.resolve()),
                "bridge_method": bridge_state["method"],
                "boundary": bridge_state["boundary"],
                "summary": summary,
            })
        finally:
            handle.remove()
            del bridge
            runner.close()


if __name__ == "__main__":
    main()
