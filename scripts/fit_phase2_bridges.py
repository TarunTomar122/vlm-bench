#!/usr/bin/env python3
"""Measure per-depth feature gaps and fit frozen low-rank repair bridges."""

import argparse
import json
import os
from pathlib import Path

import torch

from vlm_bench.benchmark import BaselineRunner, VisionBlockIdentity
from vlm_bench.io import read_jsonl, write_json
from vlm_bench.phase2 import FeatureMoments, feature_gap_metrics, sample_tokens


def _atomic_torch_save(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


class GapCalibrationRunner(BaselineRunner):
    def __init__(self, config: dict, manifest: Path, output_dir: Path, data_root: Path) -> None:
        super().__init__(config, manifest, output_dir, data_root)
        self.original_blocks = list(self.model.visual.blocks)

    def _set_route(self, blocks: list[int]) -> None:
        skipped = set(blocks)
        for index, original in enumerate(self.original_blocks):
            self.model.visual.blocks[index] = VisionBlockIdentity() if index in skipped else original

    def capture_visual_states(self, inputs: dict, blocks: list[int]) -> list[torch.Tensor]:
        self._set_route(blocks)
        captured: list[torch.Tensor | None] = [None] * len(self.model.visual.blocks)
        handles = []
        for index, block in enumerate(self.model.visual.blocks):
            def capture(_module, _args, output, block_index=index):
                captured[block_index] = output.detach()

            handles.append(block.register_forward_hook(capture))
        try:
            with torch.inference_mode():
                self.model.visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
        finally:
            for handle in handles:
                handle.remove()
        if any(state is None for state in captured):
            raise RuntimeError("Failed to capture every vision block output")
        return [state for state in captured if state is not None]

    def close(self) -> None:
        self._set_route([])
        super().close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=Path("configs/baseline_qwen25_vl_3b.json"))
    parser.add_argument("--phase2-config", type=Path, default=Path("configs/phase2_feature_gap.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed-v2"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--route", action="append", dest="route_names")
    parser.add_argument("--ranks")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    model_config = json.loads(args.model_config.read_text(encoding="utf-8"))
    phase2_config = json.loads(args.phase2_config.read_text(encoding="utf-8"))
    routes = json.loads(args.routes.read_text(encoding="utf-8"))
    if args.route_names:
        routes = [route for route in routes if route["phase2_name"] in set(args.route_names)]
    ranks = [int(value) for value in args.ranks.split(",")] if args.ranks else phase2_config["ranks"]
    rows = list(read_jsonl(args.manifest))
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows or not routes:
        raise ValueError("Calibration requires at least one row and one route")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner = GapCalibrationRunner(model_config, args.manifest, args.output_dir / "runtime", args.data_root)
    try:
        hidden_size = int(runner.model.visual.config.hidden_size)
        for route_index, route in enumerate(routes, start=1):
            route_dir = args.output_dir / route["phase2_name"]
            route_dir.mkdir(parents=True, exist_ok=True)
            completed_path = route_dir / "fit_summary.json"
            if completed_path.exists() and all((route_dir / f"bridge-rank-{rank:03d}.pt").exists() for rank in ranks):
                print(f"[{route_index}/{len(routes)}] {route['phase2_name']} already complete", flush=True)
                continue

            checkpoint_path = route_dir / "moments-checkpoint.pt"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                moments = FeatureMoments.from_state_dict(checkpoint["moments"], torch.device("cuda"))
                processed = set(checkpoint["processed_ids"])
            else:
                moments = FeatureMoments(hidden_size, torch.device("cuda"))
                processed = set()
            gap_path = route_dir / "feature_gaps.jsonl"
            mode = "a" if gap_path.exists() else "w"
            with gap_path.open(mode, encoding="utf-8", buffering=1) as gap_handle:
                for position, row in enumerate(rows, start=1):
                    if row["id"] in processed:
                        continue
                    inputs, _, _, _ = runner._inputs(row)
                    full_states = runner.capture_visual_states(inputs, [])
                    pruned_states = runner.capture_visual_states(inputs, route["skip_vision_blocks"])
                    layer_metrics = [
                        {"block": block, **feature_gap_metrics(full, pruned)}
                        for block, (full, pruned) in enumerate(zip(full_states, pruned_states))
                    ]
                    full_final = sample_tokens(full_states[-1], int(phase2_config["max_tokens_per_example"]))
                    pruned_final = sample_tokens(pruned_states[-1], int(phase2_config["max_tokens_per_example"]))
                    moments.update(pruned_final, full_final)
                    processed.add(row["id"])
                    gap_handle.write(json.dumps({
                        "id": row["id"],
                        "capability": row["capability"],
                        "route": route["phase2_name"],
                        "blocks": route["skip_vision_blocks"],
                        "layers": layer_metrics,
                    }, sort_keys=True) + "\n")
                    checkpoint_every = int(phase2_config["checkpoint_every"])
                    if len(processed) % checkpoint_every == 0 or len(processed) == len(rows):
                        _atomic_torch_save({
                            "processed_ids": sorted(processed),
                            "moments": moments.state_dict(),
                        }, checkpoint_path)
                    if position == 1 or position % 10 == 0 or position == len(rows):
                        print(
                            f"[{route_index}/{len(routes)}] {route['phase2_name']} "
                            f"examples={len(processed)}/{len(rows)} tokens={moments.count}",
                            flush=True,
                        )
                    del inputs, full_states, pruned_states, full_final, pruned_final

            bridge_states = moments.fit(ranks, float(phase2_config["ridge"]))
            bridge_metadata = []
            for rank, state in bridge_states.items():
                state.update({
                    "route": route["phase2_name"],
                    "source_route": route["name"],
                    "skip_vision_blocks": route["skip_vision_blocks"],
                    "target_capability": route["target_capability"],
                    "boundary": "output-of-vision-block-31-before-merger",
                    "method": "centered ridge residual map followed by truncated SVD",
                })
                _atomic_torch_save(state, route_dir / f"bridge-rank-{rank:03d}.pt")
                bridge_metadata.append({
                    key: value for key, value in state.items()
                    if key not in {"left", "right", "bias"}
                })
            write_json(completed_path, {
                "route": route,
                "examples": len(processed),
                "calibration_tokens": moments.count,
                "bridges": bridge_metadata,
                "claim_boundary": "SCP-inspired calibration bridge; not a reproduction of Short-LVLM SCP",
            })
            checkpoint_path.unlink(missing_ok=True)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
