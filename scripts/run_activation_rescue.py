#!/usr/bin/env python3
"""Trace where full visual activations rescue a four-block identity ablation."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from vlm_bench.benchmark import BaselineRunner, VisionBlockIdentity
from vlm_bench.io import read_jsonl, write_json
from vlm_bench.scoring import score_prediction


SKIPPED_BLOCKS = (3, 5, 9, 28)
CONDITIONS = {
    "fully-pruned": None,
    "restore-after-03": 3,
    "restore-after-05": 5,
    "restore-after-09": 9,
    "restore-after-28": 28,
}


def _accuracy(rows: list[dict]) -> float:
    return sum(bool(row["correct"]) for row in rows) / len(rows) if rows else 0.0


def _summarize(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[f"capability:{row['capability']}"] .append(row)
        groups[f"source:{row['source']}"] .append(row)
    return {
        "examples": len(rows),
        "accuracy": _accuracy(rows),
        "groups": {
            key: {"examples": len(items), "accuracy": _accuracy(items)}
            for key, items in sorted(groups.items())
        },
    }


class ActivationRescueRunner(BaselineRunner):
    """Runs full-state capture and pruned-state rescue without loading a second model."""

    def __init__(self, config: dict, manifest: Path, output_dir: Path, data_root: Path) -> None:
        super().__init__(config, manifest, output_dir, data_root)
        self.original_blocks = {index: self.model.visual.blocks[index] for index in SKIPPED_BLOCKS}
        self.identity_blocks = {index: VisionBlockIdentity() for index in SKIPPED_BLOCKS}

    def _use_full_blocks(self) -> None:
        for index, block in self.original_blocks.items():
            self.model.visual.blocks[index] = block

    def _use_pruned_blocks(self) -> None:
        for index, block in self.identity_blocks.items():
            self.model.visual.blocks[index] = block

    def _capture_full_activations(self, inputs: dict) -> dict[int, torch.Tensor]:
        self._use_full_blocks()
        captured = {}
        handles = []
        for index, block in self.original_blocks.items():
            def capture(_module, _args, output, block_index=index):
                captured[block_index] = output.detach()

            handles.append(block.register_forward_hook(capture))
        try:
            with torch.inference_mode():
                self.model.visual(inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != set(SKIPPED_BLOCKS):
            raise RuntimeError("Failed to capture every requested vision block activation")
        return captured

    def _generate(self, inputs: dict, rescue_block: int | None, captured: dict[int, torch.Tensor]) -> str:
        self._use_pruned_blocks()
        handle = None
        if rescue_block is not None:
            def restore(_module, _args, _output):
                return captured[rescue_block]

            handle = self.identity_blocks[rescue_block].register_forward_hook(restore)
        try:
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    use_cache=True,
                )
        finally:
            if handle is not None:
                handle.remove()
        generated_tokens = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def run_rescue(self, baseline_dir: Path, limit: int | None = None) -> dict:
        rows = list(read_jsonl(self.manifest))
        if limit is not None:
            rows = rows[:limit]
        baseline = {row["id"]: row for row in read_jsonl(baseline_dir / "predictions.jsonl")}
        if not all(row["id"] in baseline for row in rows):
            raise ValueError("Baseline predictions do not cover the requested manifest rows")

        condition_paths = {name: self.output_dir / name / "predictions.jsonl" for name in CONDITIONS}
        existing = {
            name: {item["id"] for item in read_jsonl(path)} if path.exists() else set()
            for name, path in condition_paths.items()
        }
        handles = {}
        try:
            for name, path in condition_paths.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                handles[name] = path.open("a", encoding="utf-8", buffering=1)
            for position, row in enumerate(rows, start=1):
                pending = [name for name in CONDITIONS if row["id"] not in existing[name]]
                if not pending:
                    continue
                inputs, _, _, _ = self._inputs(row)
                captured = self._capture_full_activations(inputs)
                for name in pending:
                    prediction = self._generate(inputs, CONDITIONS[name], captured)
                    score = score_prediction(prediction, row["answers"], row["answer_format"])
                    result = {
                        **{key: row[key] for key in [
                            "id", "source", "suite", "split", "capability", "subtype", "answers", "answer_format",
                        ]},
                        "question": row["question"],
                        "prediction": prediction,
                        **score,
                    }
                    handles[name].write(json.dumps(result, sort_keys=True) + "\n")
                if position == 1 or position % 25 == 0 or position == len(rows):
                    print(f"[{position}/{len(rows)}] {row['id']} conditions={len(pending)}", flush=True)
        finally:
            for handle in handles.values():
                handle.close()
            self._use_full_blocks()

        summary = {"baseline_accuracy": _accuracy([baseline[row["id"]] for row in rows]), "conditions": {}}
        for name, path in condition_paths.items():
            predictions = list(read_jsonl(path))
            prediction_ids = {row["id"] for row in predictions}
            expected_ids = {row["id"] for row in rows}
            if prediction_ids != expected_ids:
                raise ValueError(f"Incomplete or unexpected prediction IDs for {name}")
            condition_summary = _summarize(predictions)
            condition_summary["accuracy_drop"] = summary["baseline_accuracy"] - condition_summary["accuracy"]
            summary["conditions"][name] = condition_summary
        write_json(self.output_dir / "summary.json", summary)
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_qwen25_vl_3b.json"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    runner = ActivationRescueRunner(config, args.manifest, args.output_dir, args.data_root)
    try:
        print(json.dumps(runner.run_rescue(args.baseline_dir, args.limit), indent=2, sort_keys=True))
    finally:
        runner.close()


if __name__ == "__main__":
    main()
