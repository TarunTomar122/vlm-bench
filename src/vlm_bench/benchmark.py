import json
import gc
import platform
import statistics
import subprocess
import sys
import time
from copy import deepcopy
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed

from .io import read_jsonl, sha256_file, write_json
from .scoring import score_prediction


ANSWER_INSTRUCTIONS = {
    "binary": "Answer only yes or no.",
    "integer": "Answer using only the integer.",
    "short_text": "Answer with only the shortest correct text. Do not explain.",
}


class VisionBlockIdentity(torch.nn.Module):
    """Skip a vision block while preserving its encoder's output contract."""

    def __init__(self, output_contract: str = "tensor") -> None:
        super().__init__()
        if output_contract not in {"tensor", "tuple_first"}:
            raise ValueError(f"Unsupported vision block output contract: {output_contract}")
        self.output_contract = output_contract

    def forward(self, *args, **kwargs):
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        if hidden_states is None:
            raise TypeError("VisionBlockIdentity requires hidden_states as the first argument")
        return (hidden_states,) if self.output_contract == "tuple_first" else hidden_states


def validate_skip_blocks(skip_blocks: list[int] | None, block_count: int) -> list[int]:
    """Normalize and validate a set of vision blocks for dynamic route switching."""
    normalized = [] if skip_blocks is None else [int(block) for block in skip_blocks]
    if len(set(normalized)) != len(normalized):
        raise ValueError("skip_vision_blocks must not contain duplicates")
    invalid = [block for block in normalized if not 0 <= block < block_count]
    if invalid:
        raise ValueError(f"Invalid vision block indices: {invalid}")
    return sorted(normalized)


VISION_ADAPTERS = {
    "qwen2_5_vl": {
        "vision_module_path": ("visual",),
        "vision_blocks_path": ("visual", "blocks"),
        "block_output_contract": "tensor",
        "input_mode": "qwen",
    },
    "smolvlm": {
        "vision_module_path": ("model", "vision_model"),
        "vision_blocks_path": ("model", "vision_model", "encoder", "layers"),
        "block_output_contract": "tuple_first",
        "input_mode": "smolvlm",
    },
}


def _attribute_path(root, parts: tuple[str, ...]):
    value = root
    for part in parts:
        value = getattr(value, part)
    return value


def resolve_vision_adapter(model, requested: str | None = None) -> dict:
    """Resolve the separately-addressable vision tower for supported VLM families."""
    adapter_name = requested or getattr(model.config, "model_type", None)
    if adapter_name not in VISION_ADAPTERS:
        supported = ", ".join(sorted(VISION_ADAPTERS))
        raise ValueError(f"Unsupported VLM vision adapter {adapter_name!r}; supported: {supported}")
    adapter = {"name": adapter_name, **VISION_ADAPTERS[adapter_name]}
    blocks = _attribute_path(model, adapter["vision_blocks_path"])
    if not isinstance(blocks, torch.nn.ModuleList) or not blocks:
        raise ValueError(f"Vision block path for {adapter_name} is not a non-empty ModuleList")
    return adapter


def _nvidia_smi() -> dict:
    fields = "name,driver_version,memory.total,power.limit"
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    name, driver, memory, power = [part.strip() for part in output.split(",")]
    return {
        "name": name,
        "driver_version": driver,
        "memory_total_mib": float(memory),
        "power_limit_w": float(power),
    }


def _quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, int(0.95 * len(ordered))))
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": ordered[p95_index],
    }


class VisionTimer:
    def __init__(self, module) -> None:
        self.start = None
        self.end = None
        self.elapsed_ms = None
        self.handles = [
            module.register_forward_pre_hook(self._pre),
            module.register_forward_hook(self._post),
        ]

    def _pre(self, _module, _args) -> None:
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def _post(self, _module, _args, _output) -> None:
        self.end.record()

    def reset(self) -> None:
        self.start = None
        self.end = None
        self.elapsed_ms = None

    def finalize(self) -> float | None:
        if self.start is None or self.end is None:
            return None
        torch.cuda.synchronize()
        self.elapsed_ms = self.start.elapsed_time(self.end)
        return self.elapsed_ms

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


class BaselineRunner:
    def __init__(self, config: dict, manifest: Path, output_dir: Path, data_root: Path) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the GPU baseline")
        self.config = config
        self.manifest = manifest.resolve()
        self.output_dir = output_dir.resolve()
        self.data_root = data_root.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        set_seed(int(config["seed"]))
        torch.backends.cudnn.benchmark = False
        adapter_name = config.get("vision_adapter")
        processor_kwargs = {"revision": config.get("revision")}
        if adapter_name != "smolvlm":
            processor_kwargs["use_fast"] = False
        for setting in ("min_pixels", "max_pixels"):
            if setting in config:
                processor_kwargs[setting] = int(config[setting])
        self.processor = AutoProcessor.from_pretrained(config["model_id"], **processor_kwargs)
        dtype = torch.bfloat16 if config["dtype"] == "bfloat16" else torch.float16
        load_started = time.perf_counter()
        model_kwargs = {
            "revision": config.get("revision"),
            "torch_dtype": dtype,
            "attn_implementation": config["attention_implementation"],
            "low_cpu_mem_usage": True,
        }
        if adapter_name == "qwen2_5_vl":
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config["model_id"], **model_kwargs
            ).to(config["device"])
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                config["model_id"], **model_kwargs
            ).to(config["device"])
        self.vision_adapter = resolve_vision_adapter(self.model, adapter_name)
        self.vision_module = _attribute_path(self.model, self.vision_adapter["vision_module_path"])
        self.vision_blocks = _attribute_path(self.model, self.vision_adapter["vision_blocks_path"])
        expected_block_count = config.get("vision_block_count")
        if expected_block_count is not None and len(self.vision_blocks) != int(expected_block_count):
            raise ValueError(
                f"Configured {expected_block_count} vision blocks, but loaded {len(self.vision_blocks)}"
            )
        self._full_model_parameters = sum(parameter.numel() for parameter in self.model.parameters())
        self._full_vision_parameters = sum(parameter.numel() for parameter in self.vision_module.parameters())
        self._block_parameters = {
            str(index): sum(parameter.numel() for parameter in block.parameters())
            for index, block in enumerate(self.vision_blocks)
        }
        # Keep the original modules outside the model tree so hundreds of routes can be tested
        # without repeatedly loading the complete checkpoint.
        self._original_vision_blocks = list(self.vision_blocks)
        legacy_skip_block = config.get("skip_vision_block")
        skip_blocks = config.get("skip_vision_blocks")
        if legacy_skip_block is not None and skip_blocks is not None:
            raise ValueError("Use either skip_vision_block or skip_vision_blocks, not both")
        if skip_blocks is None and legacy_skip_block is not None:
            skip_blocks = [legacy_skip_block]
        self.set_skip_vision_blocks(skip_blocks)
        self.model.eval()
        self.generation_config = deepcopy(self.model.generation_config)
        self.generation_config.do_sample = bool(config["do_sample"])
        self.generation_config.temperature = None
        self.generation_config.top_p = None
        self.generation_config.top_k = None
        self.generation_config.max_new_tokens = int(config["max_new_tokens"])
        self.model_load_seconds = time.perf_counter() - load_started
        self.vision_timer = VisionTimer(self.vision_module)
        self.results_path = self.output_dir / "predictions.jsonl"

    def set_skip_vision_blocks(self, skip_blocks: list[int] | None) -> None:
        """Activate a new identity route while retaining the loaded checkpoint in memory."""
        normalized = validate_skip_blocks(skip_blocks, len(self._original_vision_blocks))
        blocks = self.vision_blocks
        for index, original in enumerate(self._original_vision_blocks):
            blocks[index] = original
        for block in normalized:
            blocks[block] = VisionBlockIdentity(self.vision_adapter["block_output_contract"])
        removed = sum(self._block_parameters[str(block)] for block in normalized)
        self.parameter_counts = {
            "skipped_vision_blocks": normalized,
            "full_model": self._full_model_parameters,
            "active_model": self._full_model_parameters - removed,
            "removed_model": removed,
            "full_vision": self._full_vision_parameters,
            "active_vision": self._full_vision_parameters - removed,
            "removed_vision": removed,
            "per_vision_block": self._block_parameters,
        }

    def _prompt(self, row: dict) -> str:
        instruction = ANSWER_INSTRUCTIONS[row["answer_format"]]
        return f"{row['question'].strip()}\n{instruction}"

    def _inputs(self, row: dict) -> tuple[dict, int, int, float]:
        image_path = self.data_root / row["image"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        image_content = (
            {"type": "image", "image": image}
            if self.vision_adapter["input_mode"] == "smolvlm"
            else {"type": "image"}
        )
        messages = [
            {
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": self._prompt(row)},
                ],
            }
        ]
        started = time.perf_counter()
        if self.vision_adapter["input_mode"] == "smolvlm":
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = {
            key: value.to(self.config["device"], dtype=self.model.dtype)
            if torch.is_floating_point(value)
            else value.to(self.config["device"])
            for key, value in inputs.items()
        }
        torch.cuda.synchronize()
        preprocessing_ms = (time.perf_counter() - started) * 1000
        return inputs, width, height, preprocessing_ms

    def _predict(self, row: dict) -> dict:
        torch.cuda.reset_peak_memory_stats()
        self.vision_timer.reset()
        total_started = time.perf_counter()
        inputs, width, height, preprocessing_ms = self._inputs(row)
        generation_started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                use_cache=True,
            )
        torch.cuda.synchronize()
        generation_ms = (time.perf_counter() - generation_started) * 1000
        vision_ms = self.vision_timer.finalize()
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = generated[:, input_length:]
        prediction = self.processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        total_ms = (time.perf_counter() - total_started) * 1000
        image_token_id = getattr(self.model.config, "image_token_id", None)
        visual_tokens = (
            int((inputs["input_ids"] == image_token_id).sum().item()) if image_token_id is not None else None
        )
        score = score_prediction(prediction, row["answers"], row["answer_format"])
        return {
            **{
                key: row[key]
                for key in [
                    "id",
                    "image_sha256",
                    "source",
                    "suite",
                    "split",
                    "capability",
                    "subtype",
                ]
            },
            "question": row["question"],
            "answers": row["answers"],
            "answer_format": row["answer_format"],
            "prediction": prediction,
            **score,
            "image_width": width,
            "image_height": height,
            "input_tokens": int(input_length),
            "output_tokens": int(generated_tokens.shape[1]),
            "visual_tokens": visual_tokens,
            "preprocessing_ms": preprocessing_ms,
            "vision_encoder_ms": vision_ms,
            "generation_ms": generation_ms,
            "total_ms": total_ms,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        }

    def predict(self, row: dict) -> dict:
        """Run one prediction for orchestration scripts that manage their own checkpoints."""
        return self._predict(row)

    def _existing_ids(self) -> set[str]:
        if not self.results_path.exists():
            return set()
        return {row["id"] for row in read_jsonl(self.results_path)}

    def run(self, limit: int | None = None, suite: str | None = None, split: str | None = None) -> dict:
        rows = list(read_jsonl(self.manifest))
        if suite:
            rows = [row for row in rows if row["suite"] == suite]
        if split:
            rows = [row for row in rows if row["split"] == split]
        if limit is not None:
            rows = rows[:limit]

        existing = self._existing_ids()
        pending = [row for row in rows if row["id"] not in existing]
        if pending:
            warmup_examples = int(self.config.get("warmup_examples", 0))
            for index in range(warmup_examples):
                self._predict(pending[index % len(pending)])
        mode = "a" if self.results_path.exists() else "w"
        with self.results_path.open(mode, encoding="utf-8", buffering=1) as handle:
            for index, row in enumerate(pending, start=1):
                result = self._predict(row)
                handle.write(json.dumps(result, sort_keys=True) + "\n")
                if index == 1 or index % 25 == 0 or index == len(pending):
                    print(
                        f"[{index}/{len(pending)}] {row['id']} correct={result['correct']} "
                        f"total_ms={result['total_ms']:.1f}",
                        flush=True,
                    )

        all_results = [row for row in read_jsonl(self.results_path) if row["id"] in {item["id"] for item in rows}]
        summary = summarize(all_results)
        write_json(self.output_dir / "summary.json", summary)
        metadata = {
            "config": self.config,
            "manifest": str(self.manifest),
            "manifest_sha256": sha256_file(self.manifest),
            "model_load_seconds": self.model_load_seconds,
            "model_config_commit": getattr(self.model.config, "_commit_hash", None),
            "gpu": _nvidia_smi(),
            "platform": platform.platform(),
            "python": sys.version,
            "packages": {
                name: version(name)
                for name in ["torch", "transformers", "datasets", "accelerate", "pillow", "qwen-vl-utils"]
            },
            "examples_requested": len(rows),
            "examples_completed": len(all_results),
            "parameters": self.parameter_counts,
            "vision_adapter": {
                "name": self.vision_adapter["name"],
                "blocks": len(self._original_vision_blocks),
                "block_output_contract": self.vision_adapter["block_output_contract"],
            },
        }
        write_json(self.output_dir / "run_metadata.json", metadata)
        return summary

    def close(self) -> None:
        self.vision_timer.close()
        # Explicitly release each reloaded ablation model before the next block.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        del self.model
        del self.processor
        del self.generation_config
        del self._original_vision_blocks
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def summarize(rows: list[dict]) -> dict:
    def aggregate(items: list[dict]) -> dict:
        image_groups = defaultdict(list)
        for item in items:
            image_groups[item["image_sha256"]].append(bool(item["correct"]))
        return {
            "examples": len(items),
            "accuracy": sum(bool(item["correct"]) for item in items) / len(items) if items else None,
            "all_questions_per_image_accuracy": (
                sum(all(values) for values in image_groups.values()) / len(image_groups)
                if image_groups
                else None
            ),
            "latency_ms": {
                "vision_encoder": _quantiles([item["vision_encoder_ms"] for item in items if item["vision_encoder_ms"] is not None]),
                "generation": _quantiles([item["generation_ms"] for item in items]),
                "total": _quantiles([item["total_ms"] for item in items]),
            },
            "tokens": {
                "input": _quantiles([item["input_tokens"] for item in items]),
                "output": _quantiles([item["output_tokens"] for item in items]),
                "visual": _quantiles(
                    [item["visual_tokens"] for item in items if item["visual_tokens"] is not None]
                ),
            },
            "peak_allocated_mib": max((item["peak_allocated_mib"] for item in items), default=None),
            "peak_reserved_mib": max((item["peak_reserved_mib"] for item in items), default=None),
        }

    grouped = defaultdict(list)
    for row in rows:
        grouped[f"capability:{row['capability']}"] .append(row)
        grouped[f"source:{row['source']}"] .append(row)
        grouped[f"suite:{row['suite']}"] .append(row)
        grouped[f"split:{row['split']}"] .append(row)
        grouped[f"subtype:{row['source']}:{row['subtype']}"] .append(row)
    return {
        "overall": aggregate(rows),
        "groups": {name: aggregate(items) for name, items in sorted(grouped.items())},
        "counts": dict(Counter(row["capability"] for row in rows)),
    }
