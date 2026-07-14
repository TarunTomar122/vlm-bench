#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from vlm_bench.heldout_builder import HeldoutDatasetBuilder
from vlm_bench.validation import validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the sealed external held-out capability set")
    parser.add_argument("--raw-root", type=Path, default=Path("data/heldout-raw-cache"))
    parser.add_argument("--output-root", type=Path, default=Path("data/external-heldout-v1"))
    parser.add_argument("--reference-manifest", type=Path, default=Path("data/processed-v2/manifests/all.jsonl"))
    parser.add_argument("--per-capability", type=int, default=250)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    builder = HeldoutDatasetBuilder(
        args.raw_root, args.output_root, args.reference_manifest, args.per_capability,
    )
    summary = builder.build(download=not args.skip_download)
    validation = validate_manifest(
        args.output_root / "manifests" / "heldout.jsonl", args.output_root,
    )
    print(json.dumps({"summary": summary, "validation": validation}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
