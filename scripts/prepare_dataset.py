#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from vlm_bench.dataset_builder import DatasetBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic VLM capability manifests")
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--external-per-capability", type=int, default=300)
    parser.add_argument(
        "--attribute-per-capability",
        type=int,
        default=0,
        help="Add this many high-agreement VQAv2 color examples (0 preserves the V1 manifest).",
    )
    args = parser.parse_args()
    builder = DatasetBuilder(
        args.output_root,
        args.external_per_capability,
        args.attribute_per_capability,
    )
    print(json.dumps(builder.build_all(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
