#!/usr/bin/env python3
"""Seal a fresh IIIT5K OCR transfer set excluded from every route-selection stage."""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path

from datasets import load_dataset
from PIL import Image

from vlm_bench.heldout_builder import pixel_hash_file
from vlm_bench.io import read_jsonl, sha256_file, write_json, write_jsonl
from vlm_bench.validation import validate_manifest


DATASET_ID = "MiXaiLL76/IIIT5K_OCR"
REVISION = "d0a25a5bd51d121ae00ac59bfbabdc15381bd9f5"
SEED = 20260716


def pixel_hash(image: Image.Image) -> str:
    converted = image.convert("RGB")
    try:
        digest = hashlib.sha256()
        digest.update(f"{converted.width}x{converted.height}:RGB:".encode())
        digest.update(converted.tobytes())
        return digest.hexdigest()
    finally:
        converted.close()


def reference_pixels(manifests: list[Path]) -> set[str]:
    output = set()
    for manifest in manifests:
        root = manifest.parent.parent
        for row in read_jsonl(manifest):
            output.add(pixel_hash_file(root / row["image"]))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("data/fresh-ocr-iiit5k-v1"))
    parser.add_argument("--reference-manifests", type=Path, nargs="+", default=[
        Path("data/processed-v2/manifests/all.jsonl"),
        Path("data/external-heldout-v1/manifests/heldout.jsonl"),
    ])
    parser.add_argument("--examples", type=int, default=250)
    args = parser.parse_args()
    if args.examples <= 0:
        raise ValueError("--examples must be positive")
    output_root = args.output_root.resolve()
    manifest_path = output_root / "manifests" / "heldout.jsonl"
    if manifest_path.exists():
        validation = validate_manifest(manifest_path, output_root)
        print({"status": "already-sealed", "validation": validation})
        return

    excluded = reference_pixels([path.resolve() for path in args.reference_manifests])
    dataset = load_dataset(DATASET_ID, split="test", revision=REVISION)
    candidates = []
    for index, item in enumerate(dataset):
        text = str(item["text"]).strip()
        if not text:
            continue
        image = item["image"].convert("RGB")
        try:
            digest = pixel_hash(image)
            if digest in excluded:
                continue
            key = hashlib.sha256(f"{SEED}:{index}:{digest}".encode()).hexdigest()
            candidates.append((key, index, text, digest, image.copy()))
        finally:
            image.close()
    candidates.sort(key=lambda item: item[0])
    selected = candidates[:args.examples]
    if len(selected) != args.examples:
        raise RuntimeError(f"Only {len(selected)} unique non-overlapping IIIT5K examples are available")
    if len({item[3] for item in selected}) != len(selected):
        raise RuntimeError("Selected IIIT5K set contains duplicate decoded pixels")

    rows = []
    for _, index, text, decoded_hash, image in selected:
        try:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", compress_level=1)
            payload = buffer.getvalue()
        finally:
            image.close()
        encoded_hash = hashlib.sha256(payload).hexdigest()
        relative = Path("images") / "iiit5k" / f"{encoded_hash}.png"
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        rows.append({
            "id": f"iiit5k/test/{index}",
            "image": relative.as_posix(),
            "image_sha256": encoded_hash,
            "image_pixel_sha256": decoded_hash,
            "question": "What text is shown in the image? Answer with only the text.",
            "answers": [text],
            "answer_format": "short_text",
            "capability": "ocr",
            "subtype": "cropped_word_recognition",
            "source": "iiit5k",
            "source_split": "test",
            "suite": "fresh_external_heldout",
            "split": "heldout",
            "metadata": {"dataset_id": DATASET_ID, "dataset_revision": REVISION, "original_index": index},
        })
    rows.sort(key=lambda row: row["id"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(manifest_path, rows)
    validation = validate_manifest(manifest_path, output_root)
    write_json(output_root / "summary.json", {
        "status": "sealed fresh OCR transfer evaluation; prohibited for route selection",
        "dataset_id": DATASET_ID,
        "dataset_revision": REVISION,
        "split": "test",
        "selection": "SHA-256 seeded order after decoded-pixel deduplication",
        "seed": SEED,
        "reference_manifests": [{"path": str(path), "sha256": sha256_file(path)} for path in args.reference_manifests],
        "reference_pixel_overlap": 0,
        "examples": len(rows),
        "manifest_sha256": sha256_file(manifest_path),
        "validation": validation,
    })


if __name__ == "__main__":
    main()
