"""Build a source-independent held-out capability benchmark from pinned parquet files."""

from __future__ import annotations

import hashlib
import io
import random
import re
import shutil
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pyarrow.parquet as pq
from PIL import Image

from .io import read_jsonl, sha256_file, write_json, write_jsonl


SEED = 20260714
COLORS = (
    "black", "blue", "brown", "gray", "grey", "green", "orange", "pink",
    "purple", "red", "white", "yellow",
)
COLOR_PATTERN = re.compile(rf"\b({'|'.join(COLORS)})\b", re.IGNORECASE)
COLOR_SUBJECT_PATTERN = re.compile(
    rf"^is (?P<subject>.+?) (?:{'|'.join(COLORS)}) in this image\?$", re.IGNORECASE,
)


@dataclass(frozen=True)
class RawSource:
    local_name: str
    dataset_id: str
    revision: str
    filename: str
    sha256: str

    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/datasets/{self.dataset_id}/resolve/"
            f"{self.revision}/{self.filename}"
        )


RAW_SOURCES = (
    RawSource(
        "countbenchqa.parquet", "vikhyatk/CountBenchQA",
        "76d600309e9d6147bd3713b4cd431518ac1206c8", "data/test-00000-of-00001.parquet",
        "a3345e936f1c78742038c906314fb440a5b8e206d66468a64d3cdb05d81190bd",
    ),
    RawSource(
        "cvbench-2d.parquet", "nyu-visionx/CV-Bench",
        "bc284db50d036958861cb60cdd7b77612052ce0d", "test_2d.parquet",
        "33196034ef4bf3265cae4a7ff5c4071b2ff1cc21123e8e285c6a91393897ecbc",
    ),
    RawSource(
        "amber-existence.parquet", "MM-Hallu/amber-benchmark",
        "a9c4e6f0cb388c98806321422d83bec0cddab7b2",
        "discriminative-existence-00000-of-00001.parquet",
        "919b405cb459863c93dc9ad4997944d4ee9957438601a96fe8c359c8c64dbf6e",
    ),
    RawSource(
        "amber-attribute.parquet", "MM-Hallu/amber-benchmark",
        "a9c4e6f0cb388c98806321422d83bec0cddab7b2",
        "discriminative-attribute-00000-of-00001.parquet",
        "06d987c0f1a4641a5debd9f6438a2a550aa3b23010334d078980b40f399b1c98",
    ),
    RawSource(
        "textvqa-validation-0.parquet", "lmms-lab/textvqa",
        "9c0699cd19768ac5ab97568f6b3cbac4c0062884",
        "data/validation-00000-of-00003.parquet",
        "491649b5b8add2c5bde13a1e36707002fcd11898699255d65e2e568ac4fbddff",
    ),
    RawSource(
        "textvqa-validation-1.parquet", "lmms-lab/textvqa",
        "9c0699cd19768ac5ab97568f6b3cbac4c0062884",
        "data/validation-00001-of-00003.parquet",
        "909ad12a378217431548a7f4b66fda163a6def9d21e70844c96449e64206c2dd",
    ),
    RawSource(
        "textvqa-validation-2.parquet", "lmms-lab/textvqa",
        "9c0699cd19768ac5ab97568f6b3cbac4c0062884",
        "data/validation-00002-of-00003.parquet",
        "a770d03ec95fef58def04cda949d97669b7666caaf174af927856d75684e6fe6",
    ),
)


def prepared_image(image_value: dict) -> tuple[bytes, str, str]:
    """Return an RGB PNG, its file hash, and an encoding-invariant pixel hash."""
    payload = image_value.get("bytes")
    if payload is None:
        path = image_value.get("path")
        if not path:
            raise ValueError("Embedded parquet image has neither bytes nor a path")
        payload = Path(path).read_bytes()
    with Image.open(io.BytesIO(payload)) as image:
        converted = image.convert("RGB")
        try:
            pixel_hasher = hashlib.sha256()
            pixel_hasher.update(f"{converted.width}x{converted.height}:RGB:".encode())
            pixel_hasher.update(converted.tobytes())
            buffer = io.BytesIO()
            converted.save(buffer, format="PNG", compress_level=1)
        finally:
            converted.close()
    encoded = buffer.getvalue()
    return encoded, hashlib.sha256(encoded).hexdigest(), pixel_hasher.hexdigest()


def pixel_hash_file(path: Path) -> str:
    with Image.open(path) as image:
        converted = image.convert("RGB")
        try:
            hasher = hashlib.sha256()
            hasher.update(f"{converted.width}x{converted.height}:RGB:".encode())
            hasher.update(converted.tobytes())
            return hasher.hexdigest()
        finally:
            converted.close()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def choice_answer(answer: str, choices: list[str]) -> str:
    match = re.fullmatch(r"\(([A-Z])\)", answer.strip())
    if not match:
        raise ValueError(f"Unsupported multiple-choice answer: {answer!r}")
    index = ord(match.group(1)) - ord("A")
    if index >= len(choices):
        raise ValueError(f"Answer {answer!r} is outside {len(choices)} choices")
    return str(choices[index])


def modal_answer(answers: Iterable[str]) -> tuple[str, int]:
    counts = Counter(str(answer).strip().lower() for answer in answers)
    if not counts:
        raise ValueError("Cannot select a modal answer from an empty list")
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]


def color_subject(question: str) -> str | None:
    match = COLOR_SUBJECT_PATTERN.fullmatch(question.strip())
    if not match:
        return None
    subject = re.sub(r"^(?:the|a|an)\s+", "", match.group("subject"), flags=re.IGNORECASE)
    return subject.strip() or None


def stratified_select(
    candidates: list[dict], count: int, key: Callable[[dict], str], excluded_hashes: set[str]
) -> list[dict]:
    """Select deterministically across strata while retaining one QA per image."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        if candidate["image_sha256"] not in excluded_hashes:
            groups[key(candidate)].append(candidate)
    rng = random.Random(SEED)
    for group, rows in groups.items():
        rows.sort(key=lambda row: (row["stable_key"], row["locator"]))
        rng.seed(f"{SEED}:{group}")
        rng.shuffle(rows)

    selected: list[dict] = []
    used = set(excluded_hashes)
    ordered_groups = sorted(groups)
    while len(selected) < count:
        changed = False
        for group in ordered_groups:
            while groups[group]:
                candidate = groups[group].pop()
                if candidate["image_sha256"] in used:
                    continue
                selected.append(candidate)
                used.add(candidate["image_sha256"])
                changed = True
                break
            if len(selected) == count:
                break
        if not changed:
            break
    if len(selected) != count:
        raise RuntimeError(f"Selected {len(selected)} examples, expected {count}")
    return selected


class HeldoutDatasetBuilder:
    def __init__(
        self, raw_root: Path, output_root: Path, reference_manifest: Path,
        per_capability: int = 250,
    ) -> None:
        self.raw_root = raw_root.resolve()
        self.output_root = output_root.resolve()
        self.reference_manifest = reference_manifest.resolve()
        self.per_capability = per_capability
        reference_rows = list(read_jsonl(self.reference_manifest))
        reference_root = self.reference_manifest.parent.parent
        reference_images = {
            row["image_sha256"]: reference_root / row["image"] for row in reference_rows
        }
        self.reference_pixel_hashes = {
            pixel_hash_file(path) for path in reference_images.values()
        }

    def download(self) -> None:
        self.raw_root.mkdir(parents=True, exist_ok=True)
        for source in RAW_SOURCES:
            destination = self.raw_root / source.local_name
            if destination.exists():
                if sha256_file(destination) != source.sha256:
                    raise ValueError(f"Raw file hash mismatch: {destination}")
                continue
            temporary = destination.with_suffix(destination.suffix + ".part")
            with urllib.request.urlopen(source.url) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            if sha256_file(temporary) != source.sha256:
                temporary.unlink()
                raise ValueError(f"Downloaded raw file hash mismatch: {source.local_name}")
            temporary.replace(destination)

    def _paths(self, prefix: str) -> list[Path]:
        paths = sorted(self.raw_root.glob(f"{prefix}*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No parquet files found for {prefix!r} in {self.raw_root}")
        return paths

    def _iter_rows(self, paths: Iterable[Path], columns: list[str]):
        for path in paths:
            index = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=128, columns=columns):
                for row in batch.to_pylist():
                    yield f"{path.name}:{index}", row
                    index += 1

    def _image_key(self, image_value: dict) -> str:
        payload = image_value.get("bytes")
        if payload is not None:
            return hashlib.sha256(payload).hexdigest()
        path = image_value.get("path")
        if not path:
            raise ValueError("Embedded parquet image has neither bytes nor a path")
        return sha256_file(Path(path))

    def _candidate(self, locator: str, row: dict, **fields) -> dict:
        return {
            "locator": locator,
            "stable_key": hashlib.sha256(f"{SEED}:{locator}".encode()).hexdigest(),
            # Candidate selection only needs a stable image identity. The authoritative
            # canonical PNG hash is computed for selected rows during materialization.
            "image_sha256": self._image_key(row["image"]),
            **fields,
        }

    def _textvqa(self) -> list[dict]:
        candidates = []
        for locator, row in self._iter_rows(
            self._paths("textvqa-validation-"), ["image", "image_id", "question_id", "question", "answers"]
        ):
            answer, agreement = modal_answer(row["answers"])
            if (
                agreement < 7 or not answer or answer == "unanswerable"
                or "does not require reading text" in answer
            ):
                continue
            candidates.append(self._candidate(
                locator, row, source="textvqa", source_id=str(row["question_id"]),
                question=row["question"], answers=[answer], answer_format="short_text",
                capability="ocr", subtype="scene_text_vqa",
                metadata={"image_id": row["image_id"], "annotator_agreement": agreement},
            ))
        return stratified_select(candidates, self.per_capability, lambda row: str(len(row["answers"][0].split())), set())

    def _countbench(self) -> list[dict]:
        candidates = []
        for locator, row in self._iter_rows(
            self._paths("countbenchqa"), ["image", "question", "number", "text"]
        ):
            candidates.append(self._candidate(
                locator, row, source="countbenchqa", source_id=locator.rsplit(":", 1)[1],
                question=row["question"], answers=[row["number"]], answer_format="integer",
                capability="counting", subtype=f"count_{row['number']}",
                metadata={"original_caption": row["text"]},
            ))
        return stratified_select(candidates, self.per_capability, lambda row: row["subtype"], set())

    def _cvbench_spatial(self) -> list[dict]:
        candidates = []
        columns = ["image", "idx", "task", "source", "source_dataset", "question", "choices", "answer"]
        for locator, row in self._iter_rows(self._paths("cvbench-2d"), columns):
            if row["task"] != "Relation" or row["source"] != "ADE20K":
                continue
            answer = choice_answer(row["answer"], row["choices"])
            candidates.append(self._candidate(
                locator, row, source="cvbench_ade20k", source_id=str(row["idx"]),
                question=row["question"], answers=[answer], answer_format="short_text",
                capability="spatial", subtype=answer.lower(),
                metadata={"dataset": row["source_dataset"], "original_choices": row["choices"]},
            ))
        return stratified_select(candidates, self.per_capability, lambda row: row["subtype"], set())

    def _amber(self) -> tuple[list[dict], list[dict]]:
        attribute_candidates = []
        columns = ["image", "id", "query", "truth"]
        for locator, row in self._iter_rows(self._paths("amber-attribute"), columns):
            color = COLOR_PATTERN.search(row["query"])
            if not color:
                continue
            normalized_color = "gray" if color.group(1).lower() == "grey" else color.group(1).lower()
            attribute_candidates.append(self._candidate(
                locator, row, source="amber_attribute", source_id=str(row["id"]),
                question=row["query"], answers=[row["truth"]], answer_format="binary",
                capability="attribute", subtype=f"color_{normalized_color}",
                metadata={"color": normalized_color},
            ))
        attributes = stratified_select(
            attribute_candidates, self.per_capability,
            lambda row: f"{row['answers'][0]}::{row['subtype']}", set(),
        )

        excluded = {row["image_sha256"] for row in attributes}
        object_candidates = []
        for locator, row in self._iter_rows(self._paths("amber-attribute"), columns):
            subject = color_subject(row["query"])
            if row["truth"] != "yes" or subject is None:
                continue
            object_candidates.append(self._candidate(
                locator, row, source="amber_existence_positive", source_id=str(row["id"]),
                question=f"Is {subject} present in this image?", answers=["yes"], answer_format="binary",
                capability="object", subtype="existence_positive",
                metadata={"derived_from": "true AMBER color-attribute question", "subject": subject},
            ))
        for locator, row in self._iter_rows(self._paths("amber-existence"), columns):
            object_candidates.append(self._candidate(
                locator, row, source="amber_existence_negative", source_id=str(row["id"]),
                question=row["query"], answers=[row["truth"]], answer_format="binary",
                capability="object", subtype="existence_negative",
                metadata={},
            ))
        objects = stratified_select(
            object_candidates, self.per_capability, lambda row: row["answers"][0], excluded,
        )
        return attributes, objects

    def _materialize(self, selected: list[dict]) -> None:
        by_locator = {row["locator"]: row for row in selected}
        found = set()
        for path in sorted(self.raw_root.glob("*.parquet")):
            for locator, raw in self._iter_rows([path], ["image"]):
                if locator not in by_locator:
                    continue
                candidate = by_locator[locator]
                payload, digest, pixel_digest = prepared_image(raw["image"])
                candidate["image_sha256"] = digest
                candidate["image_pixel_sha256"] = pixel_digest
                relative = Path("images") / candidate["source"] / f"{digest}.png"
                destination = self.output_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    destination.write_bytes(payload)
                candidate["image"] = relative.as_posix()
                found.add(locator)
        missing = set(by_locator) - found
        if missing:
            raise RuntimeError(f"Failed to materialize selected rows: {sorted(missing)[:5]}")

    def build(self, download: bool = True) -> dict:
        if download:
            self.download()
        textvqa = self._textvqa()
        countbench = self._countbench()
        spatial = self._cvbench_spatial()
        attributes, objects = self._amber()
        selected = textvqa + countbench + spatial + attributes + objects
        self._materialize(selected)
        hashes = [row["image_sha256"] for row in selected]
        pixel_hashes = [row["image_pixel_sha256"] for row in selected]
        if len(pixel_hashes) != len(set(pixel_hashes)):
            raise RuntimeError("Held-out capabilities share images")
        if set(pixel_hashes) & self.reference_pixel_hashes:
            raise RuntimeError("Held-out images overlap the existing benchmark")

        rows = []
        for candidate in selected:
            row = {key: value for key, value in candidate.items() if key not in {"locator", "stable_key"}}
            row.update({
                "id": f"{candidate['source']}/{candidate['source_id']}",
                "source_split": "public_evaluation",
                "suite": "external_heldout",
                "split": "heldout",
            })
            rows.append(row)
        rows.sort(key=lambda row: row["id"])
        manifest_root = self.output_root / "manifests"
        manifest_root.mkdir(parents=True, exist_ok=True)
        write_jsonl(manifest_root / "heldout.jsonl", rows)
        for capability in sorted({row["capability"] for row in rows}):
            write_jsonl(
                manifest_root / f"heldout-{capability}.jsonl",
                [row for row in rows if row["capability"] == capability],
            )
        summary = {
            "status": "sealed external evaluation; do not use for route or rank selection",
            "seed": SEED,
            "per_capability": self.per_capability,
            "total_examples": len(rows),
            "unique_images": len(set(hashes)),
            "deduplication": "SHA-256 over RGB dimensions, mode, and decoded pixel bytes",
            "reference_manifest": display_path(self.reference_manifest),
            "reference_manifest_sha256": sha256_file(self.reference_manifest),
            "counts_by_capability": dict(Counter(row["capability"] for row in rows)),
            "counts_by_source": dict(Counter(row["source"] for row in rows)),
            "raw_sources": [source.__dict__ for source in RAW_SOURCES],
        }
        write_json(manifest_root / "summary.json", summary)
        summary["manifest_sha256"] = sha256_file(manifest_root / "heldout.jsonl")
        write_json(manifest_root / "summary.json", summary)
        return summary
