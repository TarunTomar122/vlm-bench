import hashlib
import io
import json
import random
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from huggingface_hub import HfApi
from PIL import Image

from .io import read_jsonl, sha256_file, write_json, write_jsonl


SEED = 20260713
OCR_RECOGNITION_TYPES = {
    "Artistic Text Recognition",
    "Digit String Recognition",
    "Handwriting Recognition",
    "Irregular Text Recognition",
    "Non-Semantic Text Recognition",
    "Regular Text Recognition",
}
VQAV2_COLORS = (
    "black", "blue", "brown", "gray", "green", "orange", "pink", "purple",
    "red", "white", "yellow",
)
VQAV2_ARCHIVE_URLS = {
    "questions": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "annotations": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
}
MME_CAPABILITIES = {
    "existence": "object",
    "count": "counting",
    "position": "spatial",
    "color": "attribute",
    "ocr": "ocr",
}


def _stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def _answer_format(capability: str) -> str:
    if capability in {"object", "spatial", "attribute"}:
        return "binary"
    if capability == "counting":
        return "integer"
    return "short_text"


class DatasetBuilder:
    def __init__(
        self,
        output_root: Path,
        external_per_capability: int = 300,
        attribute_per_capability: int = 0,
    ) -> None:
        self.output_root = output_root.resolve()
        self.image_root = self.output_root / "images"
        self.manifest_root = self.output_root / "manifests"
        self.external_per_capability = external_per_capability
        self.attribute_per_capability = attribute_per_capability
        self.rows: list[dict] = []
        self.source_revisions: dict[str, str] = {}
        self.api = HfApi()

    def _revision(self, dataset_id: str) -> str:
        revision = self.api.dataset_info(dataset_id).sha
        self.source_revisions[dataset_id] = revision
        return revision

    def _save_image(self, image: Image.Image, source: str, source_id: str) -> tuple[str, str]:
        converted = image.convert("RGB")
        try:
            buffer = io.BytesIO()
            converted.save(buffer, format="PNG", optimize=True)
            payload = buffer.getvalue()
        finally:
            converted.close()
        digest = hashlib.sha256(payload).hexdigest()
        relative = Path("images") / source / f"{digest}.png"
        destination = self.output_root / relative
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        return relative.as_posix(), digest

    def _add(
        self,
        *,
        source: str,
        source_id: str,
        source_split: str,
        suite: str,
        image: Image.Image,
        question: str,
        answers: Iterable[object],
        capability: str,
        subtype: str,
        metadata: dict,
        answer_format: str | None = None,
    ) -> None:
        try:
            image_path, image_sha = self._save_image(image, source, source_id)
        finally:
            image.close()
        split = "development" if _stable_int(image_sha) % 2 == 0 else "test"
        self.rows.append(
            {
                "id": f"{source}/{source_id}",
                "image": image_path,
                "image_sha256": image_sha,
                "question": question.strip(),
                "answers": [str(answer) for answer in answers],
                "answer_format": answer_format or _answer_format(capability),
                "capability": capability,
                "subtype": subtype,
                "source": source,
                "source_split": source_split,
                "suite": suite,
                "split": split,
                "metadata": metadata,
            }
        )

    def add_mme(self) -> None:
        dataset_id = "lmms-lab/MME"
        revision = self._revision(dataset_id)
        dataset = load_dataset(dataset_id, split="test", revision=revision)
        for index, row in enumerate(dataset):
            category = str(row["category"]).strip().lower()
            if category not in MME_CAPABILITIES:
                continue
            capability = MME_CAPABILITIES[category]
            self._add(
                source="mme",
                source_id=f"{row.get('question_id') or 'question'}-{index}",
                source_split="test",
                suite="controlled",
                image=row["image"],
                question=row["question"],
                answers=[row["answer"]],
                capability=capability,
                subtype=category,
                metadata={"dataset_id": dataset_id},
                answer_format="binary",
            )

    def add_vqav2_color(self) -> None:
        """Add unambiguous, high-agreement color questions from VQAv2 validation."""
        dataset_id = "VQAv2 validation (official release)"
        archive_root = self.output_root / "raw" / "vqav2"
        archive_root.mkdir(parents=True, exist_ok=True)
        archives = {}
        for name, url in VQAV2_ARCHIVE_URLS.items():
            path = archive_root / f"{name}.zip"
            if not path.exists():
                with urllib.request.urlopen(url) as response:
                    path.write_bytes(response.read())
            archives[name] = path
            self.source_revisions[f"vqav2_{name}_url"] = url
            self.source_revisions[f"vqav2_{name}_sha256"] = sha256_file(path)
        with zipfile.ZipFile(archives["questions"]) as archive:
            questions = json.load(archive.open("v2_OpenEnded_mscoco_val2014_questions.json"))["questions"]
        with zipfile.ZipFile(archives["annotations"]) as archive:
            annotations = json.load(archive.open("v2_mscoco_val2014_annotations.json"))["annotations"]
        annotations_by_id = {int(row["question_id"]): row for row in annotations}
        targets = {
            color: self.attribute_per_capability // len(VQAV2_COLORS)
            for color in VQAV2_COLORS
        }
        for color in VQAV2_COLORS[: self.attribute_per_capability % len(VQAV2_COLORS)]:
            targets[color] += 1
        selected = Counter()
        color_question = re.compile(r"^what color (?:is|are) .+\?*$", re.IGNORECASE)
        candidates = sorted(questions, key=lambda row: _stable_int(str(row["question_id"])))
        for row in candidates:
            question = str(row["question"]).strip()
            if not color_question.match(question):
                continue
            annotation = annotations_by_id[int(row["question_id"])]
            answers = [str(answer["answer"]).strip().lower() for answer in annotation["answers"]]
            counts = Counter(answers)
            answer, agreement = counts.most_common(1)[0]
            answer = "gray" if answer == "grey" else answer
            if answer not in targets or selected[answer] >= targets[answer] or agreement < 9:
                continue
            image_url = (
                "https://s3.amazonaws.com/images.cocodataset.org/val2014/"
                f"COCO_val2014_{int(row['image_id']):012d}.jpg"
            )
            with urllib.request.urlopen(image_url) as response:
                image = Image.open(io.BytesIO(response.read()))
            self._add(
                source="vqav2_color",
                source_id=str(row["question_id"]),
                source_split="validation",
                suite="external",
                image=image,
                question=question,
                answers=[answer],
                capability="attribute",
                subtype="color",
                metadata={
                    "dataset_id": dataset_id,
                    "image_id": row["image_id"],
                    "image_url": image_url,
                    "annotator_agreement": agreement,
                },
                answer_format="short_text",
            )
            selected[answer] += 1
            if sum(selected.values()) == self.attribute_per_capability:
                break
        if sum(selected.values()) != self.attribute_per_capability:
            raise RuntimeError(
                f"VQAv2 produced {sum(selected.values())} color examples, expected {self.attribute_per_capability}; "
                f"per-color counts: {dict(selected)}"
            )

    def _stratified_indices(self, rows: list[dict], count: int, key_fn) -> list[int]:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            groups[str(key_fn(row))].append(index)
        rng = random.Random(SEED)
        for indices in groups.values():
            rng.shuffle(indices)
        selected: list[int] = []
        ordered_groups = sorted(groups)
        while len(selected) < min(count, len(rows)):
            changed = False
            for group in ordered_groups:
                if groups[group]:
                    selected.append(groups[group].pop())
                    changed = True
                    if len(selected) == count:
                        break
            if not changed:
                break
        return selected

    def add_ocrbench(self) -> None:
        dataset_id = "echo840/OCRBench"
        revision = self._revision(dataset_id)
        dataset = load_dataset(dataset_id, split="test", revision=revision)
        source_datasets = dataset["dataset"]
        question_types = dataset["question_type"]
        rows = [
            {"index": index, "dataset": source_datasets[index], "question_type": question_types[index]}
            for index in range(len(dataset))
            if question_types[index] in OCR_RECOGNITION_TYPES
        ]
        indices = self._stratified_indices(
            rows,
            self.external_per_capability,
            lambda row: f"{row['dataset']}::{row['question_type']}",
        )
        for selected_index in indices:
            index = rows[selected_index]["index"]
            row = dataset[index]
            answers = row["answer"] if isinstance(row["answer"], list) else [row["answer"]]
            self._add(
                source="ocrbench",
                source_id=str(index),
                source_split="test",
                suite="external",
                image=row["image"],
                question=row["question"],
                answers=answers,
                capability="ocr",
                subtype=f"{row['dataset']}::{row['question_type']}",
                metadata={"dataset_id": dataset_id},
                answer_format="short_text",
            )

    def add_tallyqa(self) -> None:
        dataset_id = "vikhyatk/tallyqa-test"
        revision = self._revision(dataset_id)
        dataset = load_dataset(dataset_id, split="test", revision=revision, streaming=True)
        targets = {True: self.external_per_capability // 2, False: self.external_per_capability // 2}
        if self.external_per_capability % 2:
            targets[False] += 1
        counts = Counter()
        for image_index, row in enumerate(dataset):
            candidates = sorted(
                row["qa"],
                key=lambda qa: _stable_int(f"{image_index}:{qa['question']}"),
            )
            chosen = None
            for qa in candidates:
                simple = bool(qa["is_simple"])
                if counts[simple] < targets[simple]:
                    chosen = qa
                    break
            if chosen is None:
                if all(counts[key] >= target for key, target in targets.items()):
                    break
                continue
            simple = bool(chosen["is_simple"])
            counts[simple] += 1
            self._add(
                source="tallyqa",
                source_id=f"{image_index}-{_stable_int(chosen['question'])}",
                source_split="test",
                suite="external",
                image=row["image"],
                question=chosen["question"],
                answers=[chosen["answer"]],
                capability="counting",
                subtype="simple" if simple else "complex",
                metadata={"dataset_id": dataset_id, "data_source": chosen["data_source"]},
                answer_format="integer",
            )
            if all(counts[key] >= target for key, target in targets.items()):
                break
        if sum(counts.values()) != self.external_per_capability:
            raise RuntimeError(f"TallyQA produced {sum(counts.values())} examples, expected {self.external_per_capability}")

    def add_vsr(self) -> None:
        dataset_id = "cambridgeltl/vsr_random"
        revision = self._revision(dataset_id)
        url = f"https://huggingface.co/datasets/{dataset_id}/resolve/{revision}/test.jsonl"
        with urllib.request.urlopen(url) as response:
            rows = [json.loads(line) for line in response.read().decode("utf-8").splitlines() if line]
        indices = self._stratified_indices(
            rows,
            self.external_per_capability,
            lambda row: f"{row['relation']}::{row['label']}",
        )
        for index in indices:
            row = rows[index]
            image_url = str(row["image_link"]).replace(
                "http://images.cocodataset.org/",
                "https://s3.amazonaws.com/images.cocodataset.org/",
            )
            with urllib.request.urlopen(image_url) as response:
                image = Image.open(io.BytesIO(response.read())).convert("RGB")
            self._add(
                source="vsr",
                source_id=str(index),
                source_split="test",
                suite="external",
                image=image,
                question=f"Is this statement true for the image? {row['caption']}",
                answers=["yes" if int(row["label"]) == 1 else "no"],
                capability="spatial",
                subtype=str(row["relation"]),
                metadata={"dataset_id": dataset_id, "image_url": image_url},
                answer_format="binary",
            )

    def add_pope(self) -> None:
        dataset_id = "lmms-lab/POPE"
        revision = self._revision(dataset_id)
        per_split = {
            "adversarial": self.external_per_capability // 3,
            "popular": self.external_per_capability // 3,
            "random": self.external_per_capability - 2 * (self.external_per_capability // 3),
        }
        for split, count in per_split.items():
            dataset = load_dataset(dataset_id, "Full", split=split, revision=revision)
            rows = [{"answer": answer} for answer in dataset["answer"]]
            indices = self._stratified_indices(rows, count, lambda row: row["answer"])
            for index in indices:
                row = dataset[index]
                self._add(
                    source="pope",
                    source_id=f"{split}-{row['question_id']}",
                    source_split=split,
                    suite="external",
                    image=row["image"],
                    question=row["question"],
                    answers=[row["answer"]],
                    capability="object",
                    subtype=split,
                    metadata={"dataset_id": dataset_id, "image_source": row["image_source"]},
                    answer_format="binary",
                )

    def write(self) -> dict:
        if not self.rows:
            raise RuntimeError("No examples were prepared")
        ids = [row["id"] for row in self.rows]
        if len(ids) != len(set(ids)):
            duplicates = [item for item, count in Counter(ids).items() if count > 1]
            raise RuntimeError(f"Duplicate manifest IDs: {duplicates[:5]}")
        rows = sorted(self.rows, key=lambda row: row["id"])
        self.manifest_root.mkdir(parents=True, exist_ok=True)
        write_jsonl(self.manifest_root / "all.jsonl", rows)
        for suite in {row["suite"] for row in rows}:
            write_jsonl(self.manifest_root / f"{suite}.jsonl", [row for row in rows if row["suite"] == suite])
        for split in {row["split"] for row in rows}:
            write_jsonl(self.manifest_root / f"{split}.jsonl", [row for row in rows if row["split"] == split])

        smoke: list[dict] = []
        capabilities = sorted({row["capability"] for row in rows})
        for capability in capabilities:
            candidates = [
                row for row in rows if row["capability"] == capability and row["split"] == "development"
            ]
            candidates.sort(key=lambda row: _stable_int(row["id"]))
            smoke.extend(candidates[:20])
        write_jsonl(self.manifest_root / "smoke.jsonl", sorted(smoke, key=lambda row: row["id"]))

        summary = {
            "seed": SEED,
            "external_per_capability": self.external_per_capability,
            "attribute_per_capability": self.attribute_per_capability,
            "total_examples": len(rows),
            "unique_images": len({row["image_sha256"] for row in rows}),
            "counts_by_suite": dict(Counter(row["suite"] for row in rows)),
            "counts_by_capability": dict(Counter(row["capability"] for row in rows)),
            "counts_by_source": dict(Counter(row["source"] for row in rows)),
            "counts_by_split": dict(Counter(row["split"] for row in rows)),
            "source_revisions": self.source_revisions,
        }
        write_json(self.manifest_root / "summary.json", summary)
        summary["manifest_sha256"] = sha256_file(self.manifest_root / "all.jsonl")
        write_json(self.manifest_root / "summary.json", summary)
        return summary

    def build_all(self) -> dict:
        stages = [
            ("mme", "lmms-lab/MME", self.add_mme),
            ("ocrbench", "echo840/OCRBench", self.add_ocrbench),
            ("tallyqa", "vikhyatk/tallyqa-test", self.add_tallyqa),
            ("vsr", "cambridgeltl/vsr_random", self.add_vsr),
            ("pope", "lmms-lab/POPE", self.add_pope),
        ]
        if self.attribute_per_capability:
            stages.append(("vqav2_color", "VQAv2 validation", self.add_vqav2_color))
        checkpoint_root = self.output_root / "checkpoints"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        for source, dataset_id, build_stage in stages:
            checkpoint = checkpoint_root / f"{source}.jsonl"
            if checkpoint.exists():
                stage_rows = list(read_jsonl(checkpoint))
                target_count = (
                    self.attribute_per_capability if source == "vqav2_color" else self.external_per_capability
                )
                if source != "mme" and len(stage_rows) > target_count:
                    indices = self._stratified_indices(
                        stage_rows,
                        target_count,
                        lambda row: f"{row['subtype']}::{row['answers'][0] if row['answer_format'] == 'binary' else ''}",
                    )
                    stage_rows = [stage_rows[index] for index in indices]
                self.rows.extend(stage_rows)
                if dataset_id.startswith("VQAv2"):
                    archive_root = self.output_root / "raw" / "vqav2"
                    for name, url in VQAV2_ARCHIVE_URLS.items():
                        path = archive_root / f"{name}.zip"
                        if not path.exists():
                            raise RuntimeError(f"Missing pinned VQAv2 archive required by checkpoint: {path}")
                        self.source_revisions[f"vqav2_{name}_url"] = url
                        self.source_revisions[f"vqav2_{name}_sha256"] = sha256_file(path)
                else:
                    self._revision(dataset_id)
                continue
            previous_count = len(self.rows)
            build_stage()
            write_jsonl(checkpoint, self.rows[previous_count:])
        return self.write()
