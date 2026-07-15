#!/usr/bin/env python3
"""Prepare frozen, source-aware manifests for robust route search."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vlm_bench.io import read_jsonl, sha256_file, write_json, write_jsonl  # noqa: E402


REQUIRED_FIELDS = {"id", "image_sha256", "capability", "source", "split"}


def _stable_key(seed: int, label: str, *parts: str) -> str:
    value = "\0".join((str(seed), label, *parts)).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _stratum_key(capability: str, source: str) -> str:
    return f"{capability}::{source}"


def _counts(rows: Iterable[dict], fields: tuple[str, ...]) -> dict[str, int]:
    counts = Counter(
        "::".join(str(row[field]) for field in fields) for row in rows
    )
    return dict(sorted(counts.items()))


def _manifest_stats(rows: list[dict]) -> dict[str, object]:
    return {
        "examples": len(rows),
        "images": len({row["image_sha256"] for row in rows}),
        "capability_example_counts": _counts(rows, ("capability",)),
        "source_example_counts": _counts(rows, ("source",)),
        "capability_source_example_counts": _counts(rows, ("capability", "source")),
        "capability_source_image_counts": {
            key: len(images)
            for key, images in sorted(_stratum_images(rows).items())
        },
    }


def _stratum_images(rows: Iterable[dict]) -> dict[str, set[str]]:
    images: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        images[_stratum_key(row["capability"], row["source"])].add(
            row["image_sha256"]
        )
    return images


def _validate_source(rows: list[dict], search_split: str, selection_split: str) -> None:
    if not rows:
        raise ValueError("source manifest is empty")

    missing = [
        (index, sorted(REQUIRED_FIELDS - row.keys()))
        for index, row in enumerate(rows, start=1)
        if REQUIRED_FIELDS - row.keys()
    ]
    if missing:
        index, fields = missing[0]
        raise ValueError(f"row {index} is missing required fields: {fields}")

    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        duplicates = sorted(key for key, count in Counter(ids).items() if count > 1)
        raise ValueError(f"duplicate example ids in source manifest: {duplicates[:5]}")

    expected_splits = {search_split, selection_split}
    actual_splits = {row["split"] for row in rows}
    if actual_splits != expected_splits:
        raise ValueError(
            f"expected exactly source splits {sorted(expected_splits)}, "
            f"found {sorted(actual_splits)}"
        )

    image_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        image_splits[row["image_sha256"]].add(row["split"])
    overlapping = sorted(
        image for image, splits in image_splits.items() if len(splits) > 1
    )
    if overlapping:
        raise ValueError(
            "source split is not image-disjoint; overlapping image hashes include "
            f"{overlapping[:5]}"
        )


def _balanced_scout(
    rows: list[dict],
    *,
    quotas: dict[str, int],
    seed: int,
    label: str,
) -> list[dict]:
    available = _stratum_images(rows)
    if not available or set(available) != set(quotas):
        raise ValueError(f"no search rows available for scout {label!r}")

    image_strata: dict[str, set[str]] = defaultdict(set)
    for stratum, images in available.items():
        for image in images:
            image_strata[image].add(stratum)

    shared = sorted(
        (image for image, strata in image_strata.items() if len(strata) > 1),
        key=lambda image: (_stable_key(seed, label, "shared", image), image),
    )
    if len(shared) > 20:
        raise ValueError(
            f"scout {label!r} has {len(shared)} cross-stratum images; "
            "exact complete-group balancing supports at most 20"
        )

    single_by_stratum: dict[str, list[str]] = {}
    for stratum, images in sorted(available.items()):
        quota = quotas[stratum]
        if quota <= 0:
            raise ValueError(f"scout quota for {stratum} must be positive")
        if len(images) < quota:
            raise ValueError(
                f"scout {label!r} requests {quota} images from {stratum}, "
                f"but only {len(images)} are available"
            )
        single_by_stratum[stratum] = sorted(
            (image for image in images if len(image_strata[image]) == 1),
            key=lambda image: (_stable_key(seed, label, stratum, image), image),
        )

    selected_shared: tuple[str, ...] | None = None
    for size in range(len(shared) + 1):
        for candidate in combinations(shared, size):
            shared_counts = Counter(
                stratum for image in candidate for stratum in image_strata[image]
            )
            if all(
                shared_counts[stratum] <= quota
                and quota - shared_counts[stratum] <= len(single_by_stratum[stratum])
                for stratum, quota in quotas.items()
            ):
                selected_shared = candidate
                break
        if selected_shared is not None:
            break
    if selected_shared is None:
        raise ValueError(f"cannot satisfy complete-image scout quotas for {label!r}")

    selected_images = set(selected_shared)
    shared_counts = Counter(
        stratum for image in selected_shared for stratum in image_strata[image]
    )
    for stratum, quota in sorted(quotas.items()):
        needed = quota - shared_counts[stratum]
        selected_images.update(single_by_stratum[stratum][:needed])

    # Selecting by global image identity retains every row attached to an image,
    # including the rare images that contribute to more than one stratum.
    return [row for row in rows if row["image_sha256"] in selected_images]


def _preserves_complete_image_groups(scout: list[dict], source: list[dict]) -> bool:
    scout_ids: dict[str, set[str]] = defaultdict(set)
    source_ids: dict[str, set[str]] = defaultdict(set)
    for row in scout:
        scout_ids[row["image_sha256"]].add(row["id"])
    for row in source:
        source_ids[row["image_sha256"]].add(row["id"])
    return all(ids == source_ids[image] for image, ids in scout_ids.items())


def _file_record(path: Path, rows: list[dict]) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        **_manifest_stats(rows),
    }


def _load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"seed", "data", "scouts"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"config is missing required keys: {sorted(missing)}")
    return config


def prepare(config_path: Path, manifest_path: Path, output_dir: Path) -> dict[str, object]:
    config = _load_config(config_path)
    actual_manifest_sha256 = sha256_file(manifest_path)
    expected_manifest_sha256 = config["data"].get("manifest_sha256")
    if expected_manifest_sha256 and actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "source manifest SHA-256 does not match the frozen config: "
            f"expected {expected_manifest_sha256}, found {actual_manifest_sha256}"
        )

    rows = list(read_jsonl(manifest_path))
    search_split = str(config["data"]["search_source_split"])
    selection_split = str(config["data"]["selection_source_split"])
    _validate_source(rows, search_split, selection_split)

    expected_examples = config["data"].get("expected_examples")
    if expected_examples is not None and len(rows) != int(expected_examples):
        raise ValueError(
            f"source manifest has {len(rows)} rows; frozen config expects "
            f"{expected_examples}"
        )
    expected_images = config["data"].get("expected_images")
    actual_images = len({row.get("image_sha256") for row in rows})
    if expected_images is not None and actual_images != int(expected_images):
        raise ValueError(
            f"source manifest has {actual_images} images; frozen config expects "
            f"{expected_images}"
        )
    search = [row for row in rows if row["split"] == search_split]
    selection = [row for row in rows if row["split"] == selection_split]
    search_images = {row["image_sha256"] for row in search}
    selection_images = {row["image_sha256"] for row in selection}
    search_ids = {row["id"] for row in search}
    selection_ids = {row["id"] for row in selection}

    seed = int(config["seed"])
    generic_quota = int(config["scouts"]["generic_images_per_capability_source"])
    task_target_quota = int(config["scouts"]["task_target_images_per_source"])
    task_collateral_quota = int(
        config["scouts"]["task_collateral_images_per_non_target_stratum"]
    )
    development_evaluation_limit = int(
        config["scouts"]["development_evaluation_max_images_per_capability_source"]
    )
    search_strata = _stratum_images(search)
    generic_scout = _balanced_scout(
        search,
        quotas={stratum: generic_quota for stratum in search_strata},
        seed=seed,
        label="generic",
    )

    capabilities = sorted({row["capability"] for row in rows})
    task_scouts = {
        capability: _balanced_scout(
            search,
            quotas={
                stratum: (
                    task_target_quota
                    if stratum.startswith(f"{capability}::")
                    else task_collateral_quota
                )
                for stratum in search_strata
            },
            seed=seed,
            label=f"task:{capability}",
        )
        for capability in capabilities
    }
    development_evaluation = _balanced_scout(
        search,
        quotas={
            stratum: min(development_evaluation_limit, len(images))
            for stratum, images in search_strata.items()
        },
        seed=seed,
        label="development-evaluation",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    search_path = output_dir / "search.jsonl"
    selection_path = output_dir / "selection.jsonl"
    development_evaluation_path = output_dir / "development-evaluation.jsonl"
    generic_path = output_dir / "scout-generic.jsonl"
    write_jsonl(search_path, search)
    write_jsonl(selection_path, selection)
    write_jsonl(development_evaluation_path, development_evaluation)
    write_jsonl(generic_path, generic_scout)

    task_paths: dict[str, Path] = {}
    for capability, scout in task_scouts.items():
        path = output_dir / f"scout-task-{capability}.jsonl"
        write_jsonl(path, scout)
        task_paths[capability] = path

    generic_counts = _stratum_images(generic_scout)
    development_evaluation_counts = _stratum_images(development_evaluation)
    task_counts = {
        capability: _stratum_images(scout)
        for capability, scout in task_scouts.items()
    }
    all_ids = {row["id"] for row in rows}
    assertions = {
        "search_selection_example_overlap_is_zero": len(search_ids & selection_ids) == 0,
        "search_selection_image_overlap_is_zero": len(search_images & selection_images) == 0,
        "search_selection_partition_all_source_examples": search_ids | selection_ids == all_ids,
        "search_selection_preserve_complete_image_groups": (
            _preserves_complete_image_groups(search, rows)
            and _preserves_complete_image_groups(selection, rows)
        ),
        "search_preserves_development_rows": len(search) == sum(
            row["split"] == search_split for row in rows
        ),
        "selection_preserves_test_rows": len(selection) == sum(
            row["split"] == selection_split for row in rows
        ),
        "generic_scout_is_search_subset": {
            row["id"] for row in generic_scout
        } <= search_ids,
        "generic_scout_balanced_by_capability_source": all(
            len(images) == generic_quota for images in generic_counts.values()
        ) and set(generic_counts) == set(search_strata),
        "generic_scout_preserves_complete_image_groups": (
            _preserves_complete_image_groups(generic_scout, search)
        ),
        "development_evaluation_is_search_subset": {
            row["id"] for row in development_evaluation
        } <= search_ids,
        "development_evaluation_has_expected_stratum_coverage": all(
            len(development_evaluation_counts[stratum])
            == min(development_evaluation_limit, len(images))
            for stratum, images in search_strata.items()
        ) and set(development_evaluation_counts) == set(search_strata),
        "development_evaluation_preserves_complete_image_groups": (
            _preserves_complete_image_groups(development_evaluation, search)
        ),
        "task_scouts_are_search_subsets": all(
            {row["id"] for row in scout} <= search_ids
            for scout in task_scouts.values()
        ),
        "task_scouts_cover_every_capability_source": all(
            set(counts) == set(search_strata) for counts in task_counts.values()
        ),
        "task_scouts_balance_target_sources": all(
            all(
                len(images) == task_target_quota
                for stratum, images in counts.items()
                if stratum.startswith(f"{capability}::")
            )
            for capability, counts in task_counts.items()
        ),
        "task_scouts_balance_collateral_strata": all(
            all(
                len(images) == task_collateral_quota
                for stratum, images in counts.items()
                if not stratum.startswith(f"{capability}::")
            )
            for capability, counts in task_counts.items()
        ),
        "task_scouts_preserve_complete_image_groups": all(
            _preserves_complete_image_groups(scout, search)
            for scout in task_scouts.values()
        ),
    }
    if not all(assertions.values()):
        failed = sorted(key for key, value in assertions.items() if not value)
        raise AssertionError(f"preparation assertions failed: {failed}")

    summary_path = output_dir / "summary.json"
    summary = {
        "schema_version": 1,
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "seed": seed,
        },
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": actual_manifest_sha256,
            **_manifest_stats(rows),
        },
        "partition_policy": {
            "search_source_split": search_split,
            "selection_source_split": selection_split,
            "resampled": False,
            "selection_example_fraction": len(selection) / len(rows),
            "selection_image_fraction": len(selection_images)
            / len(search_images | selection_images),
        },
        "outputs": {
            "search": _file_record(search_path, search),
            "selection": _file_record(selection_path, selection),
            "development_evaluation": _file_record(
                development_evaluation_path, development_evaluation
            ),
            "scout_generic": _file_record(generic_path, generic_scout),
            "scout_task": {
                capability: _file_record(task_paths[capability], task_scouts[capability])
                for capability in capabilities
            },
        },
        "overlap": {
            "search_selection_examples": len(search_ids & selection_ids),
            "search_selection_images": len(search_images & selection_images),
        },
        "assertions": assertions,
    }
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "robust_route_search.json",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = _load_config(config_path)
    manifest_path = (
        args.manifest.resolve()
        if args.manifest
        else (REPO_ROOT / config["data"]["manifest"]).resolve()
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (REPO_ROOT / config["data"]["prepared_dir"]).resolve()
    )
    summary = prepare(config_path, manifest_path, output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
