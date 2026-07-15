#!/usr/bin/env python3
"""Run resumable, deterministic GPU search for robust fixed-budget routes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Iterable

from vlm_bench.benchmark import BaselineRunner
from vlm_bench.io import read_jsonl, sha256_file
from vlm_bench.robust_search import (
    RouteCandidate,
    crossover_routes,
    generic_robust_objective,
    jaccard_route_stability,
    mutate_route,
    normalize_route,
    select_pareto_survivors,
    source_balanced_paired_metrics,
    task_robust_objective,
)


FAMILIES = ("generic", "attribute", "counting", "object", "ocr", "spatial")
DEFAULT_SEARCH_CONFIG = Path("configs/robust_route_search.json")
DEFAULT_MODEL_CONFIG = Path("configs/baseline_qwen25_vl_3b.json")
DEFAULT_OUTPUT_DIR = Path("results/robust-route-search-qwen25-vl-3b")
DEFAULT_BASELINE = DEFAULT_OUTPUT_DIR / "baseline" / "predictions.jsonl"
DEFAULT_PHASE3_ROOT = Path("results/phase3-interaction-search-qwen25-vl-3b")
DEFAULT_SENSITIVITY = Path("results/task-route-design-qwen25-vl-3b/sensitivity.json")
DEFAULT_ROUTES = Path("results/task-route-design-qwen25-vl-3b/routes.json")
DEFAULT_DATA_ROOT = Path("data/processed-v2")
DEFAULT_ALL_MANIFEST = DEFAULT_DATA_ROOT / "manifests" / "all.jsonl"
DEFAULT_DEVELOPMENT_MANIFEST = DEFAULT_DATA_ROOT / "manifests" / "development.jsonl"
DEFAULT_TEST_MANIFEST = DEFAULT_DATA_ROOT / "manifests" / "test.jsonl"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def route_key(route: Iterable[int]) -> str:
    blocks = normalize_route(route)
    return f"k{len(blocks):02d}-" + "-".join(f"b{block:02d}" for block in blocks)


def nested(config: dict, *paths: str, default: Any = None) -> Any:
    for dotted in paths:
        current: Any = config
        for part in dotted.split("."):
            if not isinstance(current, dict) or part not in current:
                break
            current = current[part]
        else:
            return current
    return default


def required_int(config: dict, *paths: str) -> int:
    value = nested(config, *paths)
    if value is None:
        raise ValueError(f"Frozen config is missing one of: {', '.join(paths)}")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Expected a positive integer at {paths}; got {value!r}")
    return value


def configured_path(config: dict, *paths: str, default: Path) -> Path:
    value = nested(config, *paths)
    return Path(value) if value is not None else default


def prepared_dir(config: dict) -> Path:
    return configured_path(
        config,
        "data.prepared_dir",
        "prepared_dir",
        default=DEFAULT_DATA_ROOT / "robust-route-search" / "prepared",
    )


def first_existing(candidates: Iterable[Path], label: str) -> Path:
    ordered = list(dict.fromkeys(candidates))
    for path in ordered:
        if path.exists():
            return path
    rendered = "\n  ".join(str(path) for path in ordered)
    raise FileNotFoundError(f"Could not resolve {label}; tried:\n  {rendered}")


def resolve_stage_manifests(config: dict, family: str) -> dict[str, Path]:
    root = prepared_dir(config)
    scout_value = nested(
        config,
        f"prepared_manifests.scout.{family}",
        f"manifests.scout.{family}",
        f"scouts.manifests.{family}",
    )
    development_value = nested(
        config,
        "prepared_manifests.development_evaluation",
        "manifests.development_evaluation",
        "data.development_evaluation_manifest",
    )
    selection_value = nested(
        config,
        "prepared_manifests.selection",
        "manifests.selection",
        "data.selection_manifest",
    )
    scout_candidates = [Path(scout_value)] if scout_value else []
    scout_candidates.extend(
        [
            root / f"scout-{family}.jsonl",
            root / f"scout-task-{family}.jsonl",
            root / f"{family}-scout.jsonl",
            root / "scout" / f"{family}.jsonl",
        ]
    )
    development_candidates = [Path(development_value)] if development_value else []
    development_candidates.append(root / "development-evaluation.jsonl")
    selection_candidates = [Path(selection_value)] if selection_value else []
    selection_candidates.extend([root / "selection.jsonl", root / "test.jsonl"])
    return {
        "scout": first_existing(scout_candidates, f"{family} scout manifest"),
        "development": first_existing(
            development_candidates,
            "stage-2 development-evaluation manifest",
        ),
        "selection": first_existing(selection_candidates, "stage-3 selection manifest"),
    }


def reject_sealed_path(path: Path) -> None:
    lowered = str(path.resolve()).lower()
    if "external-heldout" in lowered or "/heldout" in lowered:
        raise ValueError(f"Sealed external-heldout input is forbidden: {path}")


def load_unique_rows(path: Path) -> list[dict]:
    reject_sealed_path(path)
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Manifest contains duplicate IDs: {path}")
    return rows


def load_prediction_map(path: Path) -> dict[str, dict]:
    reject_sealed_path(path)
    indexed: dict[str, dict] = {}
    for row in read_jsonl(path):
        row_id = str(row["id"])
        if row_id in indexed:
            raise ValueError(f"Duplicate prediction ID {row_id!r} in {path}")
        indexed[row_id] = row
    return indexed


def validate_stage_rows(
    manifests: dict[str, Path],
    rows: dict[str, list[dict]],
    baseline: dict[str, dict],
) -> None:
    development_ids = {str(row["id"]) for row in read_jsonl(DEFAULT_DEVELOPMENT_MANIFEST)}
    test_ids = {str(row["id"]) for row in read_jsonl(DEFAULT_TEST_MANIFEST)}
    all_ids = {str(row["id"]) for row in read_jsonl(DEFAULT_ALL_MANIFEST)}
    if development_ids & test_ids or development_ids | test_ids != all_ids:
        raise RuntimeError("Authoritative processed-v2 development/test partition is inconsistent")
    for stage in ("scout", "development"):
        ids = {str(row["id"]) for row in rows[stage]}
        if not ids <= development_ids:
            raise ValueError(f"{manifests[stage]} contains rows outside processed-v2 development")
    selection_ids = {str(row["id"]) for row in rows["selection"]}
    if not selection_ids <= test_ids:
        raise ValueError(f"{manifests['selection']} contains rows outside processed-v2 test")
    required = {str(row["id"]) for stage_rows in rows.values() for row in stage_rows}
    missing = sorted(required - baseline.keys())
    if missing:
        raise ValueError(
            f"Baseline does not yet cover {len(missing)} prepared IDs; first missing IDs: {missing[:5]}"
        )


def objective_to_dict(objective: Any) -> dict:
    return {
        **asdict(objective),
        "objective_vector": list(objective.objective_vector()),
        "selection_score": objective.selection_score,
    }


def objective_from_dict(value: dict) -> Any:
    from vlm_bench.robust_search import RobustObjective

    return RobustObjective(
        kind=value["kind"],
        mean_drop_pp=float(value["mean_drop_pp"]),
        worst_source_drop_pp=float(value["worst_source_drop_pp"]),
        source_variability_pp=float(value["source_variability_pp"]),
        collateral_drop_pp=float(value.get("collateral_drop_pp", 0.0)),
        target_capability=value.get("target_capability"),
    )


class PredictionCache:
    """Append-only route/example cache with lazy import from Phase3 routes."""

    def __init__(self, path: Path, phase3_root: Path | None, fingerprint: str) -> None:
        self.path = path
        self.fingerprint = fingerprint
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: dict[tuple[str, str], dict] = {}
        self.phase3: dict[tuple[int, ...], list[Path]] = {}
        self._load_and_repair()
        if phase3_root is not None and phase3_root.exists():
            self._index_phase3(phase3_root)

    def _load_and_repair(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("rb+") as handle:
            while True:
                offset = handle.tell()
                raw = handle.readline()
                if not raw:
                    break
                try:
                    record = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    if handle.read(1):
                        raise ValueError(f"Corrupt non-final cache record in {self.path}")
                    handle.truncate(offset)
                    break
                prediction = record["prediction"]
                if record.get("fingerprint") != self.fingerprint:
                    raise ValueError(
                        f"Cached prediction fingerprint does not match the frozen inputs: {self.path}"
                    )
                key = (str(record["route_key"]), str(prediction["id"]))
                if key in self.rows and self.rows[key] != prediction:
                    raise ValueError(f"Conflicting cached prediction for {key}")
                self.rows[key] = prediction

    def _index_phase3(self, root: Path) -> None:
        for metadata_path in sorted(root.rglob("route.json")):
            predictions_path = metadata_path.with_name("predictions.jsonl")
            if not predictions_path.exists():
                continue
            try:
                metadata = load_json(metadata_path)
                route = normalize_route(metadata["blocks"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            self.phase3.setdefault(route, []).append(predictions_path)

    def _append(self, route: tuple[int, ...], prediction: dict, source: str) -> None:
        key = (route_key(route), str(prediction["id"]))
        if key in self.rows:
            existing = self.rows[key]
            comparable = ("prediction", "correct", "capability", "source")
            if any(existing.get(field) != prediction.get(field) for field in comparable):
                raise ValueError(f"Conflicting cached prediction for {key}")
            return
        record = {
            "blocks": list(route),
            "cache_version": 1,
            "fingerprint": self.fingerprint,
            "prediction": prediction,
            "route_key": key[0],
            "source": source,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.rows[key] = prediction

    def import_phase3(self, route: tuple[int, ...], wanted_ids: set[str]) -> None:
        missing = wanted_ids - {
            row_id for cached_route, row_id in self.rows if cached_route == route_key(route)
        }
        for path in self.phase3.get(route, []):
            if not missing:
                break
            for prediction in read_jsonl(path):
                row_id = str(prediction["id"])
                if row_id in missing:
                    self._append(route, prediction, f"phase3:{path}")
                    missing.remove(row_id)

    def predictions_for(
        self,
        runner: BaselineRunner,
        route: tuple[int, ...],
        rows: list[dict],
        label: str,
    ) -> dict[str, dict]:
        wanted_ids = {str(row["id"]) for row in rows}
        self.import_phase3(route, wanted_ids)
        runner.set_skip_vision_blocks(list(route))
        pending = [
            row for row in rows if (route_key(route), str(row["id"])) not in self.rows
        ]
        for index, row in enumerate(pending, start=1):
            prediction = runner.predict(row)
            self._append(route, prediction, "robust-search")
            if index == 1 or index % 25 == 0 or index == len(pending):
                print(f"[{label}] {index}/{len(pending)} new predictions", flush=True)
        return {
            str(row["id"]): self.rows[(route_key(route), str(row["id"]))]
            for row in rows
        }


def ranking_for_family(sensitivity: dict, family: str) -> list[int]:
    if family == "generic":
        return [int(block) for block in sensitivity["generic_macro_ranking_least_to_most_sensitive"]]
    rankings = sensitivity["capability_rankings_least_to_most_sensitive"]
    return [int(block) for block in rankings[family]]


def resize_prior(route: Iterable[int], ranking: list[int], k: int) -> tuple[int, ...]:
    present = set(normalize_route(route))
    ordered = [block for block in ranking if block in present]
    ordered.extend(block for block in ranking if block not in present)
    return normalize_route(ordered[:k], k=k, allowed_blocks=ranking)


def recursive_blocks(value: Any) -> Iterable[list[int]]:
    if isinstance(value, dict):
        blocks = value.get("blocks")
        if isinstance(blocks, list) and all(isinstance(block, int) for block in blocks):
            yield blocks
        for child in value.values():
            yield from recursive_blocks(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_blocks(child)


def phase3_family_section(summary: dict, family: str) -> Any:
    if family == "generic":
        return summary
    capabilities = summary.get("capabilities")
    if isinstance(capabilities, dict) and family in capabilities:
        return capabilities[family]
    beam_routes = summary.get("beam_routes")
    if isinstance(beam_routes, dict) and family in beam_routes:
        return beam_routes[family]
    return summary


def prior_routes(
    family: str,
    k: int,
    sensitivity: dict,
    routes: list[dict],
    phase3_summaries: list[dict],
) -> list[tuple[int, ...]]:
    ranking = ranking_for_family(sensitivity, family)
    priors: list[tuple[int, ...]] = [normalize_route(ranking[:k], k=k)]
    for item in routes:
        assignments = item.get("assignments", [])
        matches = any(
            (
                assignment.get("route_type") == "generic_macro"
                if family == "generic"
                else assignment.get("route_type") == "task_specific"
                and assignment.get("capability") == family
            )
            for assignment in assignments
        )
        if matches:
            priors.append(resize_prior(item["skip_vision_blocks"], ranking, k))
    for summary in phase3_summaries:
        for blocks in recursive_blocks(phase3_family_section(summary, family)):
            priors.append(resize_prior(blocks, ranking, k))
    return list(dict.fromkeys(priors))


def random_route(rng: random.Random, allowed_blocks: tuple[int, ...], k: int) -> tuple[int, ...]:
    return normalize_route(rng.sample(allowed_blocks, k), k=k, allowed_blocks=allowed_blocks)


def initial_population(
    priors: list[tuple[int, ...]],
    allowed_blocks: tuple[int, ...],
    k: int,
    population_size: int,
    seed: int,
) -> list[tuple[int, ...]]:
    rng = random.Random(f"initial:{seed}:k{k}")
    population = list(priors[: max(1, population_size // 2)])
    seen = set(population)
    while len(population) < population_size:
        candidate = random_route(rng, allowed_blocks, k)
        if candidate not in seen:
            population.append(candidate)
            seen.add(candidate)
    return population


def next_population(
    survivors: list[tuple[int, ...]],
    allowed_blocks: tuple[int, ...],
    population_size: int,
    seed: int,
    generation: int,
) -> list[tuple[int, ...]]:
    population = list(survivors)
    seen = set(population)
    attempt = 0
    while len(population) < population_size:
        left = survivors[attempt % len(survivors)]
        right = survivors[(attempt * 5 + 1) % len(survivors)]
        crossed = crossover_routes(left, right, seed=f"cross:{seed}:{generation}:{attempt}")
        candidate = mutate_route(
            crossed,
            allowed_blocks,
            seed=f"mutate:{seed}:{generation}:{attempt}",
        )
        if candidate not in seen:
            population.append(candidate)
            seen.add(candidate)
        attempt += 1
        if attempt > population_size * 100:
            rng = random.Random(f"fallback:{seed}:{generation}:{attempt}")
            candidate = random_route(rng, allowed_blocks, len(left))
            if candidate not in seen:
                population.append(candidate)
                seen.add(candidate)
    return population


def objective_for(
    family: str,
    baseline: dict[str, dict],
    predictions: dict[str, dict],
    rows: list[dict],
) -> tuple[Any, dict]:
    ids = [str(row["id"]) for row in rows]
    paired = source_balanced_paired_metrics(
        {row_id: baseline[row_id] for row_id in ids},
        {row_id: predictions[row_id] for row_id in ids},
    )
    objective = (
        generic_robust_objective(paired)
        if family == "generic"
        else task_robust_objective(paired, family)
    )
    return objective, {"groups": [asdict(group) for group in paired.groups]}


class FamilyJob:
    def __init__(
        self,
        family: str,
        config: dict,
        model_config: dict,
        manifests: dict[str, Path],
        rows: dict[str, list[dict]],
        baseline: dict[str, dict],
        sensitivity: dict,
        routes: list[dict],
        phase3_summaries: list[dict],
        output_dir: Path,
        input_hashes: dict[str, Any],
        phase3_root: Path | None,
    ) -> None:
        self.family = family
        self.config = config
        self.model_config = model_config
        self.manifests = manifests
        self.rows = rows
        self.baseline = baseline
        self.sensitivity = sensitivity
        self.routes = routes
        self.phase3_summaries = phase3_summaries
        self.job_dir = output_dir / "families" / family
        self.state_path = self.job_dir / "state.json"
        self.progress_path = self.job_dir / "progress.json"
        self.summary_path = self.job_dir / "summary.json"
        self.input_hashes = input_hashes
        fingerprint_payload = json.dumps(
            input_hashes, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        cache_fingerprint = hashlib.sha256(fingerprint_payload).hexdigest()
        self.cache = PredictionCache(
            self.job_dir / "route-predictions.jsonl",
            phase3_root,
            cache_fingerprint,
        )
        self.state = self._load_state()
        self.runner: BaselineRunner | None = None

    def _load_state(self) -> dict:
        if self.state_path.exists():
            state = load_json(self.state_path)
            if state.get("input_hashes") != self.input_hashes:
                raise ValueError(
                    f"Cannot resume {self.family}: frozen config or input manifest hashes changed"
                )
            state["status"] = "running"
            state.pop("error", None)
            return state
        return {
            "family": self.family,
            "status": "running",
            "started_at": now(),
            "input_hashes": self.input_hashes,
            "evaluations": {"scout": {}, "development": {}, "selection": {}},
            "searches": {},
        }

    def checkpoint(self, current: dict | None = None) -> None:
        self.state["updated_at"] = now()
        atomic_write_json(self.state_path, self.state)
        counts = {
            stage: len(evaluations)
            for stage, evaluations in self.state["evaluations"].items()
        }
        atomic_write_json(
            self.progress_path,
            {
                "family": self.family,
                "status": self.state["status"],
                "current": current,
                "completed_candidates": counts,
                "updated_at": self.state["updated_at"],
            },
        )

    def evaluate(self, stage: str, route: tuple[int, ...], label: str) -> RouteCandidate:
        key = route_key(route)
        existing = self.state["evaluations"][stage].get(key)
        if existing is not None:
            return RouteCandidate(route, objective_from_dict(existing["objective"]))
        if self.runner is None:
            raise RuntimeError("Runner has not been initialized")
        predictions = self.cache.predictions_for(self.runner, route, self.rows[stage], label)
        objective, paired_metrics = objective_for(
            self.family, self.baseline, predictions, self.rows[stage]
        )
        self.state["evaluations"][stage][key] = {
            "blocks": list(route),
            "completed_at": now(),
            "examples": len(self.rows[stage]),
            "objective": objective_to_dict(objective),
            "paired_metrics": paired_metrics,
        }
        self.checkpoint({"stage": stage, "route_key": key, "status": "complete"})
        return RouteCandidate(route, objective)

    def run(self, manifest_for_runner: Path, data_root: Path) -> dict:
        budgets = tuple(int(value) for value in nested(self.config, "route.k_values", "budgets", default=[]))
        if budgets != (4, 6, 8):
            raise ValueError(f"Frozen route.k_values must be [4, 6, 8]; got {list(budgets)}")
        seeds = tuple(int(value) for value in nested(self.config, "optimizer.seeds", "seeds", default=[]))
        if len(seeds) != 3 or len(set(seeds)) != 3:
            raise ValueError(f"Frozen optimizer must contain exactly three unique seeds; got {seeds}")
        population_size = required_int(self.config, "optimizer.population_size", "population_size")
        generations = required_int(self.config, "optimizer.generations", "generations")
        development_finalists = required_int(
            self.config,
            "optimizer.development_finalists_per_seed",
            "development_finalists_per_seed",
        )
        selection_finalists = required_int(
            self.config,
            "optimizer.selection_finalists",
            "selection.selection_finalists",
            "selection_finalists",
        )
        if development_finalists > population_size:
            raise ValueError("development_finalists_per_seed cannot exceed population_size")
        allowed_blocks = tuple(
            int(value)
            for value in nested(self.config, "route.allowed_blocks", default=list(range(32)))
        )
        normalize_route(allowed_blocks)
        protocol = {
            "budgets": list(budgets),
            "development_finalists_per_seed": development_finalists,
            "generations": generations,
            "population_size": population_size,
            "seeds": list(seeds),
            "selection_finalists": selection_finalists,
        }
        existing_protocol = self.state.get("protocol")
        if existing_protocol is not None and existing_protocol != protocol:
            raise ValueError("Cannot resume with a changed optimizer protocol")
        self.state["protocol"] = protocol
        self.checkpoint({"status": "loading-model"})
        self.runner = BaselineRunner(
            self.model_config,
            manifest_for_runner,
            self.job_dir / "runtime",
            data_root,
        )
        try:
            warmup = int(self.runner.config.get("warmup_examples", 0))
            for _ in range(warmup):
                self.runner.predict(self.rows["scout"][0])
            budget_summaries: dict[str, dict] = {}
            for k in budgets:
                priors = prior_routes(
                    self.family,
                    k,
                    self.sensitivity,
                    self.routes,
                    self.phase3_summaries,
                )
                seed_development_finalists: dict[str, list[tuple[int, ...]]] = {}
                seed_searches: dict[str, dict] = {}
                for seed in seeds:
                    population = initial_population(priors, allowed_blocks, k, population_size, seed)
                    generation_summaries = []
                    final_candidates: list[RouteCandidate] = []
                    for generation in range(generations):
                        candidates = [
                            self.evaluate(
                                "scout",
                                route,
                                f"{self.family} k{k} seed={seed} generation={generation + 1}",
                            )
                            for route in population
                        ]
                        survivor_count = max(development_finalists, population_size // 2)
                        survivor_count = min(survivor_count, len(candidates))
                        survivors = list(select_pareto_survivors(candidates, survivor_count))
                        survivors.sort(
                            key=lambda candidate: (
                                candidate.objective.selection_score,
                                candidate.route,
                            )
                        )
                        generation_summaries.append(
                            {
                                "generation": generation + 1,
                                "population": [list(candidate.route) for candidate in candidates],
                                "survivors": [list(candidate.route) for candidate in survivors],
                            }
                        )
                        final_candidates = survivors
                        if generation + 1 < generations:
                            population = next_population(
                                [candidate.route for candidate in survivors],
                                allowed_blocks,
                                population_size,
                                seed,
                                generation + 1,
                            )
                    selected = select_pareto_survivors(
                        final_candidates,
                        min(development_finalists, len(final_candidates)),
                    )
                    ordered = sorted(
                        selected,
                        key=lambda candidate: (
                            candidate.objective.selection_score,
                            candidate.route,
                        ),
                    )
                    seed_development_finalists[str(seed)] = [candidate.route for candidate in ordered]
                    seed_searches[str(seed)] = {
                        "generations": generation_summaries,
                        "scout_finalists": [
                            {
                                "blocks": list(candidate.route),
                                "objective": objective_to_dict(candidate.objective),
                            }
                            for candidate in ordered
                        ],
                    }
                    self.state["searches"].setdefault(str(k), {})[str(seed)] = seed_searches[str(seed)]
                    self.checkpoint({"budget": k, "seed": seed, "status": "scout-complete"})

                unique_development_routes = list(
                    dict.fromkeys(
                        route
                        for seed in seeds
                        for route in seed_development_finalists[str(seed)]
                    )
                )
                development_candidates = [
                    self.evaluate("development", route, f"{self.family} k{k} development finalist")
                    for route in unique_development_routes
                ]
                finalist_count = min(selection_finalists, len(development_candidates))
                selected_for_selection = sorted(
                    development_candidates,
                    key=lambda candidate: (candidate.objective.selection_score, candidate.route),
                )
                selected_for_selection = selected_for_selection[:finalist_count]
                selection_candidates = [
                    self.evaluate("selection", candidate.route, f"{self.family} k{k} selection")
                    for candidate in selected_for_selection
                ]
                selection_candidates.sort(
                    key=lambda candidate: (candidate.objective.selection_score, candidate.route)
                )
                frozen = selection_candidates[0]
                seed_winners = []
                for seed in seeds:
                    available = [
                        candidate
                        for candidate in development_candidates
                        if candidate.route in seed_development_finalists[str(seed)]
                    ]
                    available.sort(
                        key=lambda candidate: (candidate.objective.selection_score, candidate.route)
                    )
                    seed_winners.append(available[0].route)
                budget_summaries[str(k)] = {
                    "development_finalists": [
                        {
                            "blocks": list(candidate.route),
                            "objective": objective_to_dict(candidate.objective),
                        }
                        for candidate in sorted(
                            development_candidates,
                            key=lambda candidate: (
                                candidate.objective.selection_score,
                                candidate.route,
                            ),
                        )
                    ],
                    "frozen_route": {
                        "blocks": list(frozen.route),
                        "objective": objective_to_dict(frozen.objective),
                    },
                    "seed_development_winners": [list(route) for route in seed_winners],
                    "seed_route_stability_jaccard": jaccard_route_stability(seed_winners),
                    "selection_candidates": [
                        {
                            "blocks": list(candidate.route),
                            "objective": objective_to_dict(candidate.objective),
                        }
                        for candidate in selection_candidates
                    ],
                }
                partial = self._summary("running", budget_summaries)
                atomic_write_json(self.summary_path, partial)
                self.checkpoint({"budget": k, "status": "complete"})
            summary = self._summary("complete", budget_summaries)
            self.state["status"] = "complete"
            self.state["completed_at"] = now()
            atomic_write_json(self.summary_path, summary)
            self.checkpoint({"status": "complete"})
            return summary
        except Exception as error:
            self.state["status"] = "failed"
            self.state["error"] = f"{type(error).__name__}: {error}"
            self.checkpoint({"status": "failed"})
            raise
        finally:
            if self.runner is not None:
                self.runner.close()
                self.runner = None

    def _summary(self, status: str, budgets: dict[str, dict]) -> dict:
        return {
            "schema_version": 1,
            "family": self.family,
            "status": status,
            "evidence_status": (
                "processed-v2 method-selection evidence; not sealed external-heldout evidence"
            ),
            "input_hashes": self.input_hashes,
            "protocol": self.state.get("protocol"),
            "budgets": budgets,
            "updated_at": now(),
        }


def input_paths(config: dict) -> dict[str, Path]:
    return {
        "model_config": configured_path(
            config, "model_config", "paths.model_config", default=DEFAULT_MODEL_CONFIG
        ),
        "baseline": configured_path(
            config,
            "baseline_predictions",
            "paths.baseline_predictions",
            default=DEFAULT_BASELINE,
        ),
        "all_manifest": configured_path(
            config, "data.manifest", "manifest", default=DEFAULT_ALL_MANIFEST
        ),
        "data_root": configured_path(config, "data.root", "data_root", default=DEFAULT_DATA_ROOT),
        "sensitivity": configured_path(
            config, "priors.sensitivity", "sensitivity", default=DEFAULT_SENSITIVITY
        ),
        "routes": configured_path(config, "priors.routes", "routes", default=DEFAULT_ROUTES),
        "phase3_root": configured_path(
            config, "priors.phase3_root", "phase3_root", default=DEFAULT_PHASE3_ROOT
        ),
        "output_dir": configured_path(config, "output_dir", default=DEFAULT_OUTPUT_DIR),
    }


def load_phase3_summaries(root: Path) -> list[dict]:
    summaries = []
    for name in ("search_summary.json", "validation_summary.json", "pairwise_summary.json"):
        path = root / name
        if path.exists():
            summaries.append(load_json(path))
    return summaries


def hashes_for(
    search_config_path: Path,
    paths: dict[str, Path],
    manifests: dict[str, Path],
) -> dict[str, Any]:
    hashed = {
        "search_config": sha256_file(search_config_path),
        "model_config": sha256_file(paths["model_config"]),
        "baseline_predictions": sha256_file(paths["baseline"]),
        "all_manifest": sha256_file(paths["all_manifest"]),
        "sensitivity": sha256_file(paths["sensitivity"]),
        "routes": sha256_file(paths["routes"]),
        "phase3_summaries": {
            name: sha256_file(paths["phase3_root"] / name)
            for name in ("search_summary.json", "validation_summary.json", "pairwise_summary.json")
            if (paths["phase3_root"] / name).exists()
        },
        "prepared_manifests": {
            stage: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for stage, path in manifests.items()
        },
    }
    expected = nested(load_json(search_config_path), "data.manifest_sha256")
    if expected is not None and expected != hashed["all_manifest"]:
        raise ValueError("configs/robust_route_search.json manifest hash does not match all.jsonl")
    return hashed


def run_family(family: str, search_config_path: Path) -> Path:
    config = load_json(search_config_path)
    paths = input_paths(config)
    manifests = resolve_stage_manifests(config, family)
    for path in [*paths.values(), *manifests.values()]:
        if path != paths["output_dir"]:
            reject_sealed_path(path)
    rows = {stage: load_unique_rows(path) for stage, path in manifests.items()}
    baseline = load_prediction_map(paths["baseline"])
    validate_stage_rows(manifests, rows, baseline)
    model_config = load_json(paths["model_config"])
    sensitivity = load_json(paths["sensitivity"])
    routes = load_json(paths["routes"])
    input_hashes = hashes_for(search_config_path, paths, manifests)
    reuse_phase3 = bool(nested(config, "cache.reuse_phase3", "reuse_phase3", default=True))
    job = FamilyJob(
        family,
        config,
        model_config,
        manifests,
        rows,
        baseline,
        sensitivity,
        routes,
        load_phase3_summaries(paths["phase3_root"]),
        paths["output_dir"],
        input_hashes,
        paths["phase3_root"] if reuse_phase3 else None,
    )
    job.run(paths["all_manifest"], paths["data_root"])
    return job.summary_path


def finalize(search_config_path: Path) -> Path:
    config = load_json(search_config_path)
    paths = input_paths(config)
    summaries: dict[str, dict] = {}
    for family in FAMILIES:
        path = paths["output_dir"] / "families" / family / "summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing family summary: {path}")
        summary = load_json(path)
        if summary.get("status") != "complete":
            raise ValueError(f"Family {family} is not complete")
        summaries[family] = summary
    config_hash = sha256_file(search_config_path)
    model_hash = sha256_file(paths["model_config"])
    all_manifest_hash = sha256_file(paths["all_manifest"])
    baseline_hash = sha256_file(paths["baseline"])
    sensitivity_hash = sha256_file(paths["sensitivity"])
    routes_hash = sha256_file(paths["routes"])
    phase3_hashes = {
        name: sha256_file(paths["phase3_root"] / name)
        for name in ("search_summary.json", "validation_summary.json", "pairwise_summary.json")
        if (paths["phase3_root"] / name).exists()
    }
    for family, summary in summaries.items():
        hashes = summary["input_hashes"]
        expected = {
            "search_config": config_hash,
            "model_config": model_hash,
            "all_manifest": all_manifest_hash,
            "baseline_predictions": baseline_hash,
            "sensitivity": sensitivity_hash,
            "routes": routes_hash,
            "phase3_summaries": phase3_hashes,
        }
        for name, value in expected.items():
            if hashes.get(name) != value:
                raise ValueError(f"Family {family} has stale {name} hash")
        for stage, record in hashes["prepared_manifests"].items():
            manifest_path = Path(record["path"])
            reject_sealed_path(manifest_path)
            if sha256_file(manifest_path) != record["sha256"]:
                raise ValueError(f"Family {family} has stale prepared {stage} manifest")
    compiled = {
        "schema_version": 1,
        "status": "frozen",
        "frozen_at": now(),
        "evidence_status": (
            "processed-v2 development and test method-selection evidence; "
            "no sealed external-heldout access"
        ),
        "hashes": {
            "search_config": config_hash,
            "model_config": model_hash,
            "all_manifest": all_manifest_hash,
            "baseline_predictions": baseline_hash,
            "sensitivity": sensitivity_hash,
            "routes": routes_hash,
            "phase3_summaries": phase3_hashes,
            "prepared_manifests_by_family": {
                family: summary["input_hashes"]["prepared_manifests"]
                for family, summary in summaries.items()
            },
        },
        "families": {
            family: {
                "budgets": {
                    budget: {
                        "blocks": details["frozen_route"]["blocks"],
                        "selection_objective": details["frozen_route"]["objective"],
                        "seed_development_winners": details["seed_development_winners"],
                        "seed_route_stability_jaccard": details[
                            "seed_route_stability_jaccard"
                        ],
                    }
                    for budget, details in summary["budgets"].items()
                },
                "mean_seed_route_stability_jaccard": sum(
                    details["seed_route_stability_jaccard"]
                    for details in summary["budgets"].values()
                )
                / len(summary["budgets"]),
            }
            for family, summary in summaries.items()
        },
    }
    output_path = paths["output_dir"] / "frozen_routes.json"
    atomic_write_json(output_path, compiled)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--family", choices=FAMILIES)
    mode.add_argument("--finalize", action="store_true")
    parser.add_argument("--search-config", type=Path, default=DEFAULT_SEARCH_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = finalize(args.search_config) if args.finalize else run_family(args.family, args.search_config)
    print(path.resolve(), flush=True)


if __name__ == "__main__":
    main()
