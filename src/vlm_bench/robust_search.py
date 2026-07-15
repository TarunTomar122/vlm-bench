"""Pure, deterministic helpers for robust fixed-budget route search.

All objective values are accuracy drops in percentage points, so lower is
better. Source balancing gives every ``(capability, source)`` cell equal
weight, regardless of the number of examples in that cell.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from numbers import Integral
from statistics import fmean, pstdev
from types import MappingProxyType
from typing import Any, Literal


Route = tuple[int, ...]
ObjectiveKind = Literal["generic", "task"]

# Frozen selection weights. Changing these changes the search protocol.
GENERIC_SELECTION_WEIGHTS = MappingProxyType({
    "mean_drop_pp": 0.50,
    "worst_source_drop_pp": 0.30,
    "source_variability_pp": 0.20,
})
TASK_SELECTION_WEIGHTS = MappingProxyType({
    "mean_drop_pp": 0.45,
    "worst_source_drop_pp": 0.30,
    "collateral_drop_pp": 0.15,
    "source_variability_pp": 0.10,
})


def normalize_route(
    blocks: Iterable[int],
    *,
    k: int | None = None,
    allowed_blocks: Iterable[int] | None = None,
) -> Route:
    """Return a sorted route, rejecting duplicates and non-integer blocks."""
    normalized = []
    for block in blocks:
        if isinstance(block, bool) or not isinstance(block, Integral):
            raise ValueError(f"Route blocks must be integers; got {block!r}")
        normalized.append(int(block))

    if len(set(normalized)) != len(normalized):
        raise ValueError("Route blocks must be unique")
    route = tuple(sorted(normalized))

    if k is not None:
        if isinstance(k, bool) or not isinstance(k, Integral) or k < 0:
            raise ValueError(f"Route size must be a non-negative integer; got {k!r}")
        if len(route) != k:
            raise ValueError(f"Expected a K={k} route; got {len(route)} blocks")

    if allowed_blocks is not None:
        allowed = set(normalize_route(allowed_blocks))
        outside = tuple(block for block in route if block not in allowed)
        if outside:
            raise ValueError(f"Route contains blocks outside the candidate pool: {outside}")
    return route


def _seed_bytes(seed: int | str | bytes) -> bytes:
    if isinstance(seed, bytes):
        return seed
    if isinstance(seed, (int, str)) and not isinstance(seed, bool):
        return str(seed).encode("utf-8")
    raise ValueError("seed must be an int, str, or bytes")


def _seeded_order(values: Iterable[int], seed: int | str | bytes, label: str) -> list[int]:
    seed_value = _seed_bytes(seed)

    def key(value: int) -> tuple[bytes, int]:
        payload = b"\0".join((seed_value, label.encode("ascii"), str(value).encode("ascii")))
        return hashlib.sha256(payload).digest(), value

    return sorted(values, key=key)


def swap_mutation(
    route: Iterable[int],
    candidate_pool: Iterable[int],
    *,
    seed: int | str | bytes,
    swaps: int = 1,
) -> Route:
    """Replace ``swaps`` route blocks while preserving the exact route size K."""
    parent = normalize_route(route)
    pool = normalize_route(candidate_pool)
    if not parent:
        raise ValueError("Cannot mutate an empty route")
    normalize_route(parent, allowed_blocks=pool)
    if isinstance(swaps, bool) or not isinstance(swaps, Integral) or not 1 <= swaps <= len(parent):
        raise ValueError(f"swaps must be between 1 and K={len(parent)}")

    additions = tuple(block for block in pool if block not in set(parent))
    if len(additions) < swaps:
        raise ValueError(f"Candidate pool has only {len(additions)} blocks available for {swaps} swaps")

    removed = set(_seeded_order(parent, seed, "remove")[:swaps])
    added = _seeded_order(additions, seed, "add")[:swaps]
    return normalize_route((*((block for block in parent if block not in removed)), *added), k=len(parent))


def fixed_k_crossover(
    parent_a: Iterable[int],
    parent_b: Iterable[int],
    *,
    seed: int | str | bytes,
) -> Route:
    """Cross two same-K set routes, retaining shared blocks and exactly K total."""
    left = normalize_route(parent_a)
    right = normalize_route(parent_b)
    if len(left) != len(right):
        raise ValueError(f"Parents must have the same K; got {len(left)} and {len(right)}")
    if not left:
        return ()

    shared = set(left) & set(right)
    alternatives = (set(left) | set(right)) - shared
    needed = len(left) - len(shared)
    selected = _seeded_order(alternatives, seed, "crossover")[:needed]
    return normalize_route((*shared, *selected), k=len(left))


@dataclass(frozen=True)
class PairedSourceMetric:
    """Paired accuracy result for one capability/source cell."""

    capability: str
    source: str
    examples: int
    baseline_accuracy: float
    variant_accuracy: float
    accuracy_drop_pp: float
    lost_correct: int
    recovered_correct: int


@dataclass(frozen=True)
class SourceBalancedMetrics:
    """Deterministically ordered paired metrics grouped by capability and source."""

    groups: tuple[PairedSourceMetric, ...]

    def for_capability(self, capability: str) -> tuple[PairedSourceMetric, ...]:
        return tuple(group for group in self.groups if group.capability == capability)

    @property
    def capabilities(self) -> tuple[str, ...]:
        return tuple(sorted({group.capability for group in self.groups}))

    def by_group(self) -> dict[tuple[str, str], PairedSourceMetric]:
        return {(group.capability, group.source): group for group in self.groups}


def _index_rows(
    rows: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    id_key: str,
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    keyed_input = isinstance(rows, Mapping)
    items = rows.items() if keyed_input else ((None, row) for row in rows)
    for supplied_id, row in items:
        if not isinstance(row, Mapping):
            raise ValueError(f"Prediction rows must be mappings; got {type(row).__name__}")
        if keyed_input:
            if id_key in row and str(row[id_key]) != str(supplied_id):
                raise ValueError(f"Prediction id {row[id_key]!r} does not match mapping key {supplied_id!r}")
            raw_id = supplied_id
        else:
            if id_key not in row:
                raise ValueError(f"Prediction row is missing field {id_key!r}")
            raw_id = row[id_key]
        if raw_id is None or str(raw_id) == "":
            raise ValueError("Prediction ids cannot be empty")
        row_id = str(raw_id)
        if row_id in indexed:
            raise ValueError(f"Duplicate prediction id: {row_id}")
        indexed[row_id] = row
    return indexed


def _correct_value(row: Mapping[str, Any], key: str, row_id: str) -> bool:
    try:
        value = row[key]
    except KeyError as error:
        raise ValueError(f"Prediction {row_id!r} is missing field {key!r}") from error
    if not isinstance(value, bool):
        raise ValueError(f"Prediction {row_id!r} field {key!r} must be boolean")
    return value


def source_balanced_paired_metrics(
    baseline_rows: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    variant_rows: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    *,
    id_key: str = "id",
    capability_key: str = "capability",
    source_key: str = "source",
    correct_key: str = "correct",
) -> SourceBalancedMetrics:
    """Compare exactly paired rows and aggregate within capability/source cells."""
    baseline = _index_rows(baseline_rows, id_key)
    variant = _index_rows(variant_rows, id_key)
    if not baseline:
        raise ValueError("At least one paired prediction is required")
    if baseline.keys() != variant.keys():
        missing = sorted(baseline.keys() - variant.keys())
        extra = sorted(variant.keys() - baseline.keys())
        raise ValueError(f"Prediction ids are not paired; missing={missing[:5]}, extra={extra[:5]}")

    grouped: dict[tuple[str, str], list[tuple[bool, bool]]] = {}
    for row_id in sorted(baseline):
        base = baseline[row_id]
        candidate = variant[row_id]
        try:
            base_group = (str(base[capability_key]), str(base[source_key]))
            candidate_group = (str(candidate[capability_key]), str(candidate[source_key]))
        except KeyError as error:
            raise ValueError(f"Prediction {row_id!r} is missing field {error.args[0]!r}") from error
        base_correct = _correct_value(base, correct_key, row_id)
        candidate_correct = _correct_value(candidate, correct_key, row_id)
        if base_group != candidate_group:
            raise ValueError(
                f"Prediction {row_id!r} changed group from {base_group!r} to {candidate_group!r}"
            )
        grouped.setdefault(base_group, []).append((base_correct, candidate_correct))

    metrics = []
    for (capability, source), pairs in sorted(grouped.items()):
        examples = len(pairs)
        baseline_accuracy = sum(base for base, _ in pairs) / examples
        variant_accuracy = sum(candidate for _, candidate in pairs) / examples
        metrics.append(
            PairedSourceMetric(
                capability=capability,
                source=source,
                examples=examples,
                baseline_accuracy=baseline_accuracy,
                variant_accuracy=variant_accuracy,
                accuracy_drop_pp=100.0 * (baseline_accuracy - variant_accuracy),
                lost_correct=sum(base and not candidate for base, candidate in pairs),
                recovered_correct=sum(not base and candidate for base, candidate in pairs),
            )
        )
    return SourceBalancedMetrics(tuple(metrics))


@dataclass(frozen=True)
class RobustObjective:
    """Generic or task-specific source-balanced minimization objective.

    Generic objectives set ``collateral_drop_pp`` to zero because no capability
    is designated as the target; that field is excluded from the generic score.
    """

    kind: ObjectiveKind
    mean_drop_pp: float
    worst_source_drop_pp: float
    source_variability_pp: float
    collateral_drop_pp: float = 0.0
    target_capability: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"generic", "task"}:
            raise ValueError(f"Unknown objective kind: {self.kind!r}")
        values = (
            self.mean_drop_pp,
            self.worst_source_drop_pp,
            self.source_variability_pp,
            self.collateral_drop_pp,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Objective values must be finite")
        if self.source_variability_pp < 0:
            raise ValueError("Source variability cannot be negative")
        if self.kind == "generic" and self.target_capability is not None:
            raise ValueError("A generic objective cannot have a target capability")
        if self.kind == "task" and not self.target_capability:
            raise ValueError("A task objective requires a target capability")

    @property
    def selection_score(self) -> float:
        """Return the frozen deterministic scalar score; lower is better."""
        weights = TASK_SELECTION_WEIGHTS if self.kind == "task" else GENERIC_SELECTION_WEIGHTS
        # Integer percentages make mathematically equal decimal-weight sums tie
        # exactly instead of depending on binary-float addition order.
        terms = [
            round(100 * weights["mean_drop_pp"]) * self.mean_drop_pp,
            round(100 * weights["worst_source_drop_pp"]) * self.worst_source_drop_pp,
            round(100 * weights["source_variability_pp"]) * self.source_variability_pp,
        ]
        if self.kind == "task":
            terms.append(round(100 * weights["collateral_drop_pp"]) * self.collateral_drop_pp)
        return math.fsum(terms) / 100.0

    def objective_vector(self) -> tuple[float, ...]:
        if self.kind == "generic":
            return (self.mean_drop_pp, self.worst_source_drop_pp, self.source_variability_pp)
        return (
            self.mean_drop_pp,
            self.worst_source_drop_pp,
            self.collateral_drop_pp,
            self.source_variability_pp,
        )


def _drop_summary(groups: Sequence[PairedSourceMetric]) -> tuple[float, float, float]:
    if not groups:
        raise ValueError("Objective requires at least one capability/source group")
    drops = [group.accuracy_drop_pp for group in groups]
    return fmean(drops), max(drops), pstdev(drops)


def generic_robust_objective(metrics: SourceBalancedMetrics) -> RobustObjective:
    """Build a generic objective with every capability/source cell weighted equally."""
    mean_drop, worst_drop, variability = _drop_summary(metrics.groups)
    return RobustObjective("generic", mean_drop, worst_drop, variability)


def task_robust_objective(metrics: SourceBalancedMetrics, target_capability: str) -> RobustObjective:
    """Build a target objective and equally weighted non-target collateral term."""
    target = metrics.for_capability(target_capability)
    if not target:
        raise ValueError(f"No groups found for target capability {target_capability!r}")
    collateral = tuple(group for group in metrics.groups if group.capability != target_capability)
    if not collateral:
        raise ValueError("Task objective requires at least one non-target collateral group")
    mean_drop, worst_drop, variability = _drop_summary(target)
    collateral_drop = fmean(group.accuracy_drop_pp for group in collateral)
    return RobustObjective(
        "task",
        mean_drop,
        worst_drop,
        variability,
        collateral_drop,
        target_capability,
    )


def selection_score(objective: RobustObjective) -> float:
    """Functional form of :attr:`RobustObjective.selection_score`."""
    return objective.selection_score


@dataclass(frozen=True)
class RouteCandidate:
    route: Route
    objective: RobustObjective

    def __post_init__(self) -> None:
        object.__setattr__(self, "route", normalize_route(self.route))


def _dominates(left: RouteCandidate, right: RouteCandidate) -> bool:
    left_values = left.objective.objective_vector()
    right_values = right.objective.objective_vector()
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def nondominated_sort(candidates: Iterable[RouteCandidate]) -> tuple[tuple[RouteCandidate, ...], ...]:
    """Return deterministic Pareto fronts for minimization objectives."""
    ordered = tuple(sorted(candidates, key=lambda candidate: candidate.route))
    if len({candidate.route for candidate in ordered}) != len(ordered):
        raise ValueError("Candidate routes must be unique")
    kinds = {candidate.objective.kind for candidate in ordered}
    if len(kinds) > 1:
        raise ValueError("Cannot Pareto-sort generic and task objectives together")
    targets = {candidate.objective.target_capability for candidate in ordered}
    if len(targets) > 1:
        raise ValueError("Cannot Pareto-sort different target capabilities together")

    dominates: dict[RouteCandidate, list[RouteCandidate]] = {candidate: [] for candidate in ordered}
    domination_count = {candidate: 0 for candidate in ordered}
    first_front = []
    for candidate in ordered:
        for other in ordered:
            if candidate is other:
                continue
            if _dominates(candidate, other):
                dominates[candidate].append(other)
            elif _dominates(other, candidate):
                domination_count[candidate] += 1
        if domination_count[candidate] == 0:
            first_front.append(candidate)

    fronts: list[tuple[RouteCandidate, ...]] = []
    current = tuple(first_front)
    while current:
        fronts.append(current)
        following = []
        for candidate in current:
            for other in dominates[candidate]:
                domination_count[other] -= 1
                if domination_count[other] == 0:
                    following.append(other)
        current = tuple(sorted(following, key=lambda candidate: candidate.route))
    return tuple(fronts)


def crowding_distances(front: Sequence[RouteCandidate]) -> dict[Route, float]:
    """Compute deterministic NSGA-II crowding distance for one Pareto front."""
    ordered = tuple(sorted(front, key=lambda candidate: candidate.route))
    if len({candidate.route for candidate in ordered}) != len(ordered):
        raise ValueError("Candidate routes must be unique")
    distances = {candidate.route: 0.0 for candidate in ordered}
    if not ordered:
        return distances
    if len(ordered) <= 2:
        return {candidate.route: math.inf for candidate in ordered}

    dimensions = len(ordered[0].objective.objective_vector())
    for dimension in range(dimensions):
        ranked = sorted(
            ordered,
            key=lambda candidate: (candidate.objective.objective_vector()[dimension], candidate.route),
        )
        low = ranked[0].objective.objective_vector()[dimension]
        high = ranked[-1].objective.objective_vector()[dimension]
        if low == high:
            continue
        distances[ranked[0].route] = math.inf
        distances[ranked[-1].route] = math.inf
        for index in range(1, len(ranked) - 1):
            route = ranked[index].route
            if math.isinf(distances[route]):
                continue
            previous = ranked[index - 1].objective.objective_vector()[dimension]
            following = ranked[index + 1].objective.objective_vector()[dimension]
            distances[route] += (following - previous) / (high - low)
    return distances


def select_pareto_survivors(
    candidates: Iterable[RouteCandidate],
    survivor_count: int,
) -> tuple[RouteCandidate, ...]:
    """Select survivors by front, crowding, scalar score, then route order."""
    if isinstance(survivor_count, bool) or not isinstance(survivor_count, Integral) or survivor_count < 0:
        raise ValueError("survivor_count must be a non-negative integer")
    fronts = nondominated_sort(candidates)
    available = sum(len(front) for front in fronts)
    if survivor_count > available:
        raise ValueError(f"Requested {survivor_count} survivors from {available} candidates")

    selected: list[RouteCandidate] = []
    for front in fronts:
        remaining = survivor_count - len(selected)
        if remaining <= 0:
            break
        if len(front) <= remaining:
            selected.extend(front)
            continue
        distances = crowding_distances(front)
        ranked = sorted(
            front,
            key=lambda candidate: (
                -distances[candidate.route],
                candidate.objective.selection_score,
                candidate.route,
            ),
        )
        selected.extend(ranked[:remaining])
    return tuple(selected)


def route_jaccard(route_a: Iterable[int], route_b: Iterable[int]) -> float:
    """Return Jaccard similarity for two normalized routes."""
    left = set(normalize_route(route_a))
    right = set(normalize_route(route_b))
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def jaccard_route_stability(routes: Iterable[Iterable[int]]) -> float:
    """Return mean pairwise route Jaccard; zero or one route is perfectly stable."""
    normalized = tuple(normalize_route(route) for route in routes)
    if len(normalized) <= 1:
        return 1.0
    similarities = [
        route_jaccard(normalized[left], normalized[right])
        for left in range(len(normalized))
        for right in range(left + 1, len(normalized))
    ]
    return fmean(similarities)


# Readable aliases for callers that use operation-oriented naming.
mutate_route = swap_mutation
crossover_routes = fixed_k_crossover
