import math
import unittest

from vlm_bench.robust_search import (
    PairedSourceMetric,
    RobustObjective,
    RouteCandidate,
    SourceBalancedMetrics,
    crowding_distances,
    fixed_k_crossover,
    generic_robust_objective,
    jaccard_route_stability,
    nondominated_sort,
    normalize_route,
    route_jaccard,
    select_pareto_survivors,
    selection_score,
    source_balanced_paired_metrics,
    swap_mutation,
    task_robust_objective,
)


def assert_raises(error_type: type[Exception], function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__}")


def test_normalize_route_is_canonical_and_strict() -> None:
    assert normalize_route([9, 1, 4], k=3, allowed_blocks=range(10)) == (1, 4, 9)
    assert_raises(ValueError, normalize_route, [1, 1])
    assert_raises(ValueError, normalize_route, [1, 2.0])
    assert_raises(ValueError, normalize_route, [False, 2])
    assert_raises(ValueError, normalize_route, [1, 2], k=3)
    assert_raises(ValueError, normalize_route, [1, 3], allowed_blocks=[0, 1, 2])


def test_swap_mutation_is_seeded_order_independent_and_fixed_k() -> None:
    first = swap_mutation([7, 1, 4], [9, 7, 6, 4, 3, 1], seed="fold-2")
    second = swap_mutation([4, 7, 1], [1, 3, 4, 6, 7, 9], seed="fold-2")
    assert first == second
    assert len(first) == 3
    assert len(set(first)) == 3
    assert len(set(first) ^ {1, 4, 7}) == 2


def test_swap_mutation_supports_multiple_swaps_and_rejects_impossible_requests() -> None:
    result = swap_mutation([0, 1, 2, 3], range(8), seed=17, swaps=2)
    assert len(result) == 4
    assert len(set(result) & {0, 1, 2, 3}) == 2
    assert_raises(ValueError, swap_mutation, [0, 1], [0, 1], seed=1)
    assert_raises(ValueError, swap_mutation, [0, 1], [0, 1, 2], seed=1, swaps=0)


def test_fixed_k_crossover_is_canonical_deterministic_and_keeps_shared_blocks() -> None:
    first = fixed_k_crossover([1, 3, 5, 7], [1, 2, 5, 8], seed=4)
    second = fixed_k_crossover([7, 5, 3, 1], [8, 5, 2, 1], seed=4)
    assert first == second
    assert len(first) == 4
    assert {1, 5} <= set(first) <= {1, 2, 3, 5, 7, 8}
    assert_raises(ValueError, fixed_k_crossover, [1], [1, 2], seed=0)
    assert fixed_k_crossover([], [], seed=0) == ()


def _paired_rows() -> tuple[list[dict], list[dict]]:
    baseline = [
        {"id": "a1", "capability": "attribute", "source": "large", "correct": True},
        {"id": "a2", "capability": "attribute", "source": "large", "correct": True},
        {"id": "a3", "capability": "attribute", "source": "small", "correct": True},
        {"id": "o1", "capability": "object", "source": "negative", "correct": False},
        {"id": "o2", "capability": "object", "source": "steady", "correct": True},
    ]
    variant = [
        {"id": "o2", "capability": "object", "source": "steady", "correct": True},
        {"id": "a3", "capability": "attribute", "source": "small", "correct": True},
        {"id": "a2", "capability": "attribute", "source": "large", "correct": False},
        {"id": "o1", "capability": "object", "source": "negative", "correct": True},
        {"id": "a1", "capability": "attribute", "source": "large", "correct": False},
    ]
    return baseline, variant


def test_paired_metrics_are_grouped_sorted_and_source_balanced() -> None:
    baseline, variant = _paired_rows()
    metrics = source_balanced_paired_metrics(baseline, variant)
    assert [(group.capability, group.source) for group in metrics.groups] == [
        ("attribute", "large"),
        ("attribute", "small"),
        ("object", "negative"),
        ("object", "steady"),
    ]
    groups = metrics.by_group()
    assert groups[("attribute", "large")].accuracy_drop_pp == 100.0
    assert groups[("attribute", "large")].lost_correct == 2
    assert groups[("object", "negative")].accuracy_drop_pp == -100.0
    assert groups[("object", "negative")].recovered_correct == 1
    assert generic_robust_objective(metrics).mean_drop_pp == 0.0


def test_paired_metrics_accept_mappings_and_reject_unpaired_or_changed_metadata() -> None:
    baseline, variant = _paired_rows()
    baseline_map = {row["id"]: {key: value for key, value in row.items() if key != "id"} for row in baseline}
    variant_map = {row["id"]: {key: value for key, value in row.items() if key != "id"} for row in variant}
    assert source_balanced_paired_metrics(baseline_map, variant_map).capabilities == ("attribute", "object")

    assert_raises(ValueError, source_balanced_paired_metrics, baseline, variant[:-1])
    changed = [dict(row) for row in variant]
    changed[0]["source"] = "other"
    assert_raises(ValueError, source_balanced_paired_metrics, baseline, changed)

    invalid_correct = [dict(row) for row in variant]
    invalid_correct[0]["correct"] = "true"
    assert_raises(ValueError, source_balanced_paired_metrics, baseline, invalid_correct)


def test_generic_objective_and_frozen_selection_score() -> None:
    baseline, variant = _paired_rows()
    objective = generic_robust_objective(source_balanced_paired_metrics(baseline, variant))
    assert objective.kind == "generic"
    assert objective.mean_drop_pp == 0.0
    assert objective.worst_source_drop_pp == 100.0
    assert math.isclose(objective.source_variability_pp, math.sqrt(5000.0))
    expected = 0.50 * 0.0 + 0.30 * 100.0 + 0.20 * math.sqrt(5000.0)
    assert math.isclose(objective.selection_score, expected)
    assert selection_score(objective) == objective.selection_score
    assert objective.objective_vector() == (0.0, 100.0, objective.source_variability_pp)


def test_task_objective_uses_target_sources_and_non_target_collateral() -> None:
    baseline, variant = _paired_rows()
    objective = task_robust_objective(
        source_balanced_paired_metrics(baseline, variant),
        "attribute",
    )
    assert objective.mean_drop_pp == 50.0
    assert objective.worst_source_drop_pp == 100.0
    assert objective.source_variability_pp == 50.0
    assert objective.collateral_drop_pp == -50.0
    assert objective.selection_score == 50.0
    assert objective.objective_vector() == (50.0, 100.0, -50.0, 50.0)


def test_objectives_reject_missing_target_collateral_and_non_finite_values() -> None:
    group = PairedSourceMetric("ocr", "one", 1, 1.0, 1.0, 0.0, 0, 0)
    metrics = SourceBalancedMetrics((group,))
    assert_raises(ValueError, task_robust_objective, metrics, "object")
    assert_raises(ValueError, task_robust_objective, metrics, "ocr")
    assert_raises(ValueError, RobustObjective, "generic", math.nan, 0.0, 0.0)


def _generic_candidate(route: int, mean: float, worst: float, variability: float) -> RouteCandidate:
    return RouteCandidate((route,), RobustObjective("generic", mean, worst, variability))


def test_nondominated_sort_assigns_deterministic_fronts() -> None:
    balanced = _generic_candidate(1, 2.0, 2.0, 2.0)
    low_mean = _generic_candidate(2, 1.0, 4.0, 2.0)
    dominated = _generic_candidate(3, 3.0, 5.0, 3.0)
    fronts = nondominated_sort([dominated, low_mean, balanced])
    assert [[candidate.route for candidate in front] for front in fronts] == [
        [(1,), (2,)],
        [(3,)],
    ]


def test_pareto_survivors_use_crowding_score_and_route_tie_breaks() -> None:
    candidates = [
        _generic_candidate(3, 2.0, 2.0, 2.5),
        _generic_candidate(1, 0.0, 4.0, 1.0),
        _generic_candidate(4, 3.0, 1.0, 3.0),
        _generic_candidate(2, 1.0, 3.0, 2.0),
    ]
    distances = crowding_distances(tuple(candidates))
    assert math.isinf(distances[(1,)])
    assert math.isinf(distances[(4,)])
    survivors = select_pareto_survivors(reversed(candidates), 2)
    assert [candidate.route for candidate in survivors] == [(1,), (4,)]

    tied = [_generic_candidate(2, 0.0, 0.0, 0.0), _generic_candidate(1, 0.0, 0.0, 0.0)]
    assert [candidate.route for candidate in select_pareto_survivors(tied, 1)] == [(1,)]


def test_pareto_selection_validates_scope_and_count() -> None:
    generic = _generic_candidate(1, 0.0, 0.0, 0.0)
    task = RouteCandidate((2,), RobustObjective("task", 0.0, 0.0, 0.0, 0.0, "ocr"))
    assert_raises(ValueError, nondominated_sort, [generic, task])
    assert_raises(ValueError, nondominated_sort, [generic, generic])
    assert_raises(ValueError, select_pareto_survivors, [generic], 2)


def test_jaccard_similarity_and_mean_pairwise_stability() -> None:
    assert route_jaccard([3, 1, 2], [2, 3, 4]) == 0.5
    assert route_jaccard([], []) == 1.0
    assert jaccard_route_stability([]) == 1.0
    assert jaccard_route_stability([[1, 2]]) == 1.0
    assert math.isclose(
        jaccard_route_stability([[1, 2], [2, 3], [1, 2]]),
        (1 / 3 + 1 / 3 + 1) / 3,
    )


def load_tests(loader, tests, pattern):
    """Expose the dependency-free function tests to ``unittest discover``."""
    del loader, tests, pattern
    suite = unittest.TestSuite()
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            suite.addTest(unittest.FunctionTestCase(function, description=name))
    return suite
