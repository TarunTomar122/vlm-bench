import torch

from vlm_bench.phase2 import (
    FeatureMoments,
    LowRankResidualBridge,
    feature_gap_metrics,
    select_four_block_routes,
    split_by_image,
)


def test_select_four_block_routes() -> None:
    routes = []
    capabilities = ("attribute", "counting", "object", "ocr", "spatial")
    for index, capability in enumerate(capabilities):
        routes.append({
            "name": f"task-{index}",
            "skip_vision_blocks": [index, 10, 11, 12],
            "assignments": [{"route_type": "task_specific", "capability": capability, "budget": 4}],
        })
    routes.append({
        "name": "generic-source",
        "skip_vision_blocks": [1, 2, 3, 4],
        "assignments": [{"route_type": "generic_macro", "capability": None, "budget": 4}],
    })
    selected = select_four_block_routes(routes)
    assert [route["phase2_name"] for route in selected] == [
        "task-attribute", "task-counting", "task-object", "task-ocr", "task-spatial", "generic",
    ]


def test_split_by_image_is_exact_and_disjoint() -> None:
    rows = []
    for capability in ("attribute", "counting", "object", "ocr", "spatial"):
        for index in range(5):
            rows.append({
                "id": f"{capability}-{index}",
                "image": f"{capability}-{index}.png",
                "image_sha256": f"{capability}-{index}",
                "capability": capability,
            })
    calibration, evaluation = split_by_image(rows, calibration_per_capability=2, seed=7)
    assert len(calibration) == 10
    assert len(evaluation) == 15
    assert {row["image_sha256"] for row in calibration}.isdisjoint(
        {row["image_sha256"] for row in evaluation}
    )


def test_low_rank_bridge_recovers_known_residual_map() -> None:
    generator = torch.Generator().manual_seed(4)
    x = torch.randn(200, 4, generator=generator, dtype=torch.float64)
    left = torch.randn(4, 2, generator=generator, dtype=torch.float64)
    right = torch.randn(2, 4, generator=generator, dtype=torch.float64)
    bias = torch.randn(4, generator=generator, dtype=torch.float64)
    full = x + x @ left @ right + bias
    moments = FeatureMoments(4, torch.device("cpu"))
    moments.update(x, full)
    bridge = LowRankResidualBridge(moments.fit([2], ridge=1e-10)[2])
    repaired = bridge(x.float())
    assert feature_gap_metrics(full.float(), repaired)["relative_l2"] < 1e-4
