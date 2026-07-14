"""Shared primitives for Phase 2 feature-gap and repair experiments."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn.functional as F


CAPABILITIES = ("attribute", "counting", "object", "ocr", "spatial")


def select_four_block_routes(routes: list[dict]) -> list[dict]:
    """Return the five task routes and generic control in a stable order."""
    selected = []
    for route in routes:
        if len(route["skip_vision_blocks"]) != 4:
            continue
        for assignment in route.get("assignments", []):
            route_type = assignment["route_type"]
            capability = assignment.get("capability")
            if route_type == "task_specific" and capability in CAPABILITIES:
                selected.append({
                    **route,
                    "phase2_name": f"task-{capability}",
                    "route_type": route_type,
                    "target_capability": capability,
                })
            elif route_type == "generic_macro":
                selected.append({
                    **route,
                    "phase2_name": "generic",
                    "route_type": route_type,
                    "target_capability": None,
                })
    by_name = {route["phase2_name"]: route for route in selected}
    expected = {"generic", *(f"task-{capability}" for capability in CAPABILITIES)}
    if set(by_name) != expected or len(selected) != len(expected):
        raise ValueError(f"Expected exactly one Phase 2 route per role; found {sorted(by_name)}")
    return [by_name[f"task-{capability}"] for capability in CAPABILITIES] + [by_name["generic"]]


def split_by_image(
    rows: list[dict],
    calibration_per_capability: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Build deterministic capability-balanced splits without image overlap."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        image_key = row.get("image_sha256") or row["image"]
        groups[image_key].append(row)

    def group_order(item: tuple[str, list[dict]]) -> str:
        return hashlib.sha256(f"{seed}:{item[0]}".encode()).hexdigest()

    calibration_keys = set()
    counts = {capability: 0 for capability in CAPABILITIES}
    for image_key, group in sorted(groups.items(), key=group_order):
        additions = {
            capability: sum(row["capability"] == capability for row in group)
            for capability in CAPABILITIES
        }
        useful = any(counts[capability] < calibration_per_capability and additions[capability] for capability in CAPABILITIES)
        overshoots = any(
            counts[capability] + additions[capability] > calibration_per_capability
            for capability in CAPABILITIES
        )
        if useful and not overshoots:
            calibration_keys.add(image_key)
            for capability in CAPABILITIES:
                counts[capability] += additions[capability]
        if all(count == calibration_per_capability for count in counts.values()):
            break

    if any(count != calibration_per_capability for count in counts.values()):
        raise ValueError(f"Could not construct exact image-disjoint calibration counts: {counts}")
    calibration = []
    evaluation = []
    for row in rows:
        image_key = row.get("image_sha256") or row["image"]
        (calibration if image_key in calibration_keys else evaluation).append(row)
    return calibration, evaluation


def sample_tokens(hidden: torch.Tensor, maximum: int) -> torch.Tensor:
    """Deterministically cap each image's token contribution to bridge fitting."""
    if hidden.ndim != 2:
        raise ValueError(f"Expected [tokens, hidden] tensor, received {tuple(hidden.shape)}")
    if hidden.shape[0] <= maximum:
        return hidden
    indices = torch.linspace(0, hidden.shape[0] - 1, maximum, device=hidden.device).round().long()
    return hidden.index_select(0, indices)


def feature_gap_metrics(full: torch.Tensor, variant: torch.Tensor) -> dict[str, float]:
    """Measure matched token-state displacement without retaining activations."""
    if full.shape != variant.shape:
        raise ValueError(f"Feature shapes differ: {tuple(full.shape)} != {tuple(variant.shape)}")
    full32 = full.float()
    variant32 = variant.float()
    residual = variant32 - full32
    denominator = torch.linalg.vector_norm(full32).clamp_min(1e-12)
    return {
        "mean_token_cosine": float(F.cosine_similarity(full32, variant32, dim=-1).mean().item()),
        "relative_l2": float((torch.linalg.vector_norm(residual) / denominator).item()),
        "residual_rms": float(residual.square().mean().sqrt().item()),
        "full_rms": float(full32.square().mean().sqrt().item()),
    }


@dataclass
class FeatureMoments:
    """Sufficient statistics for a centered ridge map X -> (Y - X)."""

    hidden_size: int
    device: torch.device

    def __post_init__(self) -> None:
        shape = (self.hidden_size, self.hidden_size)
        self.count = 0
        self.sum_x = torch.zeros(self.hidden_size, dtype=torch.float64, device=self.device)
        self.sum_delta = torch.zeros_like(self.sum_x)
        self.xtx = torch.zeros(shape, dtype=torch.float64, device=self.device)
        self.xtdelta = torch.zeros(shape, dtype=torch.float64, device=self.device)

    def update(self, pruned: torch.Tensor, full: torch.Tensor) -> None:
        if pruned.shape != full.shape or pruned.ndim != 2 or pruned.shape[1] != self.hidden_size:
            raise ValueError("Bridge calibration tensors have incompatible shapes")
        x = pruned.to(device=self.device, dtype=torch.float64)
        delta = full.to(device=self.device, dtype=torch.float64) - x
        self.count += x.shape[0]
        self.sum_x += x.sum(dim=0)
        self.sum_delta += delta.sum(dim=0)
        self.xtx += x.T @ x
        self.xtdelta += x.T @ delta

    def state_dict(self) -> dict:
        return {
            "hidden_size": self.hidden_size,
            "count": self.count,
            "sum_x": self.sum_x.cpu(),
            "sum_delta": self.sum_delta.cpu(),
            "xtx": self.xtx.cpu(),
            "xtdelta": self.xtdelta.cpu(),
        }

    @classmethod
    def from_state_dict(cls, state: dict, device: torch.device) -> "FeatureMoments":
        moments = cls(int(state["hidden_size"]), device)
        moments.count = int(state["count"])
        for name in ("sum_x", "sum_delta", "xtx", "xtdelta"):
            setattr(moments, name, state[name].to(device=device, dtype=torch.float64))
        return moments

    def fit(self, ranks: list[int], ridge: float) -> dict[int, dict]:
        if self.count <= self.hidden_size:
            raise ValueError("Insufficient calibration tokens for a full-rank ridge fit")
        mean_x = self.sum_x / self.count
        mean_delta = self.sum_delta / self.count
        covariance_x = self.xtx - self.count * torch.outer(mean_x, mean_x)
        covariance_x_delta = self.xtdelta - self.count * torch.outer(mean_x, mean_delta)
        scale = covariance_x.diagonal().mean().clamp_min(1e-12)
        regularized = covariance_x + ridge * scale * torch.eye(
            self.hidden_size, dtype=torch.float64, device=self.device
        )
        weight = torch.linalg.solve(regularized, covariance_x_delta)
        u, singular_values, vh = torch.linalg.svd(weight, full_matrices=False)
        output = {}
        for rank in ranks:
            if not 0 < rank <= self.hidden_size:
                raise ValueError(f"Invalid bridge rank {rank} for hidden size {self.hidden_size}")
            left = u[:, :rank] * singular_values[:rank]
            right = vh[:rank, :]
            truncated = left @ right
            bias = mean_delta - mean_x @ truncated
            output[rank] = {
                "rank": rank,
                "left": left.float().cpu(),
                "right": right.float().cpu(),
                "bias": bias.float().cpu(),
                "ridge": ridge,
                "calibration_tokens": self.count,
                "retained_weight_energy": float(
                    singular_values[:rank].square().sum() / singular_values.square().sum().clamp_min(1e-12)
                ),
            }
        return output


class LowRankResidualBridge(torch.nn.Module):
    """Frozen final-boundary repair: h <- h + hAB + b."""

    def __init__(self, state: dict) -> None:
        super().__init__()
        self.register_buffer("left", state["left"])
        self.register_buffer("right", state["right"])
        self.register_buffer("bias", state["bias"])

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        dtype = hidden.dtype
        residual = hidden.float() @ self.left.float() @ self.right.float() + self.bias.float()
        return hidden + residual.to(dtype=dtype)
