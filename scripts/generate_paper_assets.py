#!/usr/bin/env python3
"""Generate publication figures and tables from committed result JSON.

The script intentionally contains no hand-entered accuracy measurements. Every plotted
result is read from the frozen Qwen/SmolVLM analysis artifacts. Architecture parameter
counts are read from the latency audits. This keeps the paper package auditable without
requiring model weights or a GPU.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


ROOT = Path(__file__).resolve().parents[1]
QWEN_ANALYSIS = ROOT / "results/robust-route-search-qwen25-vl-3b/analysis/analysis.json"
QWEN_ROUTES = ROOT / "results/robust-route-search-qwen25-vl-3b/frozen_routes.json"
SMOL_ANALYSIS = ROOT / "results/robust-route-search-smolvlm2-2b-k4/analysis/analysis.json"
SMOL_ROUTES = ROOT / "results/robust-route-search-smolvlm2-2b-k4/frozen_routes.json"
CROSS_MODEL = ROOT / "results/cross-model-replication-k4/report.json"
QWEN_LATENCY = ROOT / "results/task-route-latency-locked-qwen25-vl-3b/summary.json"
SMOL_LATENCY = ROOT / "results/fixed-clock-latency-smolvlm2-2b-k4/k4/summary.json"

CAPABILITIES = ["attribute", "counting", "object", "ocr", "spatial"]
METHOD_ORDER = [
    "generic-independent",
    "contiguous",
    "random-mean",
    "evolved-generic",
    "evolved-task",
]
METHOD_LABELS = {
    "generic-independent": "Independent",
    "contiguous": "Contiguous",
    "random-mean": "Random mean",
    "evolved-generic": "Evolved generic",
    "evolved-task": "Evolved task",
}
COLORS = {
    "ink": "#17211b",
    "muted": "#637168",
    "paper": "#f5f0e6",
    "grid": "#d7d0c3",
    "green": "#176b55",
    "lime": "#9bbf55",
    "orange": "#e36b3d",
    "blue": "#4776a8",
    "red": "#b7473d",
    "gold": "#c89a3b",
}


def load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": COLORS["ink"],
            "axes.linewidth": 0.8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "legend.frameon": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "vlm-vision-paths",
            "pdf.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        metadata = None
        if suffix == "pdf":
            epoch = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)
            metadata = {"CreationDate": epoch, "ModDate": epoch}
        elif suffix == "svg":
            metadata = {"Date": None}
        fig.savefig(
            out_dir / f"generated-{name}.{suffix}",
            dpi=240 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.08,
            metadata=metadata,
        )
    svg_path = out_dir / f"generated-{name}.svg"
    normalized_svg = "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines())
    svg_path.write_text(normalized_svg + "\n", encoding="utf-8")
    plt.close(fig)


def random_mean(conditions: dict[str, Any]) -> float:
    values = [
        value["overall"]["accuracy"]
        for name, value in conditions.items()
        if name.startswith("random-")
    ]
    if not values:
        raise ValueError("No random controls found")
    return float(np.mean(values))


def condition_accuracies(analysis: dict[str, Any], budget: int) -> dict[str, float]:
    conditions = analysis["budgets"][str(budget)]["conditions"]
    output = {
        name: value["overall"]["accuracy"] * 100
        for name, value in conditions.items()
        if not name.startswith("random-")
    }
    output["random-mean"] = random_mean(conditions) * 100
    return output


def model_routes(frozen: dict[str, Any], budgets: list[int]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for family, family_data in frozen["families"].items():
        output[family] = {}
        for budget in budgets:
            point = family_data["budgets"].get(str(budget))
            if point:
                output[family][str(budget)] = {
                    "blocks": point["blocks"],
                    "seed_development_winners": point["seed_development_winners"],
                    "seed_route_stability_jaccard": point["seed_route_stability_jaccard"],
                }
    return output


def build_data() -> dict[str, Any]:
    qwen = load(QWEN_ANALYSIS)
    smol = load(SMOL_ANALYSIS)
    cross = load(CROSS_MODEL)
    qwen_routes = load(QWEN_ROUTES)
    smol_routes = load(SMOL_ROUTES)
    qwen_latency = load(QWEN_LATENCY)
    smol_latency = load(SMOL_LATENCY)

    qwen_results: dict[str, Any] = {}
    for budget in (4, 6, 8):
        qwen_results[str(budget)] = {
            "accuracy": condition_accuracies(qwen, budget),
            "task_advantage": qwen["budgets"][str(budget)]["comparisons"][
                "evolved_task_minus_evolved_generic"
            ],
        }
    smol_results = {
        "4": {
            "accuracy": condition_accuracies(smol, 4),
            "task_advantage": smol["budgets"]["4"]["comparisons"][
                "evolved_task_minus_evolved_generic"
            ],
            "evolution_vs_independent": {
                "generic": smol["budgets"]["4"]["comparisons"][
                    "evolved_generic_minus_generic_independent"
                ]["overall"],
                "task": smol["budgets"]["4"]["comparisons"][
                    "evolved_task_minus_task_independent"
                ]["overall"],
            },
        }
    }

    qwen_parameter_route = qwen_latency["routes"][0]["parameters"]
    smol_parameter_route = smol_latency["routes"][0]["parameters"]
    parameter_data = {
        "Qwen2.5-VL-3B": {
            "full_model": qwen_parameter_route["full_model"],
            "full_vision": qwen_parameter_route["full_vision"],
            "removed_k4": qwen_parameter_route["removed_model"],
            "vision_removed_percent": 100
            * qwen_parameter_route["removed_vision"]
            / qwen_parameter_route["full_vision"],
            "total_removed_percent": 100
            * qwen_parameter_route["removed_model"]
            / qwen_parameter_route["full_model"],
        },
        "SmolVLM2-2.2B": {
            "full_model": smol_parameter_route["full_model"],
            "full_vision": smol_parameter_route["full_vision"],
            "removed_k4": smol_parameter_route["removed_model"],
            "vision_removed_percent": 100
            * smol_parameter_route["removed_vision"]
            / smol_parameter_route["full_vision"],
            "total_removed_percent": 100
            * smol_parameter_route["removed_model"]
            / smol_parameter_route["full_model"],
        },
    }

    data = {
        "schema_version": 1,
        "claim": (
            "Evolutionary search finds better vision-block combinations than independent, "
            "contiguous, or random pruning across two VLMs; task-specific routing gains depend "
            "on model, capability, and pruning budget."
        ),
        "evidence_boundary": cross["status"],
        "model_revisions": {
            "Qwen/Qwen2.5-VL-3B-Instruct": "66285546d2b821cf421d4f5eb2576359d3770cd3",
            "HuggingFaceTB/SmolVLM2-2.2B-Instruct": "482adb537c021c86670beed01cd58990d01e72e4",
        },
        "selection_examples": qwen["examples"],
        "models": {
            "qwen": {
                "name": "Qwen2.5-VL-3B-Instruct",
                "vision_blocks": 32,
                "full_accuracy": qwen["full"]["overall"]["accuracy"] * 100,
                "budgets": qwen_results,
                "routes": model_routes(qwen_routes, [4, 6, 8]),
            },
            "smol": {
                "name": "SmolVLM2-2.2B-Instruct",
                "vision_blocks": 27,
                "full_accuracy": smol["full"]["overall"]["accuracy"] * 100,
                "budgets": smol_results,
                "routes": model_routes(smol_routes, [4]),
            },
        },
        "fresh_ocr_transfer": cross["fresh_smol_ocr_transfer"],
        "latency": {
            "smol_k4": {
                "measurement_mode": cross["smol_latency_measurement_mode"],
                **cross["smol_fixed_clock_generic_latency"]["4"],
            },
            "qwen_note": (
                "The committed Qwen K4 latency audit predates the final evolved K4/K6/K8 "
                "routes, so it is not used as a final-route latency curve."
            ),
        },
        "parameters": parameter_data,
        "source_artifacts": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in (
                QWEN_ANALYSIS,
                QWEN_ROUTES,
                SMOL_ANALYSIS,
                SMOL_ROUTES,
                CROSS_MODEL,
                QWEN_LATENCY,
                SMOL_LATENCY,
            )
        ],
    }

    # Fail loudly if the authoritative artifacts no longer match the frozen paper story.
    assert round(data["models"]["qwen"]["budgets"]["6"]["accuracy"]["evolved-task"], 4) == 81.2785
    assert round(data["models"]["smol"]["budgets"]["4"]["accuracy"]["evolved-generic"], 4) == 72.4886
    assert round(data["fresh_ocr_transfer"]["ocr_minus_generic_k4"]["mean_pp"], 1) == -13.6
    return data


def plot_method(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    steps = [
        (0.25, 3.15, "Initialize", "task-budget routes"),
        (3.5, 3.15, "Evaluate", "source-aware loss"),
        (6.75, 3.15, "Select", "Pareto finalists"),
        (6.75, 0.85, "Evolve", "mutate and crossover"),
        (3.5, 0.85, "Freeze", "three-seed winner"),
        (0.25, 0.85, "Audit", "held-out transfer"),
    ]
    for index, (x, y, title, subtitle) in enumerate(steps):
        color = COLORS["green"] if index in (2, 4, 5) else COLORS["ink"]
        box = patches.FancyBboxPatch(
            (x, y), 2.0, 1.05, boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor="white", edgecolor=color, linewidth=1.8
        )
        ax.add_patch(box)
        ax.text(x + 1.0, y + 0.69, title, ha="center", va="center", weight="bold", fontsize=11)
        ax.text(x + 1.0, y + 0.31, subtitle, ha="center", va="center", color=COLORS["muted"], fontsize=8.5)

    arrow = {"arrowstyle": "->", "color": COLORS["orange"], "lw": 1.8}
    ax.annotate("", xy=(3.42, 3.68), xytext=(2.3, 3.68), arrowprops=arrow)
    ax.annotate("", xy=(6.67, 3.68), xytext=(5.55, 3.68), arrowprops=arrow)
    ax.annotate("", xy=(7.75, 1.98), xytext=(7.75, 3.1), arrowprops=arrow)
    ax.annotate("", xy=(5.55, 1.38), xytext=(6.67, 1.38), arrowprops=arrow)
    ax.annotate("", xy=(2.3, 1.38), xytext=(3.42, 1.38), arrowprops=arrow)

    ax.annotate(
        "repeat",
        xy=(4.5, 3.1),
        xytext=(7.45, 2.3),
        ha="center",
        color=COLORS["muted"],
        fontsize=8.5,
        arrowprops={
            "arrowstyle": "->",
            "connectionstyle": "arc3,rad=0.28",
            "color": COLORS["muted"],
            "lw": 1.2,
        },
    )
    ax.text(6.1, 1.63, "finish", ha="center", color=COLORS["muted"], fontsize=8.5)
    ax.set_title("Evolutionary search at a matched skip budget", loc="left", pad=3)
    save_figure(fig, out_dir, "method-overview")


def plot_budget_accuracy(data: dict[str, Any], out_dir: Path) -> None:
    qwen = data["models"]["qwen"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    budgets = np.array([0, 4, 6, 8])
    generic = [qwen["full_accuracy"]] + [qwen["budgets"][str(k)]["accuracy"]["evolved-generic"] for k in budgets[1:]]
    task = [qwen["full_accuracy"]] + [qwen["budgets"][str(k)]["accuracy"]["evolved-task"] for k in budgets[1:]]
    independent = [qwen["full_accuracy"]] + [qwen["budgets"][str(k)]["accuracy"]["generic-independent"] for k in budgets[1:]]
    random = [qwen["full_accuracy"]] + [qwen["budgets"][str(k)]["accuracy"]["random-mean"] for k in budgets[1:]]
    ax.plot(budgets, independent, "o--", color=COLORS["blue"], label="Independent ranking", lw=1.7)
    ax.plot(budgets, random, "o--", color=COLORS["muted"], label="Random mean", lw=1.7)
    ax.plot(budgets, generic, "o-", color=COLORS["green"], label="Evolved generic", lw=2.5)
    ax.plot(budgets, task, "o-", color=COLORS["orange"], label="Evolved task policy", lw=2.5)
    ax.axhline(qwen["full_accuracy"], color=COLORS["ink"], lw=0.8, alpha=0.4)
    ax.set_xticks(budgets, ["Full", "K4", "K6", "K8"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(35, 86)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    ax.set_title("Qwen: route search preserves accuracy as pruning grows", loc="left")
    ax.legend(ncol=2, loc="lower left", fontsize=8.5)
    ax.text(0, -0.17, "SmolVLM2 is omitted because only K4 was completed.", transform=ax.transAxes,
            fontsize=8.5, color=COLORS["muted"])
    save_figure(fig, out_dir, "qwen-accuracy-by-budget")


def plot_controls(data: dict[str, Any], out_dir: Path) -> None:
    models = [data["models"]["qwen"], data["models"]["smol"]]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    palette = [COLORS["blue"], COLORS["gold"], COLORS["muted"], COLORS["green"], COLORS["orange"]]
    for ax, model in zip(axes, models):
        values = [model["budgets"]["4"]["accuracy"][method] for method in METHOD_ORDER]
        bars = ax.bar(np.arange(len(values)), values, color=palette, width=0.72)
        ax.axhline(model["full_accuracy"], color=COLORS["ink"], ls="--", lw=1, label="Full model")
        ax.set_xticks(np.arange(len(values)), [METHOD_LABELS[m] for m in METHOD_ORDER], rotation=28, ha="right")
        ax.set_title(model["name"], loc="left", fontsize=11)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.6, f"{value:.1f}", ha="center", fontsize=8)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(42, 87)
    fig.suptitle("Matched K4 controls: search beats naive route construction", x=0.06, ha="left", weight="bold", fontsize=14)
    save_figure(fig, out_dir, "matched-k4-controls")


def plot_heatmap(data: dict[str, Any], out_dir: Path) -> None:
    qwen_comp = data["models"]["qwen"]["budgets"]["6"]["task_advantage"]["capabilities"]
    smol_comp = data["models"]["smol"]["budgets"]["4"]["task_advantage"]["capabilities"]
    matrix = np.array([[qwen_comp[c]["mean_pp"] for c in CAPABILITIES], [smol_comp[c]["mean_pp"] for c in CAPABILITIES]])
    fig, ax = plt.subplots(figsize=(8.7, 2.8))
    image = ax.imshow(matrix, cmap="RdYlGn", vmin=-14, vmax=14, aspect="auto")
    ax.set_xticks(range(len(CAPABILITIES)), [c.title() if c != "ocr" else "OCR" for c in CAPABILITIES])
    ax.set_yticks([0, 1], ["Qwen K6", "SmolVLM2 K4"])
    for row, source in enumerate((qwen_comp, smol_comp)):
        for col, capability in enumerate(CAPABILITIES):
            point = source[capability]
            confirmed = point["ci95_low_pp"] > 0 or point["ci95_high_pp"] < 0
            label = f"{point['mean_pp']:+.1f}{'*' if confirmed else ''}"
            color = "white" if abs(point["mean_pp"]) > 7 else COLORS["ink"]
            ax.text(col, row, label, ha="center", va="center", color=color, weight="bold")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Task route minus generic route (pp)")
    ax.set_title("Capability-specific gains do not transfer uniformly across models", loc="left")
    ax.text(0, -0.28, "* paired 95% interval excludes zero", transform=ax.transAxes,
            fontsize=8.5, color=COLORS["muted"])
    save_figure(fig, out_dir, "cross-model-capability-heatmap")


def plot_transfer(data: dict[str, Any], out_dir: Path) -> None:
    conditions = data["fresh_ocr_transfer"]["conditions"]
    labels = ["Full", "Generic K4", "OCR-specific K4"]
    values = [conditions[key]["candidate_accuracy"] * 100 for key in ("full", "generic-k4", "ocr-k4")]
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    bars = ax.bar(labels, values, color=[COLORS["ink"], COLORS["green"], COLORS["red"]], width=0.62)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.7, f"{value:.1f}%", ha="center", weight="bold")
    delta = data["fresh_ocr_transfer"]["ocr_minus_generic_k4"]
    ax.annotate(
        f"{delta['mean_pp']:.1f} pp\n95% CI [{delta['ci95_low_pp']:.1f}, {delta['ci95_high_pp']:.1f}]",
        xy=(2, values[2]), xytext=(1.42, 80.3), ha="center", color=COLORS["red"], weight="bold",
        arrowprops={"arrowstyle": "->", "color": COLORS["red"]},
    )
    ax.set_ylim(65, 99)
    ax.set_ylabel("Exact-match accuracy (%)")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    ax.set_title("Sealed IIIT5K transfer: the SmolVLM OCR route fails", loc="left")
    ax.text(0, -0.17, "250 examples; routes frozen before any IIIT5K evaluation.",
            transform=ax.transAxes, fontsize=8.5, color=COLORS["muted"])
    save_figure(fig, out_dir, "fresh-ocr-transfer")


def plot_stability(data: dict[str, Any], out_dir: Path) -> None:
    qwen_routes = data["models"]["qwen"]["routes"]
    smol_routes = data["models"]["smol"]["routes"]
    labels = ["Generic", "Attribute", "Counting", "Object", "OCR", "Spatial"]
    families = [label.lower() for label in labels]
    qwen_k4 = [qwen_routes[f]["4"]["seed_route_stability_jaccard"] for f in families]
    smol_k4 = [smol_routes[f]["4"]["seed_route_stability_jaccard"] for f in families]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    width = 0.36
    ax.bar(x - width / 2, qwen_k4, width, color=COLORS["green"], label="Qwen K4")
    ax.bar(x + width / 2, smol_k4, width, color=COLORS["orange"], label="SmolVLM2 K4")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Mean pairwise Jaccard across seed winners")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    ax.legend()
    ax.set_title("Low route stability warns against a fixed capability map", loc="left")
    save_figure(fig, out_dir, "route-stability")


def plot_efficiency(data: dict[str, Any], out_dir: Path) -> None:
    names = list(data["parameters"])
    vision = [data["parameters"][name]["vision_removed_percent"] for name in names]
    total = [data["parameters"][name]["total_removed_percent"] for name in names]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    width = 0.34
    axes[0].bar(x - width / 2, vision, width, color=COLORS["green"], label="Vision tower")
    axes[0].bar(x + width / 2, total, width, color=COLORS["orange"], label="Whole model")
    axes[0].set_xticks(x, names)
    axes[0].set_ylabel("Parameters removed (%)")
    axes[0].set_ylim(0, 17)
    axes[0].grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    axes[0].legend(fontsize=8.5)
    for i, (v, t) in enumerate(zip(vision, total)):
        axes[0].text(i - width / 2, v + 0.35, f"{v:.1f}", ha="center", fontsize=8)
        axes[0].text(i + width / 2, t + 0.35, f"{t:.1f}", ha="center", fontsize=8)

    latency = data["latency"]["smol_k4"]
    speedups = [latency["vision_speedup_percent"], latency["total_speedup_percent"]]
    bars = axes[1].bar(["Vision encoder", "End-to-end"], speedups, color=[COLORS["green"], COLORS["orange"]], width=0.58)
    for bar, value in zip(bars, speedups):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.3, f"+{value:.1f}%", ha="center", weight="bold")
    axes[1].set_ylim(0, 10.5)
    axes[1].set_ylabel("SmolVLM2 K4 speedup (%)")
    axes[1].grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    axes[1].text(0.5, -0.18, "Unlocked same-VM fallback; not fixed-clock or edge-device evidence.",
                 transform=axes[1].transAxes, ha="center", fontsize=8, color=COLORS["muted"])
    fig.suptitle("K4 removes vision depth, not most total model parameters", x=0.06, ha="left", weight="bold", fontsize=14)
    save_figure(fig, out_dir, "efficiency-summary")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_tables(data: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for model_key in ("qwen", "smol"):
        model = data["models"][model_key]
        for budget, point in model["budgets"].items():
            overall = point["task_advantage"]["overall"]
            rows.append(
                {
                    "model": model["name"],
                    "budget": f"K{budget}",
                    "full_accuracy_pct": f"{model['full_accuracy']:.2f}",
                    "evolved_generic_pct": f"{point['accuracy']['evolved-generic']:.2f}",
                    "evolved_task_pct": f"{point['accuracy']['evolved-task']:.2f}",
                    "task_minus_generic_pp": f"{overall['mean_pp']:+.2f}",
                    "ci95_low_pp": f"{overall['ci95_low_pp']:.2f}",
                    "ci95_high_pp": f"{overall['ci95_high_pp']:.2f}",
                }
            )
    write_csv(out_dir / "generated-main-results.csv", rows)

    markdown = [
        "| Model | Budget | Full | Evolved generic | Evolved task | Task - generic (pp) | Paired 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    latex = [
        "\\begin{tabular}{llrrrrl}",
        "\\toprule",
        "Model & Budget & Full & Generic & Task & $\\Delta$ (pp) & 95\\% CI \\\\",
        "\\midrule",
    ]
    for row in rows:
        short_model = "Qwen2.5-VL-3B" if row["model"].startswith("Qwen") else "SmolVLM2-2.2B"
        interval = f"[{row['ci95_low_pp']}, {row['ci95_high_pp']}]"
        markdown.append(
            f"| {short_model} | {row['budget']} | {row['full_accuracy_pct']} | "
            f"{row['evolved_generic_pct']} | {row['evolved_task_pct']} | "
            f"{row['task_minus_generic_pp']} | {interval} |"
        )
        latex.append(
            f"{short_model} & {row['budget']} & {row['full_accuracy_pct']} & "
            f"{row['evolved_generic_pct']} & {row['evolved_task_pct']} & "
            f"{row['task_minus_generic_pp']} & {interval} \\\\"
        )
    latex.extend(["\\bottomrule", "\\end{tabular}"])
    (out_dir / "generated-main-results.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (out_dir / "generated-main-results.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")

    capability_rows = []
    for model_key, budget in (("qwen", "6"), ("smol", "4")):
        model = data["models"][model_key]
        points = model["budgets"][budget]["task_advantage"]["capabilities"]
        for capability in CAPABILITIES:
            point = points[capability]
            capability_rows.append(
                {
                    "model": model["name"],
                    "budget": f"K{budget}",
                    "capability": capability,
                    "task_minus_generic_pp": f"{point['mean_pp']:+.3f}",
                    "ci95_low_pp": f"{point['ci95_low_pp']:.3f}",
                    "ci95_high_pp": f"{point['ci95_high_pp']:.3f}",
                    "confirmed": str(point["ci95_low_pp"] > 0 or point["ci95_high_pp"] < 0).lower(),
                }
            )
    write_csv(out_dir / "generated-capability-results.csv", capability_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "paper")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    setup_style()
    data = build_data()
    data_dir = output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "paper-data.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    figures = output_root / "figures"
    plot_method(figures)
    plot_budget_accuracy(data, figures)
    plot_controls(data, figures)
    plot_heatmap(data, figures)
    plot_transfer(data, figures)
    plot_stability(data, figures)
    plot_efficiency(data, figures)
    write_tables(data, output_root / "tables")
    if output_root == (ROOT / "paper").resolve():
        site_assets = ROOT / "docs/assets"
        site_assets.mkdir(parents=True, exist_ok=True)
        for png in figures.glob("generated-*.png"):
            shutil.copy2(png, site_assets / png.name)
    print(f"Generated 7 figure sets and paper tables under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
