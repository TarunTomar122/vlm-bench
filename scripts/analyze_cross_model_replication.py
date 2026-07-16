#!/usr/bin/env python3
"""Produce a concise, evidence-bounded Qwen versus SmolVLM2 replication report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from vlm_bench.io import write_json


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def model_summary(name: str, analysis: dict) -> dict:
    output = {"model": name, "full_accuracy": analysis["full"]["overall"]["accuracy"], "budgets": {}}
    for k in sorted(analysis["budgets"], key=int):
        result = analysis["budgets"][k]
        comparison = result["comparisons"]["evolved_task_minus_evolved_generic"]["overall"]
        output["budgets"][k] = {
            "generic_accuracy": result["conditions"]["evolved-generic"]["overall"]["accuracy"],
            "task_accuracy": result["conditions"]["evolved-task"]["overall"]["accuracy"],
            "task_minus_generic_pp": comparison["mean_pp"],
            "ci95_low_pp": comparison["ci95_low_pp"],
            "ci95_high_pp": comparison["ci95_high_pp"],
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", type=Path, default=Path("results/robust-route-search-qwen25-vl-3b/analysis/analysis.json"))
    parser.add_argument("--smol", type=Path, default=Path("results/robust-route-search-smolvlm2-2b-k4/analysis/analysis.json"))
    parser.add_argument("--fresh-ocr", type=Path, default=Path("results/fresh-ocr-iiit5k-smolvlm2-2b-k4/analysis.json"))
    parser.add_argument("--latency-root", type=Path, default=Path("results/fixed-clock-latency-smolvlm2-2b-k4"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/cross-model-replication-k4"))
    args = parser.parse_args()
    qwen = model_summary("Qwen2.5-VL-3B-Instruct", load(args.qwen))
    smol = model_summary("SmolVLM2-2.2B-Instruct", load(args.smol))
    fresh = load(args.fresh_ocr)
    clock_control = load(args.latency_root / "clock-control.json")
    latency_measurement_mode = clock_control.get("measurement_mode", "unknown")
    latency = {}
    for path in sorted(args.latency_root.glob("k*/summary.json"), key=lambda item: int(item.parent.name[1:])):
        k = int(path.parent.name[1:])
        summary = load(path)
        route = summary["routes"][0]
        latency[str(k)] = {
            "baseline_vision_encoder_median_ms": summary["baseline_vision_encoder_median_ms"],
            "baseline_total_median_ms": summary["baseline_total_median_ms"],
            "vision_speedup_percent": route["vision_speedup_percent"],
            "total_speedup_percent": route["total_speedup_percent"],
        }
    report = {
        "status": "two-model method replication; method-selection and fresh transfer evidence separated",
        "models": [qwen, smol],
        "fresh_smol_ocr_transfer": fresh,
        "smol_fixed_clock_generic_latency": latency,
        "smol_latency_measurement_mode": latency_measurement_mode,
        "interpretation_rule": "A positive task-minus-generic value favors the capability-conditional policy at the same K. Intervals crossing zero are not treated as confirmed advantages.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "report.json", report)
    rows = []
    for model in (qwen, smol):
        for k, values in model["budgets"].items():
            rows.append(
                f"| {model['model']} | K{k} | {100 * values['generic_accuracy']:.2f}% | "
                f"{100 * values['task_accuracy']:.2f}% | {values['task_minus_generic_pp']:+.2f} pp | "
                f"[{values['ci95_low_pp']:.2f}, {values['ci95_high_pp']:.2f}] |"
            )
    fresh_keys = sorted(key for key in fresh if key.startswith("ocr_minus_generic_k"))
    if len(fresh_keys) != 1:
        raise ValueError("Fresh OCR analysis must define exactly one OCR-minus-generic budget")
    fresh_key = fresh_keys[0]
    fresh_budget = fresh_key.removeprefix("ocr_minus_generic_k")
    fresh_advantage = fresh[fresh_key]
    markdown = "\n".join([
        "# Cross-Model Vision-Block Route Replication",
        "",
        "## Matched-K Method-Selection Results",
        "",
        "| Model | Budget | Evolved generic | Evolved task policy | Task minus generic | Paired 95% interval |",
        "|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Fresh OCR Transfer",
        "",
        f"Full: {100 * fresh['conditions']['full']['candidate_accuracy']:.2f}%; "
        f"generic K{fresh_budget}: {100 * fresh['conditions'][f'generic-k{fresh_budget}']['candidate_accuracy']:.2f}%; "
        f"OCR K{fresh_budget}: {100 * fresh['conditions'][f'ocr-k{fresh_budget}']['candidate_accuracy']:.2f}%.",
        "",
        f"Frozen OCR route minus frozen generic route = {fresh_advantage['mean_pp']:+.2f} pp "
        f"with paired 95% interval [{fresh_advantage['ci95_low_pp']:.2f}, {fresh_advantage['ci95_high_pp']:.2f}].",
        "",
        "## SmolVLM2 Generic-Route Latency",
        "",
        f"Measurement mode: `{latency_measurement_mode}`.",
        "",
        "| Budget | Vision speedup | End-to-end speedup |",
        "|---:|---:|---:|",
        *[
            f"| K{k} | {latency[str(k)]['vision_speedup_percent']:+.2f}% | {latency[str(k)]['total_speedup_percent']:+.2f}% |"
            for k in sorted((int(value) for value in latency))
        ],
        "",
        "## Evidence Boundary",
        "",
        "The matched-K tables are image-disjoint method-selection evidence on processed-v2. The IIIT5K value is a separate sealed OCR source-transfer test and was excluded from every route-selection step. The latency table is batch-size-one RTX 4090 evidence, not a mobile-device measurement. If the measurement mode is unlocked, the cloud provider denied clock control and the measurement is only a same-VM comparison. Neither table supports a universal capability-routing claim unless the corresponding uncertainty interval excludes zero.",
        "",
    ])
    (args.output_dir / "README.md").write_text(markdown, encoding="utf-8")
    (args.output_dir / "index.html").write_text(
        "<!doctype html><meta charset=\"utf-8\"><title>Cross-Model Replication</title>"
        "<style>body{max-width:960px;margin:3rem auto;padding:0 1rem;font:16px/1.5 Georgia,serif;color:#17302c;background:#f7f2e8}pre{white-space:pre-wrap;background:#fffaf0;padding:1.4rem;border:1px solid #17302c}</style>"
        f"<pre>{html.escape(markdown)}</pre>",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
