#!/usr/bin/env python3
"""Verify the CPU-only paper package against frozen experiment evidence."""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FIGURES = {
    "method-overview",
    "qwen-accuracy-by-budget",
    "matched-k4-controls",
    "cross-model-capability-heatmap",
    "fresh-ocr-transfer",
    "route-stability",
    "efficiency-summary",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    subprocess.run([sys.executable, str(ROOT / "scripts/generate_paper_assets.py")], check=True)
    paper_data_path = ROOT / "paper/data/paper-data.json"
    data = json.loads(paper_data_path.read_text(encoding="utf-8"))

    require(data["models"]["qwen"]["vision_blocks"] == 32, "Qwen block count changed")
    require(data["models"]["smol"]["vision_blocks"] == 27, "Smol block count changed")
    require(set(data["models"]["qwen"]["budgets"]) == {"4", "6", "8"}, "Qwen budget set changed")
    require(set(data["models"]["smol"]["budgets"]) == {"4"}, "Smol must remain K4-only")
    require(round(data["fresh_ocr_transfer"]["ocr_minus_generic_k4"]["mean_pp"], 1) == -13.6, "OCR transfer changed")
    require(data["latency"]["smol_k4"]["measurement_mode"] == "unlocked_same_vm_fallback", "Latency caveat missing")
    for artifact in data["source_artifacts"]:
        path = ROOT / artifact["path"]
        require(path.is_file(), f"Missing source artifact: {path}")
        require(sha256(path) == artifact["sha256"], f"Source artifact hash changed: {path}")

    for stem in EXPECTED_FIGURES:
        for suffix in ("png", "pdf", "svg"):
            path = ROOT / f"paper/figures/generated-{stem}.{suffix}"
            require(path.is_file() and path.stat().st_size > 1000, f"Missing or empty {path}")

    required = [
        ROOT / "paper/tables/generated-main-results.csv",
        ROOT / "paper/tables/generated-main-results.md",
        ROOT / "paper/tables/generated-main-results.tex",
        ROOT / "paper/tables/generated-capability-results.csv",
        ROOT / "site/index.html",
        ROOT / "site/styles.css",
        ROOT / "paper/outline.md",
        ROOT / "paper/writing-guide.md",
        ROOT / "paper/references.bib",
        ROOT / "paper/submission-checklist.md",
    ]
    for path in required:
        require(path.is_file() and path.stat().st_size > 100, f"Missing required submission file: {path}")

    html = (ROOT / "site/index.html").read_text(encoding="utf-8")
    require("-13.6" in html, "Website omits the negative transfer result")
    require("unlocked same-VM" in html, "Website omits latency caveat")
    require("SmolVLM2 K4" in html, "Website must identify Smol's completed budget")
    print("Submission verification passed: frozen results, 21 figure files, tables, docs, and site.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
