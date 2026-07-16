# Paper Package

Working title: **Searching for Task-Specific Vision Paths: Evolutionary Block Pruning Across Vision-Language Models**

This directory is the CPU-only handoff from completed experiments to manuscript writing. Generated
files are derived from committed aggregate evidence; no model weights, images, or GPU are required.

## Start Here

1. Read [`claims-and-limitations.md`](claims-and-limitations.md) before drafting claims.
2. Follow [`outline.md`](outline.md) section by section.
3. Use [`writing-guide.md`](writing-guide.md) as paragraph prompts.
4. Insert figures from [`figures/`](figures/) and tables from [`tables/`](tables/).
5. Complete [`submission-checklist.md`](submission-checklist.md) before arXiv upload.

## Regenerate Everything

```bash
python3 -m venv .paper-venv
.paper-venv/bin/pip install -r requirements-paper.txt
make PYTHON=.paper-venv/bin/python submission
```

The command regenerates PNG, PDF, and SVG versions of seven figures, writes CSV/Markdown/LaTeX
tables, and verifies the exact frozen values used by the paper and website.

## Figure Map

| Figure | Recommended placement | Main point |
|---|---|---|
| `generated-method-overview` | Method | Matched-budget evolutionary route search |
| `generated-qwen-accuracy-by-budget` | Main results | Evolved routes degrade more gracefully than naive controls |
| `generated-matched-k4-controls` | Main results | Evolution helps on both architectures at the same K |
| `generated-cross-model-capability-heatmap` | Cross-model analysis | Capability gains are architecture-dependent |
| `generated-fresh-ocr-transfer` | Transfer | The internal SmolVLM OCR route fails on sealed IIIT5K |
| `generated-route-stability` | Analysis | Seed winners often disagree, weakening a fixed layer map claim |
| `generated-efficiency-summary` | Efficiency | K4 meaningfully reduces vision depth but little total-model size |

## Evidence Boundary

The 876-example image-disjoint split is method-selection evidence, not a new sealed benchmark.
IIIT5K is a sealed 250-example source-transfer test. SmolVLM2 has a completed K4 study only. Its
latency result is an unlocked same-VM comparison because the cloud provider denied clock control.
No final evolved Qwen K4/K6/K8 latency curve was measured, so the paper must not present one.
