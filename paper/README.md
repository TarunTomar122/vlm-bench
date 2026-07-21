# Paper Package

Title: **Searching for Task-Specific Vision Paths: Evolutionary Block Pruning Across Vision-Language Models**

This directory contains the canonical paper source and the CPU-only path from committed experiment
evidence to the manuscript. No model weights, dataset images, or GPU are required.

## Start Here

1. Read the published [arXiv record](https://arxiv.org/abs/2607.17052) and the archived [`main.pdf`](main.pdf).
2. Edit the canonical [`main.tex`](main.tex) and [`references.bib`](references.bib) only for a later version.
3. Check every claim against [`claims-and-limitations.md`](claims-and-limitations.md).
4. Use [`method.md`](method.md) to audit equations, operators, and pseudocode.
5. Read [`arxiv-metadata.md`](arxiv-metadata.md) for the announced-v1 record.
6. Review the primary-source record in [`citation-audit.md`](citation-audit.md).
7. Use [`submission-checklist.md`](submission-checklist.md) when preparing a future arXiv version or venue submission.

## Regenerate Everything

```bash
python3 -m venv .paper-venv
.paper-venv/bin/pip install -r requirements-paper.txt
make PYTHON=.paper-venv/bin/python submission
```

The command regenerates PNG, PDF, and SVG versions of eight figures, writes CSV/Markdown/LaTeX
tables, and verifies the exact frozen values used by the paper and website.

## Build The Paper

The editable manuscript is [`main.tex`](main.tex), its bibliography is
[`references.bib`](references.bib), and the verified render is [`main.pdf`](main.pdf).

```bash
make paper-pdf
```

This target regenerates the evidence-backed figures and table before compiling the manuscript with
`latexmk`, `pdflatex`, and BibTeX. The build must finish without unresolved citations, references, or
typesetting overflow warnings.

## Use With Overleaf

```bash
make overleaf-package
```

The generated `overleaf-package.zip` is intentionally ignored by Git. Upload it as a new Overleaf
project when needed. It contains `main.tex`, `references.bib`, all eight vector PDF figures, and the
generated LaTeX results table with the same relative paths used by the manuscript. Overleaf should
detect `main.tex` as the root file and compile it with pdfLaTeX.

The same source layout produced the published [arXiv v1](https://arxiv.org/abs/2607.17052). Follow
[`submission-checklist.md`](submission-checklist.md) when preparing a future version or venue submission.

```bash
make arxiv-package
```

This creates the ignored local file `arxiv-source.zip`. The v1 archive is available from the
[GitHub release](https://github.com/TarunTomar122/vision-pathways/releases/tag/arxiv-v1-candidate).
Use a newly built archive only when preparing a later arXiv version, after confirming it matches the
intended manuscript.

Run the complete clean-room package audit before uploading:

```bash
make PYTHON=.paper-venv/bin/python arxiv-preflight
```

This verifies the exact ZIP contents, metadata constraints, citations, included files, source hygiene,
clean TeX compilation, final log, page count, encryption state, and embedded fonts.

## What Is Canonical

- Keep: `main.tex`, `main.pdf`, `references.bib`, `figures/*.pdf`,
  `tables/generated-main-results.tex`, the evidence tables/data, and the research notes.
- Generated but not tracked: `overleaf-package.zip`, `figures/*.png`, `figures/*.svg`, and LaTeX
  intermediate files.
- Website-specific raster figures live once under `docs/assets/`.
- `draft.md`, `outline.md`, and `writing-guide.md` are retained as writing history, not as the
  submission source of truth.

## Figure Map

| Figure | Recommended placement | Main point |
|---|---|---|
| `generated-single-block-sensitivity` | Introduction / motivation | Individual sensitivity varies across capability and depth |
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
