# arXiv v1 Record

This sheet records the announced arXiv v1 metadata. The frozen source is the submission archived by arXiv;
future manuscript changes require a new arXiv version.

## Paper

- **Title:** Searching for Task-Specific Vision Paths: Evolutionary Block Pruning Across Vision-Language Models
- **Authors:** Tarun Tomar
- **Primary category:** `cs.CV`
- **Cross-list:** None proposed
- **arXiv identifier:** [arXiv:2607.17052](https://arxiv.org/abs/2607.17052)
- **Version:** v1, announced 19 July 2026
- **DOI:** [10.48550/arXiv.2607.17052](https://doi.org/10.48550/arXiv.2607.17052)
- **Submission processor:** pdfLaTeX
- **Top-level file:** `main.tex`
- **Source candidate tag:** `arxiv-v1-candidate`
- **Comments:** 14 pages, 8 figures. Code and aggregate evidence: https://github.com/TarunTomar122/vision-pathways

## Abstract

Vision-language models normally execute the same complete vision encoder for every question, even when OCR,
counting, object, attribute, and spatial queries may not require identical computation. We study whether
fixed-budget combinations of vision blocks can be skipped without fine-tuning. A shared K-block route skips one
searched set of exactly K blocks for every question, while a capability-specific K-block policy selects one
same-size route using a known capability label. We introduce a source-balanced evolutionary search and compare
it with independent ranking, contiguous removal, and random routes at matched budgets. Experiments use
Qwen2.5-VL-3B-Instruct, SmolVLM2-2.2B-Instruct, and an 876-example image-disjoint selection split. Search
transfers across architectures: on SmolVLM2, the searched shared four-block route beats independent construction
by 4.91 percentage points. Capability specialization is less stable. On Qwen, the six-block capability policy
beats the shared route by 2.17 points, driven by a 7.10-point OCR gain. On sealed IIIT5K, however, the SmolVLM2
OCR-specific route trails its shared route by 13.6 points. Combinatorial search reliably improves route
construction, but capability labels do not define universally transferable vision pathways.

## Archived Source

The announced source archive contains `main.tex`, `references.bib`, the generated result table, and eight PDF
figures. It is available from the [arXiv record](https://arxiv.org/abs/2607.17052) and the
[GitHub release](https://github.com/TarunTomar122/vision-pathways/releases/tag/arxiv-v1-candidate).

The acknowledgement in v1 discloses AI assistance for code, orchestration, figures, manuscript organization,
and language editing. Any future venue or arXiv version should receive a fresh human review before submission.
