# Redesign Overlapping Paper Figures

## Intended Commit Subject

`Redesign method and transfer figures`

## Problem Or Decision

The generated method overview looked like a presentation graphic rather than an academic figure, and
its curved generation-loop arrow crossed the visual hierarchy. The IIIT5K bar chart placed its paired
difference annotation over the middle bar and near the OCR value label. Both problems remained visible
in the paper and GitHub Pages assets even though the files compiled successfully.

## Exact Changes

- Reimplemented `plot_method` in `scripts/generate_paper_assets.py` as a compact two-band process
  diagram with numbered nodes, a four-step search loop, and a separate freeze-and-test sequence.
- Routed the generation loop below the search steps so no arrow crosses a node, label, or subtitle.
- Reimplemented `plot_transfer` as a horizontal point plot with direct value labels and a dedicated
  comparison line for the OCR-specific route's -13.6 percentage-point result and paired interval.
- Regenerated the PDF, PNG, and SVG versions of both figures under `paper/figures/`.
- Regenerated the corresponding GitHub Pages PNG assets under `docs/assets/`.
- Rebuilt `paper/main.pdf` and `paper/overleaf-package.zip` with the corrected vector figures.

## Alternatives Considered

- Moving the existing curved arrow and bar annotation was rejected because it would preserve the same
  oversized card and callout style rather than address the underlying visual problem.
- Using a generated bitmap illustration was rejected because these are data and method figures that
  benefit from deterministic, editable, vector output.
- Removing the figures entirely was rejected because the method sequence and held-out transfer result
  are easier to understand visually when shown without decorative chart elements.

## Verification Evidence

- `make PYTHON=/tmp/vlm-paper-venv/bin/python submission` regenerated all paper and site assets and
  passed frozen-result and publication-package verification.
- `make PYTHON=/tmp/vlm-paper-venv/bin/python overleaf-package` rebuilt the PDF and Overleaf archive.
- The standalone PNG renderings of both redesigned figures were visually inspected.
- Paper pages 5 and 10 were rasterized from `paper/main.pdf` and visually inspected at 120 DPI.
- The Overleaf ZIP was extracted into `/tmp/vlm-overleaf-redesign` and compiled independently with
  `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`.
- The clean package produced 14 pages and 22 references with no matched LaTeX warnings, unresolved
  citations, unresolved references, overfull boxes, or underfull boxes.

## Known Limitations And Unsupported Claims

- The redesign changes visual presentation only; it does not change data, route selection, confidence
  intervals, captions, or scientific claims.
- The global Matplotlib style used by the remaining six figures is unchanged.
- Figure 6 reports a paired interval for the OCR-route minus shared-route comparison, not uncertainty
  intervals for each displayed condition independently.

## Next Action Enabled

Continue manuscript review using the rebuilt `paper/main.pdf`, or import the refreshed
`paper/overleaf-package.zip` without manually replacing either figure.
