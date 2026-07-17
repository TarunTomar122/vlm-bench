# Fix Method Figure Layout

## Intended Commit Subject

`Fix method figure layout`

## Problem Or Decision

The generated method overview used a very wide single-row canvas. Although the source image contained every stage, it could be clipped or reduced to unreadable text inside the narrower paper-style website column.

## Files And Behavior Changed

- Reworked `plot_method` in `scripts/generate_paper_assets.py` from a wide single-row workflow to a compact two-row workflow.
- Preserved the full process as Initialize, Evaluate, Select, Evolve, Freeze, and Audit.
- Added concise repeat and finish branches without placing labels over workflow boxes.
- Regenerated the website PNG and the paper PNG, PDF, and SVG from the authoritative generator.

## Alternatives Considered

- Hide the overflow with CSS. This was rejected because it would continue clipping information.
- Scale the original image more aggressively. This was rejected because its text would become too small on narrow screens.
- Replace the figure only on the website with custom HTML. This was rejected because the paper and website should share the same authoritative figure.

## Verification Evidence

- `make PYTHON=/tmp/vlm-paper-venv/bin/python submission`
- `python3 -m py_compile scripts/generate_paper_assets.py scripts/verify_submission.py`
- `git --git-dir=.git-data --work-tree=. diff --check`
- Rendered the generated PNG in Chromium at a 595-pixel viewport and visually confirmed that all six stages, arrows, and labels fit.

## Known Limitations And Unsupported Claims

- The workflow is a conceptual summary and does not expose every search hyperparameter.
- The website applies grayscale styling while the generated publication asset retains its established colors.
- This layout fix does not alter any experimental result or research claim.

## Next Action Enabled

Deploy the corrected method figure through the existing GitHub Pages pipeline.
