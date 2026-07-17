# Explain Route Results Clearly

## Intended Commit Subject

`Explain route results clearly`

## Problem Or Decision

The repository reported correct numbers but relied on compressed labels such as generic K4,
evolved task, and K6. It also presented overall accuracy without clearly explaining that large
capability gains and regressions can cancel in the aggregate. Readers therefore had to infer the
actual intervention and the model-specific conclusions.

## Files And Behavior Changed

- Added explicit definitions for the full model, one shared K-block route, capability-specific
  K-block policy, K as the exact number skipped, and best found versus globally optimal.
- Expanded the README and final-status document with model-by-model interpretations for Qwen and
  SmolVLM2.
- Added the aggregate-cancellation result: Qwen six-block OCR gains are larger than the overall
  policy gain, while Smol counting and spatial gains are offset by its OCR regression.
- Updated the claims, outline, and writing guide so the eventual manuscript must preserve these
  interpretations and terminology.
- Updated the website tables, findings, captions, lessons, and transfer explanation with full route
  names and exact skip counts.
- Updated generated chart legends, axes, heatmap labels, transfer labels, and paper-table headings
  from generic/task shorthand to shared route/capability policy language.
- Updated submission verification to require the explicit Smol four-of-27 setting.

## Alternatives Considered

- Add only a glossary. This was rejected because definitions alone would not explain what the
  numbers mean for each model.
- Keep compressed chart labels and explain them only in prose. This was rejected because figures
  should remain understandable when viewed independently.
- State that task-specific routing is better. This was rejected because the result is conditional
  and the Smol OCR evidence directly contradicts a universal claim.

## Verification Evidence

- `make PYTHON=/tmp/vlm-paper-venv/bin/python submission`
- `python3 -m py_compile scripts/generate_paper_assets.py scripts/verify_submission.py`
- `git --git-dir=.git-data --work-tree=. diff --check`
- Searched the canonical README, status, claims, website, and generated tables for unexplained
  generic K or evolved task shorthand.
- Visually inspected regenerated Qwen budget and matched four-block control charts.
- Confirmed the README and website remain free of en dash and em dash characters.

## Known Limitations And Unsupported Claims

- Internal JSON and CSV schema keys retain stable names such as `evolved-generic` for compatibility.
- Compact K notation can still appear in historical artifacts after it has been defined; historical
  decision logs and result records are not rewritten.
- The interpretation does not establish globally optimal routes or localize capabilities inside
  named blocks.
- This documentation and labeling change does not alter experimental results.

## Next Action Enabled

Use the standardized terminology and interpretation hierarchy when drafting the manuscript and
presenting the work publicly.
