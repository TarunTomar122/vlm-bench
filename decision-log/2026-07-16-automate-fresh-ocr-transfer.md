# Automate Fresh OCR Transfer Evaluation

## Intended Commit Subject

`Automate frozen IIIT5K OCR transfer evaluation`

## Problem And Decision

The strongest Qwen method-selection signal was OCR at K6. The second-model replication needs an
untouched source-transfer check that cannot influence routes. Evaluate only the fully unpruned
model, the frozen generic K6 route, and the frozen OCR-specific K6 route on the separately sealed
IIIT5K manifest, after all robust selection and matched controls are complete.

## Files And Behavior Changed

- `scripts/run_fresh_ocr_transfer.py` provides resumable inference and paired bootstrap analysis
  for full, frozen generic K6, and frozen OCR K6 on an OCR-only manifest.
- `scripts/supervise_smolvlm2_replication.py` launches that evaluation only after robust analysis
  exists and the sealed manifest is present.

## Alternatives Considered

- Running IIIT5K before route finalization was rejected because it could tempt route selection on
  the new source.
- Evaluating all K values was rejected for this first transfer check because the prior signal and
  research question are specifically the matched K6 OCR comparison.
- Reusing Qwen routes was rejected because the transfer evaluation must assess SmolVLM2-frozen
  routes selected by SmolVLM2 development data.

## Verification Evidence

- The transfer script checks frozen-route status, OCR-only manifest content, exact prediction
  coverage, manifest hash, and frozen-route hash before recording its paired bootstrap result.
- The supervisor gate is analysis completion plus sealed manifest existence.

## Limitations And Non-Claims

- This is a single OCR source and a single K budget, not evidence for all capabilities or budgets.
- IIIT5K word recognition differs from scene text and document question answering.

## Next Action Enabled

Let the supervisor run the untouched OCR transfer comparison immediately after it completes the
core SmolVLM2 route-search analysis.
