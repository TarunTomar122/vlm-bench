# Publish SmolVLM2 K4 Results

## Intended Commit Subject

`Publish completed SmolVLM2 K4 results`

## Problem And Decision

The K4 route search, matched controls, sealed IIIT5K transfer, and latency audit are complete.
Publish compact, reproducible result artifacts and an evidence-bounded conclusion document to
GitHub. Keep raw prediction JSONL out of Git because it is large and already retained on the VPS
and local backup.

## Files And Behavior Changed

- Result summaries, frozen routes, control state, prepared-manifest summaries, and reports for the
  completed K4 protocol are added under `results/` and `data/`.
- The generated cross-model report now includes the actual sealed OCR transfer values in its OCR
  section.
- `CONCLUSIONS.md` states confirmed findings, null findings, K6 diagnostic status, and the unlocked
  latency limitation.

## Alternatives Considered

- Committing raw prediction JSONL was rejected because it would add tens of megabytes of generated
  outputs to normal Git history; the `.gitignore` policy excludes it and the VPS/local backup retains
  it.
- Claiming OCR-specific transfer was rejected because the sealed IIIT5K result favors generic K4.
- Claiming fixed-clock latency was rejected because the provider denied clock-control permission.

## Verification Evidence

- K4 analysis covers 876 selection examples and all six matched controls.
- Sealed IIIT5K transfer covers 250 examples with full, generic K4, and OCR K4 conditions.
- Latency covers 50 balanced examples, five repeats, 20 warmups per model, and three bracketing
  baseline measurements; its measurement mode is recorded in `clock-control.json`.

## Limitations And Non-Claims

- The internal method-selection split is not an untouched transfer benchmark.
- Latency is an unlocked same-VM RTX 4090 comparison, not a fixed-clock or edge-device result.
- K6 was not a completed experiment and is not published as a matched-control conclusion.

## Next Action Enabled

Stop the GPU instance safely, inspect the committed report, and decide whether to pursue recovery
fine-tuning or a narrower generic-route compression study.
