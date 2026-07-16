# Seal Fresh IIIT5K OCR Transfer Set

## Intended Commit Subject

`Add sealed IIIT5K OCR transfer preparation`

## Problem And Decision

The Qwen external OCR source (TextVQA) is consumed and cannot be used to select or validate the
SmolVLM2 method. Add a new OCR-only, source-distinct evaluation from IIIT5K test words. It is
selected deterministically and verified by decoded-pixel hashes against both the original V2 suite
and the consumed external suite before any model runs it.

## Files And Behavior Changed

- `scripts/prepare_iiit5k_ocr_transfer.py` loads the pinned IIIT5K dataset revision, selects 250
  unique test images by seeded hash order, materializes canonical RGB PNGs, creates a complete VLM
  manifest, validates it, and records provenance plus no-overlap assertions.

## Alternatives Considered

- Reusing TextVQA was rejected because it is already consumed external evidence.
- Letting this source enter the evolutionary search was rejected because it must remain untouched
  source-transfer evidence.
- Using encoded-file hashes alone was rejected because re-encoding can obscure duplicate pixels.

## Verification Evidence

- Hugging Face dataset metadata on the GPU reports IIIT5K splits including a 3,000-example `test`
  split with image and text fields.
- The dataset API resolved revision `d0a25a5bd51d121ae00ac59bfbabdc15381bd9f5` before preparation.
- The builder runs `validate_manifest` and reports decoded-pixel overlap explicitly.

## Limitations And Non-Claims

- IIIT5K is cropped word recognition, not scene/document VQA. It tests OCR transfer to a distinct
  visual distribution, not all OCR forms.
- This commit prepares data only and makes no model-performance claim.

## Next Action Enabled

After SmolVLM2 routes are frozen, evaluate full, frozen generic K6, and frozen OCR-specific K6 on
this sealed set without changing any route or decoding setting.
