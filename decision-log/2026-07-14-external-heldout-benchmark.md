# Prepare Sealed External Held-Out Benchmark

## Intended Commit Subject

`Prepare sealed external held-out benchmark`

## Problem And Decision

The existing V2 development and test partitions are image-disjoint, but both partitions draw from
the same OCRBench, TallyQA, VSR, POPE, and VQAv2 source families. They therefore cannot establish
that capability-specific vision routes transfer to genuinely different benchmark distributions.

The decision is to create a separate, sealed external evaluation set with 250 examples for each of
OCR, counting, spatial reasoning, object existence, and color recognition. The builder pins every
raw dataset revision and file hash, samples deterministically, retains one question per image, and
rejects decoded pixel overlap with every image in the existing V2 manifest. This set is explicitly
prohibited from route, rank, prompt, or scoring selection.

## Files And Behavior Changed

- `src/vlm_bench/heldout_builder.py` adds revision-pinned downloading, raw-file verification,
  deterministic stratified selection, RGB pixel-level deduplication, image materialization, and
  sealed manifest generation.
- `scripts/prepare_external_heldout.py` adds the one-command build and validation entry point.
- `docs/external_heldout_protocol.md` records composition, provenance, the evaluation freeze,
  adaptations, and claim limitations.
- `tests/test_heldout_builder.py` verifies multiple-choice conversion, modal-answer tie handling,
  AMBER color-subject extraction, deterministic stratification, and image exclusion.
- `data/external-heldout-v1/manifests/*.jsonl` records the 1,250 selected examples without
  redistributing their images.
- `data/external-heldout-v1/manifests/summary.json` records source revisions, raw hashes, balance,
  the reference-manifest hash, and the final manifest hash.
- `.gitignore` excludes the 1.9 GB raw cache and 1.8 GB materialized image directory while retaining
  the derived manifests in Git.

The final composition is TextVQA/OpenImages for OCR, CountBenchQA/LAION for counting, CV-Bench
ADE20K relations for spatial reasoning, and AMBER for color and existence. AMBER publishes only
negative existence probes, so 125 positive existence questions are deterministically derived from
true AMBER color-attribute subjects and matched with 125 original negative probes on disjoint
images. The color and object capabilities also use disjoint images.

## Alternatives Considered

- Reuse the existing internal test split: rejected because source-family overlap remains.
- Use additional COCO questions for spatial and object evaluation: rejected where possible because
  COCO already contributes to VSR, POPE, and VQAv2.
- Use all CV-Bench relation rows: rejected; restricting to ADE20K avoids its COCO portion.
- Use AMBER existence directly: rejected after schema inspection showed all 4,924 labels are `no`.
- Trust source IDs for deduplication: rejected because duplicates can cross datasets or encodings.
- Compare encoded PNG hashes only: replaced with decoded RGB pixel hashing because pixel hashing is
  encoding-invariant and catches the same image stored as different JPEG/PNG files.
- Download and commit images: rejected because upstream image terms vary and repository storage
  would grow by approximately 1.8 GB.

## Verification Evidence

Commands and results on `france-gpu-1x-rtx-4090-https-attached-fallen`:

```text
PYTHONPATH=src .venv/bin/python -m pytest -q
11 passed in 1.94s

PYTHONPATH=src .venv/bin/python scripts/prepare_external_heldout.py --skip-download
examples: 1250
unique_images: 1250
counts_by_capability: 250 each
counts_by_split: heldout=1250
manifest_sha256: c01c93a9f8f007bb21a11c1952ca50fa51bfdaa5232ebbd681a979156cca5a77
reference_manifest_sha256: ec3712d74a43caf8dd3d1818788ee0a92bc82e7aa597d640b00f089c6ed357c8
```

Additional assertions verified:

- object answers are exactly 125 `yes` and 125 `no`;
- no TextVQA sentinel answer saying reading is unnecessary remains;
- counting covers answers 2 through 10 with 27 or 28 examples each;
- spatial labels contain above, below, left, and right;
- all 1,250 selected decoded-pixel hashes are unique and absent from V2;
- raw cache size is 1.9 GB, materialized held-out images are 1.8 GB, and 80 GB remains free;
- `python3 -m compileall -q src scripts tests` succeeds locally;
- `git diff --check` reports no whitespace errors.

## Known Limitations And Unsupported Claims

- Public evaluation data may have appeared in Qwen2.5-VL pretraining; this is source transfer, not
  proof against training-data contamination.
- TextVQA combines OCR with reasoning over scene text and is not pure transcription.
- CV-Bench is adapted from multiple choice to semantic short answers, so these are not official
  CV-Bench leaderboard scores.
- Positive AMBER existence questions are deterministic transformations, not an official AMBER
  evaluation dimension.
- Dataset identity remains partly confounded with capability.
- Building this set does not demonstrate route generalization, accuracy retention, or speedup.
- The RTX 4090 is not itself an edge device, and no held-out model predictions are generated here.

## Next Action Enabled

After Phase 2 fixes the route, bridge rank, prompt, scorer, controls, and latency procedure, run one
sealed evaluation of the full model, identity-pruned route, repaired task route, repaired generic
route, and random control on this manifest without tuning from its results.
