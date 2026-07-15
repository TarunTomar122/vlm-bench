# External Held-Out Capability Set

## Purpose

This set tests whether fixed capability routes and a fixed repair rank transfer beyond the source
families used for discovery. It is sealed evaluation data: do not use its predictions to select
blocks, routes, bridge rank, ridge strength, prompts, or scoring rules.

The existing image-disjoint `test` split remains useful, but it shares OCRBench, TallyQA, VSR,
POPE, and VQAv2 source families with development. This external set changes the source family for
every capability.

## Composition

The deterministic default build contains 250 examples and 250 unique images per capability:

| Capability | Source | Selection |
|---|---|---|
| OCR | TextVQA validation | OpenImages scenes; at least 7/10 annotators agree; one QA per image |
| Counting | CountBenchQA | LAION-derived, manually verified counting images; balanced over counts 2-10 |
| Spatial | CV-Bench 2D Relation | ADE20K only; balanced over relation answers; COCO rows excluded |
| Object existence | AMBER existence + attributes | Balanced yes/no; positives derived from true color-subject attributes |
| Color attribute | AMBER discriminative attribute | Explicit color terms; balanced across answer/color strata |

AMBER object and color selections use disjoint images. AMBER's existence shard contains only
negative hallucination probes, so positive existence questions are deterministically derived from
true color-attribute questions by asking whether the annotated subject is present. Every selected image is decoded to RGB and
rejected if its dimensions and pixel bytes match any image in the existing V2 manifest. This
encoding-invariant check catches duplicates even when JPEG/PNG encoding differs.

Primary sources:

- TextVQA: https://textvqa.org/
- CountBenchQA implementation: https://github.com/google-research/big_vision/tree/main/big_vision/datasets/countbenchqa
- CV-Bench: https://huggingface.co/datasets/nyu-visionx/CV-Bench
- AMBER: https://github.com/junyangwang0410/AMBER

The builder pins Hugging Face revisions and raw parquet SHA-256 values. Images and raw parquet
files remain local and are excluded from Git. Upstream datasets and their source images retain
their original licenses and terms; this repository does not redistribute them.

## Build

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_external_heldout.py
```

The command validates image files and hashes after building. The final manifest is
`data/external-heldout-v1/manifests/heldout.jsonl`.

## Evaluation Freeze

The first frozen external evaluation is recorded in
`configs/external_frozen_evaluation.json`. It compares exactly four conditions: the full model,
the discovery-selected generic eight-block route, the Phase 3 capability-conditional eight-block
routes, and the earlier capability-conditional four-block routes. All use identity substitution
without repair. The manifest, model revision, prompt, decoding, and scoring implementation are
fixed by that file. No route may be changed after external predictions are inspected.

The conditions may run concurrently to reduce wall time. Therefore this run supports accuracy
transfer claims only; its latency values are diagnostic and must not be used for speed claims.
Existing fixed-clock audits remain the latency evidence.

## First Evaluation Result

The protocol was committed as `8ed9dc7` before inference. All four conditions completed 1,250
matched predictions. The full model scored 74.96%, generic K8 scored 66.00%, task K8 scored 64.72%,
and task K4 scored 68.08%.

At the matched K8 budget, task routing trailed generic by 1.28 percentage points overall with a
paired 95% bootstrap interval of [-3.60, 0.96]. Spatial was the only statistically clear task-K8
advantage at +6.40 points [1.60, 11.20]. Full results and per-capability flips are in
`results/external-frozen-qwen25-vl-3b/`.

The set is now consumed. Its outcomes may be analyzed but must not be used to alter route blocks,
candidate pools, prompts, scoring, or repair hyperparameters.

### General protocol requirements

Before running any model on this set, record the frozen:

1. four-block route for every capability;
2. generic and random controls;
3. repair method and rank;
4. prompt and decoding configuration;
5. scoring implementation;
6. latency measurement procedure.

Future sealed evaluations should include matched generic and random controls for every pruning
budget and any repaired condition. Report source-specific results because dataset identity remains
partially confounded with capability.

## Limitations

- These are public benchmarks and may have appeared in model pretraining.
- CV-Bench questions are adapted from multiple choice to their semantic relation answer because
  this project evaluates short-answer generation rather than official CV-Bench leaderboard scores.
- TextVQA measures OCR plus reasoning over scene text, not pure transcription alone.
- AMBER negatives are constructed hallucination probes and may contain linguistic regularities.
- This set tests source transfer for one model; it does not replace replication on a second VLM.
