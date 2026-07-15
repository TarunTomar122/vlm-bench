# VLM Bench: Task-Aware Vision-Encoder Pruning

This repository studies whether OCR, counting, spatial reasoning, and object recognition need
different blocks of a vision-language model's image encoder. The practical goal is to use a
validated task-by-block sensitivity map to build smaller, faster task-specific pathways for edge
deployment.

## Research Question

> Do fine-grained visual capabilities require measurably different subsets of a VLM vision
> encoder, and can capability-aware block execution outperform generic pruning at the same
> measured latency?

A score drop after bypassing a block establishes sensitivity under that intervention. It does
not by itself prove that the block stores or computes a capability. The full project therefore
adds graded suppression, block-group ablations, probing, recovery training, and matched-compute
baselines before making causal claims.

## Current Phase

The project has completed the one-block screen, controlled task-route comparisons at four pruning
budgets, a locked-clock latency audit, Phase 2 feature-gap repair, Phase 3 interaction-aware search,
and a frozen external evaluation:

1. Build a controlled screening suite and a balanced natural validation suite.
2. Run Qwen2.5-VL-3B-Instruct with fixed deterministic settings.
3. Record predictions, per-capability accuracy, latency, visual-token counts, and peak VRAM.
4. Verify scoring, determinism, resume behavior, and image-dependence controls.
5. Run every single-block vision intervention on the same fixed examples.
6. Use the paired sensitivity map to choose multi-block candidates; do not claim deployable
   pruning until those candidates pass combined ablations, repeated latency trials, and recovery
   training.
7. Measure full-versus-pruned states and fit frozen low-rank feature repairs on image-disjoint
   calibration examples. Completed; feature error decreased without reliable answer recovery.
8. Recompute candidate importance after every selected removal instead of composing independent
   one-block rankings. Completed; target-only K8 search did not transfer reliably.
9. Freeze the complete method before evaluating the sealed 1,250-example external set. Completed;
   the matched K8 task route did not beat generic K8 overall, while spatial transferred positively.

## Dataset Design

No single public benchmark cleanly balances all target capabilities. We use a discovery suite for
route construction and a separate sealed suite for final source-transfer evaluation:

| Suite | Source | Capability | Role |
|---|---|---|---|
| Controlled | MME | OCR, count, position, existence, color | Uniform yes/no screening |
| Discovery | OCRBench | OCR | Natural text recognition and text VQA |
| Discovery | TallyQA | Counting | Simple and compositional counting |
| Discovery | VSR | Spatial | Fine-grained spatial relations |
| Discovery | POPE | Object existence | Random, popular, and adversarial objects |
| Discovery | VQAv2 | Color attribute | Open-ended color questions |
| Sealed external | TextVQA | OCR | Scene-text source transfer |
| Sealed external | CountBenchQA | Counting | Balanced verified counts 2-10 |
| Sealed external | CV-Bench/ADE20K | Spatial | Four 2D relation classes |
| Sealed external | AMBER | Object and color | Balanced hallucination and attribute probes |

MME reduces answer-format confounds because its perception tasks share a binary protocol. The
discovery mixture prevents conclusions from depending on MME's small size or yes/no format. The
sealed suite contains 1,250 examples from different source families. Its first evaluation is
complete, so it is now consumed and must not be used for further method selection. Dataset images
are never committed; deterministic manifests contain source identifiers, metadata, and content
hashes.

## Primary Models

- `Qwen/Qwen2.5-VL-3B-Instruct`: primary model and first complete baseline.
- `HuggingFaceTB/SmolVLM2-2.2B-Instruct`: compact second architecture for later replication.

The first model is evaluated in BF16 with greedy decoding and SDPA. Qwen dynamic resolution is
pinned to 256-1,280 visual tokens (`200704-1003520` pixels in 28x28-pixel units) so isolated
high-resolution samples cannot consume the entire GPU or dominate latency. Model snapshots are
cached on the GPU machine and are not stored in Git.

## Repository Layout

```text
configs/                         Reproducible run settings
scripts/prepare_dataset.py       Dataset download, stratification, and manifests
scripts/validate_dataset.py      Integrity, hash, schema, and split-leakage checks
scripts/download_models.py       Pinned Hugging Face snapshot download
scripts/run_baseline.py          Deterministic VLM evaluation and telemetry
scripts/run_layer_ablation.py    Resumable one-block identity-ablation sweep
scripts/analyze_layer_ablation.py Paired capability analysis and heatmap generation
src/vlm_bench/                   Dataset, scoring, and benchmark implementation
tests/                           Fast unit tests
projectIdeas/vlm-task-aware-encoder-pruning-project/
                                 Full brief, protocol, and literature map
```

## Remote Setup

The experiment machine is Ubuntu 24.04 with one RTX 4090 (24 GB). From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-gpu.txt
PYTHONPATH=src .venv/bin/python scripts/prepare_dataset.py --output-root data/processed
PYTHONPATH=src .venv/bin/python scripts/download_models.py
PYTHONPATH=src .venv/bin/python scripts/run_baseline.py \
  --manifest data/processed/manifests/all.jsonl \
  --output-dir results/baseline-qwen25-vl-3b
```

Use `--limit` for a smoke run. Raw JSONL predictions are resumable and intentionally ignored by
Git; compact summaries and run metadata are committed.

## Measurements

Every prediction records:

- raw and normalized output;
- exact and task-aware correctness;
- preprocessing, vision-encoder, generation, and total latency;
- input, output, and visual-token counts;
- peak allocated and reserved GPU memory;
- image dimensions, source, split, capability, and subtype.

Run metadata records the model revision, dependency versions, GPU/driver, prompt policy, decoding
configuration, and dataset-manifest hash.

## Findings So Far

- The research gap is not generic pruning. Generic, domain-aware, and dynamic VLM pruning already
  exist, mostly over decoder layers, visual tokens, resolution, frames, or experts.
- The narrower question is whether named visual capabilities have different causal dependencies
  across standard vision-encoder blocks.
- A controlled plus natural dataset mixture is necessary: a single mixed benchmark either has
  too few samples or confounds capability with answer format and source distribution.
- Dynamic prompt-conditioned routing is a stretch goal. It is justified only if static
  capability-specific pathways first beat generic pruning at matched measured cost.
- The original discovery dataset has 1,480 examples and 1,136 unique images. The sealed external
  suite has 1,250 examples, exactly 250 for each of five capabilities.
- The verified Qwen2.5-VL-3B baseline scored 81.62% overall: 81.00% OCRBench, 72.00% TallyQA,
  79.33% VSR, 87.67% POPE, and 88.57% on the controlled MME suite.
- Median vision-encoder latency was 69.48 ms; median end-to-end latency was 197.80 ms at batch
  size one. Peak reserved VRAM was 8,318 MiB after pinning visual resolution.
- The primary model exposes 32 vision blocks and a 668.7M-parameter vision tower, making an
  exhaustive block sweep practical. No deployable block-pruning conclusion has been drawn yet.
- The completed one-block sweep shows broad sensitivity at blocks 0 and 15 (17.36 and 21.08
  percentage-point overall accuracy drops) and a concentrated OCR dependence across several
  intermediate and late blocks. Under a conservative screen of at most 1.0 point overall loss
  and 2.0 points loss on every capability, only block 28 is a candidate for the next combined
  ablation. This is a screening result, not a pruning result.
- The progressive combined study confirms that effects are non-additive. Skipping blocks 5 and
  28 stays within 0.95 points of baseline but has no measured end-to-end speedup; skipping four
  blocks improves matched end-to-end latency by 1.48% but loses 3.72 points overall and 7.94
  points on OCR.
- The activation-rescue trace for skipped blocks 3, 5, 9, and 28 recovers 0.74 points when the
  full state after block 3 is restored, 2.43 points after block 5, and 3.58 points after block 9.
  Restoring after block 28 exactly reproduces baseline outputs. This supports a cumulative
  refinement interpretation, especially for OCR, rather than a one-block/one-capability map.
  These pathways are not deployable universal pruners without a task-specific objective and
  recovery training.
- Capability-specific routes have already been tested at 4, 8, 12, and 16 removed blocks. At eight
  blocks, target-task losses were 10.18-37.30 points; at sixteen blocks they were 25.70-72.97
  points. Independent one-block rankings therefore do not compose into aggressive routes.
- Four-block task routes remove 11.79% of vision parameters but only 2.10% of total parameters.
  Locked-clock measurements show 6.47-7.28% vision speedup and 1.39-4.41% end-to-end speedup.
  Object and OCR beat generic four-block pruning by 4.79 and 5.95 points respectively, with paired
  95% intervals above zero; the other three task advantages remain uncertain.
- Phase 2 fitted rank-8, rank-32, and rank-128 final-boundary residual bridges using 200 calibration
  and 704 evaluation examples. Every bridge reduced relative feature error, but answer recovery was
  negligible or negative except for +0.79 points on object and +0.69 points on OCR. Final-state L2
  similarity is therefore not an adequate behavioral-recovery objective.
- Phase 3 target-only beam search did not reliably construct K8 routes: only attribute stayed near
  its intended validation-loss range, and object, OCR, and spatial lost 14.93-23.53 points.
- The frozen external full-model accuracy is 74.96%. Generic K8 scored 66.00% and task K8 scored
  64.72%, so task K8 trailed by 1.28 points overall (95% interval [-3.60, 0.96]). Spatial was the
  only clear matched-K8 task win at +6.40 points [1.60, 11.20].
- Task K4 scored 68.08% and preserved OCR and spatial better than generic K8, but that comparison
  uses different pruning budgets and is not evidence of matched-compute superiority.

## Detailed Documentation

- [Project brief](projectIdeas/vlm-task-aware-encoder-pruning-project/project_brief.md)
- [Experiment protocol](projectIdeas/vlm-task-aware-encoder-pruning-project/experiment_protocol.md)
- [Research landscape](projectIdeas/vlm-task-aware-encoder-pruning-project/research_landscape.md)
- [Dataset card](docs/dataset_card.md)
- [Verified baseline report](results/baseline-qwen25-vl-3b/README.md)
- [One-block ablation analysis](results/ablation-qwen25-vl-3b/README.md)
- [Combined-ablation analysis](results/combined-qwen25-vl-3b/README.md)
- [Activation-rescue analysis](results/activation-rescue-qwen25-vl-3b/README.md)
- [Current research status](docs/current_status.md)
- [Task-specific route analysis](results/task-route-analysis-qwen25-vl-3b/README.md)
- [Phase 2 feature-gap protocol](docs/phase2_feature_gap_protocol.md)
- [Phase 2 feature-gap results](results/phase2-feature-gap-qwen25-vl-3b/analysis/README.md)
- [Phase 3 interaction-search protocol](docs/phase3_interaction_search_protocol.md)
- [External held-out protocol](docs/external_heldout_protocol.md)
- [Frozen external evaluation](results/external-frozen-qwen25-vl-3b/README.md)
