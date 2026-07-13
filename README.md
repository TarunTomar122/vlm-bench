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

The current milestone has established the unpruned baseline and completed the first one-block
identity-ablation screen:

1. Build a controlled screening suite and a balanced natural validation suite.
2. Run Qwen2.5-VL-3B-Instruct with fixed deterministic settings.
3. Record predictions, per-capability accuracy, latency, visual-token counts, and peak VRAM.
4. Verify scoring, determinism, resume behavior, and image-dependence controls.
5. Run every single-block vision intervention on the same fixed examples.
6. Use the paired sensitivity map to choose multi-block candidates; do not claim deployable
   pruning until those candidates pass combined ablations, repeated latency trials, and recovery
   training.

## Dataset Design

No single public benchmark cleanly balances all target capabilities. We use two complementary
suites:

| Suite | Source | Capability | Role |
|---|---|---|---|
| Controlled | MME | OCR, count, position, existence, color | Uniform yes/no screening |
| External | OCRBench | OCR | Natural text recognition and text VQA |
| External | TallyQA | Counting | Simple and compositional counting |
| External | VSR | Spatial | Fine-grained spatial relations |
| External | POPE | Object existence | Random, popular, and adversarial objects |

MME reduces answer-format confounds because its perception tasks share a binary protocol. The
external suite prevents conclusions from depending on MME's small size or yes/no format. Dataset
images are never committed; deterministic manifests contain source identifiers, metadata, and
content hashes.

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
- The finalized dataset has 1,480 examples and 1,136 unique images. The external suite is balanced
  at 300 direct OCR, counting, spatial, and object-existence examples per capability.
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

## Detailed Documentation

- [Project brief](projectIdeas/vlm-task-aware-encoder-pruning-project/project_brief.md)
- [Experiment protocol](projectIdeas/vlm-task-aware-encoder-pruning-project/experiment_protocol.md)
- [Research landscape](projectIdeas/vlm-task-aware-encoder-pruning-project/research_landscape.md)
- [Dataset card](docs/dataset_card.md)
- [Verified baseline report](results/baseline-qwen25-vl-3b/README.md)
- [One-block ablation analysis](results/ablation-qwen25-vl-3b/README.md)
