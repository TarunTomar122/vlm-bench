# Searching for Task-Specific Vision Paths

**Evolutionary block pruning across Qwen2.5-VL and SmolVLM2**

[Paper PDF](https://raw.githubusercontent.com/TarunTomar122/vision-pathways/main/paper/main.pdf) | [Research website](https://taruntomar122.github.io/vision-pathways/) | [LaTeX source](paper/main.tex)

This repository studies a narrow question: when a VLM vision encoder must skip the same number of
transformer blocks, can combinatorial search find better routes than independent, contiguous, or
random pruning, and do named visual capabilities benefit from different routes?

## Result in One Paragraph

Evolutionary search consistently finds better block combinations than naive route construction
across two architectures. Capability-specific routing is not universally better. When Qwen skips
six blocks, choosing a different six-block route for each capability improves overall accuracy by
2.17 percentage points over using one shared six-block route `[0.00, 4.34]`, driven by OCR. When
SmolVLM2 skips four blocks, the capability-specific policy is only +0.80 points overall
`[-1.94, 3.54]`: counting and spatial improve while OCR falls. On a sealed 250-example IIIT5K
transfer set, the Smol OCR-specific four-block route is 13.6 points worse than its shared four-block
route. The defensible conclusion is that **route search generalizes, but a universal
capability-to-layer map does not**.

![Cross-model capability heatmap](docs/assets/generated-cross-model-capability-heatmap.png)

## How to Read the Route Names

- **Full model:** skip zero vision blocks and execute the complete vision encoder.
- **One shared K-block route:** evolutionary search selects one set of exactly K blocks. The same
  set is skipped for OCR, counting, spatial, object, and attribute questions.
- **Capability-specific K-block policy:** evolutionary search selects five different routes, each
  skipping exactly K blocks. The route matching the known capability label is used for each
  question.
- **K4, K6, and K8:** skip exactly four, six, or eight vision blocks, respectively.
- **Best found route:** the strongest route evaluated under the frozen search procedure. It is not
  a proof of the global optimum over every possible block combination.

These routes identify combinations that are less damaging under identity skipping. They do not
show that a named capability is stored inside the skipped or retained blocks.

## Exactly How the Evolutionary Search Works

This is a **fixed-cardinality, multi-objective genetic search** over block subsets. For a model with
`L` vision blocks, a candidate route is a set

$$
S \subseteq \{0, \ldots, L-1\}, \qquad |S|=K,
$$

where every block in `S` is replaced by identity. The chromosome can therefore be viewed as an
`L`-bit vector with exactly `K` ones. Search never changes `K`, model weights, prompts, or decoding.

For capability `c` and dataset source `d`, let `A^0_{c,d}` be full-model accuracy and
`A^S_{c,d}` be accuracy after skipping route `S`. The paired accuracy drop in percentage points is

$$
\Delta_{c,d}(S)=100\left(A^0_{c,d}-A^S_{c,d}\right).
$$

Lower is better. A negative drop means the skipped route actually answered more examples correctly
than the full model in that cell. Every capability-source cell receives equal weight, regardless of
its number of examples.

For one shared route, define `mu` as the mean of all cell drops, `w` as the largest cell drop, and
`sigma` as their population standard deviation. The scalar finalist loss is

$$
L_{shared}(S)=0.50\mu+0.30w+0.20\sigma.
$$

For a route targeting capability `t`, compute `mu_t`, `w_t`, and `sigma_t` using only sources for
that capability. Let `kappa_t` be the mean drop across every non-target capability-source cell. Its
loss is

$$
L_t(S)=0.45\mu_t+0.30w_t+0.15\kappa_t+0.10\sigma_t.
$$

The algorithm then proceeds as follows:

1. **Initialize.** Fill up to half the population with priors from single-block sensitivity and
   earlier route searches. Fill the remainder with deterministic random K-block subsets.
2. **Evaluate.** Run every route on a source-balanced scout and compute the objective vector
   `(mu, w, sigma)` for a shared route or `(mu_t, w_t, kappa_t, sigma_t)` for a capability route.
3. **Select survivors.** Keep half the population using nondominated Pareto fronts. Within a partial
   front, use NSGA-II crowding distance, then the scalar loss, then lexicographic block order.
4. **Crossover.** Each child keeps blocks shared by both parents, then deterministically samples
   from their remaining union until it again contains exactly K blocks.
5. **Mutate.** Every child swaps exactly one included block for one excluded block. Survivors are
   copied unchanged, duplicate children are rejected, and random fallback routes prevent a stall.
6. **Repeat and freeze.** Qwen used 16 routes, 3 evaluated generations, and 3 seeds. The lean Smol
   replication used 12 routes, 2 evaluated generations, and the same 3 seeds. Each seed contributes
   at most two scout finalists. Deduplicated finalists are reranked on 300 development rows, at most
   three advance to all 876 selection rows, and the minimum matching scalar loss is frozen.

There is no learned router and no gradient update here. The evolutionary part searches a large
discrete space of fixed-size block combinations while preserving several kinds of accuracy damage
during selection. Full notation and pseudocode are in the
[paper method](paper/method.md) and [frozen protocol](research-docs/robust_route_search_protocol.md).

For a concrete K4 example, parents `{2, 5, 8, 11}` and `{2, 6, 8, 14}` force the child to retain
their shared blocks `{2, 8}`. Crossover chooses two more blocks from `{5, 6, 11, 14}`, then mutation
removes one of those four child blocks and inserts one allowed block not already present. The result
always contains exactly four skipped blocks.

## Main Results

| Model | Blocks skipped | Full model | One shared route | Capability-specific policy | Policy - shared | Paired 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-3B | 4 of 32 | 83.68 | 81.28 | 81.39 | +0.11 pp | [-1.83, 2.05] |
| Qwen2.5-VL-3B | 6 of 32 | 83.68 | 79.11 | 81.28 | +2.17 pp | [0.00, 4.34] |
| Qwen2.5-VL-3B | 8 of 32 | 83.68 | 75.91 | 76.60 | +0.68 pp | [-1.71, 3.08] |
| SmolVLM2-2.2B | 4 of 27 | 82.65 | 72.49 | 73.29 | +0.80 pp | [-1.94, 3.54] |

## What the Results Mean

### Qwen2.5-VL

- Skipping one shared set of four blocks reaches 81.28%, only 2.40 points below the full model.
  Using capability-specific four-block routes changes almost nothing overall: +0.11 points.
- When six blocks are skipped, the capability-specific policy reaches 81.28%, which is 2.17 points
  above the shared six-block route. It therefore matches the shared four-block route while skipping
  two additional blocks.
- The Qwen six-block overall gain is moderate, but OCR improves by 7.10 points `[0.65, 14.19]`.
  Aggregate accuracy understates that capability-specific gain.
- Eight skipped blocks are more aggressive. The capability-specific policy improves only 0.68
  points over the shared route, and its interval crosses zero.

### SmolVLM2

- Skipping one shared set of four blocks reaches 72.49%, 10.16 points below the full model.
  SmolVLM2 is therefore more fragile than Qwen under this intervention.
- Evolutionary search still matters: the searched shared four-block route beats selecting four
  blocks from independent one-block rankings by 4.91 points `[1.83, 7.99]`.
- The capability-specific four-block policy is only 0.80 points better overall than the shared
  route, but that average hides large changes: counting is +7.18, spatial is +9.39, and OCR is
  -13.55 points.
- On sealed IIIT5K, the OCR-specific four-block route remains 13.6 points below the shared
  four-block route. The OCR route does not transfer across sources.

The central interpretive lesson is that **similar overall accuracy can hide meaningfully different
capability profiles**. Positive and negative task-level effects can cancel in the aggregate.

The searched SmolVLM2 shared four-block route removes 14.76% of vision parameters, 2.71% of the
full model, and measures +8.60% vision and +4.19% end-to-end speedup. The latency result is an
unlocked same-VM RTX 4090 comparison, not an edge-device claim.

## Reproduce the Paper Assets

No GPU or model download is required. The command reads committed aggregate JSON and regenerates
all paper figures in PNG/PDF/SVG, tables in CSV/Markdown/LaTeX, the website assets, and a checked
paper data manifest.

```bash
python3 -m venv .paper-venv
.paper-venv/bin/pip install -r requirements-paper.txt
make PYTHON=.paper-venv/bin/python submission
```

## Repository Map

```text
paper/                      Canonical LaTeX manuscript, PDF, references, vector figures and tables
docs/                       GitHub Pages research website and generated web assets
research-docs/              Dataset card, final status and frozen experimental protocols
results/                    Committed aggregate evidence and historical experiment reports
configs/                    Frozen model, dataset, search and evaluation configurations
data/*/manifests/           Reproducible metadata and hashes; dataset images are not committed
scripts/                    Dataset, inference, analysis, search and paper-generation commands
src/vlm_bench/              Dataset, scoring and VLM evaluation implementation
tests/                      CPU unit tests
decision-log/               Commit-level research and engineering decisions
```

Start with the [paper package](paper/README.md), [final conclusions](results/cross-model-replication-k4/CONCLUSIONS.md),
and [GitHub Pages website](docs/index.html).

## Experimental Design

- **Models:** Qwen2.5-VL-3B-Instruct (32 vision blocks) and SmolVLM2-2.2B-Instruct (27 blocks).
- **Intervention:** replace a selected set of vision-transformer blocks with identity; no fine-tuning.
- **Capabilities:** attribute/color, counting, object existence, OCR, and spatial relations.
- **Discovery data:** 1,780 examples and 1,431 unique images from MME, OCRBench, TallyQA, VSR,
  POPE, and VQAv2 Color.
- **Method-selection data:** 876 image-disjoint examples with source-aware objectives.
- **Controls:** independent rankings, contiguous removal, three random routes, one searched shared
  route, and searched capability-specific routes, always compared with the same number of skipped
  blocks.
- **Uncertainty:** paired bootstrap 95% intervals over the same examples.
- **Transfer:** consumed external Qwen suite plus a post-freeze 250-example IIIT5K Smol OCR audit.

Exact model revisions:

- Qwen: `66285546d2b821cf421d4f5eb2576359d3770cd3`
- SmolVLM2: `482adb537c021c86670beed01cd58990d01e72e4`

## Evidence Boundary

Processed-v2 development/test partitions are image-disjoint method-development and method-selection
evidence, not a new sealed benchmark. SmolVLM2 has a completed K4 replication only; its stopped K6
run is diagnostic. A consistent latency audit of the final evolved Qwen K4/K6/K8 routes was not
collected, so this repository does not claim a complete accuracy-latency Pareto curve. Identity
skipping reduces executed vision depth but does not by itself create a smaller serialized checkpoint.

## Core Documentation

- [Current manuscript PDF](paper/main.pdf)
- [Canonical LaTeX source](paper/main.tex)
- [Paper claim and limitations](paper/claims-and-limitations.md)
- [Formal evolutionary-search method](paper/method.md)
- [Manuscript outline](paper/outline.md)
- [Section-by-section writing guide](paper/writing-guide.md)
- [Related work map and BibTeX](paper/related-work.md)
- [Submission checklist](paper/submission-checklist.md)
- [Dataset card](research-docs/dataset_card.md)
- [Robust route-search protocol](research-docs/robust_route_search_protocol.md)
- [Final cross-model report](results/cross-model-replication-k4/README.md)
- [Qwen matched-K analysis](results/robust-route-search-qwen25-vl-3b/analysis/README.md)
- [SmolVLM2 K4 analysis](results/robust-route-search-smolvlm2-2b-k4/analysis/README.md)

## Historical Pipeline

The full GPU pipeline remains available for audit and extension. See `requirements-gpu.txt`, frozen
configs, and the protocol documents. Raw predictions, dataset images, model snapshots, and caches
are intentionally excluded from Git; compact summaries, routes, hashes, and paired analyses are
committed. The final paper package does not require rerunning inference.

## Tracked Artifact Policy

The repository keeps the canonical LaTeX source, BibTeX database, PDF figures, generated LaTeX table,
compiled manuscript PDF, website PNGs, aggregate result JSON, protocols, configs, and code. It does
not track Overleaf/arXiv ZIP exports, duplicate paper PNG/SVG figures, LaTeX intermediate files, model
weights, third-party images, or large prediction caches. Run `make overleaf-package` whenever a fresh
source ZIP is needed locally.
