# Manuscript Outline

Target: 7-9 main-text pages, excluding references and appendix.

## Abstract (150-190 words)

State the deployment problem, the matched-budget research question, evolutionary route search,
two-model/five-capability setup, strongest positive result, strongest negative transfer result,
and the restrained conclusion. Include exact values for Smol evolution vs independent, Qwen K6
task advantage, and IIIT5K negative transfer. Avoid a generic claim that task routing works.

## 1. Introduction (0.8-1.0 page)

1. VLMs run a fixed vision stack for every query despite large differences between OCR, counting,
   object presence, attributes, and spatial relations.
2. Structured depth skipping can reduce real dense computation, but independent layer scores need
   not compose because residual blocks interact.
3. Formulate the question: at the same number of skipped vision blocks, can combinatorial search
   beat naive pruning, and do capability-specific routes beat a single generic route?
4. Preview the answer: search is robustly useful; task specialization is conditional and unstable.
5. End with three contributions: protocol/search, two-model matched-K evidence, and transfer/stability
   analysis that exposes the limits of capability routing.

## 2. Related Work (0.65-0.8 page)

Organize by mechanism, not chronology:

- Depth/structured pruning in language models: ShortGPT and SliceGPT.
- Visual-token reduction: DynamicViT, Token Merging, SparseVLM, VScan, FlowCut.
- VLM layer skipping and domain adaptation: FlashVLM and domain-aware decoder pruning.
- Position this work: vision-encoder block combinations, matched budgets, named capabilities,
  evolutionary rather than independent ranking, and explicit cross-model/source-transfer tests.

Do not claim that task-aware VLM efficiency is absent. The distinction is the intervention target
and the empirical question, not ownership of the broad idea.

## 3. Method (0.9-1.1 pages)

### 3.1 Identity block skipping

Define a route as a set of vision-transformer block indices replaced by identity. Explain that K is
the number skipped and all comparisons within a row use identical K. On first use, write one shared
K-block route for one route used by every question, and capability-specific K-block policy for one
same-size route per capability. State no weights are updated and best found does not mean globally
optimal.

### 3.2 Generic and capability objectives

Describe source-balanced mean drop, worst-source drop, variability, and collateral loss. Explain
why source balancing prevents a large source from dominating route selection. Include the exact
cell-drop definition, Pareto vectors, and frozen scalar losses from `method.md`.

### 3.3 Evolutionary search

Insert `generated-method-overview.pdf`. Describe initialization, evaluation, Pareto selection,
mutation/crossover, three seeds, finalist selection, and route freezing. Refer exact optimizer values
to `method.md` and the frozen protocol/config. State that each child receives fixed-K crossover and
one swap mutation, while survivors pass through unchanged. Distinguish Qwen's 16 by 3 search from
the lean SmolVLM2 12 by 2 search.

### 3.4 Baselines and uncertainty

Define independent ranking, contiguous removal, three random routes, evolved generic, and evolved
task policy. Explain paired bootstrap intervals and the rule that an interval crossing zero is not a
confirmed advantage.

## 4. Experimental Setup (0.8-1.0 page)

### Models

Qwen2.5-VL-3B has 32 vision blocks; SmolVLM2-2.2B has 27. Include exact revisions from
`paper/data/paper-data.json`. Both use deterministic greedy inference and identity skipping.

### Data

Describe the 1,780-example/1,431-image discovery corpus and five capability buckets. The
876-example image-disjoint selection split contains 166 attribute, 181 counting, 193 object, 155
OCR, and 181 spatial examples. Name source datasets. Clearly distinguish development, selection,
consumed external Qwen evaluation, and sealed 250-example IIIT5K transfer.

### Metrics

Report overall/capability accuracy, paired point differences with 95% intervals, route stability
(pairwise Jaccard), parameter fractions, and measured batch-size-one latency.

## 5. Main Results (1.2-1.4 pages)

Insert `generated-main-results.tex`, `generated-qwen-accuracy-by-budget.pdf`, and
`generated-matched-k4-controls.pdf`.

Lead with evolution versus naive construction. Then discuss task versus generic separately. The
correct hierarchy is:

1. Evolved routes dominate random/contiguous/independent controls at aggressive Qwen budgets.
2. Qwen K6 has the largest overall task-policy advantage (+2.17 points), but its interval touches
   zero and most capability intervals remain uncertain.
3. SmolVLM K4 replicates the search benefit but not a universal task-policy benefit (+0.80 points,
   interval crossing zero). Explain that this small aggregate masks +7.18 counting, +9.39 spatial,
   and -13.55 OCR effects.

## 6. Capability and Transfer Analysis (0.9-1.1 pages)

Insert `generated-cross-model-capability-heatmap.pdf`, `generated-fresh-ocr-transfer.pdf`, and
optionally `generated-route-stability.pdf`.

Explain the sign reversals: Qwen K6 OCR improves, SmolVLM K4 OCR deteriorates, while Smol counting
and spatial improve. State explicitly that similar overall policy accuracy can hide large positive
and negative capability changes that cancel in the aggregate. Use this to reject a universal
capability-to-depth map. Then present IIIT5K as the decisive source-transfer test: a route optimized
as OCR-specific can overfit source mixture or architecture-specific shortcuts.

## 7. Efficiency (0.55-0.7 page)

Insert `generated-efficiency-summary.pdf`. Separate vision-tower reduction from whole-model
reduction. Report Smol latency with the unlocked same-VM caveat. State that Qwen parameter removal
is known, while a consistent final evolved K4/K6/K8 latency series was not measured.

## 8. Discussion (0.55-0.75 page)

Four takeaways:

1. Block interactions make route construction a combinatorial problem.
2. Safe pruning budgets are architecture-dependent; the smaller model is more fragile at K4.
3. Aggregate parity does not imply capability-level parity; report both overall and bucket metrics.
4. Capability labels alone may be insufficient routing variables; source, image complexity, and
   prompt format may matter more.

Propose follow-up work: learned per-input router, multi-source route objectives, post-pruning
recovery, and hardware-aware search. Keep these as future directions, not current contributions.

## 9. Limitations and Conclusion (0.4-0.55 page)

Use the limitation ingredients in `claims-and-limitations.md`. Conclude in two sentences: route
search generalizes; fixed capability routes do not yet. Do not end by overselling the one positive
OCR result because the cross-model transfer evidence directly qualifies it.

## Appendix

- Complete frozen block IDs and seed winners.
- Dataset source/subtype counts and manifest hashes.
- Search hyperparameters and objective definitions.
- Per-capability confidence intervals at every Qwen budget.
- Earlier ablation, feature-repair, and interaction-search diagnostics.
- Reproduction commands and hardware/software details.
