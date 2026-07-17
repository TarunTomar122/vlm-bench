# Claims and Limitations

## Primary Claim

Evolutionary search finds substantially better vision-block combinations than independent,
contiguous, or random pruning across Qwen2.5-VL-3B and SmolVLM2-2.2B. Task-specific routing is not
universally superior: its benefit depends on architecture, capability, and pruning budget.

## Required Terminology

- Write **one shared K-block route** for one searched set of exactly K skipped blocks used for every
  question.
- Write **capability-specific K-block policy** for five searched routes, each skipping exactly K
  blocks, selected using the known capability label.
- Define K4, K6, and K8 as exactly four, six, and eight skipped vision blocks on first use.
- Describe selected routes as **best found under the frozen search**, not globally optimal.
- Do not shorten these terms to generic K4 or task K6 before the full meaning has been established.

## Interpretation Hierarchy

1. The strongest cross-model result is that evolutionary search constructs better same-budget
   routes than independent, contiguous, or random selection.
2. Overall policy accuracy can conceal large capability-level effects. Similar aggregate accuracy
   does not imply that shared and capability-specific policies behave similarly.
3. Qwen with six blocks skipped is the strongest positive specialization result: the capability
   policy is +2.17 points overall and OCR is +7.10 points over the shared six-block route.
4. Smol with four blocks skipped is the clearest cancellation result: +0.80 points overall combines
   +7.18 counting, +9.39 spatial, and -13.55 OCR effects.
5. The negative IIIT5K transfer prevents interpreting the Smol OCR route as a stable OCR pathway.

## Supported Statements

- On Qwen, searched shared routes outperform routes constructed from independent one-block ranking
  by 1.03 points with four blocks skipped, 1.03 points with six skipped, and 3.54 points with eight
  skipped on the 876-example selection split.
- With six Qwen blocks skipped, the capability-specific policy is 2.17 points above the shared
  route with paired 95% interval `[0.00, 4.34]`; OCR contributes the clearest capability result at
  +7.10 points `[0.65, 14.19]`.
- With four SmolVLM2 blocks skipped, the searched shared route beats independent construction by
  4.91 points `[1.83, 7.99]`, and the searched capability policy beats independently constructed
  task routes by 8.90 points `[5.82, 11.99]`.
- Smol capability routing is mixed: counting is +7.18 points and spatial +9.39, while OCR is
  -13.55 relative to the shared four-block route.
- On sealed IIIT5K, the Smol full model, shared four-block route, and OCR-specific four-block route
  score 94.8%, 86.4%, and 72.8%; the OCR route is -13.6 points below the shared route
  `[-19.2, -8.4]`.
- The Smol shared four-block route removes 14.76% of vision parameters and 2.71% of total parameters;
  measured speedups are 8.60% vision and 4.19% end-to-end in an unlocked same-VM RTX 4090
  comparison.

## Statements to Avoid

- Do not say that named capabilities are stored in particular blocks. Skipping measures causal
  sensitivity under one intervention, not representational storage.
- Do not claim universal task-aware routing. Most overall intervals cross zero, and SmolVLM OCR
  transfers negatively.
- Do not call processed-v2 a sealed held-out benchmark. It was used for development and final
  method selection with image-disjoint partitions.
- Do not plot nonexistent SmolVLM K6/K8 values or describe the stopped K6 diagnostic as complete.
- Do not claim a final accuracy-latency Pareto frontier for Qwen. Equivalent final evolved-route
  latency was not collected across K4/K6/K8.
- Do not claim edge-device performance. Measurements were batch-size-one on one RTX 4090 VPS.
- Do not equate skipped parameter count with a physically smaller saved checkpoint. Identity
  skipping avoids execution; materializing a smaller checkpoint is future engineering work.

## Limitations Paragraph Ingredients

Two VLM architectures and five capabilities were evaluated. Route search used one dataset mixture
and a finite evolutionary budget. Only the SmolVLM OCR route received a genuinely fresh
post-freeze source-transfer test. No fine-tuning, learned router, mobile hardware, energy
measurement, or full decoder compression was studied. Accuracy is exact/task-aware correctness on
short-answer tasks and does not cover long-form generation. Confidence intervals quantify paired
example uncertainty, not model-training or dataset-family uncertainty.
