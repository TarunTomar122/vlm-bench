# Claims and Limitations

## Primary Claim

Evolutionary search finds substantially better vision-block combinations than independent,
contiguous, or random pruning across Qwen2.5-VL-3B and SmolVLM2-2.2B. Task-specific routing is not
universally superior: its benefit depends on architecture, capability, and pruning budget.

## Supported Statements

- On Qwen, evolved generic routes outperform generic independent ranking by 1.03 points at K4,
  1.03 points at K6, and 3.54 points at K8 on the 876-example selection split.
- At Qwen K6, the task policy is 2.17 points above evolved generic with paired 95% interval
  `[0.00, 4.34]`; OCR contributes the clearest capability result at +7.10 points
  `[0.65, 14.19]`.
- On SmolVLM2 K4, evolved generic beats generic independent by 4.91 points `[1.83, 7.99]`, and
  evolved task beats task-independent by 8.90 points `[5.82, 11.99]`.
- SmolVLM task routing is mixed: counting is +7.18 points and spatial +9.39, while OCR is -13.55.
- On sealed IIIT5K, SmolVLM full/generic-K4/OCR-K4 score 94.8/86.4/72.8%; the OCR route is
  -13.6 points below generic K4 `[-19.2, -8.4]`.
- SmolVLM generic K4 removes 14.76% of vision parameters and 2.71% of total parameters; measured
  speedups are 8.60% vision and 4.19% end-to-end in an unlocked same-VM RTX 4090 comparison.

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
