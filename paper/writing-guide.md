# Writing Guide and Prompts

## One-Sentence Story

Combinatorial search is a reliable way to find removable vision blocks, but capability-specific
routes are model- and source-dependent rather than a stable map of visual skills.

## Abstract Prompt

> Write a 170-word technical abstract. Begin with fixed-depth inefficiency in VLM vision encoders.
> Define identity block routes and matched-K evolutionary search. Name Qwen2.5-VL-3B,
> SmolVLM2-2.2B, five capabilities, and the 876-example selection split. Report Smol evolved
> generic versus independent (+4.91 pp), Qwen K6 task versus generic (+2.17 pp), and sealed IIIT5K
> OCR task versus generic (-13.6 pp). Conclude that search generalizes but task specialization does
> not universally transfer. Do not claim edge deployment or capability localization.

Terminology requirement: define a shared K-block route as one searched route used for every
question and a capability-specific K-block policy as one same-size route per capability. Explain K
as the exact number skipped. Do not use generic K4 or task K6 before these definitions.

## Introduction Prompt

> Draft five concise paragraphs following the numbered Introduction outline. Motivate actual dense
> depth reduction, distinguish block selection from token pruning, and frame two falsifiable
> questions. State the mixed answer in paragraph four. End with exactly three contribution bullets.
> Cite Qwen2.5-VL, SmolVLM2, ShortGPT, DynamicViT, and recent VLM layer/token pruning.

## Method Prompt

> Explain route S as a subset of vision blocks and K=|S|. Define identity skipping in equations.
> Explain generic and capability source-balanced objectives, including collateral drop. Describe
> three-seed evolutionary search with Pareto selection, mutation, crossover, final selection, and
> freezing. Define all matched-K controls and paired bootstrap confidence intervals. Use the method
> diagram and avoid implying that skipped blocks are physically deleted from the checkpoint.

## Results Prompt

> Write results in claim-evidence-caveat order. First compare evolutionary search with independent,
> contiguous, and random controls. Then compare evolved task and generic routes only at equal K.
> Report point changes as percentage points, not percentages. Explicitly say when intervals cross
> or touch zero. Treat Qwen K4/K6/K8 and Smol K4 separately; never interpolate Smol K6/K8.

Add the aggregate-cancellation insight: Qwen with six blocks skipped is +2.17 overall but +7.10 on
OCR, while Smol with four skipped is only +0.80 overall because +7.18 counting and +9.39 spatial
are offset by -13.55 OCR. Similar overall accuracy does not imply similar capability behavior.

## Transfer Prompt

> Interpret the cross-model heatmap without assigning abilities to single blocks. Contrast Qwen K6
> OCR (+7.10 pp) with Smol K4 OCR (-13.55 pp), then report fresh IIIT5K (-13.6 pp). Explain three
> plausible mechanisms: source overfitting, architecture-specific interactions, and route-selection
> noise. State that the experiment cannot distinguish them conclusively.

## Efficiency Prompt

> Separate skipped vision parameters from total model parameters and latency from storage. Report
> Qwen K4 11.79% vision/2.10% total parameter execution and Smol K4 14.76% vision/2.71% total.
> Report Smol +8.60% vision and +4.19% end-to-end speedup as an unlocked same-VM RTX 4090 result.
> State that final-route Qwen K4/K6/K8 latency is unavailable and do not construct a Pareto curve.

## Final Editing Pass

Search the manuscript for these terms and check every occurrence:

- **held out**: say which split/source and whether it was ever used for method selection.
- **significant**: use only if the paired interval excludes zero; prefer "confirmed under this
  interval" over a formal hypothesis-testing claim.
- **faster/smaller**: specify vision encoder versus whole model and execution versus checkpoint.
- **task-aware**: identify whether it means a capability-conditional frozen policy or a dynamic
  per-input router. This work studies the former.
- **causal**: identity skipping is an intervention, but it does not localize stored information.
