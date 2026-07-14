# Decision: Evaluate task-specific vision-block routes against matched pruning controls

- Date: 2026-07-14
- Intended commit subject: `Evaluate task-specific vision pruning routes`
- Status: Accepted as an exploratory development-set result

## Context

The completed one-block sweep showed that individual Qwen2.5-VL vision blocks have different apparent importance for attribute, counting, object, OCR, and spatial questions. Earlier progressive pruning of blocks 3, 5, 9, and 28 removed four blocks with a 3.72-percentage-point overall loss, approximately 4.89% vision speedup, and approximately 1.48% end-to-end speedup under the original timing method.

That experiment did not establish whether capability-specific block choices outperform generic or random choices at the same pruning budget. It also had weak attribute coverage: only 60 MME color questions. The project needed to answer two questions:

1. Does a larger open-ended color set materially change the attribute-layer map?
2. Do task-specific block rankings produce better multi-block routes than generic, contiguous, or random pruning?

## Decision

Treat all previously observed examples as discovery data and run a structured route-search experiment rather than repeating the old 1,480 examples unnecessarily.

The experiment has three stages:

1. Run the unmodified model and all 32 one-block identity interventions on only the 300 newly added VQAv2 color examples.
2. Merge those color predictions with the existing 1,480-example sweep, rank blocks independently for each capability, and construct nested 4-, 8-, 12-, and 16-block routes.
3. Evaluate every route on the 904-example V2 development manifest and compare task-specific choices against matched generic, contiguous, and random controls.

Full-attention blocks were not automatically protected. A route was allowed to include blocks 7, 15, 23, or 31 when the discovery measurements ranked them as low-impact for that capability.

## Model And Data

- Model: `Qwen/Qwen2.5-VL-3B-Instruct`
- Revision: `66285546d2b821cf421d4f5eb2576359d3770cd3`
- Precision: BF16
- Attention implementation: SDPA
- Decoding: deterministic greedy generation
- Vision blocks: 32
- Parameters per vision block: 19,702,200
- Full model parameters: 3,754,622,976
- Full vision parameters: 668,684,288
- Discovery examples: 1,780
- Route-evaluation examples: 904 development rows

Discovery capability counts were 360 attribute, 360 counting, 360 object, 340 OCR, and 360 spatial examples.

## Route Construction

For each capability, blocks were sorted from smallest to largest single-block accuracy drop. The first 4, 8, 12, and 16 blocks formed nested task-specific routes.

The controls were:

- Four generic routes ranked by macro-average drop across the five capabilities.
- Four contiguous routes whose windows minimized the sum of generic single-block scores.
- Twelve random routes: three deterministic seeds at each of the four budgets.

This produced 40 unique configurations:

- 20 task-specific routes.
- 4 generic routes.
- 4 contiguous routes.
- 12 random routes.

Every route generated predictions for all 904 development examples so target-task preservation and collateral damage to other capabilities could both be measured.

## Implementation Changes

- `src/vlm_bench/benchmark.py`
  - Records full, active, and removed model and vision parameter counts.
  - Records the exact parameter count for every vision block.
- `scripts/run_combined_ablation.py`
  - Preserves route assignments in result records.
  - Records active parameters and peak allocated/reserved GPU memory.
  - Associates latency measurements with route results.
- `scripts/build_task_routes.py`
  - Merges the old and new discovery predictions.
  - Computes capability and generic rankings.
  - Generates task-specific and matched-control route configurations.
  - Produces the reused development baseline prediction set.
- `scripts/analyze_task_routes.py`
  - Expands route assignments into capability-level comparisons.
  - Computes deterministic paired 95% bootstrap intervals over examples.
  - Distinguishes fixed-clock audited latency from diagnostic dynamic-clock latency.
- `scripts/audit_route_latency.py`
  - Randomizes route measurement order.
  - Brackets routes with baseline measurements at the start, middle, and end.
  - Uses 20 model warmups and five repeats over 50 balanced examples.
- `results/color-capability-qwen25-vl-3b/`
  - Contains the compact 300-example baseline and 32-block color sweep.
- `results/task-route-design-qwen25-vl-3b/`
  - Contains sensitivity rankings and all 40 route definitions.
- `results/task-route-benchmark-qwen25-vl-3b/`
  - Contains the compact 40-route benchmark summary.
- `results/task-route-analysis-qwen25-vl-3b/`
  - Contains the readable report, paired statistical analysis, and flat CSV.
- `results/task-route-latency-locked-qwen25-vl-3b/`
  - Contains the final latency audit with fixed graphics and memory clocks.

Raw predictions remain on the GPU host and are intentionally not committed because they are large. Compact summaries preserve route definitions, aggregate metrics, paired lost/recovered counts, parameter accounting, latency repeats, and analysis outputs.

## Color Sweep Result

The unmodified model scored 96.67% on 300 VQAv2 color questions.

Block 0 was uniquely sensitive: skipping it caused a 25.33-point loss. The full-attention blocks had heterogeneous color effects:

| Full-attention block | Color drop |
|---:|---:|
| 7 | 1.00 pp |
| 15 | 3.33 pp |
| 23 | 0.33 pp |
| 31 | 2.33 pp |

Therefore, full attention describes architecture but does not by itself determine task-specific removability.

## Multi-Block Accuracy Result

The four-block routes are the only practically interesting untrained operating point:

| Task-specific route | Target-task drop | Generic drop | Contiguous drop | Random mean drop |
|---|---:|---:|---:|---:|
| Attribute | 2.06 pp | 1.03 pp | 3.61 pp | 4.64 pp |
| Counting | 0.00 pp | -0.56 pp | 2.79 pp | 5.03 pp |
| Object | -1.80 pp | 2.99 pp | 10.78 pp | 3.19 pp |
| OCR | 5.95 pp | 11.89 pp | 15.14 pp | 38.20 pp |
| Spatial | 2.79 pp | 5.59 pp | 3.91 pp | 5.40 pp |

At eight removed blocks, target-task losses were already 11.86 points for attribute, 12.29 for counting, 10.18 for object, 37.30 for OCR, and 12.85 for spatial. Twelve- and sixteen-block routes degraded further. Independent one-block rankings therefore do not compose into aggressive pruning routes; block interactions dominate beyond mild pruning.

## Paired Statistical Result

Positive values mean the task-specific four-block route was more accurate than the control on the same examples.

| Task | Advantage over generic | Paired 95% bootstrap interval | Advantage over random mean | Paired 95% bootstrap interval |
|---|---:|---:|---:|---:|
| Attribute | -1.03 pp | [-4.12, 1.55] | 2.58 pp | [-0.34, 5.50] |
| Counting | -0.56 pp | [-5.59, 4.47] | 5.03 pp | [0.00, 10.06] |
| Object | 4.79 pp | [1.20, 8.38] | 4.99 pp | [2.40, 7.58] |
| OCR | 5.95 pp | [0.54, 11.89] | 32.25 pp | [26.85, 37.84] |
| Spatial | 2.79 pp | [-1.12, 7.26] | 2.61 pp | [-2.05, 7.08] |

Object and OCR routes show a positive paired advantage over generic pruning on this development set. Attribute, counting, and spatial differences versus generic pruning remain uncertain. All task routes beat the random mean by point estimate, but only object and OCR have intervals clearly above zero; counting reaches zero at the lower bound.

## Parameter And Latency Result

Replacing four blocks removes 78,808,800 parameters from the active runtime module tree:

- 11.79% of vision-encoder parameters.
- 2.10% of total model parameters.

Initial dynamic-clock timing suggested 12-15% vision speedup for several routes but produced contradictory values for equal-compute routes. Those numbers were rejected.

A final audit used:

- Graphics clock locked at 2505 MHz.
- Memory clock locked at 10501 MHz.
- Persistence mode enabled during the audit.
- Three bracketed baseline models.
- Randomized route order.
- 20 warmups per model.
- Five repeats over 50 balanced examples.
- Automatic clock and persistence reset on exit.

Audited task-route results were:

| Route | Vision speedup | End-to-end speedup |
|---|---:|---:|
| Attribute, four blocks | 6.86% | 2.87% |
| Counting, four blocks | 6.78% | 1.39% |
| Object, four blocks | 6.47% | 2.15% |
| OCR, four blocks | 7.28% | 4.41% |
| Spatial, four blocks | 6.52% | 1.77% |

The ten audited four-block routes ranged from 5.62% to 7.55% vision speedup. End-to-end timing is less comparable because routes can produce different output token sequences. One random route remained slower end-to-end despite a positive vision speedup.

The closing locked-clock baseline also became slower than the opening and middle baselines, ranging from 73.97 to 80.74 ms versus approximately 71.3 to 72.6 ms earlier. The report uses the median across all 15 baseline repeats, but this residual drift is a reason to report a speed range rather than excessive decimal precision.

## Main Interpretation

The strongest defensible conclusion is narrow:

> Capability-aware one-block maps can identify useful mild four-block routes for some tasks, especially object recognition and OCR, but independent block rankings fail as an aggressive pruning strategy because block interactions dominate at eight or more removed blocks.

For edge deployment, four-block vision pruning alone is insufficient. It removes only 2.10% of total model parameters and provides roughly 1-4% end-to-end speedup. The scientific value is the task-specific accuracy comparison, not a claim that this technique alone makes the full VLM edge-ready.

## Alternatives Considered

- Rerun all 1,480 old examples for the color extension. Rejected because those exact predictions already existed; only the 300 new color rows required inference.
- Protect all full-attention blocks. Rejected because the hypothesis under test is task dependence, and the color sweep showed large differences among full-attention blocks.
- Test only the previously chosen four-block route. Rejected because it would not establish scaling behavior or comparison against controls.
- Use one random control. Rejected in favor of three deterministic random routes per budget.
- Trust the original route latency pass. Rejected after equal-size routes reported implausibly different speedups.
- Run bootstrap analysis concurrently with latency measurement. Started accidentally, then stopped. The incomplete timing output was deleted and the audit restarted without competing CPU work.
- Lock only graphics clocks. Tested and rejected because some equal-compute controls still showed impossible slowdowns while memory clocks remained dynamic.
- Report the largest measured speedup. Rejected in favor of the fully locked-clock audit and a conservative range.

## Verification

- Completed 300 baseline color predictions.
- Completed 9,600 one-block color-ablation predictions: 32 blocks by 300 examples.
- Completed all 40 route configurations on 904 development examples: 36,160 predictions.
- Completed three repeated latency trials per original route over 100 balanced examples.
- Completed the final fully locked audit for ten four-block routes plus three bracketed baselines, with five repeats over 50 examples per model.
- Confirmed all 40 route records and the final analysis summary exist.
- Confirmed the GPU returned to persistence disabled, dynamic graphics/memory clocks, 1 MiB allocated memory, and 0% utilization.
- Python compilation passed for every changed script.
- `git diff --check` passed before staging.

## Limitations And Risks

- Route rankings used all 1,780 discovery examples, including examples previously labeled test. The 904-example route benchmark overlaps discovery data and is not held-out evidence.
- The paired bootstrap intervals quantify example-level uncertainty on this development set; they do not account for dataset-source uncertainty or route-selection bias.
- Three random controls per budget are insufficient to characterize the complete random-route distribution.
- Single-block ranking is a simplistic route-search method and ignores interactions by construction.
- OCR still loses 5.95 points at the four-block budget, even though it outperforms controls.
- Attribute coverage is improved but still measures color rather than the full space of visual attributes.
- Identity replacement removes parameters from the active runtime module tree but does not produce a serialized compact checkpoint.
- All measurements use one Qwen model and one RTX 4090. No edge-device measurement or second-model replication exists.
- End-to-end latency includes generated-answer behavior and should not be interpreted as pure compute reduction.
- Residual baseline timing drift remained even with graphics and memory clocks locked.
- No recovery fine-tuning or distillation was performed.

## Claims This Commit Does Not Support

- It does not show that a capability is stored in one block.
- It does not show that full-attention blocks are always removable or always necessary.
- It does not show that task-aware pruning universally beats generic pruning.
- It does not establish held-out generalization.
- It does not establish edge-device speedup.
- It does not produce a deployable compact VLM.

## Next Step

Focus on the four-block regime and replace independent ranking with interaction-aware route search. Measure selected block pairs or use greedy backward elimination on discovery data, freeze a small set of object/OCR/general routes, optionally recover them with distillation or lightweight fine-tuning, serialize physical compact checkpoints, and evaluate only those frozen routes on genuinely unseen multi-capability data and a second VLM.
