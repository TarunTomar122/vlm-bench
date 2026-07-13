# One-Block Vision Encoder Ablation

## Result

The unpruned baseline is 81.62% on 1480 fixed examples. Each intervention replaces exactly one of Qwen2.5-VL-3B's 32 vision blocks with an identity mapping.

The sweep identifies sensitivity, not deployable pruning. A candidate must still pass a multi-block ablation, repeated latency measurement, and recovery fine-tuning before it can be called safe.

## Most Sensitive Blocks

| Block | Accuracy drop | Lost correct | Recovered |
|---:|---:|---:|---:|
| 15 | 21.08 pp | 359 | 47 |
| 0 | 17.36 pp | 324 | 67 |
| 31 | 7.77 pp | 156 | 41 |
| 14 | 7.50 pp | 149 | 38 |
| 16 | 4.46 pp | 97 | 31 |
| 23 | 4.26 pp | 115 | 52 |
| 7 | 3.99 pp | 109 | 50 |
| 1 | 3.65 pp | 98 | 44 |

## Screening Candidates

Threshold: at most 1.0 percentage point overall accuracy drop and at most 2.0 points on every capability. These are candidates for the next, multi-block experiment only.

| Block | Overall drop | Largest capability drop |
|---:|---:|---:|
| 28 | 0.14 pp | 1.67 pp (attribute) |

## Capability Map

`capability_accuracy_drop_heatmap.png` plots accuracy drop in percentage points for every capability and block. Positive means the block was useful under this intervention; negative means the ablation happened to improve this finite benchmark sample.

Latency from this one-pass sweep is diagnostic only. It is not used to rank candidates because individual block timings vary at the millisecond level; the next experiment needs repeated, matched-compute multi-block latency trials.
