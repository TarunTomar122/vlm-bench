# Frozen External Evaluation

The 1,250 examples come from source families excluded from route selection. All routes were committed before inference. Positive task-versus-generic values mean the conditional route is more accurate.

| Condition | Overall accuracy | Drop from full | Attribute drop | Counting drop | Object drop | OCR drop | Spatial drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 74.96% | 0.00 pp | 0.00 pp | 0.00 pp | 0.00 pp | 0.00 pp | 0.00 pp |
| generic-k8 | 66.00% | 8.96 pp | -0.80 pp | 8.00 pp | 5.20 pp | 25.60 pp | 6.80 pp |
| task-k8 | 64.72% | 10.24 pp | 0.80 pp | 19.20 pp | 8.40 pp | 22.40 pp | 0.40 pp |
| task-k4 | 68.08% | 6.88 pp | 0.40 pp | 18.40 pp | 3.20 pp | 10.80 pp | 1.60 pp |

## Conditional Versus Generic K8

| Route | Overall advantage | 95% interval | Attribute | Counting | Object | OCR | Spatial |
|---|---:|---:|---:|---:|---:|---:|---:|
| task-k8 | -1.28 pp | [-3.60, 0.96] | -1.60 pp | -11.20 pp | -3.20 pp | 3.20 pp | 6.40 pp |
| task-k4 | 2.08 pp | [-0.24, 4.32] | -1.20 pp | -10.40 pp | 2.00 pp | 14.80 pp | 5.20 pp |

## Interpretation

- Task K8 versus generic K8 is the matched-compute test. Task K8 does not win overall, and its confidence interval crosses zero.
- Spatial is the only statistically clear task-K8 advantage. Counting is a statistically clear task-K8 failure.
- Task K4 has the highest compressed-model accuracy, but it removes half as many blocks as generic K8 and is not a matched-compute control.
- The external set is now consumed. These outcomes must not be used to alter routes or hyperparameters.

Concurrent execution invalidates latency comparisons from this run. Use the existing fixed-clock latency audit for speed claims.
