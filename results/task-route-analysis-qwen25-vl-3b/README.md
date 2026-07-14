# Task-Specific Vision Routes

These are development-set route-search results, not held-out claims. Four-block latency uses the fixed-clock audit; larger budgets retain diagnostic dynamic-clock timing.

| Task | Removed blocks | Task route drop | Generic drop | Contiguous drop | Random mean drop | Vision speedup | Total speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| attribute | 4 | 2.06 pp | 1.03 pp | 3.61 pp | 4.64 pp | 6.86% | 2.87% |
| attribute | 8 | 11.86 pp | 7.22 pp | 43.30 pp | 38.66 pp | 21.73% | 17.01% |
| attribute | 12 | 17.53 pp | 22.16 pp | 68.04 pp | 42.27 pp | 30.64% | 18.88% |
| attribute | 16 | 67.53 pp | 52.58 pp | 46.39 pp | 59.45 pp | 33.45% | 14.30% |
| counting | 4 | 0.00 pp | -0.56 pp | 2.79 pp | 5.03 pp | 6.78% | 1.39% |
| counting | 8 | 12.29 pp | 2.79 pp | 30.17 pp | 29.05 pp | 20.48% | 15.78% |
| counting | 12 | 25.14 pp | 24.02 pp | 42.46 pp | 35.38 pp | 29.86% | 17.62% |
| counting | 16 | 31.28 pp | 34.08 pp | 43.02 pp | 37.24 pp | 39.47% | 20.11% |
| object | 4 | -1.80 pp | 2.99 pp | 10.78 pp | 3.19 pp | 6.47% | 2.15% |
| object | 8 | 10.18 pp | 10.18 pp | 38.32 pp | 27.35 pp | 20.60% | 15.35% |
| object | 12 | 27.54 pp | 22.16 pp | 39.52 pp | 29.74 pp | 27.24% | 15.02% |
| object | 16 | 39.52 pp | 38.32 pp | 36.53 pp | 35.93 pp | 40.38% | 20.74% |
| ocr | 4 | 5.95 pp | 11.89 pp | 15.14 pp | 38.20 pp | 7.28% | 4.41% |
| ocr | 8 | 37.30 pp | 32.43 pp | 71.89 pp | 68.29 pp | 18.55% | 15.24% |
| ocr | 12 | 71.35 pp | 68.65 pp | 72.97 pp | 72.61 pp | 28.27% | 16.57% |
| ocr | 16 | 72.97 pp | 72.97 pp | 72.97 pp | 72.97 pp | 40.65% | 21.35% |
| spatial | 4 | 2.79 pp | 5.59 pp | 3.91 pp | 5.40 pp | 6.52% | 1.77% |
| spatial | 8 | 12.85 pp | 10.61 pp | 29.61 pp | 23.28 pp | 20.75% | 16.14% |
| spatial | 12 | 22.35 pp | 26.26 pp | 29.61 pp | 29.24 pp | 29.36% | 18.13% |
| spatial | 16 | 25.70 pp | 29.05 pp | 27.93 pp | 29.05 pp | 32.00% | 13.30% |

A task route is useful only if it preserves its target capability better than controls at comparable measured speed. Identity substitution removes the selected block parameters from the active runtime module tree, but this experiment does not yet serialize a standalone compact checkpoint.

## Four-Block Paired Accuracy Advantage

Positive means the task-specific route is more accurate than the control. Intervals are paired 95% bootstrap intervals over examples.

| Task | Versus generic | 95% interval | Versus random mean | 95% interval |
|---|---:|---:|---:|---:|
| attribute | -1.03 pp | [-4.12, 1.55] | 2.58 pp | [-0.34, 5.50] |
| counting | -0.56 pp | [-5.59, 4.47] | 5.03 pp | [0.00, 10.06] |
| object | 4.79 pp | [1.20, 8.38] | 4.99 pp | [2.40, 7.58] |
| ocr | 5.95 pp | [0.54, 11.89] | 32.25 pp | [26.85, 37.84] |
| spatial | 2.79 pp | [-1.12, 7.26] | 2.61 pp | [-2.05, 7.08] |
