# Phase 2 Feature-Gap Repair

This is a development-set mechanism experiment. The bridge is SCP-inspired but is not a reproduction of Short-LVLM SCP.

## Target-Capability Results

| Capability | Rank | Identity drop | Repaired drop | Repair gain | Task vs repaired generic |
|---|---:|---:|---:|---:|---:|
| attribute | 8 | 0.65 pp | 0.65 pp | +0.00 pp | -0.65 pp |
| attribute | 32 | 0.65 pp | 1.30 pp | -0.65 pp | -0.65 pp |
| attribute | 128 | 0.65 pp | 0.65 pp | +0.00 pp | +0.00 pp |
| counting | 8 | 0.00 pp | 3.60 pp | -3.60 pp | -4.32 pp |
| counting | 32 | 0.00 pp | 4.32 pp | -4.32 pp | -5.04 pp |
| counting | 128 | 0.00 pp | 1.44 pp | -1.44 pp | -2.88 pp |
| object | 8 | 0.00 pp | 0.79 pp | -0.79 pp | +4.72 pp |
| object | 32 | 0.00 pp | -0.79 pp | +0.79 pp | +6.30 pp |
| object | 128 | 0.00 pp | 0.79 pp | -0.79 pp | +4.72 pp |
| ocr | 8 | 4.83 pp | 4.83 pp | +0.00 pp | +6.90 pp |
| ocr | 32 | 4.83 pp | 4.14 pp | +0.69 pp | +7.59 pp |
| ocr | 128 | 4.83 pp | 6.21 pp | -1.38 pp | +4.14 pp |
| spatial | 8 | 4.32 pp | 6.47 pp | -2.16 pp | -2.16 pp |
| spatial | 32 | 4.32 pp | 6.47 pp | -2.16 pp | -2.88 pp |
| spatial | 128 | 4.32 pp | 6.47 pp | -2.16 pp | -4.32 pp |

## Evaluation Feature Gap

| Route | Identity relative L2 | Rank 8 | Rank 32 | Rank 128 |
|---|---:|---:|---:|---:|
| task-attribute | 0.7716 | 0.6149 | 0.6149 | 0.6149 |
| task-counting | 0.7122 | 0.6650 | 0.6650 | 0.6650 |
| task-object | 0.7246 | 0.6534 | 0.6534 | 0.6533 |
| task-ocr | 0.8096 | 0.6599 | 0.6599 | 0.6598 |
| task-spatial | 0.6874 | 0.6260 | 0.6260 | 0.6260 |
| generic | 0.7283 | 0.5995 | 0.5995 | 0.5995 |

Positive repair gain means the repaired task route answered more evaluation examples correctly than its uncorrected identity route. Confidence intervals and paired counts are stored in `analysis.json`.
