# Robust Route Search Results

These are processed-v2 method-selection results, not a new sealed external evaluation.
All route comparisons at a given K skip exactly the same number of vision blocks.

## Matched-K Summary

| K | Full | Evolved generic | Evolved task policy | Task vs generic | Generic independent | Task independent | Contiguous | Random mean (range) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 83.68% | 81.28% | 81.39% | +0.11 pp | 80.25% | 81.28% | 79.91% | 72.37% (64.16-76.71%) |
| 6 | 83.68% | 79.11% | 81.28% | +2.17 pp | 78.08% | 77.51% | 68.26% | 64.80% (63.24-66.89%) |
| 8 | 83.68% | 75.91% | 76.60% | +0.68 pp | 72.37% | 68.15% | 40.98% | 46.96% (37.44-64.38%) |

## Capability-Specific Advantage

### K4

| Capability | Evolved task | Evolved generic | Advantage | 95% interval |
|---|---:|---:|---:|---:|
| attribute | 95.78% | 95.78% | +0.00 pp | [-3.01, 3.01] |
| counting | 72.93% | 72.93% | +0.00 pp | [-4.97, 4.97] |
| object | 86.53% | 84.97% | +1.55 pp | [-1.55, 5.18] |
| ocr | 72.26% | 73.55% | -1.29 pp | [-7.10, 4.52] |
| spatial | 79.01% | 79.01% | +0.00 pp | [-4.42, 4.42] |

### K6

| Capability | Evolved task | Evolved generic | Advantage | 95% interval |
|---|---:|---:|---:|---:|
| attribute | 93.37% | 92.17% | +1.20 pp | [-3.01, 5.42] |
| counting | 74.59% | 73.48% | +1.10 pp | [-3.87, 6.63] |
| object | 87.56% | 87.56% | +0.00 pp | [-3.11, 3.11] |
| ocr | 70.97% | 63.87% | +7.10 pp | [0.65, 14.19] |
| spatial | 79.01% | 76.80% | +2.21 pp | [-2.21, 6.63] |

### K8

| Capability | Evolved task | Evolved generic | Advantage | 95% interval |
|---|---:|---:|---:|---:|
| attribute | 91.57% | 93.37% | -1.81 pp | [-5.42, 2.41] |
| counting | 69.06% | 69.06% | +0.00 pp | [-5.52, 5.52] |
| object | 84.97% | 83.94% | +1.04 pp | [-3.63, 5.70] |
| ocr | 58.06% | 55.48% | +2.58 pp | [-4.52, 9.68] |
| spatial | 77.35% | 75.69% | +1.66 pp | [-3.31, 7.18] |

## Evidence Boundary

The development/test image split is disjoint, but both partitions informed earlier
single-block discovery. These results can compare frozen route-construction methods; they
cannot establish untouched source transfer. The previously consumed external benchmark
was not accessed by this search or analysis.
