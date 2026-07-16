# Robust Route Search Results

These are processed-v2 method-selection results, not a new sealed external evaluation.
All route comparisons at a given K skip exactly the same number of vision blocks.

## Matched-K Summary

| K | Full | Evolved generic | Evolved task policy | Task vs generic | Generic independent | Task independent | Contiguous | Random mean (range) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 82.65% | 72.49% | 73.29% | +0.80 pp | 67.58% | 64.38% | 71.58% | 54.53% (50.23-61.30%) |

## Capability-Specific Advantage

### K4

| Capability | Evolved task | Evolved generic | Advantage | 95% interval |
|---|---:|---:|---:|---:|
| attribute | 89.76% | 90.36% | -0.60 pp | [-4.82, 3.61] |
| counting | 69.06% | 61.88% | +7.18 pp | [1.10, 13.26] |
| object | 83.94% | 84.46% | -0.52 pp | [-5.18, 4.15] |
| ocr | 48.39% | 61.94% | -13.55 pp | [-21.94, -5.16] |
| spatial | 72.38% | 62.98% | +9.39 pp | [2.76, 16.02] |

## Evidence Boundary

The development/test image split is disjoint, but both partitions informed earlier
single-block discovery. These results can compare frozen route-construction methods; they
cannot establish untouched source transfer. The previously consumed external benchmark
was not accessed by this search or analysis.
