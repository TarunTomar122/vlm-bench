# Phase 3 Interaction-Aware Route Search

This is a development-set search. It is not external held-out evidence.

## Search-Selected Routes

| Capability | Blocks | Search drop |
|---|---|---:|
| attribute | `[3, 6, 8, 9, 16, 20, 22, 25]` | 3.00 pp |
| counting | `[6, 9, 12, 21, 24, 25, 28, 30]` | 1.00 pp |
| object | `[2, 5, 9, 10, 17, 23, 25, 30]` | -4.00 pp |
| ocr | `[4, 5, 9, 11, 25, 26, 27, 28]` | 18.00 pp |
| spatial | `[3, 10, 16, 19, 22, 26, 28, 29]` | 3.00 pp |

## Image-Disjoint Development Validation

| Capability | Primary blocks | Target drop | Macro drop |
|---|---|---:|---:|
| attribute | `[3, 6, 8, 9, 16, 20, 22, 25]` | 3.19 pp | 15.58 pp |
| counting | `[6, 9, 12, 21, 24, 25, 28, 30]` | 5.06 pp | 18.24 pp |
| object | `[2, 5, 9, 10, 17, 23, 25, 30]` | 14.93 pp | 25.84 pp |
| ocr | `[4, 5, 9, 11, 25, 26, 27, 28]` | 23.53 pp | 12.00 pp |
| spatial | `[3, 10, 16, 19, 22, 26, 28, 29]` | 15.19 pp | 20.82 pp |

## Strongest Harmful Pair Interaction

| Capability | Blocks | Interaction |
|---|---|---:|
| attribute | `[8, 9]` | +4.00 pp |
| counting | `[27, 28]` | +7.00 pp |
| object | `[21, 23]` | +9.00 pp |
| ocr | `[6, 28]` | +5.00 pp |
| spatial | `[26, 29]` | +5.00 pp |

The search/validation split is image-disjoint, but the original one-block ranking used the
larger discovery set. These results therefore remain method-development evidence. The sealed
external benchmark must not be used until the route and recovery method are frozen.
