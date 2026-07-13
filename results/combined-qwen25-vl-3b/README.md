# Progressive Combined Vision-Block Ablation

## Result

This experiment tests whether individually low-sensitivity blocks remain removable together.
Every candidate was evaluated on the same 1,480 examples. Latency is the median of three repeats
over the same balanced 100-example subset (20 examples per capability), measured against an
equally repeated unpruned baseline.

| Pathway | Skipped blocks | Accuracy loss | Median paired vision speedup | Median paired total speedup |
|---|---|---:|---:|---:|
| strict-1 | 28 | 0.14 pp | 0.46% | -0.32% |
| ocr-aware-2 | 5, 28 | 0.95 pp | 2.22% | -0.29% |
| general-3 | 5, 9, 28 | 2.97 pp | 2.42% | -3.98% |
| general-4 | 3, 5, 9, 28 | 3.72 pp | 4.89% | 1.48% |

## Interpretation

- Block 28 can be skipped with negligible accuracy impact, but one block does not produce a
  measurable end-to-end speedup.
- Skipping blocks 5 and 28 together keeps overall loss below one point, but remains latency
  neutral. It is an accuracy candidate, not a deployment candidate.
- The three-block pathway loses 2.97 points overall and is slower end-to-end. Its slower output
  generation shows why a single latency pass is not enough to infer a compute benefit.
- The four-block pathway skips 12.5% of vision blocks and improves median paired vision latency
  by 4.89% and total latency by 1.48%, but loses 3.72 points overall and 7.94 points on OCR. It is
  not acceptable as a universal pathway without recovery training.

The one-block screen therefore identified useful candidates, but their effects are non-additive:
blocks that are individually low-sensitivity can jointly damage OCR and object/spatial reasoning.
The next experiment should optimize a named deployment target. For example, a non-OCR pathway
may accept OCR loss, while a universal pathway must preserve OCR and should use recovery
fine-tuning or a different compression mechanism.

## Score Resolution

The paired-score audit recomputed every candidate correctness value from its raw prediction,
reference answer, and answer format. It found zero mismatches, and every candidate had the same
1,480 example IDs as the baseline. The arithmetic is therefore correct for this benchmark.

Small changes are not equally informative across capabilities:

| Capability | Examples | One-answer score increment |
|---|---:|---:|
| Attribute | 60 | 1.67 pp |
| Counting | 360 | 0.28 pp |
| Object | 360 | 0.28 pp |
| OCR | 340 | 0.29 pp |
| Spatial | 360 | 0.28 pp |

Attribute differences of one or two points are consequently too coarse to interpret. The larger
OCR, counting, object, and spatial changes in the three- and four-block paths represent 12--27
net answer changes and are materially more credible, though they still need a held-out or larger
replication before becoming a general claim.

`summary.json` contains the paired accuracy changes and all three latency repeats. Raw candidate
predictions remain on the GPU machine and are intentionally excluded from Git.
