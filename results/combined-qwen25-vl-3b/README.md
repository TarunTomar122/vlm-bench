# Progressive Combined Vision-Block Ablation

## Result

This experiment tests whether individually low-sensitivity blocks remain removable together.
Every candidate was evaluated on the same 1,480 examples. Latency is the median of three repeats
over the same balanced 100-example subset (20 examples per capability), measured against an
equally repeated unpruned baseline.

| Pathway | Skipped blocks | Accuracy | Accuracy drop | Vision latency | Total latency |
|---|---|---:|---:|---:|---:|
| Unpruned baseline | none | 81.62% | 0.00 pp | 70.06 ms | 197.47 ms |
| strict-1 | 28 | 81.49% | 0.14 pp | 69.34 ms | 197.30 ms |
| ocr-aware-2 | 5, 28 | 80.68% | 0.95 pp | 68.68 ms | 197.68 ms |
| general-3 | 5, 9, 28 | 78.65% | 2.97 pp | 68.69 ms | 206.02 ms |
| general-4 | 3, 5, 9, 28 | 77.91% | 3.72 pp | 66.70 ms | 193.70 ms |

## Interpretation

- Block 28 can be skipped with negligible accuracy impact, but one block does not produce a
  measurable end-to-end speedup.
- Skipping blocks 5 and 28 together keeps overall loss below one point, but remains latency
  neutral. It is an accuracy candidate, not a deployment candidate.
- The three-block pathway loses 2.97 points overall and is slower end-to-end. Its slower output
  generation shows why a single latency pass is not enough to infer a compute benefit.
- The four-block pathway skips 12.5% of vision blocks and improves median vision latency by
  5.04% and total latency by 1.95%, but loses 3.72 points overall and 7.94 points on OCR. It is
  not acceptable as a universal pathway without recovery training.

The one-block screen therefore identified useful candidates, but their effects are non-additive:
blocks that are individually low-sensitivity can jointly damage OCR and object/spatial reasoning.
The next experiment should optimize a named deployment target. For example, a non-OCR pathway
may accept OCR loss, while a universal pathway must preserve OCR and should use recovery
fine-tuning or a different compression mechanism.

`summary.json` contains the paired accuracy changes and all three latency repeats. Raw candidate
predictions remain on the GPU machine and are intentionally excluded from Git.
