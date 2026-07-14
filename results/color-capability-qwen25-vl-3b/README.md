# VQAv2 Color Capability Sweep

Qwen2.5-VL-3B-Instruct reached 96.67% accuracy on 300 balanced, high-consensus, open-ended VQAv2 color questions. Each of the model's 32 vision blocks was then replaced individually by an identity mapping.

## Main Result

Block 0 is uniquely important for this color task: skipping it reduces accuracy by 25.33 percentage points. Most other single-block interventions cause no more than a 1.67-point loss.

The four full-attention blocks are not uniformly necessary for color:

| Block | Color accuracy drop |
|---:|---:|
| 7 | 1.00 pp |
| 15 | 3.33 pp |
| 23 | 0.33 pp |
| 31 | 2.33 pp |

This supports task-dependent block sensitivity, not the stronger claim that a capability is stored in one block. Small negative drops are finite-sample flips and must not be interpreted as genuine improvements without replication.

The full machine-readable sweep is in `summary.json`; `heatmap.csv` and `capability_layer_heatmap.png` provide compact views.
