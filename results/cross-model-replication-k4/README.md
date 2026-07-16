# Cross-Model Vision-Block Route Replication

## Matched-K Method-Selection Results

| Model | Budget | Evolved generic | Evolved task policy | Task minus generic | Paired 95% interval |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL-3B-Instruct | K4 | 81.28% | 81.39% | +0.11 pp | [-1.83, 2.05] |
| Qwen2.5-VL-3B-Instruct | K6 | 79.11% | 81.28% | +2.17 pp | [0.00, 4.34] |
| Qwen2.5-VL-3B-Instruct | K8 | 75.91% | 76.60% | +0.68 pp | [-1.71, 3.08] |
| SmolVLM2-2.2B-Instruct | K4 | 72.49% | 73.29% | +0.80 pp | [-1.94, 3.54] |

## Fresh OCR Transfer

Full: 94.80%; generic K4: 86.40%; OCR K4: 72.80%.

Frozen OCR route minus frozen generic route = -13.60 pp with paired 95% interval [-19.20, -8.40].

## SmolVLM2 Generic-Route Latency

Measurement mode: `unlocked_same_vm_fallback`.

| Budget | Vision speedup | End-to-end speedup |
|---:|---:|---:|
| K4 | +8.60% | +4.19% |

## Evidence Boundary

The matched-K tables are image-disjoint method-selection evidence on processed-v2. The IIIT5K value is a separate sealed OCR source-transfer test and was excluded from every route-selection step. The latency table is batch-size-one RTX 4090 evidence, not a mobile-device measurement. If the measurement mode is unlocked, the cloud provider denied clock control and the measurement is only a same-VM comparison. Neither table supports a universal capability-routing claim unless the corresponding uncertainty interval excludes zero.
