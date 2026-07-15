# Capability Benchmark Dataset Card

## Purpose

This derived benchmark supports within-model comparisons under vision-encoder interventions. It
is not intended as a new general-purpose VLM leaderboard. The primary unit is an image-question
pair with a short, deterministically scorable answer and an explicit visual capability label.

## Composition

### Controlled suite

The controlled suite uses five MME perception categories: OCR, count, position, existence, and
color. MME uses paired binary questions and a consistent response protocol, making it useful for
screening whether task-by-block patterns exist without changing answer format between tasks.

Source: https://mme-benchmark.github.io/home_page.html

### Discovery natural suite

The original natural suite samples 300 examples for each core capability:

| Capability | Dataset | Sampling |
|---|---|---|
| OCR | OCRBench | All direct text-recognition examples; excludes VQA, KIE, and formulas |
| Counting | TallyQA | Equal simple and complex questions, at most one selected QA per image row |
| Spatial | VSR random test | Round-robin over relation and binary label |
| Object existence | POPE | Approximately equal random, popular, and adversarial splits; balanced answers |

Sources:

- OCRBench: https://github.com/qywh2023/OCRbench
- TallyQA: https://ojs.aaai.org/index.php/AAAI/article/view/4815
- VSR: https://github.com/cambridgeltl/visual-spatial-reasoning
- POPE: https://github.com/RUCAIBox/POPE

### Sealed external suite

The source-transfer suite has 250 examples for each of five capabilities: TextVQA for OCR,
CountBenchQA for counting, CV-Bench/ADE20K for spatial relations, and disjoint AMBER subsets for
object existence and color attributes. All 1,250 images are unique and have zero decoded-pixel
overlap with V2. Its first frozen evaluation was completed on 2026-07-15, so this set is consumed
and cannot be used for additional route selection.

## Determinism

- Global sampling seed: `20260713`.
- Every Hugging Face source is resolved to a commit hash before loading.
- Images are converted to RGB PNG and named by SHA-256 content hash.
- Development/test assignment is a deterministic function of the image hash.
- The same image can never occur in both development and test.
- A smoke manifest contains up to 20 development examples per capability.

The generated `summary.json` records exact source revisions, counts, and the full manifest hash.

## Storage and Redistribution

Third-party images are downloaded only onto the experiment machine and are excluded from Git.
The repository publishes preparation code, source references, and derived manifests. Users remain
responsible for the licenses and terms of each upstream dataset and its underlying image source.
VSR metadata is CC BY 4.0; some source images in these benchmarks originate from COCO, Visual
Genome, or other datasets with their own terms.

## Known Limitations

- Dataset identity remains partly confounded with capability in both natural suites.
- MME is small and binary, with a 50% random-answer baseline.
- OCRBench contains multiple OCR task styles, not only direct transcription.
- VSR and POPE can contain object-recognition difficulty in addition to their nominal capability.
- TallyQA complex questions include attribute and relation reasoning before counting.
- Public benchmarks may have appeared in model training data.

Results should therefore be interpreted from paired full-model/intervention changes, replicated
across controlled and external suites, rather than as absolute capability measurements.
