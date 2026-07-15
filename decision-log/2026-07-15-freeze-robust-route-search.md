# Freeze source-aware robust route search

## Intended commit subject

`Freeze source-aware robust route search`

## Problem and decision

The existing capability-specific K8 routes did not beat generic K8 on the consumed external
benchmark, and target-only Phase 3 search was unstable across dataset sources. The next experiment
must determine whether robust generic or capability-specific routes can preserve accuracy at K4,
K6, and K8 without tuning on the external outcomes or fine-tuning model weights.

This commit freezes a source-balanced evolutionary search before candidate inference. The existing
processed-v2 development partition is search-only and the image-disjoint processed-v2 test
partition is method-selection-only. They contain MME plus one dedicated source for every
capability. Both partitions participated in earlier single-block discovery, so this is explicitly
method-selection evidence rather than a new sealed test. The 1,250-row external benchmark is
consumed and prohibited from route selection.

## Files and behavior changed

- `configs/robust_route_search.json` freezes K4/K6/K8, source-aware manifests, objective weights,
  population 16, three evaluated generations, three seeds, progressive finalist counts, cache
  provenance, and the no-fine-tuning boundary.
- `scripts/prepare_robust_route_search.py` preserves the existing image-disjoint source partitions
  and creates deterministic generic, task-plus-collateral, and 300-row development-evaluation
  manifests with hashes and overlap assertions.
- `src/vlm_bench/robust_search.py` implements fixed-K mutation/crossover, source-balanced paired
  metrics, generic/task robust objectives, Pareto selection, crowding distance, and route-stability
  metrics.
- `scripts/run_robust_route_search.py` implements resumable family jobs, Phase 3 cache import,
  append-only fingerprinted prediction caches, progressive scout/development/selection evaluation,
  K4/K6/K8 route freezing, and six-family finalization.
- `tests/test_robust_search.py` covers the pure optimizer and metric helpers.
- `.gitignore` excludes per-route cache metadata alongside raw prediction artifacts; compact
  aggregate summaries remain commit-eligible.
- `data/processed-v2/manifests/` and
  `data/processed-v2/robust-route-search/prepared/` contain the frozen source and derived manifests.
- `results/robust-route-search-qwen25-vl-3b/baseline/` records compact metadata for the complete
  1,780-row full-model reference. Raw predictions remain ignored by Git.
- `README.md`, `docs/current_status.md`, and `docs/robust_route_search_protocol.md` document the
  commands, evidence boundary, runtime budget, and current research state.
- Compact Phase 3 summaries are retained as prior-route provenance; raw route predictions remain
  ignored and are reused only through route-plus-example cache keys.

## Alternatives considered

- A 64-member, 40-generation search was rejected because it would waste days of GPU time and
  encourage search-set overfitting. The frozen 16-by-3 design plus three seeds is a practical first
  robustness test.
- Re-splitting only the 904-row development set was rejected because tiny MME strata would leave as
  few as four to eight images for selection. Preserving development/test provides 1,431 disjoint
  images and materially better small-source support.
- Evaluating every development finalist on all 904 search rows was rejected for cost. A frozen
  source-balanced 300-row development-evaluation tier sits between scouts and the complete 876-row
  selection partition.
- Reusing the consumed external set for selection was rejected as test contamination.
- Fine-tuning, LoRA, distillation, token pruning, and multi-model replication are intentionally out
  of scope until static route selection is established.

## Verification evidence

- `python3 scripts/prepare_robust_route_search.py` completed with 17 assertions true.
- Search/selection overlap is zero examples and zero decoded-image hashes.
- Frozen counts are 904 search rows, 300 development-evaluation rows, 876 selection rows, 133
  generic scout rows, and 32-38 rows per task scout including collateral strata.
- Source manifest SHA-256 is
  `ec3712d74a43caf8dd3d1818788ee0a92bc82e7aa597d640b00f089c6ed357c8`.
- Selection manifest SHA-256 is
  `343d7e7e85baf2cc1ed6f6d5b973f8a1d1b1edb1f243e594ffd410b27dba0f82`.
- The complete full-model cache contains 1,780 unique predictions. Accuracy is 84.62% on search and
  83.68% on selection.
- `PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_robust_search.py' -v`
  passed 13 tests.
- The complete remote suite passed: `PYTHONPATH=src .venv/bin/python -m pytest -q` reported
  `31 passed`.
- Remote preflight resolved all six family manifests, verified all 1,780 baseline IDs, and checked
  the frozen input hashes without loading the model.
- `python3 -m py_compile scripts/prepare_robust_route_search.py
  scripts/run_robust_route_search.py src/vlm_bench/robust_search.py` passed.

## Limitations and unsupported claims

- No route candidate has been inferred at this commit boundary; the commit freezes the method.
- The method-selection split is image-disjoint but not historically untouched because earlier
  single-block rankings used all 1,780 rows.
- Two source families per capability are insufficient for a broad distribution-shift claim.
- Evolutionary search is an optimizer, not the research novelty, and does not prove global
  optimality over all block combinations.
- Fixed-K routes have effectively equal compute; search can improve accuracy at a given K but
  cannot make one K8 block subset faster than another K8 subset.
- This does not establish edge-device latency, model-family transfer, or causal localization.

## Next action enabled

Run generic and five capability family jobs in three two-process waves on the RTX 4090, finalize
the frozen K4/K6/K8 route registry, and compare task routes with the generic route at matched K on
the method-selection partition before designing a new sealed source-transfer test.
