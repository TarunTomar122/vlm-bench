# Publish robust route-search results and research report

## Intended commit subject

`Publish robust route-search results and research report`

## Decision

The frozen source-aware K4/K6/K8 route search, all six family searches, and every required
matched-K control have completed. The repository needs compact, reproducible artifacts that state
the result without committing large raw prediction caches or overstating method-selection evidence
as an untouched external evaluation.

## Files and behavior changed

- Added compact robust-route artifacts under `results/robust-route-search-qwen25-vl-3b/`:
  frozen route registry, family summaries, control state, analysis JSON/Markdown, supervisor state,
  baseline metadata, and the generated static report site.
- Updated `docs/current_status.md` with the completed source-aware matched-K outcome, its paired
  intervals, the K6 OCR signal, controls, evidence boundary, and the next research decision.
- Updated `projectIdeas/vlm-task-aware-encoder-pruning-project/research_landscape.md` with the
  result-driven novelty boundary and the replication-first research question.
- Updated `README.md` with a concise final result and links to the compact analysis and static
  report.

The raw route-prediction JSONL caches are intentionally not committed. They are resumable on the
GPU machine and backed up off-instance, while `.gitignore` prevents them from expanding the
repository.

## Alternatives considered

- Claim a task-aware pruning win from the K6 aggregate result. Rejected: the 876-row selection
  partition was image-disjoint but had informed prior one-block discovery, and the K6 lower
  bootstrap boundary is exactly zero.
- Treat the consumed 1,250-example external suite as final validation for this search. Rejected:
  it was intentionally excluded from all robust-search, control, and analysis code.
- Commit raw predictions for maximum transparency. Rejected: they are large, include dataset- and
  model-output artifacts, and are already preserved outside Git; compact summaries are sufficient
  to reproduce the reported aggregates when combined with the frozen configuration and manifests.

## Verification

- Confirmed all six family states are `complete` on the GPU machine.
- Confirmed `frozen_routes.json` exists and the 18 matched-K controls are `complete`.
- Validated `analysis/analysis.json`: 876 paired selection examples, K4/K6/K8 budgets, and all
  eight conditions per budget.
- Ran `python3 -m py_compile` for the route runner, control runner, supervisor, analysis, and
  report generator.
- Confirmed the static report contains the computed K4/K6/K8 values and that its heatmap assets
  resolve to committed local files.

## Limitations and unsupported claims

- These findings are method-selection evidence, not a fresh sealed source-transfer result.
- The result is limited to Qwen2.5-VL-3B identity substitution and does not prove a deployable
  compact checkpoint, edge-device speedup, model-family transfer, or causal localization of a
  capability to selected blocks.
- K6 OCR is the only clearly positive capability-level interval in this final run; it requires
  replication on a newly sealed source family and a second VLM.

## Next action

Freeze the route-selection rule, construct a new sealed source family, and replicate the exact
matched-K protocol on a second VLM before proposing a learned task router or combining this work
with token or decoder compression.
