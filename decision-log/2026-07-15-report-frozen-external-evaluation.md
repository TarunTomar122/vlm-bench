# Intended commit subject

Report the frozen external route evaluation

# Problem or decision

The committed external protocol needed to determine whether capability-conditional routes transfer
better than generic pruning. All four frozen conditions have completed, so the repository must
record the result, its statistical boundaries, and the fact that the external set is now consumed.

# Exact changes

- Added compact summaries, run metadata, paired analysis, and the generated report under
  `results/external-frozen-qwen25-vl-3b/`. Raw predictions remain ignored by Git and are retained
  on the VPS and off-instance backup.
- Updated `docs/current_status.md` with Phase 3 development outcomes, the complete external table,
  matched-K8 confidence intervals, evidence boundaries, and next research steps.
- Updated `README.md`, `docs/external_heldout_protocol.md`, and `docs/dataset_card.md` to mark the
  external set consumed and replace stale pre-evaluation claims.
- Updated the generated external report to distinguish the matched K8 test from the unmatched K4
  accuracy/compute tradeoff.

# Alternatives considered

- Claiming task K4 as the winner was rejected because it is less compressed than generic K8; the
  comparison is not matched compute.
- Tuning counting or object routes after seeing the external failures was rejected because that
  would leak held-out outcomes into method selection.
- Adding a generic-K4 condition after inspecting results was rejected because it was not included
  in the committed four-condition freeze.
- Reporting only overall accuracy was rejected because capability-specific transfer is the research
  question and the average hides opposite OCR, spatial, and counting effects.

# Verification

- Rebuilt 1,250 images from pinned source revisions and recovered the exact committed manifest hash
  `c01c93a9f8f007bb21a11c1952ca50fa51bfdaa5232ebbd681a979156cca5a77`.
- Completed exactly 1,250 predictions for each of full, generic K8, task K8, and task K4.
- Required exact prediction-ID equality before analysis.
- Computed paired accuracy flips and 10,000-sample bootstrap intervals with seed `20260715`.
- Full model: 74.96%; generic K8: 66.00%; task K8: 64.72%; task K4: 68.08%.
- Task K8 versus generic K8: -1.28 points overall, 95% interval [-3.60, 0.96].

# Limitations and unsupported claims

- Only one VLM architecture was evaluated.
- Public benchmarks may occur in model pretraining.
- Capability remains partly confounded with source dataset.
- Concurrent timings do not support speed claims.
- Task K4 was not compared with generic K4 externally and cannot support a matched-budget claim.
- The results do not establish causal localization, edge-device performance, or a serialized compact
  checkpoint.

# Next action

Replicate the frozen intervention on a second VLM and design a development-only, multi-fold route
objective that penalizes worst-task damage and unstable interactions before using a new sealed set.
