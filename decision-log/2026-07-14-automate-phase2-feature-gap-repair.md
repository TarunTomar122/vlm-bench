# Decision: Automate Phase 2 feature-gap measurement and repair

- Date: 2026-07-14
- Intended commit subject: `Automate feature-gap repair experiment`
- Status: Accepted for implementation and smoke validation

## Problem And Decision

Phase 1 measured task-dependent accuracy effects from identity-skipping vision blocks. Those effects
conflate loss of useful computation with downstream distribution mismatch. Short-LVLM identifies
inter-layer feature gaps as a central failure mode for direct LVLM layer pruning, so the project
requires an explicit representation-repair control before interpreting blocks as capability-bearing.

Build a resumable Phase 2 pipeline that measures per-depth feature displacement and tests a frozen
low-rank final-boundary repair. Label it SCP-inspired rather than claiming a Short-LVLM
reproduction because that project's public repository did not expose its TIS/SCP implementation at
the time of implementation.

## Files And Behavior Changed

- `configs/phase2_feature_gap.json` fixes split, fitting, rank, and checkpoint settings.
- `src/vlm_bench/phase2.py` provides route selection, image-disjoint splitting, feature metrics,
  online sufficient statistics, and a low-rank residual bridge.
- `scripts/prepare_phase2.py` freezes the six routes and creates calibration/evaluation manifests.
- `scripts/fit_phase2_bridges.py` records per-depth gaps and fits resumable bridges.
- `scripts/validate_phase2_bridges.py` measures feature reconstruction on evaluation images.
- `scripts/run_phase2_repair.py` generates resumable repaired-route predictions.
- `scripts/analyze_phase2.py` produces paired task/generic comparisons and bootstrap intervals.
- `scripts/run_phase2_pipeline.py` runs or resumes all stages and records machine-readable state.
- `docs/phase2_feature_gap_protocol.md` defines claims, controls, interpretation, and commands.
- `tests/test_phase2.py` checks fixed-route selection, image disjointness, and bridge recovery.

## Alternatives Considered

- Interpret identity-ablation loss directly as stored task information. Rejected because feature
  shock is an unresolved causal alternative.
- Claim an exact SCP reproduction from the paper equations alone. Rejected because the public
  implementation was unavailable and Qwen2.5-VL uses a separate vision encoder architecture.
- Fit task routes using only their target capability. Rejected because it would confound route
  selection with task-specific repair data.
- Retain raw activation tensors. Rejected because online sufficient statistics and per-example
  scalar gap traces are enough for the planned bridge and are far smaller.
- Run only one bridge rank. Rejected because ranks 8, 32, and 128 expose the repair-capacity tradeoff.

## Verification

- Python compilation passed for all Phase 2 modules and scripts in the pinned GPU environment.
- `PYTHONPATH=src .venv/bin/pytest -q tests/test_phase2.py tests/test_scoring.py` passed all seven
  tests.
- Preparation produced 200 calibration and 704 evaluation examples, exactly 40 calibration
  examples per capability, and zero image overlap.
- The GPU smoke pipeline fitted a rank-8 task-attribute bridge from eight examples and 2,048
  sampled tokens, validated features on one example per capability, and generated two repaired
  answers.
- The smoke bridge reduced relative-L2 feature error on all five validation examples. This is a
  mechanics check only because the calibration sample is intentionally tiny.
- GPU state returned to 1 MiB used, 0% utilization, persistence disabled, and dynamic clocks after
  smoke completion.
- `git diff --check` and the staged whitespace check passed before commit.

## Limitations, Risks, And Unsupported Claims

- This bridge is not Short-LVLM SCP and does not reproduce its reported results.
- A final-boundary linear repair may miss nonlinear or earlier-layer effects.
- Development data support mechanism selection, not held-out generalization.
- Phase 1 route selection already used overlapping discovery information.
- Repair success does not prove a skipped block lacked useful computation; it shows that a compact
  surrogate can approximate enough of its cumulative effect.
- Repair failure does not prove computation is irrecoverable.
- No compact standalone checkpoint or edge-device result is produced by setup alone.

## Next Action Enabled

Run the complete Phase 2 pipeline, determine whether task-specific advantages survive matched
feature-gap repair, then freeze the surviving route and repair design for genuinely held-out and
second-model validation.
