# Phase 2: Feature-Gap Disambiguation

## Question

The Phase 1 identity intervention establishes that removing a vision block can change task accuracy.
It does not distinguish two mechanisms:

1. The removed block performed computation required by the capability.
2. Later blocks failed because identity substitution shifted their expected feature distribution.

Phase 2 measures that representation shift and tests whether a small frozen repair can recover the
lost behavior.

## Relationship To Short-LVLM

Short-LVLM motivates using important vision-language tokens for generic layer localization and
Subspace-Compensated Pruning (SCP) for inter-layer feature gaps. Its public repository did not
contain the promised TIS/SCP implementation when this protocol was created on 2026-07-14.

This project therefore does not claim to reproduce SCP. It uses an explicitly labeled
**SCP-inspired calibration bridge** adapted to Qwen2.5-VL's separate vision encoder:

```text
full final state       Y
pruned final state     X
calibration target     D = Y - X
ridge map              W: X -> D
rank-r bridge          W_r = truncated-SVD(W)
repaired state         X + X W_r + b
```

The bridge is fitted without gradient training and inserted after vision block 31, immediately
before Qwen's visual merger. This is deliberately simpler than modifying surviving block weights.

## Fixed Routes

Use the five capability-specific four-block routes and the generic four-block route fixed by Phase
1. Do not rerank blocks using Phase 2 evaluation examples.

## Data Separation

The 904-row V2 development manifest is divided by image identity:

- Calibration: 40 examples from each capability, 200 total.
- Mechanism evaluation: all remaining examples, expected 704 total.

No image may occur in both partitions. Every route uses the same balanced calibration rows. This
prevents task routes from receiving a task-specific repair-data advantage.

These are still development data. The experiment can support mechanism and method-selection
claims, not final generalization claims.

## Measurements

For every calibration image and route, store full-versus-pruned feature displacement after all 32
vision blocks:

- mean token cosine similarity;
- relative L2 displacement;
- residual RMS;
- full-state RMS.

Fit ranks 8, 32, and 128 from at most 256 deterministic visual tokens per calibration example.
Validate final-boundary feature reconstruction on 20 evaluation examples per capability before
interpreting answer recovery.

For every evaluation answer, preserve the prediction and calculate paired:

- accuracy drop from the full model;
- harmful and beneficial flips;
- repaired-route advantage over identity removal;
- repaired task-route advantage over repaired generic pruning;
- paired bootstrap 95% intervals.

## Interpretation

- Gap decreases and accuracy recovers: distribution mismatch explains some removal damage.
- Gap decreases but accuracy does not recover: final-state L2 alignment is insufficient or the
  missing computation is not represented by this linear repair.
- Gap does not decrease: the low-rank final-boundary bridge is inadequate; this does not prove
  irrecoverable capability loss.
- Task routes still beat generic routes after matched repair: stronger evidence of task-dependent
  computational redundancy.
- Task advantage disappears after repair: Phase 1 likely measured differential feature shock more
  than task-specific computation.

## Automation And Recovery

Run the smoke test:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_phase2_pipeline.py --smoke
```

Run or resume the complete experiment:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_phase2_pipeline.py
```

Predictions append one row at a time. Bridge moments checkpoint every ten calibration examples.
Completed routes, ranks, and stages are skipped when the same command is restarted. Pipeline state
is stored under `results/phase2-feature-gap-qwen25-vl-3b/pipeline_state.json`.
