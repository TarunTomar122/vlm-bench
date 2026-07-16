# Include Fixed-Clock Latency In Final Report

## Intended Commit Subject

`Include fixed-clock latency in final replication report`

## Problem And Decision

The cross-model report was originally scheduled before the fixed-clock latency audit, which would
leave the final artifact without the practical efficiency evidence required by the protocol. Make
latency a prerequisite of final report generation and include generic K4/K6/K8 speedups directly.

## Files And Behavior Changed

- `scripts/analyze_cross_model_replication.py` reads the three fixed-clock latency summaries,
  records their medians and speedups in JSON, and adds a Markdown table.
- `scripts/supervise_smolvlm2_replication.py` now runs the latency audit before generating the
  final cross-model report.

## Alternatives Considered

- Generating a second separate report after latency was rejected because it risks readers using the
  earlier incomplete artifact.
- Inserting floating-clock timings was rejected because the protocol requires locked-clock speed
  evidence.

## Verification Evidence

- The latency wrapper writes one `summary.json` per K under the fixed-clock result root, each with
  bracketed baseline medians and route speedups.
- The supervisor checks the K8 summary only after the wrapper has completed K4, K6, and K8 in
  order, then invokes the report once.

## Limitations And Non-Claims

- The final report still distinguishes RTX 4090 batch-size-one timings from edge-device claims.
- Capability-conditional routing latency remains out of scope for this generic-route audit.

## Next Action Enabled

Produce one complete report containing same-K accuracy, fresh OCR transfer, and fixed-clock generic
route speedups after all automated stages finish.
