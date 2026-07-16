# Increase SmolVLM2 Search Concurrency

## Intended Commit Subject

`Increase SmolVLM2 search concurrency`

## Problem And Decision

The original SmolVLM2 supervisor used two search lanes as a conservative assumption. The live
two-worker ablation preflight measured 11,771 MiB total VRAM use, or about 5.9 GiB per worker, on a
24,564 MiB RTX 4090. Three simultaneous workers are therefore expected to leave roughly 6.9 GiB
headroom while shortening the long route-search stage.

## Files And Behavior Changed

- `scripts/supervise_smolvlm2_replication.py` now schedules three ordered family lanes instead of
  two. The order within each lane still prevents duplicate family launches and all search inputs,
  objectives, seeds, candidates, and route selection remain unchanged.

## Alternatives Considered

- Retaining two lanes was safer but leaves measured VRAM capacity unused during the longest stage.
- Launching all six families was rejected because six full SmolVLM2 processes exceed the measured
  VRAM budget.
- Changing the evolutionary candidate budget was rejected because it would alter the frozen
  experimental protocol rather than only reduce wall-clock idle time.

## Verification Evidence

- Live `nvidia-smi` measurement during concurrent ablation: 11,771 MiB used and 12,311 MiB free.
- Each worker had started actual inference successfully; this was not a model-load-only estimate.

## Limitations And Non-Claims

- Three-worker safety is inferred from the current image workload. The supervisor's per-family
  route caches remain resumable if an unexpected OOM occurs.
- This change improves wall-clock time only and does not change the scientific comparison.

## Next Action Enabled

After the ablations complete, run three independent route-search families concurrently and monitor
the initial VRAM peak before continuing unattended.
