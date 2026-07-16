# Add SmolVLM Processor Dependency

## Intended Commit Subject

`Add SmolVLM processor dependency`

## Problem And Decision

The pinned Transformers 4.53.2 SmolVLM processor imports `num2words` during construction. The GPU
environment did not have that package, so the pre-inference smoke test failed before model loading
or prediction. Add the explicit runtime dependency instead of treating the manually provisioned VPS
as the only source of truth.

## Files And Behavior Changed

- `requirements-gpu.txt` now pins `num2words==0.5.14`, enabling `AutoProcessor` construction for
  SmolVLM2 under the pinned Transformers version.

## Alternatives Considered

- Installing the package only on the current VPS was rejected because a replacement instance would
  reproduce the same failure.
- Bypassing the official SmolVLM processor was rejected because it risks changing prompt and image
  preprocessing semantics relative to the checkpoint.

## Verification Evidence

- The initial remote smoke attempt failed with `ImportError: Package num2words is required to run
  SmolVLM processor` from `transformers.models.smolvlm.processing_smolvlm`.
- The next gate is `pip install -r requirements-gpu.txt` followed by the same full-model and
  one-block identity smoke test.

## Limitations And Non-Claims

- This is an environment repair only. It contains no model-quality, latency, or pruning evidence.

## Next Action Enabled

Install the pinned dependency on the GPU host and continue the preflight smoke test.
