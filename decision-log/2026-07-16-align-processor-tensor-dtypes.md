# Align Processor Tensor Dtypes

## Intended Commit Subject

`Align VLM processor tensors with model dtype`

## Problem And Decision

SmolVLM2's processor emits `float32` pixel values, while the pinned model is loaded in bf16. The
first valid image-chat smoke reached the vision patch convolution and failed because CUDA convolution
requires matching input and weight dtypes. Move processor floating tensors to the loaded model dtype
while moving integer and boolean tensors to the GPU unchanged.

## Files And Behavior Changed

- `src/vlm_bench/benchmark.py` now moves all floating processor outputs to `self.model.dtype` and
  all non-floating outputs to the configured device without a dtype cast.

## Alternatives Considered

- Loading SmolVLM2 in float32 was rejected because it would invalidate the frozen bf16 memory and
  latency protocol and use substantially more VRAM.
- Casting all inputs was rejected because token IDs and attention masks must remain integer or
  boolean tensors.

## Verification Evidence

- Remote smoke failure reached `SmolVLMVisionEmbeddings.patch_embedding` and raised: input
  `torch.cuda.FloatTensor` versus bf16 convolution weights.
- The full-model and block-0 identity smoke will be rerun after deployment.

## Limitations And Non-Claims

- This fixes execution compatibility only; it contains no benchmark result.

## Next Action Enabled

Complete the full-model and block-0 smoke, then start the frozen baseline and development-only
single-block sweep.
