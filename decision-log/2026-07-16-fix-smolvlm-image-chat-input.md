# Fix SmolVLM Image Chat Input

## Intended Commit Subject

`Fix SmolVLM image chat preprocessing`

## Problem And Decision

The SmolVLM2 smoke test loaded the model but failed before generation because its processor's
tokenizing chat-template path found an image token in the rendered prompt without receiving an
image. Its API extracts visual inputs from the image content item itself. The Smol adapter must
embed the already loaded PIL image as `{"type": "image", "image": image}` while Qwen retains
its existing placeholder-plus-separate-images preprocessing path.

## Files And Behavior Changed

- `src/vlm_bench/benchmark.py` now supplies a real image object in SmolVLM chat content and keeps
  Qwen's placeholder content unchanged. The processor then creates matching pixel values and image
  tokens in one call.

## Alternatives Considered

- Passing images as a separate argument to SmolVLM's tokenizing chat-template call was rejected:
  the processor API constructs image batches from image, URL, path, or base64 keys in message
  content.
- Changing the Qwen input path was rejected because its validated processor flow is different and
  should not be altered by this replication.

## Verification Evidence

- Remote smoke failure: `ValueError: We detected 1 tokens in the text but no images/videos were
  passed` after SmolVLM2 checkpoint loading.
- Transformers 4.53.2 `processing_utils.py` was inspected on the VPS; it searches message visual
  content for `image`, `url`, `path`, or `base64` keys before calling the processor.
- The exact full-model then block-0 identity smoke test will be rerun after deployment.

## Limitations And Non-Claims

- This corrects input plumbing only. It does not report benchmark accuracy or a pruning result.

## Next Action Enabled

Rerun the two-example full-versus-block-0 smoke test and begin the frozen baseline only if both
paths generate successfully.
