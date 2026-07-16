# Related Work Map

## Structured Depth and Width Pruning

ShortGPT ranks residual blocks by input-output influence and removes low-influence LLM layers.
SliceGPT uses computational invariance to reduce dense hidden dimensions. These establish that
structured compression can yield hardware-realizable gains, but they are not designed around
vision-encoder capability interactions.

## Visual Token Reduction

DynamicViT learns input-dependent token sparsification for vision transformers. Token Merging
(ToMe) combines similar tokens without retraining. SparseVLM, VScan, FlowCut, and related LVLM
methods reduce visual-token cost at the vision-language interface or through decoder depth. They
change sequence width; this project changes vision-encoder depth, so the methods are complementary.

## Layer Selection in VLMs

FlashVLM uses visual attention to guide inference-only layer skipping during VLM decoding. Recent
domain-aware pruning studies domain-conditioned decoder-layer rankings and reports that ranking
matters most at low pruning budgets. The closest conceptual overlap is conditional redundancy, but
our intervention is the vision tower and our central comparison is combinatorial generic versus
capability-specific routes under matched K across two architectures.

## Positioning Sentence

> Prior work primarily optimizes generic depth, decoder layers, or visual tokens; we test whether
> named visual capabilities support different vision-encoder block routes and whether combinatorial
> search transfers across models and sources.

Use the BibTeX keys in `references.bib`; verify final venue fields when the manuscript is frozen.
