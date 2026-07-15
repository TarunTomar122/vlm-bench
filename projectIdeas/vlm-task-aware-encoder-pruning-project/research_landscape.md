# Research Landscape and Positioning

This is a living map of the closest work. It defines what the project may and may not claim.

## Existing Directions

| Direction | Representative work | Relationship to this project |
|---|---|---|
| Generic static VLM layer pruning | INTERLACE | Removes generally redundant layers; does not build fine-grained capability pathways |
| Visual processing in decoder layers | Activating Distributed Visual Region within LLMs | Finds distributed visual-critical decoder layers; target is not the vision encoder |
| Domain-aware layer pruning | Understanding Pruning Regimes in VLMs Through Domain-Aware Layer Selection | Closest conceptual work; math/non-math selection in the decoder rather than visual capabilities in the encoder |
| Input-dynamic layer skipping | FlashVLM: Exploiting Layer Redundancy via Visual Attention | Uses runtime attention signals; not an explicit named-capability map |
| Query-conditioned visual-token pruning | FlashVLM: Text-Guided Visual Token Selection | Selects tokens rather than encoder blocks |
| Task-aware visual routing | Task-Aware Mechanism (TAM) | Routes video experts, frames, and resolution by task; not block pruning in a standard image encoder |
| Task-optimized token pruning | TOP-RL | Adapts token pruning by task and depth; does not identify encoder-block capability requirements |
| Layer-wise encoder analysis | Layer-wise Alignment | Uses image-encoder early exits to study safety; not broad capability-aware compression |
| Intermediate visual feature reuse | DeepStack | Demonstrates utility of intermediate encoder features; focused on stronger modeling rather than pruning |

## Primary References

1. **INTERLACE: Interleaved Layer Pruning and Efficient Adaptation in Large Vision-Language
   Models.** https://arxiv.org/abs/2511.19676
2. **Activating Distributed Visual Region within LLMs for Efficient and Effective
   Vision-Language Learning.** https://aclanthology.org/2025.acl-long.1484/
3. **Understanding Pruning Regimes in VLMs Through Domain-Aware Layer Selection.**
   https://arxiv.org/abs/2603.20275
4. **FlashVLM: Exploiting Layer Redundancy via Visual Attention for Efficient Vision-Language
   Inference.** https://openreview.net/forum?id=7U9pt9yUQF
5. **FlashVLM: Text-Guided Visual Token Selection for Efficient Vision-Language Models.**
   https://arxiv.org/abs/2512.20561
6. **Task-Aware Mechanism: Hybrid MoE Vision Tower Towards Holistic Video Understanding.**
   https://openreview.net/forum?id=nVBt6ifvl0
7. **Task-Optimized Progressive Token Pruning with Reinforcement Learning for Vision Language
   Models.** https://ojs.aaai.org/index.php/AAAI/article/view/38614
8. **Layer-wise Alignment: Examining Safety Alignment Across Image Encoder Layers in VLMs.**
   https://proceedings.mlr.press/v267/bachu25a.html
9. **DeepStack: Deeply Stacking Visual Tokens is Surprisingly Simple and Effective for LMMs.**
   https://proceedings.neurips.cc/paper_files/paper/2024/file/29cd7f8331d13ede6dc6d6ef3dfacb70-Paper-Conference.pdf

## Defensible Claim

The literature shows that VLM computation can be pruned or routed generically, dynamically, and
by broad domain. It also shows that intermediate visual representations differ across depth.
What appears comparatively underexplored is a controlled, causal comparison of fine-grained
visual capabilities across **vision-encoder blocks**, followed by task-specific execution of a
standard image VLM on edge-relevant hardware.

Use "underexplored" rather than "never studied" until a complete systematic review is finished.

## Positioning After The Completed Route Search

The completed Qwen2.5-VL-3B search changes the immediate claim. On the image-disjoint
method-selection split, source-aware capability routes do **not** show a clear aggregate advantage
at K4 or K8. K6 has a narrow positive signal: the conditional policy is +2.17 percentage points
over the evolved generic route, with OCR +7.10 points and a paired 95% interval of [0.65, 14.19].
The selection split had contributed to earlier one-block discovery, so this is not fresh
source-transfer evidence.

The next defensible research question is therefore not "can a learned router be deployed?" It is:
**does a frozen, source-balanced route-selection rule reproduce the K6/OCR effect on a newly
sealed source family and a second VLM?** A negative replication would still be useful: it would
show that apparent capability pathways are source- or architecture-dependent rather than stable
compression structure.

Only after this replication should a learned task router be compared with the frozen generic route
at matched K and fixed-clock latency. Token pruning or decoder compression should remain separate
ablations until the contribution of block routing is isolated.

## Required Baselines

Any final comparison should include:

- full model;
- uniform block removal;
- random block subsets;
- generic sensitivity-based pruning;
- simple early exit or contiguous truncation;
- capability-aware pruning;
- oracle routing before learned routing.

Without generic sensitivity-based pruning at matched compute, the project cannot establish that
the capability map improves compression rather than merely identifying generic redundancy.

## Main Scientific Distinction

Three statements require different evidence:

1. **Information is decodable at a block:** supported by a probe.
2. **A block is necessary for behavior:** supported by a controlled intervention.
3. **A block computes that capability:** requires stronger causal tracing and cannot be inferred
   from a score drop alone.

The project should report the strongest statement justified by each experiment and no stronger.
