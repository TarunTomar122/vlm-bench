# Current Research Status

Status date: 2026-07-14

## Short Answer

The project has established that mild, capability-specific vision-block pruning can outperform
generic or random pruning for object recognition and OCR on development data. It has not yet
produced an edge-ready model or publication-grade held-out result.

Four removed vision blocks are the only useful untrained operating point found so far. They remove
11.79% of the vision encoder but only 2.10% of the complete Qwen2.5-VL-3B model. Depending on the
route, this yields 6.47-7.28% vision-encoder speedup and 1.39-4.41% end-to-end speedup. Removing
8, 12, or 16 blocks using independent one-block rankings causes large accuracy losses, so simply
pruning more blocks is not a viable continuation.

The first low-rank feature-repair experiment also did not solve the problem. It made final vision
features numerically closer to the full model, but generally did not recover correct answers. This
is useful negative evidence: final-boundary feature similarity is not an adequate proxy for
behavioral recovery.

## What Has Been Completed

1. Built a five-capability benchmark for attribute/color, counting, object existence, OCR, and
   spatial relations.
2. Established a deterministic Qwen2.5-VL-3B-Instruct baseline and measured latency and memory.
3. Evaluated all 32 single vision-block identity interventions.
4. Evaluated capability-specific, generic, contiguous, and random routes at 4, 8, 12, and 16
   removed blocks.
5. Repeated the four-block latency comparison under locked GPU graphics and memory clocks.
6. Ran activation-rescue traces to separate cumulative representation damage from a
   one-block/one-capability interpretation.
7. Fitted and evaluated SCP-inspired rank-8, rank-32, and rank-128 final-boundary residual bridges.
8. Built a sealed 1,250-example external benchmark from source families not used for route
   discovery. No predictions have been generated on this set.
9. Reviewed eight closely related pruning papers. Existing work includes generic VLM pruning and
   task-specific encoder pruning, so novelty cannot rest on claiming either concept is new.

## Main Experimental Results

### Baseline And Architecture

- Model: `Qwen/Qwen2.5-VL-3B-Instruct`.
- Vision blocks: 32.
- Vision parameters: 668.7 million.
- Total parameters: 3.755 billion.
- Original 1,480-example benchmark accuracy: 81.62% overall.
- Median baseline vision latency: 69.48 ms.
- Median baseline end-to-end latency: 197.80 ms.

### Four-Block Capability Routes

Accuracy changes below are percentage points relative to the full model on the 904-row development
set. A negative drop is an improvement.

| Capability | Task-route drop | Generic drop | Task-route advantage over generic | 95% paired interval | End-to-end speedup |
|---|---:|---:|---:|---:|---:|
| Attribute | 2.06 pp | 1.03 pp | -1.03 pp | [-4.12, 1.55] | 2.87% |
| Counting | 0.00 pp | -0.56 pp | -0.56 pp | [-5.59, 4.47] | 1.39% |
| Object | -1.80 pp | 2.99 pp | +4.79 pp | [1.20, 8.38] | 2.15% |
| OCR | 5.95 pp | 11.89 pp | +5.95 pp | [0.54, 11.89] | 4.41% |
| Spatial | 2.79 pp | 5.59 pp | +2.79 pp | [-1.12, 7.26] | 1.77% |

Only object and OCR have paired intervals clearly above zero against generic pruning. Attribute,
counting, and spatial do not establish a task-specific advantage.

### Larger Pruning Budgets

The target-capability drops for independently ranked task routes were:

| Capability | 4 blocks | 8 blocks | 12 blocks | 16 blocks |
|---|---:|---:|---:|---:|
| Attribute | 2.06 pp | 11.86 pp | 17.53 pp | 67.53 pp |
| Counting | 0.00 pp | 12.29 pp | 25.14 pp | 31.28 pp |
| Object | -1.80 pp | 10.18 pp | 27.54 pp | 39.52 pp |
| OCR | 5.95 pp | 37.30 pp | 71.35 pp | 72.97 pp |
| Spatial | 2.79 pp | 12.85 pp | 22.35 pp | 25.70 pp |

This experiment has already been run. It should not be repeated. The result shows that blocks that
appear independently safe are not jointly safe; interactions dominate beyond four removals.

### Phase 2 Feature Repair

Phase 2 used 200 image-disjoint calibration examples and 704 development evaluation examples. It
replaced the final pruned vision state `X` with `X + XAB + b`, where `AB` is a fitted low-rank
residual map. This was an SCP-inspired diagnostic, not Short-LVLM SCP and not an SVD replacement
for each removed block.

The bridge reduced mean relative L2 error for every route. For example, OCR decreased from 0.8096
to 0.6599, an 18.49% reduction, and the generic route decreased from 0.7283 to 0.5995, a 17.68%
reduction. Ranks 8, 32, and 128 produced almost identical feature errors, so rank above 8 added
negligible reconstruction value.

Behavior did not follow feature reconstruction:

| Capability | Identity drop | Best repaired drop | Best repair gain |
|---|---:|---:|---:|
| Attribute | 0.65 pp | 0.65 pp | 0.00 pp |
| Counting | 0.00 pp | 1.44 pp | -1.44 pp |
| Object | 0.00 pp | -0.79 pp | +0.79 pp |
| OCR | 4.83 pp | 4.14 pp | +0.69 pp |
| Spatial | 4.32 pp | 6.47 pp | -2.16 pp |

The strongest rank-32 task-versus-repaired-generic advantages were +6.30 points for object, with a
95% interval of [2.36, 11.02], and +7.59 points for OCR, with an interval of [1.38, 13.79]. These
remain development-set results and are partly caused by the generic repaired route performing
poorly; they do not show that the bridge itself is effective.

## What We Have Learned

1. Vision-block importance is not fully global. Object and OCR routes preserve their target tasks
   better than generic pruning at the same four-block compute budget.
2. A one-block sensitivity heatmap is useful for screening but insufficient for route construction.
   Independent effects do not compose at eight or more removals.
3. Four-block pruning is scientifically informative but practically small. The vision tower gets
   roughly 6-7% faster, while the full VLM gets only 1-4% faster and loses only 2.10% of its total
   parameters.
4. Making the final hidden state closer in relative L2 distance does not guarantee the same answer.
   The removed blocks likely contribute nonlinear computation, or repair must occur at each removed
   boundary rather than only after block 31.
5. The current evidence is not enough for a paper by itself. It comes from one model, development
   data used during route design, one GPU, and no physical compact checkpoint or edge-device test.

## Current Research Decision

Do not rerun the 8/12/16-block independent-ranking experiment and do not evaluate the sealed set
yet. The next decisive experiment is interaction-aware route construction on development data.

A practical first version is conditional greedy elimination:

1. Start with the full encoder.
2. Temporarily add each remaining candidate block to the current removed set.
3. Evaluate those candidates on a fixed calibration subset for one capability.
4. Permanently remove the least harmful candidate.
5. Recompute all candidate effects after every removal, continuing to eight blocks.
6. Compare the resulting four- and eight-block routes against the already measured independent,
   generic, random, and contiguous controls.

This directly tests the failure discovered above. If interaction-aware search cannot preserve
accuracy at eight removed blocks materially better than the existing route, vision-block pruning
alone is unlikely to deliver useful deployment gains. At that point the project should either
pivot toward mechanistic capability mapping across a second model or combine block routing with
visual-token and decoder compression.

## Evidence Boundaries

- The sealed external set contains 250 examples per capability and has zero image overlap with the
  V2 dataset under decoded-pixel hashing.
- The sealed set must remain unused until route search, prompts, scoring, repair, and latency
  procedures are frozen.
- No result currently establishes held-out generalization, edge-device acceleration, causal
  localization of a capability to one block, or superiority across VLM architectures.
- Public benchmark contamination remains possible even after image-level deduplication.
