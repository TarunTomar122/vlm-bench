# Current Research Status

Status date: 2026-07-15

## Short Answer

The project now has a frozen 1,250-example external evaluation. At the matched eight-block budget,
capability-conditional routing did not beat generic pruning overall: it scored 64.72% versus
66.00%, a -1.28 percentage-point advantage with a paired 95% interval of [-3.60, 0.96]. Spatial
routing did transfer, beating generic K8 by 6.40 points with an interval of [1.60, 11.20]. The
other K8 capability routes did not establish a positive external advantage.

Four removed vision blocks are the only useful untrained operating point found so far. They remove
11.79% of the vision encoder but only 2.10% of the complete Qwen2.5-VL-3B model. Depending on the
route, this yields 6.47-7.28% vision-encoder speedup and 1.39-4.41% end-to-end speedup. Removing
8, 12, or 16 blocks using independent one-block rankings causes large accuracy losses, so simply
pruning more blocks is not a viable continuation.

The conservative four-block conditional model scored 68.08%, compared with 74.96% for the full
model. It preserved OCR and spatial substantially better than generic K8, but this is not a
matched-compute comparison. Counting remained the central failure: task K4 and K8 lost 18.40 and
19.20 points from the full model on CountBenchQA.

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
   discovery and verified zero decoded-pixel overlap with V2.
9. Completed interaction-aware K8 beam search and pairwise interaction mapping on development data.
10. Froze, committed, and ran four external conditions: full, generic K8, conditional K8, and
    conditional K4. All 5,000 predictions are retained off-instance.
11. Reviewed eight closely related pruning papers. Existing work includes generic VLM pruning and
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

### Phase 3 Interaction-Aware Search

Phase 3 started from the five existing K4 task routes, searched within task-specific 16-block
candidate pools, retained a beam of three routes, and expanded to K8 using 100 search examples per
capability. On the image-disjoint development validation remainder, target-capability drops were
3.19 points for attribute, 5.06 for counting, 14.93 for object, 23.53 for OCR, and 15.19 for
spatial. Only attribute met the intended 3-5 point preservation range. Pair effects were strongly
non-additive, including +9 points of excess harm for object blocks 21 and 23.

This rejected target-only beam search as a generally reliable K8 route constructor. The routes
were nevertheless frozen for the external test so the transfer claim could be evaluated without
post-hoc selection.

### Frozen External Evaluation

The full model and three pruned conditions were evaluated on 250 examples per capability from
TextVQA, CountBenchQA, CV-Bench/ADE20K, and AMBER. Routes and the manifest hash were committed in
`8ed9dc7` before inference.

| Condition | Overall accuracy | Drop from full | Attribute drop | Counting drop | Object drop | OCR drop | Spatial drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full | 74.96% | 0.00 pp | 0.00 pp | 0.00 pp | 0.00 pp | 0.00 pp | 0.00 pp |
| Generic K8 | 66.00% | 8.96 pp | -0.80 pp | 8.00 pp | 5.20 pp | 25.60 pp | 6.80 pp |
| Task K8 | 64.72% | 10.24 pp | 0.80 pp | 19.20 pp | 8.40 pp | 22.40 pp | 0.40 pp |
| Task K4 | 68.08% | 6.88 pp | 0.40 pp | 18.40 pp | 3.20 pp | 10.80 pp | 1.60 pp |

At matched K8 compute, task routing versus generic was -1.28 points overall, with a 95% interval
of [-3.60, 0.96]. Spatial was the only clear task-K8 win: +6.40 points [1.60, 11.20]. OCR was
+3.20 points [-2.40, 8.80], while counting was -11.20 points [-18.40, -4.00]. The external set
therefore supports capability-dependent differences but rejects a universal claim that the current
task-route selection method beats generic pruning.

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

1. Vision-block importance is not fully global. Development evidence favored object and OCR at K4,
   while external matched-K8 evidence clearly favored the spatial route. The best capability is
   therefore not stable across source family and pruning budget.
2. A one-block sensitivity heatmap is useful for screening but insufficient for route construction.
   Independent effects do not compose at eight or more removals.
3. Four-block pruning is scientifically informative but practically small. The vision tower gets
   roughly 6-7% faster, while the full VLM gets only 1-4% faster and loses only 2.10% of its total
   parameters.
4. Making the final hidden state closer in relative L2 distance does not guarantee the same answer.
   The removed blocks likely contribute nonlinear computation, or repair must occur at each removed
   boundary rather than only after block 31.
5. Target-only route optimization overfits. It can preserve the target search bucket while causing
   large source-transfer losses, especially for counting and object recognition.
6. The current evidence is a credible negative/diagnostic result, not yet a complete paper. It
   comes from one model, one GPU, identity removal only, and no physical compact checkpoint or
   edge-device test.

## Current Research Decision

Do not tune any route against the external outcomes and do not rerun the already completed
independent K8/K12/K16 or Phase 3 searches as if they were new evidence.

The next research stage should test whether the observed heterogeneity is reproducible rather than
searching harder on this test set:

1. Replicate the frozen intervention and evaluation protocol on SmolVLM2 to test architecture
   dependence.
2. Replace single target-only route selection with multi-fold, multi-source selection that
   penalizes worst-capability damage and unstable pair interactions; use development data only.
3. Evaluate any new method on a newly sealed set or through nested cross-validation, never on the
   external outcomes reported here.
4. If stable capability differences replicate, train a lightweight task router and compare it
   with generic pruning at matched block budget and fixed-clock latency.
5. Combine validated block routing with visual-token pruning or decoder compression only after the
   routing contribution is isolated.

## Evidence Boundaries

- The external set contains 250 examples per capability and has zero image overlap with V2 under
  decoded-pixel hashing. It has now been consumed and cannot be reused for method selection.
- The matched-K8 result establishes source-transfer behavior for these frozen Qwen routes only. It
  does not establish superiority across VLM architectures.
- Task K4 versus generic K8 is not a matched-compute comparison. No generic-K4 external condition
  was frozen, so the external data cannot support that missing comparison post hoc.
- No result establishes edge-device acceleration or causal localization of a capability to one
  block.
- Public benchmark contamination remains possible even after image-level deduplication.
