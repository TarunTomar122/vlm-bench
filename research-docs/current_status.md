# Final Research Status

Status date: 2026-07-16

## Decision

The experiment is complete for the first arXiv preprint. No additional GPU run is required for the
current claim. The repository now treats inference as frozen evidence and provides a CPU-only path
to regenerate the paper figures, tables, and website assets.

## Final Claim

Evolutionary search finds better vision-block combinations than independent, contiguous, or random
pruning across Qwen2.5-VL-3B and SmolVLM2-2.2B. Capability-specific routing is not universally
superior: gains depend on model, capability, pruning budget, and source.

## Terminology

- **Full model:** no vision blocks are skipped.
- **One shared K-block route:** one searched set of exactly K skipped blocks is used for every
  capability.
- **Capability-specific K-block policy:** a separate searched set of exactly K skipped blocks is
  used for each of the five capability labels.
- **Best found:** selected by the frozen evolutionary-search procedure, not proven globally optimal
  over every possible block combination.

Identity skipping measures which block combinations can be bypassed with limited behavioral
damage. It does not establish that a capability is stored inside a named block.

## Completed Evidence

### Qwen2.5-VL-3B

The primary model has 32 vision blocks. Source-aware route search and matched controls were completed
with four, six, and eight skipped blocks on 876 image-disjoint method-selection examples.

| Blocks skipped | Full model | One shared route | Capability-specific policy | Policy - shared | Paired 95% CI |
|---:|---:|---:|---:|---:|---:|
| 4 of 32 | 83.68% | 81.28% | 81.39% | +0.11 pp | [-1.83, 2.05] |
| 6 of 32 | 83.68% | 79.11% | 81.28% | +2.17 pp | [0.00, 4.34] |
| 8 of 32 | 83.68% | 75.91% | 76.60% | +0.68 pp | [-1.71, 3.08] |

The shared four-block route is 2.40 points below the full model. With six blocks skipped, the
capability-specific policy reaches 81.28%, matching the shared four-block route while skipping two
additional blocks. Its +2.17 point overall advantage over the shared six-block route is driven most
clearly by OCR at +7.10 points `[0.65, 14.19]`. Attribute, counting, object, and spatial intervals
cross zero. The earlier 1,250-example external Qwen set has been consumed and remains historical
transfer evidence; it must not be used for route selection.

### SmolVLM2-2.2B

The second model has 27 vision blocks. The completed replication skips four blocks only.

- Full model: 82.65%; one searched shared four-block route: 72.49%; capability-specific four-block
  policy: 73.29%.
- Capability policy minus shared route: +0.80 points `[-1.94, 3.54]`.
- The searched shared four-block route beats independent one-block ranking by +4.91 points
  `[1.83, 7.99]`.
- The searched capability policy beats independently constructed task routes by +8.90 points
  `[5.82, 11.99]`.
- Capability policy versus shared route by capability: attribute -0.60, counting +7.18, object
  -0.52, OCR -13.55, and spatial +9.39 points. Counting, OCR, and spatial intervals exclude zero.

The Smol six-block diagnostic was stopped after its shared route lost 21.80 points. It has no completed matched
controls or transfer results and is excluded from final figures.

### Fresh OCR Transfer

The Smol full model, shared four-block route, and OCR-specific four-block route were evaluated on
250 IIIT5K examples that were excluded from every selection step. Accuracies were 94.8%, 86.4%,
and 72.8%, respectively. The OCR route was -13.6 points below the shared four-block route
`[-19.2, -8.4]`. This negative result rejects a stable cross-source OCR-path interpretation.

## Interpretation

1. **Search is the result that generalizes.** Across both architectures, evolutionary search finds
   stronger same-budget combinations than naive construction from independent block scores.
2. **Capability routing is conditional.** It is useful for Qwen OCR with six blocks skipped and for
   Smol counting and spatial with four blocks skipped, but harmful for Smol OCR.
3. **Aggregate similarity can hide large capability changes.** The Smol capability policy is only
   +0.80 points overall because counting and spatial gains are offset by the OCR loss.
4. **Architecture changes the safe budget.** Qwen tolerates four skipped blocks with a 2.40 point
   drop, while Smol loses 10.16 points from its shared four-block route.
5. **The routes are intervention-specific.** They show that selected computations can be bypassed;
   they do not localize named information inside blocks.

### Efficiency

- Skipping four Qwen blocks bypasses 78,808,800 parameters: 11.79% of the vision tower and 2.10% of
  the full model.
- Skipping four Smol blocks bypasses 60,958,016 parameters: 14.76% of the vision tower and 2.71% of
  the full model.
- The Smol shared four-block route measured +8.60% vision-encoder and +4.19% end-to-end speedup at
  batch size one.

The cloud provider denied clock locking, so the Smol latency evidence is an unlocked same-VM RTX
4090 comparison. The Qwen latency audit predates the final evolved K4/K6/K8 routes; it supports
parameter accounting and historical K4 timing, not a final evolved-route latency curve.

## Earlier Diagnostics Retained

The repository retains the complete one-block screen, progressive independent routes, activation
rescue, low-rank feature-gap repair, interaction-aware beam search, locked-clock Qwen K4 audit, and
external Qwen evaluation. These explain why the final method uses source-balanced combinatorial
search. They are supporting/appendix material, not separate main claims.

## Submission Package

- `paper/`: claim boundaries, detailed outline, writing prompts, references, generated figures,
  generated tables, and arXiv checklist.
- `docs/`: GitHub Pages research website and regenerated publication assets.
- `research-docs/`: final status, dataset documentation, and frozen experimental protocols.
- `paper/data/paper-data.json`: compact machine-readable source for every final plot/table.
- `scripts/generate_paper_assets.py`: authoritative JSON-to-paper generator.
- `scripts/verify_submission.py`: checks budgets, key values, caveats, assets, and website copy.

Run:

```bash
python3 -m venv .paper-venv
.paper-venv/bin/pip install -r requirements-paper.txt
make PYTHON=.paper-venv/bin/python submission
```

## Evidence Boundaries

- The 876-example split is image-disjoint method-selection evidence, not a sealed public test.
- IIIT5K is sealed source-transfer evidence for Smol OCR only.
- No result localizes a named capability inside one block; identity skipping measures intervention
  sensitivity and block interactions.
- No result establishes mobile/edge latency, energy use, a physically compact checkpoint, or a
  learned dynamic router.
- Exact model revisions, routes, seeds, manifests, and hashes are committed. Raw images, weights,
  and prediction caches remain intentionally outside Git.
