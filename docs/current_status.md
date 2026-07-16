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

## Completed Evidence

### Qwen2.5-VL-3B

The primary model has 32 vision blocks. Source-aware route search and matched controls were completed
at K4, K6, and K8 on 876 image-disjoint method-selection examples.

| Budget | Full | Evolved generic | Evolved task | Task - generic | Paired 95% CI |
|---:|---:|---:|---:|---:|---:|
| K4 | 83.68% | 81.28% | 81.39% | +0.11 pp | [-1.83, 2.05] |
| K6 | 83.68% | 79.11% | 81.28% | +2.17 pp | [0.00, 4.34] |
| K8 | 83.68% | 75.91% | 76.60% | +0.68 pp | [-1.71, 3.08] |

The clearest capability result is Qwen K6 OCR at +7.10 points `[0.65, 14.19]`. K6 attribute,
counting, object, and spatial intervals cross zero. The earlier 1,250-example external Qwen set has
been consumed and remains historical transfer evidence; it must not be used for route selection.

### SmolVLM2-2.2B

The second model has 27 vision blocks. The completed replication budget is K4 only.

- Full: 82.65%; evolved generic: 72.49%; evolved task policy: 73.29%.
- Task minus generic: +0.80 points `[-1.94, 3.54]`.
- Evolved generic minus generic independent: +4.91 points `[1.83, 7.99]`.
- Evolved task minus task-independent: +8.90 points `[5.82, 11.99]`.
- Task versus generic by capability: attribute -0.60, counting +7.18, object -0.52, OCR -13.55,
  and spatial +9.39 points. Counting, OCR, and spatial intervals exclude zero.

Smol K6 was stopped after a diagnostic generic route lost 21.80 points. It has no completed matched
controls or transfer results and is excluded from final figures.

### Fresh OCR Transfer

The Smol full, generic K4, and OCR-specific K4 routes were evaluated on 250 IIIT5K examples that
were excluded from every selection step. Accuracies were 94.8%, 86.4%, and 72.8%, respectively.
The OCR route was -13.6 points below generic K4 `[-19.2, -8.4]`. This negative result rejects a
stable cross-source OCR-path interpretation.

### Efficiency

- Qwen K4 skips 78,808,800 parameters: 11.79% of the vision tower and 2.10% of the full model.
- Smol K4 skips 60,958,016 parameters: 14.76% of the vision tower and 2.71% of the full model.
- Smol generic K4 measured +8.60% vision-encoder and +4.19% end-to-end speedup at batch size one.

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
- `site/`: static research website and regenerated publication assets.
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
