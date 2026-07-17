# Source-Aware Robust Route Search Protocol

## Purpose and evidence boundary

This protocol searches for generic and capability-conditional vision-block routes that are robust
to dataset source at matched pruning budgets. It performs route selection only. There is no
fine-tuning, distillation, repair, prompt tuning, or scoring-rule tuning in any phase.

The source is `data/processed-v2/manifests/all.jsonl`. Its existing `development` rows are copied
unchanged to `prepared/search.jsonl`; its existing `test` rows are copied unchanged to
`prepared/selection.jsonl`. The script does not create a new random split. Complete image groups
remain in their original partition, and preparation must fail unless search/selection image
overlap is zero. The preserved test partition is 49.21% of examples and 48.57% of unique images;
these are observed source-split fractions, not a tunable resampling target.

Both partitions contain MME plus a second source for every capability. Across the two partitions
this retains roughly twice the small MME image support available from development alone. However,
both development and test rows were used in the earlier single-block discovery sweep. Selection
therefore measures combination-level method selection under an image-disjoint split; it is not a
sealed final test and cannot support an uncontaminated generalization claim.

The external held-out benchmark has already been viewed and is consumed. It is audit-only: its
predictions may describe frozen routes after selection, but may not select routes, finalists, K,
objectives, prompts, decoding, scoring, or optimizer settings.

## Frozen search design

`configs/robust_route_search.json` freezes the preparation seed, route budgets `K = [4, 6, 8]`,
identity-substitution operation, scout limits, a population of 16, three generations, two
development finalists per seed, three selection finalists, and optimizer seeds `20260715`,
`20260716`, and `20260717`. A route at one K is not automatically a candidate or winner at another
K.

Preparation creates source-balanced scouts from search only. `scout-generic.jsonl` selects 8
images independently within every capability+source stratum. Each
`scout-task-<capability>.jsonl` selects 6 images from each target-capability source and 1 image
from every non-target capability+source stratum. Selection uses seeded SHA-256 ordering while
solving exact stratum quotas, and every search row attached to a selected global image is retained.
Thus scout files preserve complete image groups, include collateral rows needed by the task loss,
and remain balanced at the frozen target and collateral quotas. Scouts reduce early search cost;
they do not replace full selection.

On the frozen manifest these quotas produce 133 generic rows. The attribute, counting, object, OCR,
and spatial task scouts contain 33, 32, 38, 32, and 32 rows respectively, or 167 task rows total.
Preparation also creates `development-evaluation.jsonl` from search. It retains up to 24 complete
images per capability+source stratum and all images when fewer than 24 exist. The frozen result is
300 rows and 185 images: 24 images from each large source and all 12–16 development MME images per
capability.

All drops are paired against full-model accuracy on identical examples. The generic scalar loss is
`0.50 * mean drop + 0.30 * worst source drop + 0.20 * source-drop standard deviation`. The task
scalar loss is `0.45 * target mean drop + 0.30 * target worst-source drop + 0.15 * collateral mean
drop + 0.10 * target source-drop standard deviation`. Means are source-balanced rather than
row-weighted, so MME and the larger capability-specific source receive equal weight within a
capability. Evolution uses Pareto selection over the frozen loss components; the scalar losses
rank development and selection finalists. Survivor ordering follows Pareto front, NSGA-II
crowding distance, frozen scalar score, and finally lexicographic route order.

Malformed or failed generations count as incorrect. Exact scalar ties use lexicographically sorted
block indices. Each seed contributes at most two development finalists. Those candidates are
deduplicated and ranked before at most three finalists per objective and K advance to full
selection.

### Formal objective definition

For route `S`, capability `c`, and source `d`, define the paired cell drop in percentage points as

```text
Delta[c,d](S) = 100 * (full_accuracy[c,d] - route_accuracy[c,d]).
```

Lower is better and a negative value means the route outperformed the full model in that cell. For
the shared route, `mu`, `w`, and `sigma` are the equal-cell mean, maximum, and population standard
deviation over all capability-source cells. Evolution minimizes the vector `(mu, w, sigma)` and
finalist ranking minimizes

```text
L_shared(S) = 0.50 * mu + 0.30 * w + 0.20 * sigma.
```

For target capability `t`, `mu_t`, `w_t`, and `sigma_t` use only target-source cells, while
`kappa_t` is the equal-cell mean over every non-target capability-source cell. Evolution minimizes
`(mu_t, w_t, kappa_t, sigma_t)` and finalist ranking minimizes

```text
L_t(S) = 0.45 * mu_t + 0.30 * w_t + 0.15 * kappa_t + 0.10 * sigma_t.
```

### Exact evolutionary operators

A chromosome is a sorted set of exactly K unique block indices. Initialization fills at most half
the population with sensitivity and prior-route candidates, then fills the remainder with seeded
random K-subsets. Survivor selection accepts nondominated Pareto fronts in order. When a front only
partly fits, ordering is descending NSGA-II crowding distance, ascending scalar loss, then
lexicographic route order.

Survivors are copied unchanged. For every offspring, fixed-K crossover retains the intersection of
two parents and takes a seeded subset from their remaining union until K blocks are present. A
mandatory one-swap mutation then removes one included block and adds one excluded allowed block.
Duplicate children are rejected; after repeated collisions, a seeded random K-subset fills the
vacancy. Parent choice and every stochastic-looking operation are deterministic from the frozen
seed. There is no mutation probability, crossover probability, learned router, or gradient update.
After survivors are ordered by scalar loss and route order, offspring attempt `i` pairs survivor
`i mod m` with survivor `(5i + 1) mod m`, where `m` is the survivor count.

The main Qwen run uses population 16 and three evaluated generations. The lean completed SmolVLM2
K4 replication uses population 12 and two evaluated generations; it retains the same objective
definitions, three seeds, and finalist pipeline. Full derivations and pseudocode are in
`paper/method.md`.

## Fair comparisons and selection

Every robust route must be compared on identical examples against controls with exactly the same
K. Required same-K controls are generic independent single-block ranking, capability-conditional
independent single-block ranking, the least-damaging contiguous route, and three deterministic
random routes. The unpruned full model is an accuracy reference, not a matched-compute route.

Evolution uses scout answers only. At each objective and K, deduplicate the two development
finalists from each seed and evaluate all six on `development-evaluation.jsonl`. Rank those routes
by the frozen scalar loss and advance at most three to the full selection partition. The winner
minimizes the matching frozen scalar loss on selection, with lexicographic block order breaking
exact ties. Results cannot change scout construction, optimizer settings, baseline definitions,
prompts, decoding, scoring, or scalar weights. Freeze one generic route and one route per
capability at each K before any audit.

## Phases and outputs

1. **Preparation.** Run `scripts/prepare_robust_route_search.py`. Output
   `prepared/search.jsonl`, `prepared/selection.jsonl`,
   `prepared/development-evaluation.jsonl`, `prepared/scout-generic.jsonl`, five
   `prepared/scout-task-<capability>.jsonl` files, and `prepared/summary.json` with counts, SHA-256
   hashes, source strata, and overlap/subset/group-preservation assertions.
2. **Reference and controls.** Cache full-model predictions and all required same-K baseline
   predictions on each relevant scout. Output immutable prediction JSONLs plus a manifest tying
   every condition to model revision, prompt, decoding, scoring, K, and route blocks.
3. **Scout search.** Run the frozen 16-member, three-generation optimizer independently for all
   three seeds, objectives, and K values using Pareto evolution. Output resumable candidate
   predictions, generation logs, and two development finalists per seed.
4. **Development finalists and selection.** Evaluate all six per-seed finalists on the 300-row
   development-evaluation manifest, then advance at most three finalists plus matched controls to
   the complete prepared selection partition. Output per-example predictions and source-balanced
   aggregate, per-capability, per-source, and capability+source metrics. The runner's stage 2 uses
   `development-evaluation.jsonl`, never all 904 search rows. No search result may break a selection
   tie.
5. **Freeze.** Apply the stated selection rule and write a frozen route registry containing one
   generic and five task routes for each K, input/config hashes, selected metrics, and rejected
   finalists. This registry is the only route input allowed in the audit phase.
6. **Audit and report.** Re-evaluate frozen routes on search for overfit diagnostics and, if useful,
   report the already-viewed external benchmark strictly as a consumed audit. Output matched same-K
   comparisons, search-selection gaps, source slices, uncertainty intervals, failures, and explicit
   claim limitations.

The approximate pre-cache route-prediction budget is:

```text
scout:    3 K * 3 seeds * 16 population * 3 evaluated generations * (133 + 167 rows)
          = 129,600
development finalists: 3 K * 6 route families * 6 finalists * 300 rows
          = 32,400
selection: 3 K * 6 route families * 3 finalists * 876 rows
          = 47,304
total:    approximately 209,304 new route predictions before cache reuse
```

Here the three-generation cap includes the initial evaluated population. The estimate covers route
candidates and finalists, not matched controls, full-model references, retries, or audit. At the
measured planning rate it is roughly 13 GPU-hours sequentially or about 7 hours wall time with two
family workers. Cache hits should reduce actual inference. These are approximate execution
estimates, not latency or deployment claims.

## Preparation command

The script resolves repository paths itself and does not require `PYTHONPATH`:

```bash
python3 scripts/prepare_robust_route_search.py
```

The default output directory is
`data/processed-v2/robust-route-search/prepared`. Re-running with unchanged input and config must
produce identical JSONL hashes. For an isolated validation run:

```bash
python3 scripts/prepare_robust_route_search.py --output-dir /tmp/robust-route-search-prepared
```

Inspect `/tmp/robust-route-search-prepared/summary.json`; every assertion must be `true`, and
`overlap.search_selection_images` must be `0`.

## Unsupported claims

This protocol cannot establish a globally optimal route, uncontaminated held-out performance,
model-family transfer, edge-device speedup, recovery through training, or causal localization of
a capability to selected blocks. Source identity also remains partially confounded with benchmark
construction, even though the objective balances sources within each capability.
