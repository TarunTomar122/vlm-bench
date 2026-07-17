# Document the Evolutionary Search Exactly

## Intended commit subject

`Document evolutionary search exactly`

## Problem and decision

The README and public website named population size, Pareto selection, mutation, and crossover, but
did not define the route chromosome, cell-level accuracy drop, objective equations, exact genetic
operators, or finalist pipeline. The paper package had only prompts telling a future writer to add
those details. This made the central method unnecessarily hard to understand and also described the
Qwen optimizer settings as though they applied unchanged to the lean SmolVLM2 replication.

The decision is to document the implementation directly from `src/vlm_bench/robust_search.py`,
`scripts/run_robust_route_search.py`, and the frozen model-specific configurations. Public text now
calls the method a fixed-cardinality, multi-objective genetic search and distinguishes the Qwen 16
route by 3 generation run from the SmolVLM2 12 route by 2 generation run.

## Files and behavior changed

- `README.md`: added the chromosome definition, paired drop equation, shared and capability loss
  equations, six-step algorithm, model-specific budgets, and a concrete K4 crossover example.
- `paper/method.md`: added a paper-ready formal method with complete notation, Pareto dominance,
  deterministic parent scheduling, fixed-K crossover, mandatory one-swap mutation, pseudocode,
  frozen hyperparameters, and evidence boundaries.
- `paper/README.md`: made the formal method part of the required manuscript workflow.
- `paper/outline.md`: requires exact objective vectors, losses, operators, and model-specific search
  budgets in the method section.
- `paper/writing-guide.md`: points manuscript drafting to the implemented notation and prevents a
  generic optimizer description from replacing exact details.
- `research-docs/robust_route_search_protocol.md`: added formal loss definitions and the exact
  reproduction operators to the frozen protocol.
- `docs/index.html`: expanded the website method section with formulas, generation steps, a worked
  K4 example, model-specific settings, and a clear statement of what the algorithm is not.
- `docs/styles.css`: allows long method equations to use the available paper width while retaining
  horizontal overflow behavior on small screens.

No inference, route search, result aggregation, figure data, or model behavior changed.

## Alternatives considered

- Leave formulas only in the protocol. Rejected because the README, website, and paper package are
  the entry points most readers use.
- Describe only the scalar loss. Rejected because evolutionary survival uses multi-objective Pareto
  vectors; omitting that distinction would misrepresent the implementation.
- Call the method NSGA-II without qualification. Rejected because survival uses NSGA-II-style
  fronts and crowding, but reproduction and finalist ranking are repository-specific.
- Add a learned router. Rejected because that is a future experiment, not part of the completed
  search and would confuse a documentation correction with new research.

## Verification evidence

- `PYTHONPATH=src python3 -m unittest tests.test_robust_search`
  - Passed all 13 route normalization, objective, crossover, mutation, Pareto, and stability tests.
- `MPLCONFIGDIR=/tmp/vlm-mpl make PYTHON=/tmp/vlm-paper-venv/bin/python submission`
  - Regenerated seven figure sets and paper tables.
  - Submission verification passed for frozen results, 21 figure files, tables, research docs, and
    the GitHub Pages site.
- `git --git-dir=.git-data --work-tree=. diff --check`
  - Passed.
- Parsed `docs/index.html` with Python `html.parser`.
  - Passed.
- Searched `README.md` and `docs/` for en dash and em dash characters.
  - No matches.

## Limitations and unsupported claims

- This documentation does not establish that the frozen weights are optimal.
- The search is finite and seeded; it does not enumerate all `L choose K` routes.
- The capability-specific policy requires a known capability label and is not a learned per-input
  router.
- The objective weights remain protocol choices rather than theoretically optimal coefficients.
- This change supplies no new accuracy, latency, transfer, or edge-device evidence.
- Identity skipping still does not localize a capability to individual blocks.

## Next action enabled

The manuscript can now use a consistent, code-backed method section, and readers can audit precisely
how route candidates are scored, evolved, selected, and frozen before interpreting the results.
