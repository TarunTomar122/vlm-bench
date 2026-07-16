# Package arXiv Submission Assets

## Intended Commit Subject

`Package arXiv submission assets`

## Problem or Decision

The GPU experiment is complete. Continuing to add methods would blur the strongest defensible
finding: evolutionary route search improves vision-block selection across two VLMs, while
capability-specific routes are model-, budget-, and source-dependent. The repository needed to
shift from an experiment workspace to an auditable manuscript and public-project package without
inventing missing SmolVLM K6/K8 or final Qwen route-latency measurements.

## Files and Behavior Changed

- Replaced the top-level `README.md` and `docs/current_status.md` with the frozen two-model result,
  exact evidence boundaries, model revisions, final metrics, and a CPU-only reproduction path.
- Added `paper/` with the manuscript outline, paragraph-level writing prompts, claim/limitation
  guardrails, related-work map, 19-entry BibTeX file, submission checklist, generated data manifest,
  tables, and seven figure sets in PNG/PDF/SVG.
- Added `scripts/generate_paper_assets.py`, which reads authoritative committed Qwen/Smol result and
  route JSON, asserts key frozen values, and regenerates paper tables, plots, and website assets.
- Added `scripts/verify_submission.py`, which regenerates assets and checks model budgets, key
  transfer values, latency caveats, required formats, and website copy.
- Added `Makefile`, `requirements-paper.txt`, and `CITATION.cff` for a one-command, CPU-only paper
  workflow and citation metadata.
- Added a responsive, dependency-free static project website under `site/`, suitable as a direct
  static page or as the visual/content source for a Wix implementation.
- Removed tracked dissertation-scraping data, the unrelated ML-engineer list, and three abandoned
  VLM project-idea microsites. The active pruning brief/protocol/literature map and all historical
  experiment evidence remain.
- Added `.paper-venv/` to `.gitignore`.

## Alternatives Considered

- Running the missing final Qwen K4/K6/K8 latency audit was rejected because the experiment was
  explicitly called complete and the GPU is down. The package states this limitation instead.
- Plotting SmolVLM at K6/K8 was rejected because those matched-budget experiments do not exist.
- Writing a full paper draft was rejected because the requested handoff is everything around the
  manuscript; the author wants to write the prose. The outline and prompts make that work direct.
- Keeping every historical ideation/dissertation artifact was rejected because it obscured the
  submission repository. Core pruning research notes and all reproducible evidence were retained.
- Depending on a JavaScript plotting runtime was rejected in favor of static publication files and
  a small Matplotlib-only CPU dependency.

## Verification Evidence

- `/tmp/vlm-paper-venv/bin/python scripts/verify_submission.py` regenerated all assets and passed:
  frozen budgets, 21 figure files, tables, paper docs, and site checks.
- `PYTHONPATH=src /tmp/vlm-paper-venv/bin/pytest -q --ignore=tests/test_phase2.py --ignore=tests/test_phase3.py --ignore=tests/test_heldout_builder.py`
  passed 20 CPU-compatible tests.
- All seven PNG figures were inspected after generation; four initial layout defects were corrected
  and re-inspected without changing data.
- The website was rendered with headless Chromium at 1440x1000 and 390x844 and inspected visually.
- Python's HTML parser accepted `site/index.html`, and every local figure/style reference resolved.
- `python3 -m py_compile scripts/generate_paper_assets.py scripts/verify_submission.py` passed.

## Known Limitations and Unsupported Claims

- The test environment uses Python 3.14 and has no compatible installed Torch/PyArrow stack, so
  GPU/Phase 2/Phase 3/held-out-builder tests were not rerun locally. They are unchanged by this
  commit; 20 dependency-light tests and the complete submission verifier passed.
- SmolVLM2 has completed K4 evidence only. Its stopped K6 diagnostic is not a final result.
- Smol latency is an unlocked same-VM RTX 4090 comparison, not fixed-clock or edge-device evidence.
- Qwen has no equivalent final evolved-route latency series across K4/K6/K8, so the package does not
  claim an accuracy-latency Pareto frontier.
- Identity skipping reduces executed depth but does not itself produce a smaller serialized model.
- The author still must add final identity/affiliation metadata, choose licenses, obtain external
  technical review, write/compile the manuscript, and replace the website paper placeholder.

## Next Action Enabled

The author can now inspect a fixed result package, draft each paper section from the supplied
outline/prompts, obtain one external technical review, compile the LaTeX manuscript, publish the
static/Wix project page, and submit an arXiv v1 without restarting the GPU instance.
