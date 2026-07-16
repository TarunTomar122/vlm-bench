.PHONY: paper-assets verify-paper submission clean-paper-assets

PYTHON ?= python3

paper-assets:
	$(PYTHON) scripts/generate_paper_assets.py

verify-paper:
	$(PYTHON) scripts/verify_submission.py

submission: paper-assets verify-paper

clean-paper-assets:
	rm -f paper/data/paper-data.json paper/tables/generated-* paper/figures/generated-*
