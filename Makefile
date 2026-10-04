.PHONY: manifest verify figures tables syntax test smoke

PYTHON ?= python3

manifest:
	$(PYTHON) scripts/build_manifest.py

verify:
	$(PYTHON) scripts/verify_results.py
	$(PYTHON) scripts/verify_revision_results.py

figures:
	MPLBACKEND=Agg $(PYTHON) scripts/rebuild_figures.py

tables:
	$(PYTHON) scripts/make_bspc_revision_tables.py
	$(PYTHON) scripts/make_current_publication_artifacts.py

syntax:
	$(PYTHON) -m compileall -q analysis scripts tests

test:
	$(PYTHON) -m unittest discover -s tests -v
	cd analysis/kumar2024 && "$(abspath $(shell command -v $(PYTHON)))" -m unittest discover -s . -p 'test_*.py' -v

smoke: verify figures tables syntax test
