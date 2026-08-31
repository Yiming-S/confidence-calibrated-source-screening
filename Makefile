.PHONY: manifest verify figures syntax test smoke

manifest:
	python3 scripts/build_manifest.py

verify:
	python3 scripts/verify_results.py

figures:
	MPLBACKEND=Agg python3 scripts/rebuild_figures.py

syntax:
	python3 -m compileall -q analysis scripts tests

test:
	python3 -m unittest discover -s tests -v

smoke: verify figures syntax test
