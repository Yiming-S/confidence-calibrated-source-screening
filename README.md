# Confidence-Calibrated Source Screening

This repository provides the core analysis code and compact reproducibility
artifacts for confidence-calibrated, set-valued source screening in
cross-session motor-imagery EEG. The statistical problem is unusual because
all source discrepancies reuse the same empirical target, which induces
dependence across the candidate comparisons.

The associated manuscript is titled *Confidence-Calibrated Source Screening
for Cross-Session Motor-Imagery EEG under Distribution Shift*.

## What is included

- simultaneous rectangle, direct-contrast, and refinement screening code;
- primary simulations for shared-target calibration and target-limited regimes;
- covariance-space analyses for Ma2020, Stieger2021, Zhou2020, and
  BNCI2014_004;
- subject-level aggregate results used for non-inferiority analysis;
- compact data for the two reproducible overview figures;
- checksums and automated repository-integrity tests.

Raw EEG, trial-level signals, covariance caches, bootstrap draws, split-level
classifier outputs, manuscript source files, and journal-production files are not
included.

## Repository layout

```text
analysis/       Core statistical, simulation, and EEG analysis programs
data/           Official dataset records and local path conventions
results/        Compact aggregate results and integrity metadata
figures/        Three archived main-result figures
scripts/        Compact-result verification and figure reconstruction
tests/          Public-repository integrity tests
```

## Quick verification

Create an isolated Python environment and install the lightweight dependencies
needed for result verification and figure reconstruction:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-figures.txt
make verify
make figures
make test
```

`make figures` reconstructs the shared-target evidence figure into
`build/figures/` from
`results/simulations/` and the four-dataset EEG overview from
`results/figure_data/`. The Zhou2020 exclusion
certificate is an archived output: reconstructing its interval endpoints
requires the public EEG recordings and covariance calculations performed by
the full analysis pipeline.

## Full analysis

Install the complete environment with:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Every data-reading entry point accepts a local data or cache path. Inspect an
entry point before running it:

```bash
python analysis/ci_gate_ma2020_riemann.py --help
python analysis/ci_gate_stieger_riemann.py --help
python analysis/ci_gate_zhou2020_riemann.py --help
python analysis/ci_gate_bnci004_riemann.py --help
```

Dataset acquisition is described in [`data/README.md`](data/README.md). Full
runs write to the ignored `simulation_results/` directory unless another
output directory is supplied.

## Reproducibility boundary

The compact release supports three independently checkable operations:

1. verify every released result file against `results/SHA256SUMS`;
2. reproduce the two overview figures under `build/figures/` from aggregate
   figure data;
3. recompute the paired subject-level non-inferiority analysis from the
   released subject aggregates.

Full EEG preprocessing and bootstrap recalculation require separately obtained
public datasets and several gigabytes of local cache. No third-party dataset is
redistributed or relicensed here.

## Related work

The earlier applied study evaluated an operational confidence-interval overlap
gate inside cross-session transfer pipelines:

> Shen, Y., and Degras, D. (2026). A confidence-gated source selection strategy
> for cross-session transfer in brain-computer interfaces. *Frontiers in Human
> Neuroscience*, 20. <https://doi.org/10.3389/fnhum.2026.1895016>

That article and the present simultaneous-inference project are separate
research outputs.

## Citation

GitHub reads [`CITATION.cff`](CITATION.cff) directly. Repository URL, release
version, and archival DOI can be added when the first public release is made.

## Disclosure

AI-assisted tools were used for language editing and analysis-code development under the authors' direction; the authors designed the study, verified all results, and take full responsibility for the content.

## Licenses

Original source code is licensed under the MIT License in [`LICENSE`](LICENSE).
The released aggregate result files and archived figures are licensed under
CC BY 4.0 as specified in [`LICENSES/RESULTS.md`](LICENSES/RESULTS.md).
Third-party datasets remain governed by their original terms.
