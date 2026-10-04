# Confidence-Calibrated Source Screening

This repository provides the core analysis code and compact reproducibility
artifacts for confidence-calibrated, set-valued source screening in
cross-session motor-imagery EEG. All source discrepancies reuse the same
empirical target, which induces dependence across the candidate comparisons.

The associated manuscript is titled *Confidence-Calibrated Source Screening
for Cross-Session Motor-Imagery EEG under Distribution Shift*.

## What is included

- simultaneous rectangle, direct-contrast, and refinement screening code;
- primary simulations for shared-target calibration and target-limited regimes;
- covariance-space analyses for Ma2020, Stieger2021, Kumar2024, and
  BNCI2014_004;
- subject-level aggregate results used for non-inferiority analysis;
- shared-draw MCS-style comparisons, finite-bootstrap stability checks,
  and a serial fitting-cost benchmark;
- compact data for the two overview figures and the Kumar2024 exclusion example;
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
.venv/bin/python -m pip install -r requirements-verification.txt
make PYTHON=.venv/bin/python verify
make PYTHON=.venv/bin/python figures
make PYTHON=.venv/bin/python tables
make PYTHON=.venv/bin/python test
```

`make figures` reconstructs the shared-target evidence figure into
`build/figures/` from
`results/simulations/` and the four-dataset EEG overview from
`results/figure_data/`. The Kumar2024 exclusion figure is reconstructed from
the released interval endpoints and exclusion comparisons. Recalculating
those endpoints requires the public EEG recordings and full bootstrap analysis.

`make tables` reconstructs the current EEG and additional-comparison tables
under `build/tables/` from the released summaries. `make verify` checks file
integrity, subject-level aggregates, paired intervals, and current-cohort outputs.
The numerical regression tests use small synthetic covariance arrays and do
not require EEG downloads. If `make` is unavailable, run the Python commands
listed in the [Makefile](Makefile) with the same environment.

On 2026-10-03, 61 repository tests, 33 Kumar pipeline tests, seven timing
self-checks, all aggregate checks, and all figure/table generators passed.
The aggregate checks, figure/table generators, and all 94 tests also passed from
an independent code directory using the existing Python environment. Ten
generated tables matched the manuscript.
These checks use released aggregates and synthetic inputs; they do not rerun
EEG preprocessing or classification. A previous fresh install on
macOS 27.0.1 ARM64 failed while loading the pinned SciPy 1.15.3 PROPACK binary
under both Python 3.12 and 3.13, before project code ran. Fresh-install
verification on that platform remains unresolved. The hosted workflow targets
Linux; local test success alone does not establish its result.

## Full analysis

Install the complete environment with:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Every data-reading entry point accepts a local data or cache path. Inspect an
entry point before running it:

```bash
.venv/bin/python analysis/ci_gate_ma2020_riemann.py --help
.venv/bin/python analysis/ci_gate_stieger_riemann.py --help
.venv/bin/python analysis/kumar2024/preprocess.py --help
.venv/bin/python analysis/kumar2024/analyze.py --help
.venv/bin/python analysis/ci_gate_bnci004_riemann.py --help
```

Dataset acquisition is described in [`data/README.md`](data/README.md). Full
runs write to the ignored `simulation_results/` directory unless another
output directory is supplied.

## Reproducibility boundary

The compact release supports these independently checkable operations:

1. verify every released result file against `results/SHA256SUMS`;
2. reproduce the two overview figures and exclusion figure under
   `build/figures/` from released aggregates and confidence bounds;
3. recompute the paired subject-level non-inferiority analysis from the
   released subject aggregates;
4. recompute the additional MCS-style comparison means and paired intervals,
   and the budget and timing summaries, from their released subject summaries;
5. regenerate the current EEG and additional-comparison tables under `build/tables/`;
6. reconstruct both empirical-reference coverage summaries from 27 released
   subject-level rows and check the recorded sample counts and nominal levels;
7. reconstruct target-only comparisons and target-dependent summaries,
   including rolling targets, from released subject means.

## Cohort and protocol history

The current manuscript uses Ma2020, Stieger2021, Kumar2024, and BNCI2014_004.
Kumar2024 was selected after preliminary results from other cohorts had been
examined. It is an exploratory replacement cohort, not an independent
prospectively selected confirmation. Its recorded protocol, data-availability
amendment, and later supplementary plan are included with the analysis code.
The current release excludes superseded cohort results; earlier analyses remain
in the research archive and, where previously committed, the Git history.

## Classifier-matched target reference

Target-only and source-trained classifiers use the same tangent-space LSQR LDA
with automatic shrinkage and trial-covariance shrinkage 0.05. Each tangent
reference uses only that classifier's training data. Target-only training uses
labels from the screening half; source screening does not use target labels.
The compact summaries contain the corrected reference, not the earlier SVD
classifier results. Verification checks the recorded configuration and
aggregates; reproducing predictions requires the public EEG and full analysis.

Full EEG preprocessing and bootstrap recalculation require separately obtained
public datasets and several gigabytes of local cache. No third-party dataset is
redistributed or relicensed here.

## Empirical-reference coverage

The audit resamples each cohort's full-session empirical distributions, using
the original source sample sizes, half-sized target samples, and 499 shared-target
bootstrap draws. Component and contrast coverage each target 97.5%; their split
joint event targets 95%.

| Cohort | Component | Contrast | Joint | Empirical-oracle retention |
| --- | ---: | ---: | ---: | ---: |
| Kumar2024: 18 subjects, 50 outer samples each | 0.957 | 0.920 | 0.906 | 1.000 |
| BNCI2014_004: 9 subjects, 200 outer samples each | 0.979 | 0.963 | 0.957 | 1.000 |

Kumar2024 does not reach the nominal simultaneous coverage levels in this audit.
Its empirical-oracle retention is a distinct result and does not establish those
coverage levels. These estimates describe empirical-reference resampling, not
coverage under an independently known EEG population. The compact check
reconstructs subject-mean aggregations; it does not recheck the omitted outer
records or bootstrap draws. Details and commands are in
[`results/README.md`](results/README.md).

## Additional comparisons and computation checks

The summaries in `results/eeg/bspc_revision_20261002/` and
`results/eeg/kumar2024/supplemental/` describe these separate analyses:

- Ma2020 (25 subjects) and Stieger2021 (62 subjects), with 30 target half-splits
  each, comparing refinement, an MCS-style source-screening analogue, Top-m,
  a range-midpoint threshold, and all-source pooling. Refinement and MCS-style
  screening share the same bootstrap draws within each split. Paired intervals
  use subjects as the statistical unit and do not replace the primary
  non-inferiority analysis.
- A check of 99, 199, 499 and 999 bootstrap draws, using the first three subjects
  from Ma2020, Stieger2021, and Kumar2024, two target splits, and three seeds.
  The Kumar2024 extension was specified after its primary results. This measures
  finite-budget numerical sensitivity, not population coverage.
- A six-subject serial timing benchmark with three repetitions per method.
  It begins with cached trial covariances and excludes raw EEG preprocessing
  and cache loading. Fitting time decreased after screening, but total
  screening--fitting--prediction time did not decrease in this benchmark.

The accuracy intervals do not establish superiority over the MCS-style
comparator. The compact files support checks of the reported aggregations,
not reconstruction of omitted predictions, bootstrap draws or raw timings.
Entry points and rerun boundaries are documented in
[`analysis/README.md`](analysis/README.md).

## Related work

The earlier applied study evaluated an operational confidence-interval overlap
gate inside cross-session transfer pipelines:

> Shen, Y., and Degras, D. (2026). A confidence-gated source selection strategy
> for cross-session transfer in brain-computer interfaces. *Frontiers in Human
> Neuroscience*, 20. <https://doi.org/10.3389/fnhum.2026.1895016>

That article and the present simultaneous-inference project are separate
research outputs.

## Citation

GitHub reads [`CITATION.cff`](CITATION.cff) directly. Add a version and persistent
archive identifier when a fixed archival release is created; a repository URL
is not an archival DOI.

## Disclosure

AI-assisted tools were used for language editing and analysis-code development under the authors' direction; the authors designed the study, verified all results, and take full responsibility for the content.

Codex (OpenAI) and Claude (Anthropic) were used for language editing and analysis-code development.

## Licenses

Original source code is licensed under the MIT License in [`LICENSE`](LICENSE).
The released aggregate result files and archived figures are licensed under
CC BY 4.0 as specified in [`LICENSES/RESULTS.md`](LICENSES/RESULTS.md).
Third-party datasets remain governed by their original terms.
