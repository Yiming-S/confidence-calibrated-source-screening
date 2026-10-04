# Analysis entry points

## Primary simulations

- `ci_gate_protocol_sim.py`: canonical shared-target calibration experiment.
- `ci_gate_target_limited_sim.py`: target-limited regimes.
- `ci_gate_straddle_undercoverage_sim.py`: shared-target versus
  independent-target contrast coverage.
- `ci_gate_extended_baselines_sim.py`: comparator procedures, including the
  MCS-style stepdown analogue.

## Public EEG analyses

- `ci_gate_ma2020_riemann.py`
- `ci_gate_stieger_riemann.py`
- `kumar2024/preprocess.py`
- `kumar2024/analyze.py`
- `kumar2024/supplemental.py`
- `ci_gate_bnci004_riemann.py`
- `ci_gate_rolling_targets.py`
- `ci_gate_block_bootstrap_sensitivity.py`
- `ci_gate_eeg_empirical_audit.py`
- `ci_gate_noninferiority.py`

`ci_gate_realdata_case_study.py` and `ci_gate_target_only_reference.py` also
provide shared functions imported by the primary entry points.
`screening_core.py` contains the data-agnostic simultaneous-bound and
set-construction operations covered by deterministic unit tests.

`ci_gate_target_only_reference.py` calls the same `train_eval_ts` function as
the source-trained pipeline: LSQR LDA with automatic shrinkage, covariance
shrinkage 0.05, and a training-only Riemannian reference. It trains only on
the labelled target screening half. The public verifier checks its released
subject summaries without refitting:

```bash
python scripts/verify_target_only_reference.py
```

Legacy EEG output fields named `pair_ref` or `Pair/Ref` refer to the refinement
intersection. They do not assert that the direct-contrast and refinement sets
are always identical; the field names are retained for file compatibility.

### Empirical-reference reporting

The current manuscript reports the Kumar2024 and BNCI2014_004 empirical-reference
audits. Kumar2024 uses its frozen analysis backend; the general
`ci_gate_eeg_empirical_audit.py` supplies the common audit functions. For the
released subject means, use:

```bash
python scripts/verify_empirical_coverage.py
```

This check reconstructs the two coverage-summary rows and validates
subject sets, recorded outer counts, and nominal levels. It does not recreate
the omitted raw records or the table's Jaccard column. This command does not rerun
an experiment or changes an input CSV.

### Kumar2024 exploratory analysis

`kumar2024/` contains the event-based GDF preprocessing, primary analysis,
supplementary calculations, and full-run verification programs. The recorded
protocol and data-availability amendment preserve the one-second duration rule,
18-subject population, six-day source/target assignment, and random seeds.
The supplementary plan identifies calculations specified after the primary
results. The frozen backend is retained to reproduce the numerical implementation
used for those results; it is not a second set of current public-summary checks.

Inspect these entry points before supplying local recordings and cache paths:

```bash
python analysis/kumar2024/preprocess.py --help
python analysis/kumar2024/analyze.py --help
python analysis/kumar2024/supplemental.py --help
```

The `verify_results.py` and `verify_supplemental.py` programs in that directory
audit full local run outputs. They require files omitted from this compact
release. The lightweight checks in `scripts/` instead validate the released
aggregates and recorded confidence bounds.

## Additional comparisons and computation checks

- `ci_gate_bspc_mcs_comparison.py`: shared-draw downstream comparison on all
  25 Ma2020 and 62 Stieger2021 subjects, with 30 half-splits and 199 bootstrap
  draws per split. It includes the implemented MCS-style source-screening
  analogue, not the complete classical MCS procedure.
- `ci_gate_bspc_bootstrap_budget.py`: fixed-sample stability at 99, 199, 499
  and 999 draws. Smaller budgets are prefixes of the same 999-draw matrix.
- `ci_gate_bspc_runtime.py`: serial timing from cached covariance matrices,
  including all screening operations separately for each method. Do not run
  this benchmark concurrently with other computational jobs.
- `verify_bspc_revision.py`: audit of full local reruns, including split
  assignments, cache-derived training counts and input hashes. It requires
  the full local run directory and its covariance caches; the compact public
  summaries alone are not sufficient for this audit.

Examples from the repository root, after installing `requirements.txt`:

```bash
python analysis/ci_gate_bspc_mcs_comparison.py --help
python analysis/ci_gate_bspc_bootstrap_budget.py --help
python analysis/ci_gate_bspc_runtime.py --help
python analysis/verify_bspc_revision.py --help
```

Supply local caches with the data/cache options shown by each entry point.
Default output directories are inside ignored `simulation_results/`; use a
fresh output directory for a new run. The experiments record the actual
software versions, seeds, sample assignments and input hashes. The released
October 2026 summaries were generated with Python 3.13.9; the repository's
verification workflow also targets Python 3.10. Timing is machine-dependent.

For checks that require no EEG or cache files, run:

```bash
python scripts/verify_revision_results.py
python scripts/make_bspc_revision_tables.py
```

The first command reconstructs aggregations from released subject summaries;
it does not regenerate predictions, bootstrap samples or raw timing records.
The second writes the shared-draw method, paired-comparison, and runtime tables
to `build/tables/` without changing the inputs. The current EEG, non-inferiority,
coverage, block-sensitivity, and bootstrap-budget tables are generated by
`scripts/make_current_publication_artifacts.py` from the current four cohorts.

Run any entry point with `--help` before a full analysis. Data-reading scripts
require separately obtained public EEG recordings or previously built caches.
