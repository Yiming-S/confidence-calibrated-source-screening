# Compact result release

This directory contains aggregate simulation results, public-dataset subject
summaries, non-inferiority inputs, figure data, and path-free run metadata. It
contains no signal samples, trial-level features, covariance arrays, bootstrap
draws, or split-level classifier outputs.

## Contents

- `simulations/`: primary calibration, target-limited, straddling, and
  comparator summaries.
- `eeg/`: four-dataset screening and downstream aggregates, rolling-origin
  summaries, empirical-reference checks, and non-inferiority outputs.
- `eeg/kumar2024/`: all 18 subjects' primary summaries and compact supplementary
  outputs for the exploratory cohort, including exclusion-certificate bounds,
  block sensitivity, and finite-bootstrap checks.
- `eeg/target_only_reference/`: 87 latest-session target-only subject means,
  10 comparisons, and path-free correction metadata for all four analysis
  groups. All target references use the source classifier's LSQR/automatic-
  shrinkage pipeline. The metadata distinguish actual imported package versions
  from installed-distribution metadata.
- `eeg/bspc_revision_20261002/`: shared-draw MCS-style comparisons, fixed
  bootstrap-budget checks, serial timing summaries, and a portable protocol
  manifest. These are separate analyses, not replacements for the original
  primary non-inferiority inputs.
- `figure_data/`: exact aggregate input for the four-dataset EEG overview.
  The shared-target figure reads the primary simulation summaries directly;
  the Kumar2024 exclusion example reads the numerical bounds in
  `eeg/kumar2024/supplemental/certificate.json`.

Subject labels are numeric identifiers from the cited public datasets.

## Integrity

`MANIFEST.csv` records each released file's relative path, byte size, and
SHA-256 digest. `SHA256SUMS` is compatible with standard checksum tools.

From the repository root:

```bash
python scripts/verify_results.py
python scripts/verify_revision_results.py
python scripts/verify_empirical_coverage.py
python scripts/verify_target_only_reference.py
```

The manifest and checksum files exclude themselves to avoid recursive hashes.

The additional-analysis directory contains 18 compact files: method and
subject summaries, paired intervals for accuracy/session count/training-trial
count, budget and seed-stability summaries, standard-error diagnostics, timing
summaries and a protocol manifest. It omits individual target-trial predictions,
bootstrap matrices, covariance caches and raw timing repetitions. Aggregation
checks use subjects as the independent units; they do not recreate the omitted
data or independently validate classification predictions.

## Empirical-reference coverage

`eeg/empirical_reference/coverage_summary.csv` contains two cohort-level rows
reconstructed from 18 Kumar2024 and 9 BNCI2014_004 subject summaries.
The manifest records 50 and 200 outer samples per subject, respectively,
499 bootstrap draws, and nominal component/contrast/joint coverage levels of
0.975/0.975/0.95. The summaries report observed coverage of
0.957/0.920/0.906 for Kumar2024 and 0.979/0.963/0.957 for BNCI2014_004.
Empirical-oracle retention is 1.000 in both cohorts; retention does not
replace a simultaneous-coverage check.

The verifier checks both complete subject sets, compatibility with the recorded
outer counts, both cohort rows, nominal levels, and aggregation of coverage,
oracle retention, exact recovery, and set size. It does not reconstruct the
2,700 omitted outer records, bootstrap draws, or pairwise retained-set Jaccard
values. Full local raw-record validation is available through the analysis
entry point, as described in [`analysis/README.md`](../analysis/README.md).
The certificate figure uses a full target sample and 1,999 bootstrap draws;
it is not the half-target, 499-draw configuration of this coverage audit.

## Target-only correction

Only target-only accuracy and quantities compared with that reference were
recomputed. Historical-source screening, source-trained predictions, and
Ref.-versus-all-source non-inferiority inputs are unchanged. The old SVD results
are archived outside this compact release and are not mixed with the corrected
classifier results.

The compact verifier recomputes the latest-session comparisons and checks
the current subject-method rows, method-summary rows, and two rolling manuscript
summary rows. It checks target means, differences, and below-target rates; it
does not refit classifiers or reconstruct the omitted target splits.
