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
- `figure_data/`: exact aggregate input for the four-dataset EEG overview.
  The shared-target figure reads the primary simulation summaries directly;
  the archived Zhou2020 certificate metadata are in the empirical-reference
  manifest.

Subject labels are numeric identifiers from the cited public datasets.

## Integrity

`MANIFEST.csv` records each released file's relative path, byte size, and
SHA-256 digest. `SHA256SUMS` is compatible with standard checksum tools.

From the repository root:

```bash
python scripts/verify_results.py
```

The manifest and checksum files exclude themselves to avoid recursive hashes.
