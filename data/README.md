# Public EEG datasets

Raw EEG is not distributed in this repository. Obtain each dataset from its
official public record and comply with the provider's terms.

| Dataset | Source publication | Public data record |
|---|---|---|
| Ma2020 | <https://doi.org/10.1038/s41597-020-0535-2> | Harvard Dataverse `DVN/RBN3XG`: <https://doi.org/10.7910/DVN/RBN3XG> |
| Stieger2021 | <https://doi.org/10.1038/s41597-021-00883-1> | figshare record `13123148`, version 1: <https://doi.org/10.6084/m9.figshare.13123148.v1> |
| Zhou2020 | <https://doi.org/10.3389/fnhum.2021.701091> | IEEE DataPort: <https://doi.org/10.21227/f1c7-7x89> |
| BNCI2014_004 | <https://doi.org/10.1109/TNSRE.2007.906956> | BNCI Horizon 2020 accession `004-2014`: <https://bnci-horizon-2020.eu/database/data-sets> |

The repository identifies the direct public records but does not assert byte
identity between unrecorded local mirrors and a particular repository version.
For an archival rerun, record the acquisition date and checksums of the local
input files.

## Local path convention

Keep recordings and caches outside the Git repository. A data-reading script
resolves its input in this order:

1. an explicit command-line path;
2. the `EEG_DATA_ROOT` environment variable;
3. a clear command-line error.

Example:

```bash
export EEG_DATA_ROOT=/path/to/eeg-data
python analysis/ci_gate_zhou2020_riemann.py \
  --data-root "$EEG_DATA_ROOT" \
  --cache-root /path/to/cache/zhou2020 \
  --skip-download \
  --out-dir simulation_results/zhou2020
```

Do not commit raw recordings, epochs, covariance arrays, feature caches, or
download-manager directories.
