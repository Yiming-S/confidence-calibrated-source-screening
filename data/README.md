# Public EEG datasets

Raw EEG is not distributed in this repository. Obtain each dataset from its
official public record and comply with the provider's terms.

| Dataset | Source publication | Public data record |
|---|---|---|
| Ma2020 | <https://doi.org/10.1038/s41597-020-0535-2> | Harvard Dataverse `DVN/RBN3XG`: <https://doi.org/10.7910/DVN/RBN3XG> |
| Stieger2021 | <https://doi.org/10.1038/s41597-021-00883-1> | figshare record `13123148`, version 1: <https://doi.org/10.6084/m9.figshare.13123148.v1> |
| Kumar2024 | <https://doi.org/10.1093/pnasnexus/pgae076> | Zenodo record `10694880`, version 1: <https://doi.org/10.5281/zenodo.10694880> |
| BNCI2014_004 | <https://doi.org/10.1109/TNSRE.2007.906956> | BNCI Horizon 2020 accession `004-2014`: <https://bnci-horizon-2020.eu/database/data-sets> |

The Kumar2024 source archive is `Online_Offline_Race.zip`, with the
provider's MD5 checksum `89f4befec61d2bb82d09c1b6b2f43f01`. The analysis uses
bar-task runs from all 18 subjects and six recording days; racing runs are
excluded. It retains task segments lasting at least one second. The recorded
protocol and amendment identify the resulting trial population and the
exploratory status of this cohort.

For other unrecorded local mirrors, the repository does not establish byte
identity with a particular provider version. Record acquisition dates and
input checksums for each new rerun.

## Local path convention

Keep recordings and caches outside the Git repository. The Ma2020, Stieger2021,
and BNCI2014_004 entry points resolve their data root in this order:

1. an explicit command-line path;
2. the `EEG_DATA_ROOT` environment variable;
3. a clear command-line error.

For example, set the BNCI recording directory explicitly:

```bash
python analysis/ci_gate_bnci004_riemann.py \
  --data-root /path/to/004-2014 \
  --cache-root /path/to/cache/bnci2014_004 \
  --out-dir simulation_results/bnci2014_004
```

The frozen Kumar2024 entry points require explicit paths. Preprocessing uses
`--raw-root`, `--cache-root`, `--qa-dir`, and `--protocol`; analysis uses
`--cache-root`, `--out-dir`, `--protocol`, and `--simulations-root`.
Use `analysis/kumar2024/backend_snapshot` as the frozen simulations backend and
`analysis/kumar2024/protocol_minclass4.json` as the primary protocol. The
supplementary command also requires `--primary-results-root` and
`--supplemental-backend-root`; the latter points to `analysis/`.

Do not commit raw recordings, epochs, covariance arrays, feature caches, or
download-manager directories.
