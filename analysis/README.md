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
- `ci_gate_zhou2020_riemann.py`
- `ci_gate_bnci004_riemann.py`
- `ci_gate_rolling_targets.py`
- `ci_gate_block_bootstrap_sensitivity.py`
- `ci_gate_eeg_empirical_audit.py`
- `ci_gate_noninferiority.py`

`ci_gate_realdata_case_study.py` and `ci_gate_target_only_reference.py` also
provide shared functions imported by the primary entry points.
`screening_core.py` contains the data-agnostic simultaneous-bound and
set-construction operations covered by deterministic unit tests.

Run any entry point with `--help` before a full analysis. Data-reading scripts
require separately obtained public EEG recordings or previously built caches.
