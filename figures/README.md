# Archived main-result figures

The current manuscript order is:

1. `fig_eeg_overview.pdf`: four-cohort source retention and decoding results.
2. `fig_kumar2024_exclusion_certificate.pdf`: source-specific exclusion evidence.
3. `fig_shared_target_evidence.pdf`: shared-target calibration experiments.

The archived PDFs match the manuscript revision dated 2026-10-03. Use these
filenames, rather than older console figure numbers, to identify the outputs.

- `fig_shared_target_evidence.pdf` can be rebuilt from `results/simulations/`;
  `fig_eeg_overview.pdf` can be rebuilt from `results/figure_data/` and its
  underlying subject summaries.
- `fig_kumar2024_exclusion_certificate.pdf` can be rebuilt from the released
  interval endpoints and exclusion comparisons in
  `results/eeg/kumar2024/supplemental/certificate.json`. Recalculating those
  endpoints requires the original EEG preprocessing and bootstrap analysis.

Run `make figures` to rebuild all three figures under `build/figures/`, without modifying
the archived copies in this directory.
