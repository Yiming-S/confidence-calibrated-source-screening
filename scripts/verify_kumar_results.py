"""Check current Kumar aggregates and plotted exclusion bounds without EEG data.

This check validates released subject means, not omitted split predictions,
covariance caches, outer samples or bootstrap draw matrices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results/eeg/kumar2024"


def close(actual, expected, label):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.allclose(a, b, rtol=1e-11, atol=1e-12):
        raise ValueError(f"Kumar2024 {label}: does not match the released inputs")


def verify_kumar_results(folder=DEFAULT_INPUT):
    summary = json.loads((folder / "summary.json").read_text())
    subjects = summary["subject_means"]
    if (summary["status"] != "complete" or summary["analysis_role"] != "exploratory"
            or [s["subject"] for s in subjects] != list(range(1, 19))):
        raise ValueError("Kumar2024: incomplete subject set or incorrect analysis role")
    if any(s["n_splits"] != 30 or s["n_outer"] != 50 for s in subjects):
        raise ValueError("Kumar2024: split or outer-sample count changed")
    methods = {"all_sources", "top1", "top3", "mcs_stepdown", "pair_ref", "target_only"}
    downstream = pd.read_csv(folder / "downstream_by_subject.csv").set_index(["subject", "method"])
    if len(downstream) != 108 or downstream.index.duplicated().any():
        raise ValueError("Kumar2024: invalid downstream subject-method grid")
    for subject in subjects:
        if set(subject["methods"]) != methods:
            raise ValueError("Kumar2024: incorrect method inventory")
        for method, values in subject["methods"].items():
            row = downstream.loc[(subject["subject"], method)]
            close([row.balanced_accuracy, row.set_size, row.n_train],
                  [values["balanced_accuracy"], values["set_size"], values["training_trials"]],
                  "downstream subject mean")
    for method, values in summary["overall"]["methods"].items():
        for key, observed in values.items():
            close(observed, np.mean([s["methods"][method][key] for s in subjects]), f"{method}/{key}")
    for key, observed in summary["overall"]["coverage"].items():
        close(observed, np.mean([s["coverage"][key] for s in subjects]), f"coverage/{key}")
    for name, comparator in (("primary_refinement_minus_all_sources", "all_sources"),
                             ("refinement_minus_mcs", "mcs_stepdown")):
        paired = summary[name]
        d = np.array([s["methods"]["pair_ref"]["balanced_accuracy"]-
                      s["methods"][comparator]["balanced_accuracy"] for s in subjects])
        mean, se = float(d.mean()), float(d.std(ddof=1)/np.sqrt(len(d)))
        close(paired["differences"], d, name)
        close([paired["mean_difference"], paired["standard_error"]], [mean, se], name)
        close(paired["two_sided_95_ci"], mean + np.array([-1,1])*t.ppf(.975,17)*se, name)
        for margin, inference in paired["noninferiority"].items():
            lcb = mean-t.ppf(.95,17)*se
            close([inference["one_sided_95_lower"], inference["one_sided_p"]],
                  [lcb,t.sf((mean+float(margin))/se,17)], name)
            if inference["noninferior"] != (lcb > -float(margin)):
                raise ValueError("Kumar2024: non-inferiority decision mismatch")
    if set(summary["primary_refinement_minus_all_sources"]["noninferiority"]) != {"0.01","0.02"}:
        raise ValueError("Kumar2024: planned margin set changed")
    certificate = json.loads((folder / "supplemental/certificate.json").read_text())
    if (certificate["subject"], certificate["bootstrap"], certificate["source_sessions"]) != (1,1999,[1,2,3,4,5]):
        raise ValueError("Kumar2024: fixed certificate design changed")
    system=certificate["system"]
    estimate=np.asarray(certificate["estimate"])
    close(system["lower_c"], estimate-system["q_c"]*np.asarray(system["se_c"]), "component lower bounds")
    close(system["upper_c"], estimate+system["q_c"]*np.asarray(system["se_c"]), "component upper bounds")
    jj,ll=np.asarray(system["jj"]),np.asarray(system["ll"])
    close(system["lower_d"], estimate[jj]-estimate[ll]-system["q_d"]*np.asarray(system["se_d"]), "contrast lower bounds")
    rectangle=set(np.flatnonzero(np.asarray(system["lower_c"]) <= min(system["upper_c"])))
    pair=set(range(5)) - set(jj[np.asarray(system["lower_d"]) > 0])
    if set(system["sets"]["pair_ref"]) != rectangle.intersection(pair):
        raise ValueError("Kumar2024: refinement set does not equal rectangle/contrast intersection")
    for index,row in enumerate(certificate["sources"]):
        close([row["discrepancy"],row["lower_component"],row["upper_component"]],
              [estimate[index],system["lower_c"][index],system["upper_c"][index]], "plotted component endpoints")
        if row["refinement_retained"] != (index in rectangle.intersection(pair)):
            raise ValueError("Kumar2024: plotted inclusion does not match bounds")
    provenance=json.loads((folder/"provenance.json").read_text())
    if provenance["primary_summary_sha256"] != hashlib.sha256((folder/"summary.json").read_bytes()).hexdigest():
        raise ValueError("Kumar2024: original summary checksum mismatch")
    return {"status":"passed","subjects":18,"subject_method_rows":108,"paired_intervals":2,
            "coverage_subjects":18,"certificate_sources":5,"certificate_contrasts":20,
            "scope":"aggregate reconstruction and bound algebra; no raw-data or bootstrap replay"}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    print(json.dumps(verify_kumar_results(parser.parse_args().input_dir), indent=2))
