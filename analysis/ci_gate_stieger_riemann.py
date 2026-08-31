"""Covariance-space (Riemannian) Stieger2021 case study for CI-gated source
screening -- a second real-data illustration alongside Ma2020.

Stieger2021 is a longitudinal sensorimotor-rhythm BCI dataset (62 subjects,
~11 sessions each, 62-channel EEG at 1 kHz).  We use the left/right-hand
motor-imagery task (tasknumber 1, targetnumber in {1,2}); each session is
summarized by the arithmetic mean of its per-trial 8--30 Hz spatial
covariance matrices, and the screening discrepancy is the squared
affine-invariant Riemannian distance to the target-session mean covariance.
The latest available session is the target and the earlier sessions are
candidate sources.  The screening and downstream machinery is identical to
ci_gate_ma2020_riemann.py; only the loader differs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import StratifiedShuffleSplit

import mne

from ci_gate_ma2020_riemann import (
    airm2,  # noqa: F401  (imported for parity / potential use)
    riemann_estimates_and_bootstrap,
    screen_pair_ref,
    shrink,  # noqa: F401
    summarize_case,
    summarize_downstream,
    train_eval_ts,
)
from path_config import resolve_cache_root, resolve_data_root
from screening_core import build_system, pair_gate, rect_gate

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "simulation_results" / "stieger_riemann_all_s1_s62_b499"
SRATE = 1000.0


def stieger_path(data_root: Path, subject: int, session: int) -> Path:
    return data_root / "MNE-Stieger2021-data" / f"S{subject}_Session_{session}.mat"


def available_sessions(data_root: Path, subject: int, max_session: int = 11) -> list[int]:
    return [s for s in range(1, max_session + 1) if stieger_path(data_root, subject, s).exists()]


def extract_session_covs(
    path: Path, cache_path: Path, l_freq: float, h_freq: float,
    win: tuple[int, int], task: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-trial LR-task covariances + labels for one Stieger session, cached."""
    if cache_path.exists():
        z = np.load(cache_path)
        return z["covs"].astype(float), z["labels"].astype(int)

    m = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    B = m["BCI"]
    tasknum = np.array([t.tasknumber for t in B.TrialData])
    target = np.array([t.targetnumber for t in B.TrialData])
    sel = np.flatnonzero((tasknum == task) & np.isin(target, (1, 2)))
    covs, labels = [], []
    for i in sel:
        x = np.asarray(B.data[i], dtype=float)  # (C, T)
        if x.shape[1] < win[1]:
            continue  # trial ended early; skip to keep an equal-length window
        xf = mne.filter.filter_data(
            x, SRATE, l_freq, h_freq, method="iir",
            iir_params={"order": 4, "ftype": "butter"}, verbose="ERROR")
        seg = xf[:, win[0]:win[1]]
        seg = seg - seg.mean(axis=1, keepdims=True)
        covs.append(seg @ seg.T / (seg.shape[1] - 1))
        labels.append(int(target[i]) - 1)
    covs = np.asarray(covs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, covs=covs.astype(np.float32), labels=labels)
    return covs, labels


def load_subject(subject: int, args) -> tuple[list[int], int, dict, dict]:
    sessions = available_sessions(args.data_root, subject, args.max_session)
    if len(sessions) < args.min_sessions:
        raise ValueError(f"subject {subject}: only {len(sessions)} sessions")
    target = sessions[-1]
    sources = sessions[:-1]
    cache_dir = args.cache_root / f"S{subject}"
    covs, labs = {}, {}
    for s in sessions:
        covs[s], labs[s] = extract_session_covs(
            stieger_path(args.data_root, subject, s),
            cache_dir / f"ses-{s:02d}_cov.npz",
            args.l_freq, args.h_freq, (args.win_start, args.win_end), args.task)
    return sources, target, covs, labs


def run_subject_screening(subject: int, args, rng) -> dict:
    sources, target, covs, _ = load_subject(subject, args)
    src = [covs[s] for s in sources]
    tgt = covs[target]
    k = len(src)
    dh, db = riemann_estimates_and_bootstrap(src, tgt, rng, args.boot, args.shrinkage, True)
    sys = build_system(dh, db, args.alpha_c, args.alpha_d, args.alpha)
    wh, wb = riemann_estimates_and_bootstrap(src, tgt, rng, args.boot, args.shrinkage, False)
    wsys = build_system(wh, wb, args.alpha_c, args.alpha_d, args.alpha)

    def combo(system):
        rc = rect_gate(system["lower_cj"], system["upper_cj"])
        pc = pair_gate(k, system["jj"], system["lower_dj"])
        return sorted(set(rc.tolist()) & set(pc.tolist()))

    rect_s = rect_gate(sys["lower_c"], sys["upper_c"])
    pair_s = pair_gate(k, sys["jj"], sys["lower_d"])
    ref_s = sorted(set(rect_s.tolist()) & set(pair_s.tolist()))
    return {
        "subject": subject,
        "n_sources": k,
        "rect_split_size": int(rect_s.size),
        "pair_split_size": int(len(ref_s)),
        "ref_combined_size": int(len(combo(sys))),
        "rect_combined_size": int(rect_gate(sys["lower_cj"], sys["upper_cj"]).size),
        "wrong_ref_combined_size": int(len(combo(wsys))),
    }


def run_subject_downstream(subject: int, args, rng) -> list[dict]:
    sources, target, covs, labs = load_subject(subject, args)
    src = [covs[s] for s in sources]
    src_lab = [labs[s] for s in sources]
    tgt, tgt_lab = covs[target], labs[target]
    k = len(src)
    all_covs = np.concatenate(src, axis=0)
    all_lab = np.concatenate(src_lab, axis=0)
    splitter = StratifiedShuffleSplit(n_splits=args.splits, test_size=0.5,
                                      random_state=args.seed + subject)
    rows = []
    for screen_idx, test_idx in splitter.split(tgt, tgt_lab):
        screen_covs, test_covs, test_lab = tgt[screen_idx], tgt[test_idx], tgt_lab[test_idx]
        dh, db = riemann_estimates_and_bootstrap(src, screen_covs, rng, args.down_boot, args.shrinkage, True)
        pair_ref = screen_pair_ref(dh, db, k, args.alpha_c, args.alpha_d, args.alpha)
        top1 = np.argsort(dh, kind="mergesort")[:1]
        top3 = np.argsort(dh, kind="mergesort")[:3]

        def pool(idx):
            if len(idx) == 0:
                return None, None
            return (np.concatenate([src[i] for i in idx], axis=0),
                    np.concatenate([src_lab[i] for i in idx], axis=0))

        sets = {"all_sources": list(range(k)), "top1": top1, "top3": top3, "pair_ref": pair_ref}
        pools = {"all_sources": (all_covs, all_lab), "top1": pool(top1),
                 "top3": pool(top3), "pair_ref": pool(pair_ref)}
        for name, (cc, ll) in pools.items():
            acc = float("nan") if cc is None else train_eval_ts(cc, ll, test_covs, test_lab)
            rows.append({"subject": subject, "method": name,
                         "set_size": len(sets[name]), "balanced_accuracy": acc})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", type=str, default="1-20")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Covariance-cache directory (default: EEG_DATA_ROOT/CI-gate-cache/stieger2021 or OUT_DIR/cov_cache).",
    )
    p.add_argument("--max-session", type=int, default=11)
    p.add_argument("--min-sessions", type=int, default=6)
    p.add_argument("--task", type=int, default=1)
    p.add_argument("--l-freq", type=float, default=8.0)
    p.add_argument("--h-freq", type=float, default=30.0)
    p.add_argument("--win-start", type=int, default=2000)
    p.add_argument("--win-end", type=int, default=5000)
    p.add_argument("--shrinkage", type=float, default=0.1)
    p.add_argument("--boot", type=int, default=499)
    p.add_argument("--splits", type=int, default=30)
    p.add_argument("--down-boot", type=int, default=199)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--alpha-c", type=float, default=0.025)
    p.add_argument("--alpha-d", type=float, default=0.025)
    p.add_argument("--seed", type=int, default=20260707)
    p.add_argument("--downstream", action="store_true")
    args = p.parse_args()
    args.data_root = resolve_data_root(args.data_root, p)
    args.cache_root = resolve_cache_root(
        args.cache_root,
        p,
        env_suffix="CI-gate-cache/stieger2021",
        fallback=args.out_dir / "cov_cache",
    )

    lo, hi = (args.subjects.split("-") + [args.subjects])[:2]
    subjects = list(range(int(lo), int(hi) + 1))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    case_rows, used = [], []
    for s in subjects:
        try:
            print(f"[stieger] screening subject {s} ...", flush=True)
            case_rows.append(run_subject_screening(s, args, rng))
            used.append(s)
        except (FileNotFoundError, ValueError) as e:
            print(f"[stieger] skip subject {s}: {e}", flush=True)
    case = pd.DataFrame(case_rows)
    case.to_csv(args.out_dir / "stieger_riemann_case_details.csv", index=False)
    case_summary = summarize_case(case)
    case_summary.to_csv(args.out_dir / "stieger_riemann_case_summary.csv", index=False)
    print(case_summary.to_string(index=False))

    if args.downstream:
        down_rows = []
        for s in used:
            print(f"[stieger] downstream subject {s} ...", flush=True)
            down_rows.extend(run_subject_downstream(s, args, rng))
        down = pd.DataFrame(down_rows)
        down.to_csv(args.out_dir / "stieger_riemann_downstream_raw.csv", index=False)
        down_summary = summarize_downstream(down)
        down_summary.to_csv(args.out_dir / "stieger_riemann_downstream_summary.csv", index=False)
        print(down_summary.to_string(index=False))

    (args.out_dir / "manifest.json").write_text(json.dumps({
        "script": Path(__file__).name, "subjects_requested": subjects, "subjects_used": used,
        "data_root": str(args.data_root), "cache_root": str(args.cache_root),
        "out_dir": str(args.out_dir),
        "task": args.task, "window_samples": [args.win_start, args.win_end],
        "l_freq": args.l_freq, "h_freq": args.h_freq, "shrinkage": args.shrinkage,
        "screening_boot": args.boot,
        "downstream_boot": args.down_boot if args.downstream else None,
        "splits": args.splits if args.downstream else None, "seed": args.seed,
        "runtime_seconds": round(time.time() - t0, 1),
    }, indent=2))
    print(f"[stieger] done in {round(time.time()-t0,1)}s ({len(used)} subjects) -> {args.out_dir}")


if __name__ == "__main__":
    main()
