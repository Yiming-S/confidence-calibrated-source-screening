"""Covariance-space (Riemannian) Ma2020 case study for CI-gated source screening.

Covariance-native instantiation of the discrepancy-agnostic framework: each
session is summarized by the arithmetic mean of its per-trial spatial
covariance matrices (theta = E_P[m(Z)] with m(Z) = trial covariance, so it
lies in the smooth plug-in class), and the screening discrepancy is the
squared affine-invariant Riemannian distance between the source and target
mean covariances,
    Delta_k = || log( Sigma_T^{-1/2} Sigma_k Sigma_T^{-1/2} ) ||_F^2 .
Every source is compared to the same target covariance, so the shared-target
multiplier/resampling bootstrap of the paper applies verbatim.  The
downstream check is the standard Riemannian pipeline: tangent-space
projection at the training Riemannian mean followed by LDA.

Mirrors ci_gate_realdata_case_study.py (retained-set summary) and
ci_gate_ma2020_downstream.py (classification) but with a covariance-space
discrepancy and tangent-space features.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

import mne

from ci_gate_realdata_case_study import session_path
from path_config import resolve_cache_root, resolve_data_root
from screening_core import build_system, ordered_pairs, pair_gate, rect_gate

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "simulation_results" / "ma2020_riemann_all_s1_s25_b499"


# ----- covariance extraction -----

def extract_session_covs(
    path: Path, cache_path: Path, tmin: float, tmax: float, l_freq: float, h_freq: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (trial_covs [n, C, C], labels [n]) for one session, cached."""
    if cache_path.exists():
        z = np.load(cache_path)
        return z["covs"].astype(float), z["labels"].astype(int)

    raw = mne.io.read_raw_cnt(str(path), preload=True, verbose="ERROR")
    drop = [ch for ch in ("HEO", "VEO", "M2") if ch in raw.ch_names]
    if drop:
        raw.drop_channels(drop)
    raw.pick_types(eeg=True)
    raw.filter(l_freq, h_freq, method="iir",
               iir_params={"order": 4, "ftype": "butter"}, verbose="ERROR")
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    keep_ids = {k: v for k, v in event_id.items() if k in {"1", "2", "right_hand", "right_elbow"}}
    if not keep_ids:
        keep_ids = event_id
    epochs = mne.Epochs(raw, events, event_id=keep_ids, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, verbose="ERROR")
    data = epochs.get_data(copy=False)  # (n, C, T)
    n, c, t = data.shape
    data = data - data.mean(axis=2, keepdims=True)
    covs = np.einsum("nct,nkt->nck", data, data) / (t - 1)
    raw_labels = epochs.events[:, 2]
    levels = sorted(np.unique(raw_labels).tolist())
    mapping = {lev: i for i, lev in enumerate(levels)}
    labels = np.array([mapping[x] for x in raw_labels], dtype=int)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, covs=covs.astype(np.float32), labels=labels)
    return covs, labels


def shrink(cov: np.ndarray, lam: float) -> np.ndarray:
    c = cov.shape[-1]
    mu = np.trace(cov, axis1=-2, axis2=-1) / c
    eye = np.eye(c)
    if cov.ndim == 2:
        return (1.0 - lam) * cov + lam * mu * eye
    return (1.0 - lam) * cov + lam * mu[:, None, None] * eye


# ----- affine-invariant Riemannian geometry -----

def airm2(a: np.ndarray, b: np.ndarray) -> float:
    """Squared affine-invariant distance between SPD a and b."""
    w = eigh(b, a, eigvals_only=True)
    lw = np.log(np.clip(w, 1e-12, None))
    return float(np.sum(lw * lw))


def _sym(x: np.ndarray) -> np.ndarray:
    return 0.5 * (x + x.T)


def matrix_pow(cov: np.ndarray, power: float) -> np.ndarray:
    w, v = np.linalg.eigh(_sym(cov))
    w = np.clip(w, 1e-12, None)
    return (v * (w ** power)) @ v.T


def matrix_log(cov: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(_sym(cov))
    w = np.clip(w, 1e-12, None)
    return (v * np.log(w)) @ v.T


def logeuclid2(a: np.ndarray, b: np.ndarray) -> float:
    """Squared log-Euclidean distance ||log a - log b||_F^2 between SPD a and b."""
    d = matrix_log(a) - matrix_log(b)
    return float(np.sum(d * d))


def euclid2(a: np.ndarray, b: np.ndarray) -> float:
    """Squared Euclidean/Frobenius distance ||a - b||_F^2."""
    d = a - b
    return float(np.sum(d * d))


# Same-summary discrepancies on the covariance manifold: theta = arithmetic mean
# covariance in every case; only the smooth distance g(theta_k, theta_T) changes.
METRICS = {"airm": airm2, "logeuclid": logeuclid2, "euclid": euclid2}


def riemann_mean(covs: np.ndarray, n_iter: int = 5) -> np.ndarray:
    """Frechet mean of SPD matrices via the standard fixed-point iteration."""
    mean = covs.mean(axis=0)
    for _ in range(n_iter):
        mhalf = matrix_pow(mean, 0.5)
        minv = matrix_pow(mean, -0.5)
        s = np.mean([matrix_log(minv @ c @ minv) for c in covs], axis=0)
        mean = mhalf @ _sym(matrix_log_exp(s)) @ mhalf
    return _sym(mean)


def matrix_log_exp(s: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(_sym(s))
    return (v * np.exp(w)) @ v.T


def tangent_features(covs: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Tangent-space vectors at ref (upper triangle, off-diagonal * sqrt2)."""
    rinv = matrix_pow(ref, -0.5)
    c = ref.shape[0]
    iu = np.triu_indices(c)
    scale = np.sqrt(2.0) * np.ones((c, c))
    np.fill_diagonal(scale, 1.0)
    out = np.empty((covs.shape[0], iu[0].size))
    for i, cov in enumerate(covs):
        s = matrix_log(rinv @ cov @ rinv)
        out[i] = (s * scale)[iu]
    return out


# ----- screening discrepancy + shared-target bootstrap -----

def riemann_estimates_and_bootstrap(
    source_covs: list[np.ndarray],
    target_covs: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
    lam: float,
    shared_target: bool,
    dist=airm2,
) -> tuple[np.ndarray, np.ndarray]:
    k = len(source_covs)
    n_t = target_covs.shape[0]
    theta_t = shrink(target_covs.mean(axis=0), lam)
    estimates = np.array([dist(shrink(s.mean(axis=0), lam), theta_t) for s in source_covs])

    # shared target multinomial counts, reused across sources when shared_target
    tgt_counts_shared = rng.multinomial(n_t, np.full(n_t, 1.0 / n_t), size=n_boot).astype(float)
    boot = np.empty((n_boot, k))
    for j, s in enumerate(source_covs):
        n_s = s.shape[0]
        src_counts = rng.multinomial(n_s, np.full(n_s, 1.0 / n_s), size=n_boot).astype(float)
        src_mean = np.einsum("bn,ncd->bcd", src_counts, s) / n_s
        if shared_target:
            tgt_counts = tgt_counts_shared
        else:
            tgt_counts = rng.multinomial(n_t, np.full(n_t, 1.0 / n_t), size=n_boot).astype(float)
        tgt_mean = np.einsum("bn,ncd->bcd", tgt_counts, target_covs) / n_t
        for b in range(n_boot):
            boot[b, j] = dist(shrink(src_mean[b], lam), shrink(tgt_mean[b], lam))
    return estimates, boot


# ----- per-subject screening (retained sets) -----

def run_subject_screening(subject: int, args, rng) -> tuple[dict, list[np.ndarray], np.ndarray, dict]:
    source_sessions = list(range(1, args.target_session))
    all_sessions = source_sessions + [args.target_session]
    cache_dir = args.cache_root / f"sub-{subject:03d}"
    covs: dict[int, np.ndarray] = {}
    labs: dict[int, np.ndarray] = {}
    for ses in all_sessions:
        p = session_path(args.data_root, subject, ses)
        if not p.exists():
            raise FileNotFoundError(p)
        covs[ses], labs[ses] = extract_session_covs(
            p, cache_dir / f"ses-{ses:02d}_cov.npz", args.tmin, args.tmax, args.l_freq, args.h_freq)

    src = [covs[s] for s in source_sessions]
    tgt = covs[args.target_session]
    k = len(src)

    dh, db = riemann_estimates_and_bootstrap(src, tgt, rng, args.boot, args.shrinkage, True)
    sys = build_system(dh, db, args.alpha_c, args.alpha_d, args.alpha)
    wh, wb = riemann_estimates_and_bootstrap(src, tgt, rng, args.boot, args.shrinkage, False)
    wsys = build_system(wh, wb, args.alpha_c, args.alpha_d, args.alpha)

    def sizes(system):
        rect_s = rect_gate(system["lower_c"], system["upper_c"])
        pair_s = pair_gate(k, system["jj"], system["lower_d"])
        ref_s = sorted(set(rect_s.tolist()) & set(pair_s.tolist()))
        rect_c = rect_gate(system["lower_cj"], system["upper_cj"])
        pair_c = pair_gate(k, system["jj"], system["lower_dj"])
        ref_c = sorted(set(rect_c.tolist()) & set(pair_c.tolist()))
        return rect_s, pair_s, ref_s, rect_c, ref_c

    rect_s, pair_s, ref_s, rect_c, ref_c = sizes(sys)
    _, _, _, _, wref_c = sizes(wsys)
    row = {
        "subject": subject,
        "rect_split_size": int(rect_gate(sys["lower_c"], sys["upper_c"]).size),
        "pair_split_size": int(len(ref_s)),
        "ref_combined_size": int(len(ref_c)),
        "rect_combined_size": int(rect_c.size),
        "wrong_ref_combined_size": int(len(wref_c)),
    }
    return row, src, tgt, {"delta_hat": dh, "system": sys, "source_sessions": source_sessions}


# ----- downstream: tangent-space LDA -----

def screen_pair_ref(delta_hat, delta_boot, k, alpha_c, alpha_d, alpha) -> np.ndarray:
    system = build_system(delta_hat, delta_boot, alpha_c, alpha_d, alpha)
    rect_s = rect_gate(system["lower_c"], system["upper_c"])
    pair_s = pair_gate(k, system["jj"], system["lower_d"])
    return np.array(sorted(set(rect_s.tolist()) & set(pair_s.tolist())), dtype=int)


def train_eval_ts(train_covs, train_labels, test_covs, test_labels) -> float:
    if np.unique(train_labels).size < 2:
        return float("nan")
    ref = riemann_mean(shrink(train_covs, 0.05))
    xtr = tangent_features(shrink(train_covs, 0.05), ref)
    xte = tangent_features(shrink(test_covs, 0.05), ref)
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(xtr, train_labels)
    return float(balanced_accuracy_score(test_labels, clf.predict(xte)))


def run_subject_downstream(subject: int, args, rng) -> list[dict]:
    source_sessions = list(range(1, args.target_session))
    all_sessions = source_sessions + [args.target_session]
    cache_dir = args.cache_root / f"sub-{subject:03d}"
    covs, labs = {}, {}
    for ses in all_sessions:
        p = session_path(args.data_root, subject, ses)
        covs[ses], labs[ses] = extract_session_covs(
            p, cache_dir / f"ses-{ses:02d}_cov.npz", args.tmin, args.tmax, args.l_freq, args.h_freq)
    src = [covs[s] for s in source_sessions]
    src_lab = [labs[s] for s in source_sessions]
    tgt, tgt_lab = covs[args.target_session], labs[args.target_session]
    k = len(src)

    all_train_covs = np.concatenate(src, axis=0)
    all_train_lab = np.concatenate(src_lab, axis=0)

    splitter = StratifiedShuffleSplit(n_splits=args.splits, test_size=0.5, random_state=args.seed + subject)
    rows = []
    for screen_idx, test_idx in splitter.split(tgt, tgt_lab):
        screen_covs, test_covs, test_lab = tgt[screen_idx], tgt[test_idx], tgt_lab[test_idx]
        dh, db = riemann_estimates_and_bootstrap(src, screen_covs, rng, args.boot, args.shrinkage, True)
        pair_ref = screen_pair_ref(dh, db, k, args.alpha_c, args.alpha_d, args.alpha)
        top1 = np.argsort(dh, kind="mergesort")[:1]
        top3 = np.argsort(dh, kind="mergesort")[:3]

        def pool(idx):
            if len(idx) == 0:
                return None, None
            cc = np.concatenate([src[i] for i in idx], axis=0)
            ll = np.concatenate([src_lab[i] for i in idx], axis=0)
            return cc, ll

        methods = {
            "all_sources": (all_train_covs, all_train_lab),
            "top1": pool(top1),
            "top3": pool(top3),
            "pair_ref": pool(pair_ref),
        }
        for name, (cc, ll) in methods.items():
            acc = float("nan") if cc is None else train_eval_ts(cc, ll, test_covs, test_lab)
            rows.append({"subject": subject, "method": name, "set_size": 0 if cc is None else len(_uidx(name, pair_ref, top1, top3, k)), "balanced_accuracy": acc})
    return rows


def _uidx(name, pair_ref, top1, top3, k):
    return {"all_sources": list(range(k)), "top1": top1, "top3": top3, "pair_ref": pair_ref}[name]


def summarize_case(rows: pd.DataFrame) -> pd.DataFrame:
    cols = ["rect_split_size", "pair_split_size", "ref_combined_size", "rect_combined_size", "wrong_ref_combined_size"]
    names = {"rect_split_size": "Rect. split", "pair_split_size": "Pair/Ref split",
             "ref_combined_size": "Ref. combined", "rect_combined_size": "Rect. combined",
             "wrong_ref_combined_size": "Wrong-target ref."}
    out = []
    n_src = rows["rect_split_size"].max()
    for c in cols:
        out.append({"quantity": names[c], "mean": rows[c].mean(), "std": rows[c].std(),
                    "min": rows[c].min(), "median": rows[c].median(), "max": rows[c].max(),
                    "shrink_subjects_vs_rect": int((rows[c] < rows["rect_split_size"]).sum())})
    return pd.DataFrame(out)


def summarize_downstream(rows: pd.DataFrame) -> pd.DataFrame:
    per_subj = rows.groupby(["subject", "method"], as_index=False).agg(
        acc=("balanced_accuracy", "mean"), size=("set_size", "mean"))
    base = per_subj[per_subj.method == "all_sources"].set_index("subject")["acc"]
    out = []
    for m in ["all_sources", "top1", "top3", "pair_ref"]:
        sub = per_subj[per_subj.method == m].set_index("subject")
        diff = sub["acc"] - base
        out.append({"method": m, "mean_set_size": sub["size"].mean(),
                    "mean_balanced_accuracy": sub["acc"].mean(),
                    "acc_diff_vs_all": diff.mean(),
                    "negative_transfer_vs_all": float((diff < 0).mean()),
                    "subjects_improved": int((diff > 0).sum())})
    return pd.DataFrame(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subjects", type=str, default="1-25")
    p.add_argument("--target-session", type=int, default=15)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Covariance-cache directory (default: EEG_DATA_ROOT/CI-gate-cache/ma2020 or OUT_DIR/cov_cache).",
    )
    p.add_argument("--tmin", type=float, default=0.0)
    p.add_argument("--tmax", type=float, default=4.0)
    p.add_argument("--l-freq", type=float, default=8.0)
    p.add_argument("--h-freq", type=float, default=30.0)
    p.add_argument("--shrinkage", type=float, default=0.1)
    p.add_argument("--boot", type=int, default=499)
    p.add_argument("--splits", type=int, default=30)
    p.add_argument("--down-boot", type=int, default=199)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--alpha-c", type=float, default=0.025)
    p.add_argument("--alpha-d", type=float, default=0.025)
    p.add_argument("--seed", type=int, default=20260618)
    p.add_argument("--downstream", action="store_true")
    args = p.parse_args()
    args.data_root = resolve_data_root(args.data_root, p)
    args.cache_root = resolve_cache_root(
        args.cache_root,
        p,
        env_suffix="CI-gate-cache/ma2020",
        fallback=args.out_dir / "cov_cache",
    )

    lo, hi = (args.subjects.split("-") + [args.subjects])[:2]
    subjects = list(range(int(lo), int(hi) + 1))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    case_rows = []
    for s in subjects:
        print(f"[riemann] screening subject {s} ...", flush=True)
        row, *_ = run_subject_screening(s, args, rng)
        case_rows.append(row)
    case = pd.DataFrame(case_rows)
    case.to_csv(args.out_dir / "ma2020_riemann_case_details.csv", index=False)
    case_summary = summarize_case(case)
    case_summary.to_csv(args.out_dir / "ma2020_riemann_case_summary.csv", index=False)
    print(case_summary.to_string(index=False))

    screening_boot = args.boot
    if args.downstream:
        down_rows = []
        for s in subjects:
            print(f"[riemann] downstream subject {s} ...", flush=True)
            down_rows.extend(run_subject_downstream(s, args, rng))
        down = pd.DataFrame(down_rows)
        down.to_csv(args.out_dir / "ma2020_riemann_downstream_raw.csv", index=False)
        down_summary = summarize_downstream(down)
        down_summary.to_csv(args.out_dir / "ma2020_riemann_downstream_summary.csv", index=False)
        print(down_summary.to_string(index=False))

    (args.out_dir / "manifest.json").write_text(json.dumps({
        "script": Path(__file__).name, "subjects": subjects, "target_session": args.target_session,
        "data_root": str(args.data_root), "cache_root": str(args.cache_root),
        "out_dir": str(args.out_dir),
        "shrinkage": args.shrinkage, "screening_boot": screening_boot,
        "downstream_boot": args.down_boot if args.downstream else None,
        "splits": args.splits if args.downstream else None,
        "l_freq": args.l_freq, "h_freq": args.h_freq,
        "runtime_seconds": round(time.time() - t0, 1),
    }, indent=2))
    print(f"[riemann] done in {round(time.time()-t0,1)}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
