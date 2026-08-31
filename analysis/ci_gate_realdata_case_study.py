#!/usr/bin/env python3
"""Real-data MMD case study for CI-gated source screening.

This is not theorem validation.  The population discrepancies and oracle set
are unknown in real EEG data.  The script reports retained set sizes, empirical
best sessions, and samplewise relations among rectangle, pairwise, and
refinement gates.

Primary case study:
    MNE-ma2020-data, within-subject source-session screening.
    Target = session 15; sources = sessions 1--14.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mne

from screening_core import build_system, ordered_pairs, pair_gate, rect_gate
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from path_config import resolve_cache_root, resolve_data_root


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "simulation_results" / "realdata_mmd_ma2020"


def parse_subjects(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = [int(x) for x in part.split("-", 1)]
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return out


def session_path(data_root: Path, subject: int, session: int) -> Path:
    return (
        data_root
        / "MNE-ma2020-data"
        / f"sub-{subject:03d}"
        / f"sub-{subject:03d}_ses-{session:02d}_task-motorimagery_eeg.cnt"
    )


def extract_session_features(
    path: Path,
    cache_path: Path,
    tmin: float,
    tmax: float,
    l_freq: float,
    h_freq: float,
) -> np.ndarray:
    if cache_path.exists():
        return np.load(cache_path)["features"]

    raw = mne.io.read_raw_cnt(str(path), preload=True, verbose="ERROR")
    drop = [ch for ch in ("HEO", "VEO", "M2") if ch in raw.ch_names]
    if drop:
        raw.drop_channels(drop)
    raw.pick_types(eeg=True)
    raw.filter(
        l_freq,
        h_freq,
        method="iir",
        iir_params={"order": 4, "ftype": "butter"},
        verbose="ERROR",
    )
    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    keep_ids = {key: val for key, val in event_id.items() if key in {"1", "2", "right_hand", "right_elbow"}}
    if not keep_ids:
        keep_ids = event_id
    epochs = mne.Epochs(
        raw,
        events,
        event_id=keep_ids,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose="ERROR",
    )
    data = epochs.get_data(copy=False)
    features = np.log(np.var(data, axis=2) + 1e-12)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=features, ch_names=np.array(epochs.ch_names))
    return features


def gaussian_kernel_matrix(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    x2 = np.sum(x * x, axis=1)[:, None]
    y2 = np.sum(y * y, axis=1)[None, :]
    dist2 = np.maximum(x2 + y2 - 2.0 * x @ y.T, 0.0)
    return np.exp(-dist2 / (2.0 * sigma * sigma))


def u_with_counts(kernel: np.ndarray, counts: np.ndarray) -> np.ndarray:
    n = counts.sum(axis=1).astype(float)
    quad = np.einsum("bi,ij,bj->b", counts, kernel, counts, optimize=True)
    diag = counts @ np.diag(kernel)
    return (quad - diag) / (n * (n - 1.0))


def cross_with_counts(kernel: np.ndarray, counts_x: np.ndarray, counts_y: np.ndarray) -> np.ndarray:
    nx = counts_x.sum(axis=1).astype(float)
    ny = counts_y.sum(axis=1).astype(float)
    cross = np.einsum("bi,ij,bj->b", counts_x, kernel, counts_y, optimize=True)
    return cross / (nx * ny)


def multinomial_counts(rng: np.random.Generator, n_boot: int, n: int) -> np.ndarray:
    return rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(float)


def mmd_estimates_and_bootstrap(
    sources: list[np.ndarray],
    target: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
    sigma: float,
    shared_target: bool,
) -> tuple[np.ndarray, np.ndarray]:
    n_t = target.shape[0]
    k = len(sources)
    target_kernel = gaussian_kernel_matrix(target, target, sigma)
    target_ones = np.ones((1, n_t), dtype=float)
    target_u = float(u_with_counts(target_kernel, target_ones)[0])
    target_counts_shared = multinomial_counts(rng, n_boot, n_t)
    target_u_shared = u_with_counts(target_kernel, target_counts_shared)

    estimates = np.empty(k)
    boot = np.empty((n_boot, k))
    for j, source in enumerate(sources):
        n_s = source.shape[0]
        source_kernel = gaussian_kernel_matrix(source, source, sigma)
        cross_kernel = gaussian_kernel_matrix(source, target, sigma)
        source_ones = np.ones((1, n_s), dtype=float)
        source_u = float(u_with_counts(source_kernel, source_ones)[0])
        cross = float(cross_with_counts(cross_kernel, source_ones, target_ones)[0])
        estimates[j] = source_u + target_u - 2.0 * cross

        source_counts = multinomial_counts(rng, n_boot, n_s)
        source_u_boot = u_with_counts(source_kernel, source_counts)
        if shared_target:
            target_counts = target_counts_shared
            target_u_boot = target_u_shared
        else:
            target_counts = multinomial_counts(rng, n_boot, n_t)
            target_u_boot = u_with_counts(target_kernel, target_counts)
        cross_boot = cross_with_counts(cross_kernel, source_counts, target_counts)
        boot[:, j] = source_u_boot + target_u_boot - 2.0 * cross_boot
    return estimates, boot


def labels(selected: np.ndarray, source_sessions: list[int]) -> str:
    return ",".join(str(source_sessions[int(i)]) for i in selected)


def run_subject(
    subject: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    source_sessions = list(range(1, args.target_session))
    all_sessions = source_sessions + [args.target_session]
    cache_dir = args.cache_root / f"sub-{subject:03d}"

    raw_features: dict[int, np.ndarray] = {}
    for session in all_sessions:
        path = session_path(args.data_root, subject, session)
        if not path.exists():
            raise FileNotFoundError(path)
        cache_path = cache_dir / f"ses-{session:02d}_features.npz"
        raw_features[session] = extract_session_features(path, cache_path, args.tmin, args.tmax, args.l_freq, args.h_freq)

    stacked = np.vstack([raw_features[s] for s in all_sessions])
    scaler = StandardScaler()
    z = scaler.fit_transform(stacked)
    n_components = min(args.pca_components, z.shape[1], z.shape[0] - 1)
    pca = PCA(n_components=n_components, whiten=True, random_state=args.seed)
    z_pca = pca.fit_transform(z)

    split: dict[int, np.ndarray] = {}
    start = 0
    for session in all_sessions:
        n = raw_features[session].shape[0]
        split[session] = z_pca[start : start + n]
        start += n

    sources = [split[s] for s in source_sessions]
    target = split[args.target_session]

    delta_hat, delta_boot = mmd_estimates_and_bootstrap(sources, target, rng, args.boot, args.kernel_sigma, True)
    system = build_system(delta_hat, delta_boot, args.alpha_c, args.alpha_d, args.alpha)
    wrong_hat, wrong_boot = mmd_estimates_and_bootstrap(sources, target, rng, args.boot, args.kernel_sigma, False)
    wrong_system = build_system(wrong_hat, wrong_boot, args.alpha_c, args.alpha_d, args.alpha)

    k = len(sources)
    rect_split = rect_gate(system["lower_c"], system["upper_c"])
    pair_split = pair_gate(k, system["jj"], system["lower_d"])
    ref_split = np.array(sorted(set(rect_split.tolist()) & set(pair_split.tolist())), dtype=int)
    rect_comb = rect_gate(system["lower_cj"], system["upper_cj"])
    pair_comb = pair_gate(k, system["jj"], system["lower_dj"])
    ref_comb = np.array(sorted(set(rect_comb.tolist()) & set(pair_comb.tolist())), dtype=int)
    wrong_rect = rect_gate(wrong_system["lower_cj"], wrong_system["upper_cj"])
    wrong_pair = pair_gate(k, wrong_system["jj"], wrong_system["lower_dj"])
    wrong_ref = np.array(sorted(set(wrong_rect.tolist()) & set(wrong_pair.tolist())), dtype=int)

    empirical_best = int(np.flatnonzero(delta_hat == np.min(delta_hat))[0])
    domination_all = bool(
        np.all(system["q_d"] * system["se_d"] <= system["q_c"] * (system["se_c"][system["jj"]] + system["se_c"][system["ll"]]) + 1e-12)
    )

    summary = {
        "dataset": "ma2020",
        "subject": subject,
        "target_session": args.target_session,
        "n_source_sessions": len(source_sessions),
        "target_trials": int(target.shape[0]),
        "feature_dim": int(target.shape[1]),
        "empirical_best_session": source_sessions[empirical_best],
        "empirical_best_mmd2": float(delta_hat[empirical_best]),
        "rect_split_size": int(rect_split.size),
        "pair_split_size": int(pair_split.size),
        "ref_split_size": int(ref_split.size),
        "rect_combined_size": int(rect_comb.size),
        "pair_combined_size": int(pair_comb.size),
        "ref_combined_size": int(ref_comb.size),
        "wrong_ref_combined_size": int(wrong_ref.size),
        "pair_subset_rect": bool(set(pair_split.tolist()).issubset(set(rect_split.tolist()))),
        "ref_subset_rect": bool(set(ref_split.tolist()).issubset(set(rect_split.tolist()))),
        "domination_all": domination_all,
        "rect_split_sessions": labels(rect_split, source_sessions),
        "pair_split_sessions": labels(pair_split, source_sessions),
        "ref_split_sessions": labels(ref_split, source_sessions),
        "ref_combined_sessions": labels(ref_comb, source_sessions),
        "wrong_ref_combined_sessions": labels(wrong_ref, source_sessions),
    }

    detail_rows: list[dict[str, object]] = []
    for idx, session in enumerate(source_sessions):
        detail_rows.append(
            {
                "dataset": "ma2020",
                "subject": subject,
                "source_session": session,
                "target_session": args.target_session,
                "n_trials": int(sources[idx].shape[0]),
                "mmd2_hat": float(delta_hat[idx]),
                "in_rect_split": int(idx in set(rect_split.tolist())),
                "in_pair_split": int(idx in set(pair_split.tolist())),
                "in_ref_split": int(idx in set(ref_split.tolist())),
                "in_rect_combined": int(idx in set(rect_comb.tolist())),
                "in_pair_combined": int(idx in set(pair_comb.tolist())),
                "in_ref_combined": int(idx in set(ref_comb.tolist())),
                "in_wrong_ref_combined": int(idx in set(wrong_ref.tolist())),
            }
        )
    return summary, detail_rows


def write_report(out_dir: Path, summary: pd.DataFrame, detail: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = [
        "# Real-Data MMD Case Study: Ma2020",
        "",
        "This is a case study, not theorem validation.  Population discrepancies, oracle sources, and margins are unknown.",
        "",
        f"- Data root: `{args.data_root}`",
        f"- Target session: `{args.target_session}`",
        f"- Subjects: `{','.join(str(s) for s in args.subjects)}`",
        f"- Bootstrap replicates: `{args.boot}`",
        f"- Feature pipeline: {args.l_freq}-{args.h_freq} Hz log-variance, z-score, PCA whiten to `{args.pca_components}` components",
        "",
        "## Subject Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Source-Session Details",
        "",
        detail.to_markdown(index=False, floatfmt=".4f"),
    ]
    (out_dir / "ma2020_case_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Feature-cache directory (default: EEG_DATA_ROOT/CI-gate-cache/ma2020-mmd or OUT_DIR/feature_cache).",
    )
    parser.add_argument("--subjects", type=parse_subjects, default=parse_subjects("1-5"))
    parser.add_argument("--target-session", type=int, default=15)
    parser.add_argument("--boot", type=int, default=499)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--alpha-c", type=float, default=0.025)
    parser.add_argument("--alpha-d", type=float, default=0.025)
    parser.add_argument("--kernel-sigma", type=float, default=1.0)
    parser.add_argument("--pca-components", type=int, default=10)
    parser.add_argument("--tmin", type=float, default=0.5)
    parser.add_argument("--tmax", type=float, default=3.5)
    parser.add_argument("--l-freq", type=float, default=8.0)
    parser.add_argument("--h-freq", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20260503)
    args = parser.parse_args()
    args.data_root = resolve_data_root(args.data_root, parser)
    args.cache_root = resolve_cache_root(
        args.cache_root,
        parser,
        env_suffix="CI-gate-cache/ma2020-mmd",
        fallback=args.out_dir / "feature_cache",
    )
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mne.set_log_level("ERROR")
    rng = np.random.default_rng(args.seed)

    manifest = vars(args).copy()
    manifest["data_root"] = str(args.data_root)
    manifest["out_dir"] = str(args.out_dir)
    manifest["cache_root"] = str(args.cache_root)
    manifest["subjects"] = args.subjects
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    start = time.time()
    for idx, subject in enumerate(args.subjects, start=1):
        print(f"[{idx}/{len(args.subjects)}] ma2020 subject {subject} elapsed={time.time()-start:.1f}s", flush=True)
        summary, detail = run_subject(subject, args, rng)
        summary_rows.append(summary)
        detail_rows.extend(detail)
        pd.DataFrame(summary_rows).to_csv(args.out_dir / "ma2020_case_summary.csv", index=False)
        pd.DataFrame(detail_rows).to_csv(args.out_dir / "ma2020_case_details.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)
    write_report(args.out_dir, summary_df, detail_df, args)
    print(f"wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
