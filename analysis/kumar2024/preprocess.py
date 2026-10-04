#!/usr/bin/env python3
"""Event-bounded Kumar2024 covariances; no classification or outcome selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import mne
import numpy as np
from scipy.signal import butter, sosfilt


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def find_files(root):
    found = defaultdict(list)
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() != ".gdf" or "__MACOSX" in path.parts:
            continue
        matches = [re.fullmatch(r"Subject_0*(\d+)_Session_0*(\d+)_(Offline|Online)", p) for p in path.parts]
        matches = [m for m in matches if m]
        if not matches:
            continue
        if len(matches) != 1:
            raise ValueError(f"Ambiguous session directory: {path}")
        raw_subject, session, mode = matches[0].groups()
        raw_subject, session = int(raw_subject), int(session)
        if raw_subject not in (*range(1, 10), *range(11, 20)):
            continue
        if (mode == "Offline") != (session == 1):
            raise ValueError(f"Unexpected mode/session: {path}")
        subject = raw_subject if raw_subject < 10 else raw_subject - 1
        group = "GR" if subject < 10 else "PAR"
        if group not in path.parts:
            raise ValueError(f"Group mapping mismatch: {path}")
        # Bar files are directly in the day directory; racing files are separate.
        if path.parent.name != matches[0].group(0):
            continue
        found[subject, session].append(path)
    return found


def task_segments(events):
    """Match cue, task start and first terminal event, retaining short trials."""
    starts = {7691: 0, 7701: 1}
    cues = {769: 0, 770: 1}
    ends = {7692, 7702, 7693, 7703}
    out, cue, active = [], None, None
    for sample, code in events:
        if code in cues:
            if active is not None or cue is not None:
                raise ValueError("New cue before prior task ended or before prior cue was used")
            cue = (int(sample), cues[code])
        elif code in starts:
            label = starts[code]
            if active is not None or cue is None or cue[1] != label or cue[0] >= sample:
                raise ValueError(f"Task/cue mismatch at sample {sample}")
            active = (int(sample), label, cue[0])
            cue = None
        elif code in ends and active is not None:
            start, label, cue_sample = active
            if sample <= start:
                raise ValueError("Nonpositive task duration")
            out.append({"start": start, "end": int(sample), "label": label,
                        "cue": cue_sample, "end_code": int(code)})
            active = None
        elif code == 1000 and (active is not None or cue is not None):
            raise ValueError("New trial before terminal event or before prior cue was used")
    if active is not None or cue is not None:
        raise ValueError("Unterminated task or unused cue at end of file")
    return out


def extract_run(path, relative, spec):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        raw = mne.io.read_raw_gdf(path, preload=True, verbose="ERROR")
    channels = spec["eeg_channels"]
    if [x.casefold() for x in raw.ch_names[:22]] != [x.casefold() for x in channels]:
        raise ValueError(f"EEG channel order mismatch in {path}: {raw.ch_names}")
    fs = raw.info["sfreq"]
    if fs != spec["sampling_hz"]:
        raise ValueError(f"Unexpected sample rate {fs}: {path}")
    events = []
    event_counts = Counter()
    for onset, description in zip(raw.annotations.onset, raw.annotations.description):
        try:
            code = int(description)
        except ValueError:
            raise ValueError(f"Unrecognized annotation {description}: {path}")
        sample = int(raw.time_as_index([onset], use_rounding=True)[0])
        events.append((sample, code))
        event_counts[str(code)] += 1
    segments = task_segments(events)
    cue_delays = np.asarray([(x["start"] - x["cue"]) / fs for x in segments])
    if not len(cue_delays) or np.any(np.abs(cue_delays - 1.5) > 0.1):
        raise ValueError(f"Unexpected cue/task timing, expected approximately 1.5 seconds: {path}, {cue_delays.tolist()}")
    labels_before = Counter(x["label"] for x in segments)
    if len(segments) != spec["expected_trials_per_run"] or any(
        labels_before[c] != spec["expected_trials_per_class_per_run"] for c in (0, 1)
    ):
        raise ValueError(f"Unexpected task count in {path}: {len(segments)}, {labels_before}")
    data = raw.get_data(picks=list(range(22))) * spec["mne_numeric_to_microvolt"]
    if not np.isfinite(data).all():
        raise ValueError(f"Nonfinite signal: {path}")
    sos = butter(4, spec["filter_hz"], btype="bandpass", fs=fs, output="sos")
    filtered = sosfilt(sos, data, axis=1)
    n_samples = spec["epoch_samples"]
    rows, covs, labels, ids = [], [], [], []
    for i, segment in enumerate(segments):
        start, end = segment["start"], segment["end"]
        trial_id = f"{relative}:trial-{i:02d}"
        include = end - start >= n_samples
        rows.append({"trial_id": trial_id, **segment,
                     "duration_seconds": (end - start) / fs,
                     "cue_to_task_seconds": (start - segment["cue"]) / fs,
                     "included": include, "exclusion_reason": "" if include else "task_shorter_than_one_second"})
        if not include:
            continue
        stop = start + n_samples
        if start < 0 or stop > filtered.shape[1] or stop > end:
            raise ValueError(f"Epoch boundary error: {trial_id}")
        epoch = filtered[:, start:stop].copy()
        epoch -= epoch.mean(axis=1, keepdims=True)
        covariance = epoch @ epoch.T / (n_samples - 1)
        if not np.isfinite(covariance).all() or np.trace(covariance) <= 0:
            raise ValueError(f"Invalid covariance: {trial_id}")
        covs.append(covariance)
        labels.append(segment["label"])
        ids.append(trial_id)
    # Header summaries make the original malformed unit scaling inspectable.
    header = raw._raw_extras[0]
    header_fields = {}
    for key in ("physical_min", "physical_max", "digital_min", "digital_max", "units"):
        if key in header:
            header_fields[key] = np.asarray(header[key]).tolist()
    qa = {"file": relative, "sha256": sha256(path), "channels": raw.ch_names,
          "sfreq": fs, "event_counts": dict(event_counts),
          "cue_to_task_seconds_min_max": [float(cue_delays.min()), float(cue_delays.max())],
          "read_warnings": sorted(set(str(w.message) for w in caught)),
          "gdf_header": header_fields,
          "raw_channel_std_microvolt": np.std(data, axis=1).tolist(),
          "filtered_channel_std_microvolt": np.std(filtered, axis=1).tolist(),
          "trials": rows}
    return covs, labels, ids, qa


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True)
    p.add_argument("--qa-dir", type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text())
    protocol_hash = sha256(args.protocol)
    spec, analysis = protocol["preprocessing"], protocol["analysis"]
    inventory = find_files(args.raw_root)
    expected_keys = {(s, d) for s in analysis["subjects"] for d in analysis["sessions"]}
    if set(inventory) != expected_keys:
        raise ValueError(f"Subject/day inventory mismatch; missing={expected_keys-set(inventory)}, extra={set(inventory)-expected_keys}")
    for (s, d), paths in inventory.items():
        expected = spec["expected_runs_per_session"][d - 1]
        if len(paths) != expected:
            raise ValueError(f"Subject {s}, session {d}: {len(paths)} runs, expected {expected}")
    args.qa_dir.mkdir(parents=True, exist_ok=True)
    session_rows, file_rows = [], []
    for subject, session in sorted(inventory):
        covs, labels, ids = [], [], []
        session_qa = []
        for path in inventory[subject, session]:
            relative = str(path.relative_to(args.raw_root))
            c, y, trial_ids, qa = extract_run(path, relative, spec)
            covs.extend(c); labels.extend(y); ids.extend(trial_ids)
            file_rows.append(qa); session_qa.extend(qa["trials"])
        counts = Counter(labels)
        if any(counts[c] < analysis["min_trials_per_class"] for c in (0, 1)):
            raise ValueError(f"Insufficient retained trials: subject={subject}, session={session}, counts={counts}")
        out = args.cache_root / f"sub-{subject:02d}" / f"session-{session:02d}_cov.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            raise FileExistsError(f"Refusing to overwrite cache: {out}")
        np.savez_compressed(out, covs=np.asarray(covs, dtype=np.float64),
                            labels=np.asarray(labels, dtype=int), trial_ids=np.asarray(ids),
                            protocol_sha256=protocol_hash, unit="microvolt_squared",
                            channels=np.asarray(spec["eeg_channels"]))
        row = {"subject": subject, "raw_subject": subject if subject < 10 else subject + 1,
               "session": session, "group": "GR" if subject < 10 else "PAR",
               "runs": len(inventory[subject, session]), "total_trials": len(session_qa),
               "retained_trials": len(labels), "left_trials": counts[0], "right_trials": counts[1],
               "excluded_short": sum(not t["included"] for t in session_qa),
               "minimum_task_seconds": min(t["duration_seconds"] for t in session_qa),
               "cache_sha256": sha256(out)}
        session_rows.append(row)
        print(json.dumps(row), flush=True)
    manifest = {"status": "complete", "protocol_sha256": protocol_hash,
                "preprocessor_sha256": sha256(__file__), "sessions": session_rows, "files": file_rows}
    (args.qa_dir / "preprocessing_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"COMPLETE: {len(session_rows)} sessions, {len(file_rows)} bar runs", flush=True)


if __name__ == "__main__":
    main()
