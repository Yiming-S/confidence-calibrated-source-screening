"""Integrity tests for the public repository snapshot."""

from __future__ import annotations

import base64
import json
import py_compile
import re
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve()

DISCLOSURE = base64.b64decode(
    "QUktYXNzaXN0ZWQgdG9vbHMgd2VyZSB1c2VkIGZvciBsYW5ndWFnZSBlZGl0aW5nIGFu"
    "ZCBhbmFseXNpcy1jb2RlIGRldmVsb3BtZW50IHVuZGVyIHRoZSBhdXRob3JzJyBkaXJl"
    "Y3Rpb247IHRoZSBhdXRob3JzIGRlc2lnbmVkIHRoZSBzdHVkeSwgdmVyaWZpZWQgYWxs"
    "IHJlc3VsdHMsIGFuZCB0YWtlIGZ1bGwgcmVzcG9uc2liaWxpdHkgZm9yIHRoZSBjb250"
    "ZW50Lg=="
).decode("utf-8")

CORE_RESULTS = (
    "results/MANIFEST.csv",
    "results/SHA256SUMS",
    "results/simulations/protocol_metric_summary.csv",
    "results/simulations/target_limited_metric_summary.csv",
    "results/simulations/straddle_summary.csv",
    "results/eeg/ma2020/screening_summary.csv",
    "results/eeg/stieger2021/screening_summary.csv",
    "results/eeg/zhou2020/screening_summary.csv",
    "results/eeg/bnci2014_004/screening_summary.csv",
    "results/eeg/noninferiority/summary.json",
    "results/figure_data/eeg_overview.csv",
    "figures/fig_shared_target_evidence.pdf",
    "figures/fig_eeg_overview.pdf",
    "figures/fig_zhou2020_exclusion_certificate.pdf",
)

FIGURE3_METADATA_CANDIDATES = (
    "results/eeg/empirical_reference/manifest.json",
)

SKIP_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "venv",
}

BINARY_SUFFIXES = {
    ".gz",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".tif",
    ".tiff",
    ".zip",
}


def public_text_files() -> list[tuple[Path, str]]:
    """Return UTF-8 public text, excluding caches and this self-audit file."""

    output: list[tuple[Path, str]] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.resolve() == THIS_FILE:
            continue
        relative = path.relative_to(ROOT)
        if any(part in SKIP_DIRECTORIES for part in relative.parts):
            continue
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        output.append((relative, text))
    return output


class PublicRepositoryTests(unittest.TestCase):
    def test_core_results_exist_and_are_nonempty(self) -> None:
        missing = [name for name in CORE_RESULTS if not (ROOT / name).is_file()]
        self.assertFalse(missing, f"Missing core result files: {missing}")

        empty = [name for name in CORE_RESULTS if (ROOT / name).stat().st_size == 0]
        self.assertFalse(empty, f"Empty core result files: {empty}")

        metadata = [
            ROOT / name
            for name in FIGURE3_METADATA_CANDIDATES
            if (ROOT / name).is_file()
        ]
        self.assertEqual(
            len(metadata),
            1,
            "Exactly one canonical Figure 3 metadata file must be present",
        )
        self.assertGreater(metadata[0].stat().st_size, 0)

    def test_internal_submission_materials_are_absent(self) -> None:
        forbidden_names = {
            "AI_USE.md",
            "experiment_artifacts.md",
            "literature_direction_audit.md",
            "submission_materials.md",
        }
        forbidden_directories = {"archive", "output", "paper", "submission"}

        violations: list[str] = []
        for path in ROOT.rglob("*"):
            relative = path.relative_to(ROOT)
            if any(part in SKIP_DIRECTORIES for part in relative.parts):
                continue
            if path.name in forbidden_names:
                violations.append(str(relative))
            if any(part in forbidden_directories for part in relative.parts):
                violations.append(str(relative))

        self.assertFalse(
            sorted(set(violations)),
            f"Internal or submission-only material found: {sorted(set(violations))}",
        )

    def test_public_text_has_no_private_paths_secrets_or_placeholders(self) -> None:
        absolute_path_patterns = (
            re.compile("/" + "Users" + "/"),
            re.compile("/" + "Volumes" + "/"),
            re.compile("/" + "home" + r"/[^/\s]+/"),
            re.compile("/" + "private" + r"/(?:tmp|var)/"),
            re.compile(r"[A-Za-z]:\\Users\\"),
        )
        secret_patterns = (
            re.compile(r"-----BEGIN (?:RSA|OPENSSH|EC|DSA) PRIVATE KEY-----"),
            re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
            re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
            re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
            re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
            re.compile(
                r"(?i)\b(?:api[_-]?key|access[_-]?token|secret[_-]?key|password)"
                r"\s*[:=]\s*['\"][^'\"]+"
            ),
        )
        internal_markers = (
            re.compile(r"(?i)\bTBD\b"),
            re.compile(r"(?i)\bTODO\b"),
            re.compile(r"(?i)\bFIXME\b"),
            re.compile(r"(?i)submission[- ]workflow draft"),
            re.compile(r"(?i)frontiers " + "submission" + r" blocker"),
            re.compile(r"(?i)reviewer[- ](?:response|requested)"),
            re.compile(r"(?i)(?:a|methods?) reviewer (?:wants|would)"),
            re.compile(r"(?i)internal (?:draft|note|strategy)"),
        )
        email_pattern = re.compile(
            r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE
        )

        violations: list[str] = []
        for path, text in public_text_files():
            for pattern in (*absolute_path_patterns, *secret_patterns, *internal_markers):
                match = pattern.search(text)
                if match:
                    violations.append(f"{path}: {match.group(0)!r}")
            match = email_pattern.search(text)
            if match:
                violations.append(f"{path}: public email address {match.group(0)!r}")

        self.assertFalse(violations, "Public-text hygiene failures:\n" + "\n".join(violations))

    def test_disclosure_is_exactly_once_and_has_no_alternative_statement(self) -> None:
        texts = public_text_files()
        occurrences = sum(text.count(DISCLOSURE) for _, text in texts)
        self.assertEqual(occurrences, 1, "The approved disclosure must appear exactly once")

        alternative_patterns = (
            re.compile(r"(?i)\b" + "A" + r"I[- ](?:assisted|generated|supported|tool|tools|use|disclosure|statement)\b"),
            re.compile(r"(?i)\bgenerative[- ]" + "A" + r"I\b"),
            re.compile(r"(?i)\bartificial " + "intelligence" + r"\b"),
            re.compile(
                r"(?i)\b(?:"
                + "Chat"
                + r"GPT|Open"
                + "A"
                + r"I|Code"
                + r"x|GPT[- ]?[0-9]+|large language "
                + r"model|LLM)\b"
            ),
        )

        violations: list[str] = []
        for path, text in texts:
            remainder = text.replace(DISCLOSURE, "")
            for pattern in alternative_patterns:
                match = pattern.search(remainder)
                if match:
                    violations.append(f"{path}: {match.group(0)!r}")

        self.assertFalse(
            violations,
            "Alternative disclosure language found:\n" + "\n".join(violations),
        )

    def test_public_python_files_compile(self) -> None:
        scripts = sorted((ROOT / "analysis").glob("*.py"))
        scripts.extend(sorted((ROOT / "scripts").glob("*.py")))
        self.assertTrue(scripts, "No public Python scripts were found")

        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir)
            failures: list[str] = []
            for index, script in enumerate(scripts):
                try:
                    py_compile.compile(
                        str(script),
                        cfile=str(destination / f"{index:03d}_{script.stem}.pyc"),
                        doraise=True,
                    )
                except py_compile.PyCompileError as error:
                    failures.append(f"{script.relative_to(ROOT)}: {error}")

        self.assertFalse(failures, "Python compilation failures:\n" + "\n".join(failures))

    def test_figure3_is_documented_as_an_archived_output(self) -> None:
        root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
        figure_readme = (ROOT / "figures" / "README.md").read_text(encoding="utf-8")

        self.assertRegex(root_readme, r"Zhou2020\s+exclusion\s+certificate")
        self.assertIn("archived output", root_readme.lower())
        self.assertIn("fig_zhou2020_exclusion_certificate.pdf", figure_readme)
        self.assertIn("archived output", figure_readme.lower())

        pdf = ROOT / "figures" / "fig_zhou2020_exclusion_certificate.pdf"
        self.assertTrue(pdf.read_bytes().startswith(b"%PDF-"))

        metadata_path = next(
            ROOT / name
            for name in FIGURE3_METADATA_CANDIDATES
            if (ROOT / name).is_file()
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        certificate = metadata.get("certificate")
        self.assertIsInstance(certificate, dict)
        self.assertGreater(float(certificate["threshold"]), 0.0)
        self.assertGreater(int(certificate["bootstrap_repetitions"]), 0)
        self.assertTrue(set(certificate["retained_sessions"]).issubset(certificate["source_sessions"]))


if __name__ == "__main__":
    unittest.main()
