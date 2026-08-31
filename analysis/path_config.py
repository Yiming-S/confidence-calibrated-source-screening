"""Portable path resolution shared by the EEG analysis entry points."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


EEG_DATA_ROOT_ENV = "EEG_DATA_ROOT"


def _normalized(path: Path | str) -> Path:
    return Path(path).expanduser()


def resolve_data_root(
    cli_value: Path | None,
    parser: argparse.ArgumentParser,
    *,
    env_suffix: str | Path | None = None,
    option_name: str = "--data-root",
) -> Path:
    """Resolve an EEG input path as CLI > EEG_DATA_ROOT > parser error.

    ``env_suffix`` is appended only to the environment-variable value. This
    preserves the established CLI contract for scripts whose ``--data-root``
    already denotes a dataset-specific directory.
    """

    if cli_value is not None:
        return _normalized(cli_value)
    env_value = os.environ.get(EEG_DATA_ROOT_ENV)
    if env_value:
        root = _normalized(env_value)
        return root / env_suffix if env_suffix is not None else root
    parser.error(
        f"{option_name} is required unless the {EEG_DATA_ROOT_ENV} "
        "environment variable is set"
    )


def resolve_cache_root(
    cli_value: Path | None,
    parser: argparse.ArgumentParser,
    *,
    env_suffix: str | Path | None = None,
    fallback: Path | None = None,
    option_name: str = "--cache-root",
) -> Path:
    """Resolve a cache path without embedding a machine-specific default."""

    if cli_value is not None:
        return _normalized(cli_value)
    env_value = os.environ.get(EEG_DATA_ROOT_ENV)
    if env_value and env_suffix is not None:
        return _normalized(env_value) / env_suffix
    if fallback is not None:
        return _normalized(fallback)
    parser.error(
        f"{option_name} is required unless the {EEG_DATA_ROOT_ENV} "
        "environment variable is set"
    )
