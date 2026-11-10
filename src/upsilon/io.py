"""Reading the LHCb dimuon binary files and locating project paths.

Each file is a flat stream of little-endian float32 values, six per muon pair:
(mass, pt, eta, p, pt1, pt2). They are exposed as a numpy structured array so
columns can be addressed by name: ``events["mass"]``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

COLUMNS = ("mass", "pt", "eta", "p", "pt1", "pt2")
DTYPE = np.dtype([(name, "<f4") for name in COLUMNS])

#: Axis labels (LaTeX) for each observable.
LABELS = {
    "mass": r"$m_{\mu\mu}$ [GeV/$c^2$]",
    "pt": r"$p_\perp$ [GeV/$c$]",
    "eta": r"$\eta$",
    "p": r"$p$ [GeV/$c$]",
    "pt1": r"$p_{\perp,1}$ [GeV/$c$]",
    "pt2": r"$p_{\perp,2}$ [GeV/$c$]",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"
RESULT_DIR = OUTPUT_DIR / "results"

#: Short names for the provided datasets.
DATASETS = {
    "full": "ups-15.bin",
    "small": "ups-15-small.bin",
    "mc": "mc.bin",
}


def load_events(path: str | Path) -> np.ndarray:
    """Read a binary event file into a structured array with fields ``COLUMNS``."""
    raw = np.fromfile(path, dtype="<f4")
    if raw.size % len(COLUMNS):
        raise ValueError(f"{path}: {raw.size} floats is not a multiple of {len(COLUMNS)}")
    return raw.view(DTYPE)


def load(dataset: str | Path = "small") -> np.ndarray:
    """Load a dataset by short name (``"full"``, ``"small"``, ``"mc"``) or by path."""
    path = DATA_DIR / DATASETS[dataset] if dataset in DATASETS else Path(dataset)
    return load_events(path)


def select(events: np.ndarray, **ranges: tuple[float | None, float | None]) -> np.ndarray:
    """Keep events with ``lo <= events[col] < hi`` per ``col=(lo, hi)``; ``None`` is open.

    Example: ``select(events, pt=(2.0, None), eta=(2.0, 4.5))``.
    """
    mask = np.ones(len(events), dtype=bool)
    for col, (lo, hi) in ranges.items():
        if lo is not None:
            mask &= events[col] >= lo
        if hi is not None:
            mask &= events[col] < hi
    return events[mask]
