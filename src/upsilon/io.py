"""Reading the LHCb dimuon binary files, the event selection, and output paths.

Each file is a flat stream of little-endian float32 values, six per muon pair:
(mass, pt, eta, p, pt1, pt2). They are exposed as a numpy structured array so
columns can be addressed by name: ``events["mass"]``.
"""
from __future__ import annotations

import json
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


def output_subdir(dataset: str) -> str:
    """Subfolder of ``FIGURE_DIR`` and ``RESULT_DIR`` for a dataset: its short name
    ("small", "full", "mc"), or the file stem when a path to a .bin file is given."""
    return Path(dataset).stem

#: Short names for the provided datasets.
DATASETS = {
    "full": "ups-15.bin",
    "small": "ups-15-small.bin",
    "mc": "mc.bin",
}


def load_events(path: str | Path) -> np.ndarray:
    """Read a binary event file into a structured array with fields ``COLUMNS``."""
    data = np.fromfile(path, dtype="<f4")
    if data.size % len(COLUMNS):
        raise ValueError(f"{path}: {data.size} floats is not a multiple of {len(COLUMNS)}")
    return data.view(DTYPE)


def save_json(obj, name: str, dataset: str) -> Path:
    """Write ``obj`` to ``output/results/<dataset>/<name>`` (numpy arrays and numbers become lists and floats)."""
    path = RESULT_DIR / output_subdir(dataset) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
    return path


def load(dataset: str | Path = "small") -> np.ndarray:
    """Load a dataset by short name (``"full"``, ``"small"``, ``"mc"``) or by path."""
    path = DATA_DIR / DATASETS[dataset] if dataset in DATASETS else Path(dataset)
    return load_events(path)


# Selection: the softer muon's transverse momentum, min(pt1, pt2) >= MIN_MUON_PT [GeV/c].
# Choice from scripts/03_selection.py
MIN_MUON_PT = 3.0


def select_muons(events: np.ndarray, cut: float = MIN_MUON_PT) -> np.ndarray:
    """Events whose softer muon has pT >= ``cut`` (the analysis selection; apply to data and MC alike)."""
    return events[np.minimum(events["pt1"], events["pt2"]) >= cut]


def rapidity(events: np.ndarray) -> np.ndarray:
    """Dimuon rapidity y = 0.5 ln((E + p_z) / (E - p_z)), with E = sqrt(p^2 + m^2) and
    p_z = sqrt(p^2 - pT^2) (LHCb is forward, so p_z > 0). The ``eta`` column is the
    pseudorapidity, which for a massive dimuon is ~1 unit larger; JHEP06(2013)064 bins in y.
    """
    m, p, pt = (events[c].astype(float) for c in ("mass", "p", "pt"))
    pz = np.sqrt(np.clip(p**2 - pt**2, 0.0, None))
    e = np.sqrt(p**2 + m**2)
    return 0.5 * np.log((e + pz) / (e - pz))


def sideband_subtract(mass: np.ndarray, window: tuple[float, float], sidebands: tuple[tuple[float, float], ...],
                      weights: np.ndarray | None = None) -> tuple[float, float]:
    """Signal and background sums of ``weights`` (default: counts) in the mass ``window``,
    without a fit: the background under the window is the sideband sum scaled by
    window width / total sideband width, exact for a background linear in mass when the
    sidebands sit symmetrically around the window.
    """
    w = np.ones(len(mass)) if weights is None else np.asarray(weights, dtype=float)
    in_window = (mass >= window[0]) & (mass < window[1])
    in_side = np.zeros(len(mass), dtype=bool)
    for lo, hi in sidebands:
        in_side |= (mass >= lo) & (mass < hi)
    scale = (window[1] - window[0]) / sum(hi - lo for lo, hi in sidebands)
    background = scale * w[in_side].sum()
    return w[in_window].sum() - background, background
