import numpy as np

from upsilon.io import COLUMNS, load_events


def test_roundtrip(tmp_path):
    rows = np.arange(18, dtype="<f4").reshape(3, 6)
    path = tmp_path / "x.bin"
    rows.tofile(path)
    ev = load_events(path)
    assert len(ev) == 3
    assert ev.dtype.names == COLUMNS
    np.testing.assert_array_equal(ev["mass"], rows[:, 0])
    np.testing.assert_array_equal(ev["pt2"], rows[:, 5])
