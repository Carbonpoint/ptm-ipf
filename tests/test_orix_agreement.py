"""Cross-check the colour key against orix, the reference implementation.

orix implements the same EDAX/TSL key as MTEX.  Agreement here is what lets a
ptm-ipf map be compared directly with an EBSD orientation map.
"""

import numpy as np
import pytest

orix_symmetry = pytest.importorskip("orix.quaternion.symmetry")
from orix.plot.direction_color_keys import DirectionColorKeyTSL
from orix.vector import Vector3d

from ptmipf.colorkey import IPFColorKey
from ptmipf.symmetry import get_laue_group

PAIRS = [
    ("m-3m", "Oh"),
    ("6/mmm", "D6h"),
    ("4/mmm", "D4h"),
    ("-3m", "D3d"),
    ("mmm", "D2h"),
]


@pytest.mark.parametrize("ours,theirs", PAIRS)
def test_colors_match_orix(ours, theirs):
    rng = np.random.default_rng(11)
    v = rng.normal(size=(3000, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)

    mine = IPFColorKey(get_laue_group(ours)).direction2color(v)
    reference = DirectionColorKeyTSL(getattr(orix_symmetry, theirs)).direction2color(Vector3d(v))

    assert np.abs(mine - reference).max() < 1e-3
