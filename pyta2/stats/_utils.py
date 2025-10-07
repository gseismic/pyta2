from math import floor, ceil
import numpy as np


def pct_rank(x, pct):
    """
    pct: 0.5: median
    pct: 0 -> min(x)
    pct: 1 -> max(x)
    if pct == 0.5: Act as np.median
    """
    assert(pct >= 0 and pct <= 1.0)
    num = len(x)
    if num == 0:
        return np.nan
    assert(num >= 1)
    # ok for np.nan
    y = sorted(x)
    fpos = (num-1.0)*pct
    pos0 = floor(fpos)
    pos1 = ceil(fpos)
    if pos0 == pos1:  # pos1 += 1, if no considering boundary
        v = y[pos0]
    else:
        v = (pos1-fpos)*y[pos0] + (fpos-pos0)*y[pos1]
    return v
