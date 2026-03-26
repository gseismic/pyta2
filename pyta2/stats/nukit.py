# encoding: utf8
import math
import numpy as np


def value_in_pct(x, pct):
    """
    pct: 0 -> min(x)
    pct: 1 -> max(x)
    if pct == 0.5: Act as np.median
    """
    assert(pct >= 0 and pct <= 1.0)
    num = len(x)
    if num == 0:
        return np.nan
    assert(num >= 1)
    y = sorted(x)
    fpos = (num-1.0)*pct
    pos0 = math.floor(fpos)
    pos1 = math.ceil(fpos)
    if pos0 == pos1:  # pos1 += 1, if no considering boundary
        v = y[pos0]
    else:
        v = (pos1-fpos)*y[pos0] + (fpos-pos0)*y[pos1]
    return v


if __name__ == "__main__":
    import unittest
    class TestMain(unittest.TestCase):

        def testPctIn1(self):
            x = np.array([1])
            assert(value_in_pct(x, 0.5) ==  1)
            x = np.array([1, 2])
            assert(value_in_pct(x, 0.5) ==  1.5)
            x = np.array([1, 2, 3])
            assert(value_in_pct(x, 0.5) == 2)

        def testPctIn2(self):
            x = np.array([1, 2, 3, 5, 6, 7])
            assert(value_in_pct(x, 0.5) == 4)
            x = np.array([1, 2, 3, 5, 6, 7])
            assert(value_in_pct(x, 1.0) == 7)
            x = np.array([1, 2, 3, 5, 6, 7])
            assert(value_in_pct(x, 0.0) == 1)
            x = np.array([0, 1])
            assert(value_in_pct(x, 0.1) == 0.1)
            x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            assert(value_in_pct(x, 0.5) == 5)
            assert(value_in_pct(x, 0.1) == 1)

        def testPctIn3(self):
            x = np.random.random(size=(50,))
            assert(value_in_pct(x, 0) == np.min(x))
            assert(value_in_pct(x, 1) == np.max(x))
            assert(value_in_pct(x, 0.5) == np.median(x))

    unittest.main()
