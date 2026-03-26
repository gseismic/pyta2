import os
import sys
import numpy as np
import unittest

# 添加路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyta2.momentum.roc import rROC
from pyta2.momentum.vel import rVel
from pyta2.momentum.bias import rBias
from pyta2.momentum.cci import rCCI, rCCIx
from pyta2.momentum.wr import rWR, rWRx
from pyta2.momentum.kdj import rKDJ
from pyta2.momentum.tsi import rTSI

from pyta2.trend.dmi import rDMI
from pyta2.trend.bbi import rBBI
from pyta2.trend.bop import rBOP
from pyta2.trend.para_sar import rParaSAR
from pyta2.trend.ma.kama import rKAMA
from pyta2.trend.ma.zlema import rZLEMA

from pyta2.volume.obv import rOBV
from pyta2.volume.mfi import rMFI
from pyta2.volume.cmf import rCMF

class TestNewIndicators(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.n_points = 50
        self.closes = np.random.randn(self.n_points).cumsum() + 100
        self.highs  = self.closes + np.random.rand(self.n_points) * 2
        self.lows   = self.closes - np.random.rand(self.n_points) * 2
        self.opens  = self.closes + np.random.randn(self.n_points) * 0.5
        self.vols   = np.random.rand(self.n_points) * 1000000

    def test_momentum(self):
        print("\nTesting Momentum Indicators...")
        
        # ROC
        roc = rROC(10)
        val = roc.rolling(self.closes[:11])
        self.assertFalse(np.isnan(val))
        print(f"rROC(10) @11: {val:.4f}")

        # Vel
        vel = rVel(5)
        val = vel.rolling(self.closes[:6])
        self.assertFalse(np.isnan(val))
        print(f"rVel(5) @6: {val:.4f}")

        # Bias
        bias = rBias(10)
        b, ma = bias.rolling(self.closes[:10])
        self.assertFalse(np.isnan(b))
        print(f"rBias(10) @10: bias={b:.4f}, ma={ma:.4f}")

        # CCI
        ccix = rCCIx(20)
        cci, ma, dev = ccix.rolling(self.highs[:20], self.lows[:20], self.closes[:20])
        self.assertFalse(np.isnan(cci))
        print(f"rCCIx(20) @20: cci={cci:.4f}, ma={ma:.4f}, dev={dev:.4f}")
        
        # KDJ
        kdj = rKDJ(9, 3, 3)
        k, d, j = kdj.rolling(self.highs[:10], self.lows[:10], self.closes[:10])
        self.assertFalse(np.any(np.isnan([k, d, j])))
        print(f"rKDJ(9,3,3) @10: k={k:.4f}, d={d:.4f}, j={j:.4f}")

        # TSI
        tsi = rTSI(14)
        t, u, d = tsi.rolling(self.opens[:14], self.highs[:14], self.lows[:14], self.closes[:14])
        self.assertFalse(np.isnan(t))
        print(f"rTSI(14) @14: tsi={t:.4f}")

    def test_trend(self):
        print("\nTesting Trend Indicators...")
        
        # DMI
        dmi = rDMI(14)
        adx, p, m = dmi.rolling(self.highs[:15], self.lows[:15], self.closes[:15])
        self.assertFalse(np.isnan(adx))
        print(f"rDMI(14) @15: adx={adx:.4f}, +di={p:.4f}, -di={m:.4f}")

        # BBI
        bbi = rBBI(3, 6, 12, 24)
        val = bbi.rolling(self.closes[:24])
        self.assertFalse(np.isnan(val))
        print(f"rBBI @24: {val:.4f}")

        # ParaSAR
        sar = rParaSAR()
        for i in range(10):
            is_long, s_val = sar.rolling(self.highs[:i+3], self.lows[:i+3])
        self.assertIsInstance(is_long, (bool, np.bool_))
        print(f"rParaSAR @12: is_long={is_long}, sar={s_val:.4f}")

        # KAMA
        kama = rKAMA(10, 2, 30)
        val = kama.rolling(self.closes[:30])
        self.assertFalse(np.isnan(val))
        print(f"rKAMA @30: {val:.4f}")

        # ZLEMA
        zlema = rZLEMA(10)
        val = zlema.rolling(self.closes[:10])
        self.assertFalse(np.isnan(val))
        print(f"rZLEMA(10) @10: {val:.4f}")

    def test_volume(self):
        print("\nTesting Volume Indicators...")
        
        # OBV
        obv = rOBV()
        val = obv.rolling(self.closes[:2], self.vols[:2])
        self.assertFalse(np.isnan(val))
        print(f"rOBV @2: {val}")

        # MFI
        mfi = rMFI(14)
        val = mfi.rolling(self.highs[:15], self.lows[:15], self.closes[:15], self.vols[:15])
        self.assertFalse(np.isnan(val))
        print(f"rMFI(14) @15: {val:.4f}")

        # CMF
        cmf = rCMF(21)
        val = cmf.rolling(self.highs[:21], self.lows[:21], self.closes[:21], self.vols[:21])
        self.assertFalse(np.isnan(val))
        print(f"rCMF(21) @21: {val:.4f}")

if __name__ == '__main__':
    unittest.main()
