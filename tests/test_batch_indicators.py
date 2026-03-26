import pytest
import numpy as np

# 导入所有 Batch 方法
from pyta2.momentum._batch import ROC, Vel, Bias, CCI, WR, KDJ, TSI, UO, MACD, RSI
from pyta2.trend._batch import DMI, BBI, BOP, ParaSAR, SMA, EMA, WMA, HMA, DEMA, TEMA, KAMA, ZLEMA
from pyta2.volume._batch import VWAP, OBV, MFI, CMF, EMV, VPT
from pyta2.structure._batch import Boll, DC, Keltner, Ichimoku, ZigZag, NTV
from pyta2.relation._batch_ import TwinCross, MACross, Corr
from pyta2.stats._batch_ import ZScore, ER, WMean, WStd, WVar, WSkew, WKurt, WQuantile

@pytest.fixture
def test_data():
    """ 生成测试数据。 """
    np.random.seed(42)
    n = 150
    closes = np.random.randn(n).cumsum() + 100
    highs  = closes + np.random.rand(n) * 2
    lows   = closes - np.random.rand(n) * 2
    opens  = closes + np.random.randn(n) * 0.5
    vols   = np.random.rand(n) * 1000000
    return {
        'opens': opens,
        'highs': highs,
        'lows': lows,
        'closes': closes,
        'vols': vols
    }

def test_momentum_batch(test_data):
    c = test_data['closes']
    h = test_data['highs']
    l = test_data['lows']
    o = test_data['opens']
    
    assert len(ROC(c, 10)) == len(c)
    assert len(Vel(c, 5)) == len(c)
    assert len(Bias(c, 10)) == 2 # tuple
    assert len(CCI(h, l, c, 20)) == len(c)
    assert len(WR(h, l, c, 14)) == len(c)
    assert len(KDJ(h, l, c, 9, 3, 3)) == 3 # tuple
    assert len(TSI(o, h, l, c, 25)) == 3 # tuple
    assert len(UO(h, l, c, 7, 14, 28)) == len(c)
    assert len(MACD(c, 12, 26, 9)) == 3 # tuple
    assert len(RSI(c, 14)) == len(c)


def test_trend_batch(test_data):
    c = test_data['closes']
    h = test_data['highs']
    l = test_data['lows']
    o = test_data['opens']
    
    for func in [SMA, EMA, WMA, HMA, DEMA, TEMA, KAMA, ZLEMA]:
        assert len(func(c, 10)) == len(c)

    assert len(DMI(h, l, c, 14, 14)) == 3
    assert len(BBI(c, 3, 6, 12, 24)) == len(c)
    assert len(BOP(o, h, l, c, 20)) == len(c)
    assert len(ParaSAR(h, l, 0.02, 0.2)) == 2


def test_volume_batch(test_data):
    c = test_data['closes']
    h = test_data['highs']
    l = test_data['lows']
    v = test_data['vols']
    
    assert len(VWAP(h, l, c, v, 20)) == len(c)
    assert len(OBV(c, v)) == len(c)
    assert len(MFI(h, l, c, v, 14)) == len(c)
    assert len(CMF(h, l, c, v, 21)) == len(c)
    assert len(EMV(h, l, v, 14)) == len(c)
    assert len(VPT(c, v)) == len(c)


def test_stats_batch(test_data):
    c = test_data['closes']
    assert len(ZScore(c, 20)) == 3
    assert len(ER(c, 10)) == len(c)
    for func in [WMean, WStd, WVar, WSkew, WKurt]:
        assert len(func(c, 20)) == len(c)
    assert len(WQuantile(c, 20, 0.5)) == len(c)


def test_channel_batch(test_data):
    c = test_data['closes']
    h = test_data['highs']
    l = test_data['lows']
    assert len(DC(h, l, 20)) == 3
    assert len(Boll(c, 20, 2)) == 3
    assert len(Keltner(h, l, c, 20, 2)) == 3


def test_extra_batch(test_data):
    c = test_data['closes']
    h = test_data['highs']
    l = test_data['lows']
    o = test_data['opens']
    v = test_data['vols']
    
    # NTV
    res = NTV(o, h, l, c, v)
    assert len(res) == 8
    
    # Ichimoku
    res = Ichimoku(c)
    assert len(res) == 4
    
    # ZigZag
    res = ZigZag(c, delta=0.01)
    assert len(res) == 4
    assert np.any(res[0] != -1)
    
    # Corr
    c2 = c + np.random.randn(len(c))
    res = Corr(c, c2, 20)
    assert len(res) == len(c)
    
    # MACross
    res = MACross(c, 5, 10)
    assert len(res) == 3
