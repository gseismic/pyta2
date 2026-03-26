import pytest
import numpy as np

# 导入所有实现的新指标
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

@pytest.fixture
def sample_data():
    """ 生成测试数据。 """
    np.random.seed(42)
    n = 200 # 增加数据长度以支持多层级联指标
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

def run_incremental(ind, *data_streams, steps=100):
    """ 模拟逐点调用。 """
    res = None
    if steps is None:
        steps = len(data_streams[0])
    for i in range(1, steps + 1):
        args = [d[:i] for d in data_streams]
        res = ind.rolling(*args)
    return res

def test_roc(sample_data):
    roc = rROC(10)
    val = roc.rolling(sample_data['closes'][:11])
    assert not np.isnan(val)

def test_vel(sample_data):
    vel = rVel(5)
    val = vel.rolling(sample_data['closes'][:6])
    assert not np.isnan(val)

def test_bias(sample_data):
    bias = rBias(10)
    val = run_incremental(bias, sample_data['closes'], steps=20)
    b, ma = val
    assert not np.isnan(b)

def test_cci(sample_data):
    ccix = rCCIx(20)
    val = run_incremental(ccix, sample_data['highs'], sample_data['lows'], sample_data['closes'], steps=40)
    cci, ma, dev = val
    assert not np.isnan(cci)

def test_kdj(sample_data):
    kdj = rKDJ(9, 3, 3)
    val = run_incremental(kdj, sample_data['highs'], sample_data['lows'], sample_data['closes'], steps=30)
    k, d, j = val
    assert not any(np.isnan([k, d, j]))

def test_tsi(sample_data):
    tsi = rTSI(14)
    val = run_incremental(tsi, sample_data['opens'], sample_data['highs'], sample_data['lows'], sample_data['closes'], steps=50)
    t, u, d = val
    assert not np.isnan(t)

def test_dmi(sample_data):
    dmi = rDMI(14)
    val = run_incremental(dmi, sample_data['highs'], sample_data['lows'], sample_data['closes'], steps=60)
    adx, p, m = val
    assert not np.isnan(adx)

def test_bbi(sample_data):
    bbi = rBBI(3, 6, 12, 24)
    val = bbi.rolling(sample_data['closes'][:25])
    assert not np.isnan(val)

def test_para_sar(sample_data):
    sar = rParaSAR()
    val = run_incremental(sar, sample_data['highs'], sample_data['lows'], steps=20)
    is_long, s_val = val
    assert isinstance(is_long, (bool, np.bool_))
    assert not np.isnan(s_val)

def test_kama(sample_data):
    kama = rKAMA(10, 2, 30)
    val = run_incremental(kama, sample_data['closes'], steps=50)
    assert not np.isnan(val)

def test_zlema(sample_data):
    zlema = rZLEMA(10)
    val = run_incremental(zlema, sample_data['closes'], steps=100)
    assert not np.isnan(val)

def test_obv(sample_data):
    obv = rOBV()
    val = run_incremental(obv, sample_data['closes'], sample_data['vols'], steps=5)
    assert not np.isnan(val)

def test_mfi(sample_data):
    mfi = rMFI(14)
    val = run_incremental(mfi, sample_data['highs'], sample_data['lows'], sample_data['closes'], sample_data['vols'], steps=30)
    assert not np.isnan(val)

def test_cmf(sample_data):
    cmf = rCMF(21)
    # 使用 run_incremental 确保内部 NumpyDeque 预热
    val = run_incremental(cmf, sample_data['highs'], sample_data['lows'], sample_data['closes'], sample_data['vols'], steps=40)
    assert not np.isnan(val)

def test_wr(sample_data):
    # WR 推荐通过 WRx 间接测试点位计算逻辑
    wr = rWR(14)
    val = run_incremental(wr, sample_data['highs'], sample_data['lows'], sample_data['closes'], steps=30)
    assert not np.isnan(val)

def test_bop(sample_data):
    bop = rBOP(20) # 传入参数 n
    val = run_incremental(bop, sample_data['opens'], sample_data['highs'], sample_data['lows'], sample_data['closes'], steps=30)
    assert not np.isnan(val)
