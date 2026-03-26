import numpy as np
from ..base.indicator import rIndicator
from ..utils.space.box import Box
from ..utils.deque import NumpyDeque
from ..trend.ma.sma import rSMA


class rEMV(rIndicator):
    """EMV - Ease of Movement (简易波动指标)
    
    衡量股价涨跌的难易程度。

    https://help.ctrader.com/knowledge-base/zh/indicators/volume/ease-of-movement/#_1
    
    Formula:
        Mid_Move = (High + Low)/2 - (PrevHigh + PrevLow)/2
        Box_Ratio = (Volume / 100,000,000) / (High - Low)
        EMV_raw = Mid_Move / Box_Ratio
        EMV = MA(EMV_raw, n)
    """
    name = 'EMV'

    def __init__(self, n=14, **kwargs):
        assert n >= 1
        self.n = n
        self._ma = rSMA(n)
        self._cache = NumpyDeque(maxlen=n)
        super(rEMV, self).__init__(
            window=n,
            schema=[
                ('emv', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        self._ma.reset()
        self._cache.clear()

    def forward(self, highs, lows, vols):
        if len(highs) < 2:
            return np.nan
        
        mid_move = (highs[-1] + lows[-1]) / 2.0 - (highs[-2] + lows[-2]) / 2.0
        rng = highs[-1] - lows[-1]
        
        if rng == 0:
            box_ratio = 1.0 # 避免除零
        else:
            box_ratio = (vols[-1] / 100_000_000.0) / rng
            
        emv_raw = mid_move / box_ratio if box_ratio != 0 else 0.0
        self._cache.append(emv_raw)
        
        if len(self._cache) < self.n:
            return np.nan
            
        return self._ma.rolling(self._cache.values)

    @property
    def full_name(self):
        return f'{self.name}({self.n})'
