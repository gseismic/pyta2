import numpy as np
from ..base.indicator import rIndicator
from ..utils.space.box import Box
from ..utils.deque import NumpyDeque
from ..trend.ma.sma import rSMA


class rCMF(rIndicator):
    """CMF - Chaikin Money Flow

    衡量一段时间内资金流向的指标，范围 [-1, 1]。

    Reference:
        https://www.chaikinanalytics.com/chaikin-money-flow/
    """
    name = 'CMF'

    def __init__(self, n=21, **kwargs):
        assert n >= 1, f'{self.name} n must be >= 1, got {n}'
        self.n = n
        self._mfv_values = NumpyDeque(maxlen=n)
        self._vol_values = NumpyDeque(maxlen=n)
        self._ma_num = rSMA(n)
        self._ma_den = rSMA(n)
        super(rCMF, self).__init__(
            window=n,
            schema=[
                ('cmf', Box(low=-1, high=1, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        self._mfv_values.clear()
        self._vol_values.clear()
        self._ma_num.reset()
        self._ma_den.reset()

    def forward(self, highs, lows, closes, volumes):
        rng = highs[-1] - lows[-1]
        mult = (2 * closes[-1] - (highs[-1] + lows[-1])) / rng if rng != 0 else 0.0
        mfv = mult * volumes[-1]
        
        self._mfv_values.append(mfv)
        self._vol_values.append(volumes[-1])
        
        if len(self._mfv_values) < self.n:
            return np.nan
            
        sum_mfv = self._ma_num.rolling(self._mfv_values.values)
        sum_vol = self._ma_den.rolling(self._vol_values.values)
        
        if sum_vol == 0:
            return 0.0
        return sum_mfv / sum_vol

    @property
    def full_name(self):
        return f'{self.name}({self.n})'
