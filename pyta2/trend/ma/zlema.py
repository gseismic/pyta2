import numpy as np
from pyta2.base.indicator import rIndicator
from pyta2.utils.space.box import Box
from pyta2.utils.deque import NumpyDeque
from .ema import rEMA


class rZLEMA(rIndicator):
    """ZLEMA - Zero Lag EMA

    通过减去滞后分量实现的零滞后指数移动平均线。

    Reference:
        https://en.wikipedia.org/wiki/Zero_lag_exponential_moving_average
    """
    name = 'ZLEMA'

    def __init__(self, n, **kwargs):
        assert n >= 1, f'{self.name} n must be >= 1, got {n}'
        self.n = n
        self._ema1 = rEMA(n)
        self._ema2 = rEMA(n)
        self._ema3 = rEMA(n)
        self._ema4 = rEMA(n)
        self._ema5 = rEMA(n)
        
        self._val_ema1 = NumpyDeque(maxlen=n * 3)
        self._val_ema2 = NumpyDeque(maxlen=n * 3)
        self._val_ema3 = NumpyDeque(maxlen=n * 3)
        self._val_ema4 = NumpyDeque(maxlen=n * 3)
        
        super(rZLEMA, self).__init__(
            window=n,
            schema=[
                ('zlema', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        self._ema1.reset()
        self._ema2.reset()
        self._ema3.reset()
        self._ema4.reset()
        self._ema5.reset()
        self._val_ema1.clear()
        self._val_ema2.clear()
        self._val_ema3.clear()
        self._val_ema4.clear()

    def forward(self, values):
        # 参照原版 elif 1 的逻辑实现（5级深度补偿）
        v_ema1 = self._ema1.rolling(values)
        self._val_ema1.append(v_ema1)
        
        v_ema2 = self._ema2.rolling(self._val_ema1.values)
        v2 = v_ema1 + (v_ema1 - v_ema2)
        self._val_ema2.append(v2)
        
        v_ema3 = self._ema3.rolling(self._val_ema2.values)
        v3 = v2 + (v2 - v_ema3)
        self._val_ema3.append(v3)
        
        v_ema4 = self._ema4.rolling(self._val_ema3.values)
        v4 = v3 + (v3 - v_ema4)
        self._val_ema4.append(v4)
        
        v_ema5 = self._ema5.rolling(self._val_ema4.values)
        v5 = v4 + (v4 - v_ema5)
        
        return v5

    @property
    def full_name(self):
        return f'{self.name}({self.n})'
