import numpy as np
from ...utils.deque import NumpyDeque
from .base import rBaseMA
from .ema import rEMA


class rDEMA(rBaseMA):
    """
    Double Exponential Moving Average (DEMA)
    DEMA = 2 * EMA - EMA(EMA)
    """
    name = "DEMA"

    def __init__(self, n, **kwargs):
        assert n > 0, f'{self.name} n must be greater than 0, got {n}'
        self.__ema = rEMA(n)
        self.__ema_ema = rEMA(n)
        self.values_ema = NumpyDeque(n)
        # Match test_dema.py: Window = 2*n - 1
        super(rDEMA, self).__init__(n=n, window=2 * n - 1, **kwargs)

    def reset_extras(self):
        self.__ema.reset()
        self.__ema_ema.reset()
        self.values_ema.clear()

    def forward(self, values: np.ndarray):
        if not isinstance(values, np.ndarray):
            values = np.array([values])
            
        ema1 = self.__ema.rolling(values)
        self.values_ema.append(ema1)
        ema2 = self.__ema_ema.rolling(self.values_ema.values)
        
        return 2 * ema1 - ema2