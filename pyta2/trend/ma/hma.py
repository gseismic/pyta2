import numpy as np
from ...utils.deque import NumpyDeque
from .base import rBaseMA
from .wma import rWMA


class rHMA(rBaseMA):
    '''
    HMA(n) = WMA(2 * WMA(close, n/2) - WMA(close, n), sqrt(n))
    
    ref:
        https://school.stockcharts.com/doku.php?id=technical_indicators:hull_moving_average
    '''
    name = "HMA"

    def __init__(self, n, **kwargs):
        assert n > 1, f'{self.name} n must be greater than 1, got {n}'
        self.n1 = n // 2
        self.n2 = int(np.sqrt(n))
        self.fn_wma1 = rWMA(self.n1)
        self.fn_wma2 = rWMA(n)
        self.fn_wma3 = rWMA(self.n2)
        self.__raw_hma = NumpyDeque(n)
        # Dedicated test expects Window = (n-1) + n2
        window = max(self.n1 - 1, n - 1) + self.n2
        super(rHMA, self).__init__(n=n, window=window, **kwargs)

    def reset_extras(self):
        self.__raw_hma.clear()
        self.fn_wma1.reset()
        self.fn_wma2.reset()
        self.fn_wma3.reset()

    def _compute_raw_hma(self, values):
        wma1 = self.fn_wma1.rolling(values)
        wma2 = self.fn_wma2.rolling(values)
        raw = 2 * wma1 - wma2
        self.__raw_hma.append(raw)
        
    def forward(self, values: np.ndarray):
        # Use full history slice provided by rolling apply
        self._compute_raw_hma(values)
        return self.fn_wma3.rolling(self.__raw_hma.values)
