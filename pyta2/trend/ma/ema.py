import numpy as np
# from ...logger import logger
from .base import rBaseMA
import warnings

class rEMA(rBaseMA):
    name = "EMA"

    def __init__(self, n, **kwargs):
        assert n > 0, f'{self.name} n must be greater than 0, got {n}'
        self.alpha = 2.0 / (n + 1)
        super(rEMA, self).__init__(n=n, window=n, **kwargs)
    
    def reset_extras(self):
        self.ema = None
    
    def forward(self, values):
        """
        EMA = 2.0 / (n + 1) * price + (1 - 2.0 / (n + 1)) * EMA(previous)
        """
        # 允许 [nan, nan, ...,1, 1,1 ...]
        if len(values) < self.n:
            return np.nan
        
        if self.ema is None or np.isnan(self.ema):
            self.ema = np.mean(values[-self.n:])
            return self.ema
        
        value = values[-1]
        # 统一不做判断，遇到无效数值会导致后面所有的结果都为无效
        if np.isnan(value):
            warnings.warn(f'{self.name} value should not be nan after {self.ema} is calculated')
        self.ema = (value - self.ema) * self.alpha + self.ema
        return self.ema
