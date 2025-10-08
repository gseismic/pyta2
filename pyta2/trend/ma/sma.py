import numpy as np
from .base import rBaseMA

class rSMA(rBaseMA):
    name = "SMA"

    def __init__(self, n, **kwargs):
        assert n > 0, f'{self.name} n must be greater than 0, got {n}'
        super(rSMA, self).__init__(n=n, window=n, **kwargs)
    
    def reset_extras(self):
        pass
    
    def forward(self, values):
        if len(values) < self.n:
            return np.nan
        return np.mean(values[-self.n:])
