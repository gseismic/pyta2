import numpy as np
from .base import rBaseMA

class rWMA(rBaseMA):
    name = "WMA"

    def __init__(self, n, **kwargs):
        assert n > 0, f'{self.name} n must be greater than 0, got {n}'
        _sum = n * (n + 1)/2
        self._weights = np.array([i/_sum for i in range(1, n+1)])
        super(rWMA, self).__init__(n=n, window=n, **kwargs)
    
    def reset_extras(self):
        pass

    def forward(self, values):
        if len(values) < self.n:
            return np.nan
        return np.dot(values[-self.n:], self._weights)
