import numpy as np
from ..base.indicator import rIndicator
from ..utils.space.box import Box


class rZScore(rIndicator):
    """Rolling Z-Score (Standard Score)"""
    name = 'ZScore'

    def __init__(self, n, **kwargs):
        assert n > 1, f'{self.name} n must be greater than 1, got {n}'
        self.n = n 
        super(rZScore, self).__init__(
            window=n,
            schema=[
                ('zscore', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('mu', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('std', Box(low=0.0, high=np.inf, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        pass

    def forward(self, values):
        if len(values) < self.n:
            return np.nan, np.nan, np.nan
        
        subset = values[-self.n:]
        mu = np.mean(subset, axis=-1)
        std = np.std(subset, axis=-1)
        
        if std == 0:
            return 0.0, mu, std
            
        zscore = (values[-1] - mu) / std
        return zscore, mu, std

    @property
    def full_name(self):
        return f'{self.name}({self.n})'
