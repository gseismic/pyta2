import numpy as np
from ..base import rIndicator
from ..base.schema import Schema
from ..utils.space.box import Scalar

__all__ = ['rZScore']

class rZScore(rIndicator):
    name = 'ZScore'

    def __init__(self, n, **kwargs):
        assert n >= 2, f'{self.name} window must be greater than 2, got {n}'
        self.n = n 
        super(rZScore, self).__init__(
            window=self.n,
            schema=Schema([
                ('zscore', Scalar(dtype=np.float64)),
                ('mu', Scalar(dtype=np.float64)),
                ('std', Scalar(dtype=np.float64))
            ]),
            **kwargs
        )

    def reset_extras(self):
        pass

    def forward(self, values):
        if len(values) < self.required_window:
            return np.nan, np.nan, np.nan
        
        mu = np.mean(values[-self.n:], axis=-1)
        std = np.std(values[-self.n:], axis=-1)
        zscore = (values[-1] - mu) / std
        # print(f'\t{values[-1]=}\t{mu=}\t{std=}\t{zscore=}')
        return zscore, mu, std

    @property
    def full_name(self):
        return f'ZScore({self.n})'
    
    
