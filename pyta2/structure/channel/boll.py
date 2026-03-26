import numpy as np
from pyta2.base.indicator import rIndicator
from pyta2.utils.space.box import Box

class rBoll(rIndicator):
    name = "Boll"

    def __init__(self, n=20, F=2, **kwargs):
        assert n > 0, f'{self.name} n must be greater than 0, got {n}'
        assert F > 0, f'{self.name} F must be greater than 0, got {F}'
        self.n = n
        self.F = F
        super(rBoll, self).__init__(
            window=self.n,
            schema=[
                ('ub', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('mid', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('lb', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )
           
    def reset_extras(self):
        pass
       
    def forward(self, values):
        if len(values) < self.n:
            return np.nan, np.nan, np.nan
        mid = np.mean(values[-self.n:])
        std = np.std(values[-self.n:])
        ub = mid + self.F * std
        lb = mid - self.F * std
        return ub, mid, lb
    
    @property
    def full_name(self):
        return f'{self.name}({self.n},{self.F})'
    