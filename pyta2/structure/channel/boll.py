import numpy as np
from ...base import rIndicator
from ...base.schema import Schema
from ...utils.space.box import Scalar

class rBoll(rIndicator):
    name = "Boll"

    def __init__(self, n=20, F=2, **kwargs):
        if n < 1:
            raise ValueError(f'{self.name} n must be greater than 0, got {n}')
        if F <= 0:
            raise ValueError(f'{self.name} F must be greater than 0, got {F}')
        self.n = n
        self.F = F
        super(rBoll, self).__init__(
            window=self.n,
            schema=Schema([
                ('ub', Scalar()),
                ('mid', Scalar()),
                ('lb', Scalar()),
            ]),
            **kwargs
        )
           
    def reset_extras(self):
        pass
       
    def forward(self, values):
        if len(values) < self.required_window:
            return np.nan, np.nan, np.nan
        mid = np.mean(values[-self.n:])
        std = np.std(values[-self.n:])
        ub = mid + self.F * std
        lb = mid - self.F * std
        return ub, mid, lb
    
    @property
    def full_name(self):
        return f'{self.name}({self.n},{self.F})'
    