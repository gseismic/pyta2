import numpy as np
from ..base.indicator import rIndicator
from ..utils.space.box import Box


class IndicatorFunc1Call(rIndicator):
    """Base class for simple rolling statistics with a single output"""
    
    def __init__(self, callback, n, name='Stat', **kwargs):
        self.callback = callback
        self.n = n
        super(IndicatorFunc1Call, self).__init__(
            window=n,
            schema=[
                (name.lower(), Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64))
            ],
            **kwargs
        )
    
    def reset_extras(self):
        pass

    def forward(self, values):
        # We assume callback handles slicing/length checks if needed,
        # or we check window here.
        if len(values) < self.n:
            return np.nan
        return self.callback(values)

    @property
    def full_name(self):
        return f'{self.name}({self.n})'


class IndicatorFunc2Call(rIndicator):
    """Base class for statistics involving two series"""
    
    def __init__(self, callback, n, name='Stat', **kwargs):
        self.callback = callback
        self.n = n
        super(IndicatorFunc2Call, self).__init__(
            window=n,
            schema=[
                (name.lower(), Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64))
            ],
            **kwargs
        )
    
    def reset_extras(self):
        pass

    def forward(self, values1, values2):
        if len(values1) < self.n or len(values2) < self.n:
            return np.nan
        return self.callback(values1, values2)

    @property
    def full_name(self):
        return f'{self.name}({self.n})'


IndicatorFunc1Call_XXX = IndicatorFunc1Call
IndicatorFunc2Call_XXX = IndicatorFunc2Call