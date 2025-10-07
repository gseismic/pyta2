# import numpy as np
from .api import get_ma_class
from .base import rBaseMA

class rMA(rBaseMA):
    
    name = 'MA'
    def __init__(self, n, ma_type, **kwargs):
        self.fn_ma = get_ma_class(ma_type)(n)
        super(rMA, self).__init__(
            n=n,
            window=self.fn_ma.window,
            **kwargs
        )
    def reset_extras(self):
        self.fn_ma.reset()
    
    def forward(self, values):
        return self.fn_ma.rolling(values)