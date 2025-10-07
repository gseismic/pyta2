import numpy as np
from pyta2.base.indicator import rIndicator
from pyta2.utils.space.box import Box
from abc import abstractmethod

class rBaseMA(rIndicator):
    
    def __init__(self, n, window, **kwargs):
        assert n > 0, f'{self.name} n must be greater than 0, got {n}'
        super(rBaseMA, self).__init__(
            window=window,
            schema=[
                ('ma', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64))
            ],
            **kwargs
        )
        self.n = n
        
    def reset_extras(self):
        pass
    
    @abstractmethod
    def forward(self, values: np.ndarray):
        pass
    
    @property
    def full_name(self):
        return f'{self.name}({self.n})'
    