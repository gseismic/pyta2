# import numpy as np
from .base import rBaseMA
from ...utils.deque import NumPyDeque
from .ema import rEMA

class rTEMA(rBaseMA):
    """TEMA
    
    TODO:
        - [ ] 待校验 

    reference:
        https://en.wikipedia.org/wiki/Triple_exponential_moving_average
        [TEMA信号](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/triple-exponential-moving-average-tema)
    """
    name = "TEMA"

    def __init__(self, n, **kwargs): 
        assert n > 0, f'{self.name} n must be greater than 0, got {n}'
        self.__ema = rEMA(n) 
        self.values_ema = NumPyDeque(n) 

        self.__ema_ema = rEMA(n)
        self.values_ema_ema = NumPyDeque(n)

        self.__ema_ema_ema = rEMA(n)
        super(rTEMA, self).__init__(n=n, window=3*n-2, **kwargs)
    
    def reset_extras(self):
        """Reset all internal EMA calculators and their value queues."""
        self.values_ema.clear()
        self.__ema.reset()
        self.values_ema_ema.clear()
        self.__ema_ema.reset()
        self.__ema_ema_ema.reset() 

    def forward(self, values):
        """Compute Triple Exponential Moving Average.
        
        Formula: TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
        
        Args:
            values: Input values for calculation
            
        Returns:
            float: TEMA value
        """
        value_ema = self.__ema.rolling(values)
        self.values_ema.push(value_ema)

        value_ema_ema = self.__ema_ema.rolling(self.values_ema.values)
        self.values_ema_ema.push(value_ema_ema)

        value_ema_ema_ema = self.__ema_ema_ema.rolling(self.values_ema_ema.values)
        return 3.0*value_ema - 3.0*value_ema_ema + value_ema_ema_ema