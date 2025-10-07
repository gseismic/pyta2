from .base import rBaseMA
from ...utils.deque import NumPyDeque
from .ema import rEMA

class rDEMA(rBaseMA):
    """Double Exponential Moving Average (DEMA)
    
    DEMA = 2 * EMA(price) - EMA(EMA(price))
    
    The DEMA reduces the inherent lag in traditional moving averages.
    It gives more weight to recent prices while maintaining smoothness.
    
    Args:
        n (int): The period/window size for the EMA calculations
        
    References:
        https://en.wikipedia.org/wiki/Double_exponential_moving_average
    """
    name = "DEMA"

    def __init__(self, n, **kwargs):
        assert n > 0, f'{self.name} n must be greater than 0, got {n}'
        self.__ema = rEMA(n)  # First EMA
        self.values_ema = NumPyDeque(n)  # Buffer for first EMA values
        self.__ema_ema = rEMA(n)  # Second EMA
        
        # DEMA requires 2n-1 periods to be fully initialized:
        # - First EMA needs n periods
        # - Second EMA needs additional n-1 periods
        super(rDEMA, self).__init__(n=n, window=2*n-1, **kwargs)
    
    def reset_extras(self):
        self.values_ema.clear()
        self.__ema.reset()
        self.__ema_ema.reset()

    def forward(self, values):
        """
        Ema + (Ema - Ema(Ema)) = 2*Ema - Ema(Ema)
        """
        # Ema + (Ema - Ema(Ema)) = 2*Ema - Ema(Ema)
        # window = sum(invalid) +1 = n-1 + n-1 +1 = 2n-1
        value_ema = self.__ema.rolling(values) 
        self.values_ema.push(value_ema) 
        
        # Calculate EMA of EMA
        value_ema_ema = self.__ema_ema.rolling(self.values_ema.values)
        
        # DEMA = 2 * EMA(price) - EMA(EMA(price))
        return 2.0 * value_ema - value_ema_ema