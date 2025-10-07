import numpy as np
from ...utils.space import Box
from ...utils.deque import NumPyDeque
from ...trend.ma.api import get_ma_class
from ...base import rIndicator

class rATR(rIndicator):
    """Average True Range (ATR) indicator
    
    Parameters
    ----------
    n : int
        Length of ATR calculation period
    ma_type : str, default 'EMA'
        Type of moving average to use ('SMA', 'EMA', etc.)
        
    Returns
    -------
    atr : float
        ATR value
    """
    name = "ATR"

    def __init__(self, n=20, ma_type='EMA', **kwargs):
        if n < 1:
            raise ValueError(f'{self.name} n must be greater than 0, got {n}')

        self.n = n
        self.ma_type = ma_type
        self.fn_ma = get_ma_class(ma_type)(n)
        self.values_TR = NumPyDeque(maxlen=self.fn_ma.window)
        super(rATR, self).__init__(
            window=self.n,
            num_outputs=1,
            output_keys=['atr'],
            output_dtypes=[np.float64],
            output_spaces=[
                Box(low=0, high=np.inf, shape=(), dtype=np.float64),
            ],
            **kwargs
        )
           
    def reset_extras(self):
        self.fn_ma.reset()
        self.values_TR.clear()
       
    def forward_ready(self, highs, lows, closes) -> bool:
        if not (len(highs) == len(lows) == len(closes)):
            raise ValueError("Input arrays must have same length")
        return len(highs) >= self.required_window
    
    def _cache_data(self, highs, lows, closes):
        high_low = highs[-1] - lows[-1]
        if len(highs) == 1:
            TR = high_low
        else:
            high_close = abs(highs[-1] - closes[-2])
            low_close = abs(lows[-1] - closes[-2])
            TR = max(high_low, high_close, low_close)
        self.values_TR.push(TR)
    
    def pre_forward(self, highs, lows, closes):
        self._cache_data(highs, lows, closes)
    
    def safe_forward(self, highs, lows, closes):
        self._cache_data(highs, lows, closes)
        return self.fn_ma.rolling(self.values_TR.values)
    
    @property
    def null_output(self):
        return np.nan
    
    @property
    def full_name(self):
        return f'{self.name}({self.n},{self.ma_type})'
    