import numpy as np
from ...utils.space import Box
from ...utils.deque import NumPyDeque
from ...trend.ma.api import get_ma_class
from ...base import rIndicator

class rVWAP(rIndicator):
    """VWAP

    """
    name = "VWAP"

    def __init__(self, n, ma_type='SMA', **kwargs):
        if n < 1:
            raise ValueError(f'{self.name} n must be greater than 0, got {n}')
        
        self.n = n
        self.ma_type = ma_type
        self.fn_ma_value = get_ma_class(ma_type)(n)
        self.fn_ma_volume = get_ma_class(ma_type)(n)
        self.values = NumPyDeque(2 * self.fn_ma_value.window)
        super(rVWAP, self).__init__(
            window=self.n,
            num_outputs=1,
            output_keys=['vwap'],
            output_dtypes=[np.float64],
            output_spaces=[
                Box(low=0, high=np.inf, shape=(), dtype=np.float64),
            ],
            **kwargs
        )
           
    def reset_extras(self):
        self.fn_ma_value.reset()
        self.fn_ma_volume.reset()
        self.values.clear()
       
    def forward_ready(self, highs, lows, closes, volumes) -> bool:
        return len(highs) >= self.required_window
    
    def pre_forward(self, highs, lows, closes, volumes):
        tp = (highs[-1] + lows[-1] + closes[-1])/3.0
        self.values.push(tp * volumes[-1])
    
    def safe_forward(self, highs, lows, closes, volumes):
        tp = (highs[-1] + lows[-1] + closes[-1])/3.0
        self.values.push(tp * volumes[-1])
        
        avg_value = self.fn_ma_value.rolling(self.values.values)
        avg_volume = self.fn_ma_volume.rolling(volumes)
        
        vwap = avg_value / avg_volume
        # _vsum = np.sum(volumes[-self.n:])
        # if _vsum == 0:
        #     vwap = np.nan
        # else:
        #     vwap = np.dot(volumes[-self.n:], self.typicals.values)/_vsum
        return vwap
    
    @property
    def null_output(self):
        return np.nan
    
    @property
    def full_name(self):
        return f'{self.name}({self.n},{self.ma_type})'
    