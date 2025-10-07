import numpy as np
from ...utils.space import Box
from ...utils.deque import NumPyDeque
from ...trend.ma.api import get_ma_class
from ...base import rIndicator

class rDecATR(rIndicator):
    """Decomposed Average True Range (ATR) indicator
    
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
    up_atr : float
        Upward ATR value
    down_atr : float
        Downward ATR value
    """
    name = "DecATR"

    def __init__(self, n=20, ma_type='EMA', **kwargs):
        if n < 1:
            raise ValueError(f'{self.name} n must be greater than 0, got {n}')

        self.n = n
        self.ma_type = ma_type
        self.fn_up_ma = get_ma_class(ma_type)(n)
        self.fn_down_ma = get_ma_class(ma_type)(n)
        self.up_tr_values = NumPyDeque(maxlen=self.fn_up_ma.window*2)
        self.down_tr_values = NumPyDeque(maxlen=self.fn_down_ma.window*2)
        super(rDecATR, self).__init__(
            window=self.n,
            num_outputs=3,
            output_keys=['atr', 'up_atr', 'down_atr'],
            output_dtypes=[np.float64, np.float64, np.float64],
            output_spaces=[
                Box(low=0, high=np.inf, shape=(), dtype=np.float64),
                Box(low=0, high=np.inf, shape=(), dtype=np.float64),
                Box(low=0, high=np.inf, shape=(), dtype=np.float64),
            ],
            **kwargs
        )
           
    def reset_extras(self):
        self.fn_up_ma.reset()
        self.fn_down_ma.reset()
        self.up_tr_values.clear()
        self.down_tr_values.clear()
       
    def forward_ready(self, highs, lows, closes) -> bool:
        if not (len(highs) == len(lows) == len(closes)):
            raise ValueError("Input arrays must have same length")
        return len(highs) >= self.required_window
    
    def _cache_data(self, highs, lows, closes):
        high_low = highs[-1] - lows[-1]
        
        if len(highs) == 1:
            # 
            # up_tr = highs[-1] - opens[-1]
            # down_tr = opens[-1] - lows[-1]
            # TR = high_low
            up_tr = high_low/2
            down_tr = high_low/2
            TR = high_low
        else:
            prev_close = closes[-2]
            # 根据前收盘价的位置计算上行和下行TR
            if lows[-1] <= prev_close <= highs[-1]:  # 前收盘价在当前范围内
                up_tr = highs[-1] - prev_close
                down_tr = prev_close - lows[-1]
                TR = high_low
            elif prev_close < lows[-1]:  # 前收盘价低于当前最低价
                up_tr = highs[-1] - prev_close
                down_tr = 0
                TR = up_tr
            else:  # 前收盘价高于当前最高价
                up_tr = 0
                down_tr = prev_close - lows[-1]
                TR = down_tr
        
        # 验证计算结果
        assert all([
            up_tr >= 0,
            down_tr >= 0,
            TR == up_tr + down_tr,
            TR >= 0
        ]), "TR calculation validation failed"
        
        self.up_tr_values.push(up_tr)
        self.down_tr_values.push(down_tr)
    
    def pre_forward(self, highs, lows, closes):
        self._cache_data(highs, lows, closes)
    
    def safe_forward(self, highs, lows, closes):
        self._cache_data(highs, lows, closes)
        up_atr = self.fn_up_ma.rolling(self.up_tr_values.values)
        down_atr = self.fn_down_ma.rolling(self.down_tr_values.values)
        atr = up_atr + down_atr
        return atr, up_atr, down_atr
    
    @property
    def null_output(self):
        return np.nan, np.nan, np.nan
    
    @property
    def full_name(self):
        return f'{self.name}({self.n},{self.ma_type})'
    