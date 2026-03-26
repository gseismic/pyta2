import numpy as np
from pyta2.base.indicator import rIndicator
from pyta2.utils.space.box import Box
from pyta2.trend.ma.api import get_ma_class


class rVWAP(rIndicator):
    """
    VWAP - Volume Weighted Average Price
    """
    name = "VWAP"

    def __init__(self, n, ma_type='SMA', **kwargs):
        assert n >= 1, f'{self.name} window n must be at least 1, got {n}'
        self.n = n
        self.ma_type = ma_type
        # 使用指定的 MA 类型来计算价值和成交量的均值
        self.fn_ma_value = get_ma_class(ma_type)(n)
        self.fn_ma_volume = get_ma_class(ma_type)(n)
        
        super(rVWAP, self).__init__(
            window=n,
            schema=[
                ('vwap', Box(low=0.0, high=np.inf, shape=(), dtype=np.float64))
            ],
            **kwargs
        )

    def reset_extras(self):
        self.fn_ma_value.reset()
        self.fn_ma_volume.reset()

    def forward(self, highs, lows, closes, volumes):
        if len(closes) < self.n:
            return np.nan
            
        # 计算典型价格 * 成交量
        tp = (highs + lows + closes) / 3.0
        tp_v = tp * volumes
        
        # 计算各自的移动平均
        avg_value = self.fn_ma_value.rolling(tp_v)
        avg_volume = self.fn_ma_volume.rolling(volumes)
        
        if avg_volume == 0 or np.isnan(avg_volume):
            return np.nan
            
        return avg_value / avg_volume

    @property
    def full_name(self):
        return f'{self.name}({self.n},{self.ma_type})'