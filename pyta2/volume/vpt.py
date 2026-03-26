import numpy as np
from ..base.indicator import rIndicator
from ..utils.space.box import Box


class rVPT(rIndicator):
    """VPT - Volume Price Trend (成交量价格趋势)
    
    结合成交量和价格变动比例。
    
    Formula:
        VPT = Prev_VPT + Volume * (Close - PrevClose) / PrevClose
    """
    name = 'VPT'

    def __init__(self, **kwargs):
        self._vpt = 0.0
        self._prev_close = None
        super(rVPT, self).__init__(
            window=2,
            schema=[
                ('vpt', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        self._vpt = 0.0
        self._prev_close = None

    def forward(self, closes, vols):
        if len(closes) < 2:
            self._prev_close = closes[-1]
            return 0.0
            
        prev_close = self._prev_close if self._prev_close is not None else closes[-2]
        change_rate = (closes[-1] - prev_close) / prev_close
        
        self._vpt += vols[-1] * change_rate
        self._prev_close = closes[-1]
        
        return self._vpt

    @property
    def full_name(self):
        return f'{self.name}()'
