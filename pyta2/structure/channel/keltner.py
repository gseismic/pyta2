import numpy as np
from pyta2.base.indicator import rIndicator
from pyta2.utils.space.box import Box
from pyta2.stats.atr.atr import rATR
from pyta2.trend.ma.ema import rEMA
from pyta2.trend.ma.sma import rSMA


class rKeltner(rIndicator):
    """
    Keltner Channels
    A trend-following channel based on an EMA of price and ATR.
    
    Ref:
        https://www.investopedia.com/terms/k/keltnerchannel.asp
    """
    name = "Keltner"

    def __init__(self, n=20, F=2, ma_type='EMA', **kwargs):
        self.n = n or 20
        self.F = F or 2
        self.ma_type = ma_type or 'EMA'
        
        if self.ma_type.upper() == 'EMA':
            self.fn_atr = rATR(self.n)
            self.fn_ma = rEMA(self.n)
        elif self.ma_type.upper() == 'SMA':
            self.fn_atr = rATR(self.n, ma_type='SMA')
            self.fn_ma = rSMA(self.n)
        else:
            raise ValueError(f"Unsupported ma_type: {self.ma_type}")
            
        super(rKeltner, self).__init__(
            window=self.n,
            schema=[
                ('ub', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('mid', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('lb', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        self.fn_atr.reset()
        self.fn_ma.reset()

    def forward(self, highs, lows, closes):
        atr = self.fn_atr.rolling(highs, lows, closes)
        mid = self.fn_ma.rolling(closes)
        
        if len(highs) < self.n:
            return np.nan, np.nan, np.nan

        ub = mid + self.F * atr
        lb = mid - self.F * atr
        return ub, mid, lb

    @property
    def full_name(self):
        return f'{self.name}({self.n},{self.F},{self.ma_type})'
