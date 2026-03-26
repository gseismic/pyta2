import numpy as np
from pyta2.base.indicator import rIndicator
from pyta2.utils.space.box import Box


class rRSI(rIndicator):
    """Relative Strength Index (RSI)"""
    name = "RSI"

    def __init__(self, n=14, **kwargs):
        assert n > 0, f'{self.name} n must be greater than 0, got {n}'
        self.n = n
        super(rRSI, self).__init__(
            window=self.n + 1,
            schema=[
                ('rsi', Box(low=0.0, high=100.0, shape=(), dtype=np.float64))
            ],
            **kwargs
        )
        
    def reset_extras(self):
        self.prev_gain = None
        self.prev_loss = None

    def forward(self, values):
        if len(values) < self.window:
            return np.nan
            
        if self.prev_gain is None or self.prev_loss is None:
            # Wilder's initial calculation: SMA of gains and losses over the first n periods
            deltas = np.diff(values[-self.n-1:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            self.prev_gain = np.mean(gains)
            self.prev_loss = np.mean(losses)
        else:
            # Wilder's smoothing/recursive calculation
            delta = values[-1] - values[-2]
            gain = max(delta, 0)
            loss = max(-delta, 0)
            
            self.prev_gain = (self.prev_gain * (self.n - 1) + gain) / self.n
            self.prev_loss = (self.prev_loss * (self.n - 1) + loss) / self.n

        denom = self.prev_gain + self.prev_loss
        if denom == 0:
            return 50.0
            
        rsi = 100 * self.prev_gain / denom
        return np.clip(rsi, 0, 100)
    
    @property
    def full_name(self):
        return f'{self.name}({self.n})'
