import numpy as np
from ..base.indicator import rIndicator
from ..utils.space.box import Box


class rOBV(rIndicator):
    """OBV - On-Balance Volume

    通过成交量的累积来权衡价格变动的指标。

    Formula:
        If Close > PrevClose: OBV = PrevOBV + Volume
        If Close < PrevClose: OBV = PrevOBV - Volume
        If Close == PrevClose: OBV = PrevOBV
    """
    name = 'OBV'

    def __init__(self, **kwargs):
        self._obv = 0.0
        super(rOBV, self).__init__(
            window=2,
            schema=[
                ('obv', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        self._obv = 0.0

    def forward(self, closes, volumes):
        if len(closes) < 2:
            return np.nan

        if closes[-1] > closes[-2]:
            self._obv += volumes[-1]
        elif closes[-1] < closes[-2]:
            self._obv -= volumes[-1]
            
        return self._obv

    @property
    def full_name(self):
        return self.name
