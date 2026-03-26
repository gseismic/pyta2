import numpy as np
from pyta2.base.indicator import rIndicator
from pyta2.utils.space.box import Box
from pyta2.utils.deque import NumpyDeque
from pyta2.trend.ma.ema import rEMA
from pyta2.stats.atr.atr import rATR


class rDMI(rIndicator):
    """DMI - Directional Movement Index

    包含 +DI、-DI 和 ADX 三个输出，用于判断趋势方向和强度。

    Reference:
        https://www.investopedia.com/terms/p/positivedirectionalindicator.asp
        https://www.tradingview.com/wiki/Directional_Movement_(DMI)
    """
    name = 'DMI'

    def __init__(self, n=14, n_atr=14, **kwargs):
        self.n = n
        self.n_atr = n_atr
        self._ema_plus  = rEMA(n)
        self._ema_minus = rEMA(n)
        self._ema_adx   = rEMA(n)
        self._atr = rATR(n_atr)
        self._vs_plus  = NumpyDeque(maxlen=n)
        self._vs_minus = NumpyDeque(maxlen=n)
        self._vs_adx   = NumpyDeque(maxlen=n)
        super(rDMI, self).__init__(
            window=max(n, n_atr),
            schema=[
                ('adx',      Box(low=0, high=100,   shape=(), dtype=np.float64)),
                ('plus_di',  Box(low=0, high=100,   shape=(), dtype=np.float64)),
                ('minus_di', Box(low=0, high=100,   shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        self._ema_plus.reset()
        self._ema_minus.reset()
        self._ema_adx.reset()
        self._atr.reset()
        self._vs_plus.clear()
        self._vs_minus.clear()
        self._vs_adx.clear()

    def forward(self, highs, lows, closes):
        if len(highs) < 2:
            return np.nan, np.nan, np.nan

        up_move   = highs[-1] - highs[-2]
        down_move = lows[-2]  - lows[-1]

        plus_dm  = up_move   if up_move   > down_move and up_move   > 0 else 0.0
        minus_dm = down_move if down_move > up_move   and down_move > 0 else 0.0

        atr = self._atr.rolling(highs, lows, closes)
        if np.isnan(atr) or atr == 0:
            return np.nan, np.nan, np.nan

        self._vs_plus.append(plus_dm)
        self._vs_minus.append(minus_dm)

        plus_di  = self._ema_plus.rolling(self._vs_plus.values)  / atr * 100
        minus_di = self._ema_minus.rolling(self._vs_minus.values) / atr * 100

        di_sum = plus_di + minus_di
        dx = abs(plus_di - minus_di) / di_sum * 100 if di_sum != 0 else 0.0
        self._vs_adx.append(dx)
        adx = self._ema_adx.rolling(self._vs_adx.values)
        return adx, plus_di, minus_di

    @property
    def full_name(self):
        return f'{self.name}({self.n},{self.n_atr})'
