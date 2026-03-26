import numpy as np
from ..base.indicator import rIndicator
from ..utils.space.box import Box


class rTER(rIndicator):
    """TER - Total Efficiency Ratio Index by Liu Shengli

    一种提前指标
    分形效率
    gain: 多方有效占领空间
    gain - loss = 多空占领空间比
    range: 多空争议导致的市场波动

    True ER Index by Liu Shengli

    事实:
        ter := abs(body)/range
        ger := 单位时间累积[净]上涨幅度/单位时间累积[净]震荡幅度

        上涨趋势: r > 0.5
        下跌趋势: r < 0.5
        震荡趋势: r = 0.5
    指标意义:
        ter: 上涨幅度/总幅度
            过大: 代表上涨动力枯竭
            过小: 下跌动力枯竭
        dvi: 上涨下跌幅度差/总幅度
            过大: 代表上涨动力枯竭
            过小: 下跌动力枯竭
            相等: 震荡
    思想：
        价格上涨中本不需要多少成交量
        放量上涨，指标上涨
        放量，价格却不涨，为反转
    """
    name = "TER"

    def __init__(self, n, **kwargs):
        assert n >= 1
        self.n = n
        # 使用简单的增量 EMA 逻辑以解耦对 trend._moving 的依赖
        self._gain_ema_val = None
        self._loss_ema_val = None
        self._range_ema_val = None
        self.alpha = 2.0 / (n + 1)
        
        super(rTER, self).__init__(
            window=n,
            schema=[
                ('ter', Box(low=0.0, high=np.inf, shape=(), dtype=np.float64)),
                ('der', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('ger', Box(low=0.0, high=np.inf, shape=(), dtype=np.float64)),
                ('ler', Box(low=0.0, high=np.inf, shape=(), dtype=np.float64)),
                ('gain', Box(low=0.0, high=np.inf, shape=(), dtype=np.float64)),
                ('loss', Box(low=0.0, high=np.inf, shape=(), dtype=np.float64)),
                ('range', Box(low=0.0, high=np.inf, shape=(), dtype=np.float64)),
                ('gain_ma', Box(low=0.0, high=np.inf, shape=(), dtype=np.float64)),
                ('loss_ma', Box(low=0.0, high=np.inf, shape=(), dtype=np.float64)),
                ('range_ma', Box(low=0.0, high=np.inf, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        self._gain_ema_val = None
        self._loss_ema_val = None
        self._range_ema_val = None

    def _update_ema(self, ema_val, current_val):
        if ema_val is None or np.isnan(ema_val):
            return current_val
        return (current_val - ema_val) * self.alpha + ema_val

    def forward(self, opens, highs, lows, closes):
        range_val = highs[-1] - lows[-1]
        delta = closes[-1] - opens[-1]
        
        gain = max(delta, 0)
        loss = max(-delta, 0)
        
        self._gain_ema_val = self._update_ema(self._gain_ema_val, gain)
        self._loss_ema_val = self._update_ema(self._loss_ema_val, loss)
        self._range_ema_val = self._update_ema(self._range_ema_val, range_val)
        
        if len(highs) < self.window:
            return (np.nan,) * 10
            
        range_ma = self._range_ema_val
        gain_ma = self._gain_ema_val
        loss_ma = self._loss_ema_val
        
        if range_ma != 0:
            ter = (gain_ma + loss_ma) / range_ma
            der = (gain_ma - loss_ma) / range_ma
            ger = gain_ma / range_ma
            ler = loss_ma / range_ma
        else:
            ter, der, ger, ler = 0.0, 0.0, 0.0, 0.0
            
        return (ter, der, ger, ler, gain, loss, range_val, 
                gain_ma, loss_ma, range_ma)

    @property
    def full_name(self):
        return f'{self.name}({self.n})'
