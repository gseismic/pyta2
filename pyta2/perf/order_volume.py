import numpy as np
from ..base import rIndicator
from ..utils.space.box import Scalar


class rOrderVolumePerf(rIndicator):
    '''整体计算收益-by下单量回测
        
    Notes:
        - 无限可开仓资金
        - 利润表和现金流量表是一个整理，不应该拆分为cash, profit等单独的类
    '''
    name = "OrderVolumePerf"

    def __init__(self, fee_rate, **kwargs):
        """
        Args:
            fee_rate: 手续费率
        """
        assert 0 <= fee_rate <= 1.0, f'{self.name} fee_rate must be between 0 and 1, got {fee_rate}'
        
        self.fee_rate = fee_rate
        self._prev_Q = 0  # 前刻总持仓
        self._prev_price = None
        self._acc_gross_profit = 0
        self._acc_fee = 0
        self._acc_pnl = 0
        
        # 使用新的Schema格式 - 直接传递字典
        schema = {
            'acc_pnl': Scalar(low=-np.inf, high=np.inf),
            'acc_fee': Scalar(low=0, high=np.inf)
        }
        
        super(rOrderVolumePerf, self).__init__(
            window=1,
            schema=schema,
            **kwargs
        )

    def reset_extras(self):
        self._prev_Q = 0
        self._prev_price = None
        self._acc_gross_profit = 0
        self._acc_fee = 0
        self._acc_pnl = 0

    def forward(self, order_prices, order_volumes):
        '''好处: 直接考虑交易费
        Args:
            - order_prices: 交易价格 order trade order_prices
            - order_volumes: 每个价格点开仓数量
        '''
        price, qty = order_prices[-1], order_volumes[-1]
        Q = self._prev_Q + qty

        new_fee = price * abs(qty) * self.fee_rate
        if self._prev_price is None:
            new_gross_profit = 0
        else:
            new_gross_profit = self._prev_Q * (price - self._prev_price)

        self._acc_gross_profit += new_gross_profit
        self._acc_fee += new_fee
        self._prev_Q = Q
        self._prev_price = price
        
        # new_gross_profit是过去的收益，new_fee是未来收益的成本，单独相加没有意义，acc想加才是综合效果
        self._acc_pnl = self._acc_gross_profit - self._acc_fee

        return self._acc_pnl, self._acc_fee

    @property
    def full_name(self):
        return f'{self.name}(fee_rate={self.fee_rate})'
