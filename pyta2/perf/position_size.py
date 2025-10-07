import numpy as np
from ..base import rIndicator
from ..utils.space.box import Scalar
from ..utils.deque.numpy_deque import NumpyDeque
from .order_volume import rOrderVolumePerf


class rPositionSizePerf(rIndicator):
    '''无限可开仓资金
    '''
    name = "PositionSizePerf"

    def __init__(self, fee_rate, **kwargs):
        """
        Args:
            fee_rate: 手续费率
        """
        assert 0 <= fee_rate <= 1.0, f'{self.name} fee_rate must be between 0 and 1, got {fee_rate}'
        
        self._order_volume_perf = rOrderVolumePerf(fee_rate)
        self._order_volumes = NumpyDeque(maxlen=2, dtype=np.float64)
        
        # 使用新的Schema格式 - 直接传递字典
        schema = {
            'acc_pnl': Scalar(low=-np.inf, high=np.inf),
            'acc_fee': Scalar(low=0, high=np.inf)
        }
        
        super(rPositionSizePerf, self).__init__(
            window=2,
            schema=schema,
            **kwargs
        )

    def reset_extras(self):
        self._order_volume_perf.reset()
        self._order_volumes.clear()

    def forward(self, prices, position_sizes):
        '''
        Args:
            - prices: trade-prices
            - position_sizes: 每个价格点的持仓数量
        '''
        if len(prices) == 1:
            qty = position_sizes[-1] - 0.0
        else:
            qty = position_sizes[-1] - position_sizes[-2]

        self._order_volumes.push(qty)

        return self._order_volume_perf.forward(prices, self._order_volumes.values)

    @property
    def full_name(self):
        return f'{self.name}(fee_rate={self._order_volume_perf.fee_rate})'
