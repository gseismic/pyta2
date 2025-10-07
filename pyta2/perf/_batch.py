from ..utils import apply_rolling_forward
from ._rolling import *


def OrderVolumePerf(order_prices, order_volumes,  fee_rate, **kwargs):
    assert len(order_prices) == len(order_volumes)
    outputs = apply_rolling_forward(len(order_prices), rOrderVolumePerf,
                                    param_args=[fee_rate], 
                                    input_args=[order_prices, order_volumes],
                                    **kwargs)
    return outputs

def PositionSizePerf(prices, position_sizes,  fee_rate, **kwargs):
    assert len(prices) == len(position_sizes)
    outputs = apply_rolling_forward(len(prices), rPositionSizePerf,
                                    param_args=[fee_rate], 
                                    input_args=[prices, position_sizes],
                                    **kwargs)
    return outputs


__all__ = ['OrderVolumePerf', 'PositionSizePerf']
