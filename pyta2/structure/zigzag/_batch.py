import numpy as np
from ...base import forward_rolling_apply
from ._rolling import rZigZag, rZigZag_HL, rZigZag_Unit, rZigZag_HL_Unit

def ZigZag(closes, delta, use_pct=True, **kwargs):
    return forward_rolling_apply(
        len(closes), rZigZag, param_args=[delta, use_pct], 
        input_args=[np.array(closes)], 
        **kwargs
    )

def ZigZag_HL(highs, lows, delta, use_pct=True, **kwargs):
    return forward_rolling_apply(
        len(highs), rZigZag_HL, param_args=[delta, use_pct], 
        input_args=[np.array(highs), np.array(lows)], 
        **kwargs
    )

def ZigZag_Unit(units, closes, delta, **kwargs):
    return forward_rolling_apply(
        len(closes), rZigZag_Unit, param_args=[delta], 
        input_args=[np.array(units), np.array(closes)], 
        **kwargs
    )

def ZigZag_HL_Unit(units, highs, lows, closes, delta, **kwargs):
    return forward_rolling_apply(
        len(highs), rZigZag_HL_Unit, param_args=[delta], 
        input_args=[np.array(units), np.array(highs), np.array(lows), np.array(closes)], 
        **kwargs
    )

__all__ = ['ZigZag', 'ZigZag_HL', 'ZigZag_Unit', 'ZigZag_HL_Unit']
