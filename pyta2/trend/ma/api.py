from ._rolling import *
from ._batch import *

def get_ma_class(ma_type):
    return {
        'EMA': rEMA,
        'SMA': rSMA,
        'WMA': rWMA,
        'HMA': rHMA,
        'DEMA': rDEMA,
        'TEMA': rTEMA,
    }[ma_type]

def get_ma_function(ma_type):
    return {
        'EMA': EMA,
        'SMA': SMA,
        'WMA': WMA,
        'HMA': HMA,
        'DEMA': DEMA,
        'TEMA': TEMA,
    }[ma_type]
    
def get_ma_window(ma_type, n):
    obj = get_ma_class(ma_type)(n)
    return obj.window

get_ma_func = get_ma_function
__all__ = ['get_ma_class', 'get_ma_function', 'get_ma_func', 'get_ma_window']
