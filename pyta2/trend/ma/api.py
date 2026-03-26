# api.py
# 采用延迟导入以避免循环依赖

def get_ma_class(ma_type: str):
    if ma_type == 'EMA':
        from .ema import rEMA
        return rEMA
    elif ma_type == 'SMA':
        from .sma import rSMA
        return rSMA
    elif ma_type == 'WMA':
        from .wma import rWMA
        return rWMA
    elif ma_type == 'HMA':
        from .hma import rHMA
        return rHMA
    elif ma_type == 'DEMA':
        from .dema import rDEMA
        return rDEMA
    elif ma_type == 'TEMA':
        from .tema import rTEMA
        return rTEMA
    elif ma_type == 'KAMA':
        from .kama import rKAMA
        return rKAMA
    elif ma_type == 'ZLEMA':
        from .zlema import rZLEMA
        return rZLEMA
    else:
        raise ValueError(f"Unknown ma_type: {ma_type}")

def get_ma_function(ma_type: str):
    from ._batch import SMA, EMA, WMA, HMA, DEMA, TEMA, KAMA, ZLEMA
    return {
        'SMA': SMA,
        'EMA': EMA,
        'WMA': WMA,
        'HMA': HMA,
        'DEMA': DEMA,
        'TEMA': TEMA,
        'KAMA': KAMA,
        'ZLEMA': ZLEMA,
    }[ma_type]

def get_ma_window(ma_type, n):
    cls = get_ma_class(ma_type)
    return cls(n).window

get_ma_func = get_ma_function

__all__ = ['get_ma_class', 'get_ma_function', 'get_ma_func', 'get_ma_window']
