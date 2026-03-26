


def SameUpDown(x, stride, eps=0):
    """
    连续上涨的时间
    作用： 一种趋势延续度的度量
    Note: 不连续
    """
    num = len(x)
    if num <= stride:
        raise Exception('Require At least %d points' % stride+1)
    eps = abs(eps)
    y = np.zeros_like(x, dtype=np.float64)
    y[0:stride] = np.nan
    for i in range(stride, num):
        if x[i] - x[i-stride] > eps:
            delta = 1
        elif x[i] - x[i-stride] < -eps:
            delta = -1
        else:
            delta = 0
        if np.isnan(y[i-1]):
            y[i] = delta
        elif y[i-1]*delta > 0:  # 同号
            y[i] = y[i-1] + delta
        else:
            y[i] = delta
    return y


def UpDownCumsum(x, stride, eps=0):
    """
    净累计多头时间
    """
    num = len(x)
    if num <= stride:
        raise Exception('Require At least %d points' % stride+1)
    eps = abs(eps)
    y = np.zeros_like(x, dtype=np.float64)
    y[0:stride] = np.nan
    for i in range(stride, num):
        if x[i] - x[i-stride] > eps:
            delta = 1
        elif x[i] - x[i-stride] < -eps:
            delta = -1
        else:
            delta = 0
        if np.isnan(y[i-1]):
            y[i] = delta
        else:
            y[i] = y[i-1] + delta
    return y


def UpDownType(x, stride, eps=0):
    """
    意义： 多空的状态
    x: price
    y:
        上涨: １
        下跌:　-1
        平: 0
    多空的时间
    """
    num = len(x)
    if num <= stride:
        raise Exception('Require At least %d points' % stride+1)
    eps = abs(eps)
    y = np.zeros_like(x, dtype=np.float64)
    y[0:stride] = np.nan
    for i in range(stride, num):
        if x[i] - x[i-stride] > eps:
            y[i] = 1
        elif x[i] - x[i-stride] < -eps:
            y[i] = -1
        else:
            y[i] = 0
    return y


def AVGBody(open, close, N):
    """
    Ref:
        http://www.danglanglang.com/gupiao/2464
    """
    num = len(open)
    assert( num == len(close) )
    body = close - open
    values = MA(body, N)
    return values
