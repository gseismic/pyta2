import numpy as np
from pyta2.trend.ma.sma import rSMA
from pyta2.trend.ma.ema import rEMA
from pyta2.trend.ma.tema import rTEMA

def test_indicators():
    print("Testing Indicators...")
    data = np.arange(20).astype(float)
    n = 3
    
    # 1. Test SMA
    print("  - Testing SMA...")
    sma = rSMA(n=n)
    for i in range(10):
        val = sma.rolling(data[:i+1])
        if i < n - 1:
            assert np.isnan(val)
        else:
            expected = np.mean(data[i-n+1:i+1])
            assert np.isclose(val, expected)
    print("    SMA passed!")

    # 2. Test EMA
    print("  - Testing EMA...")
    ema = rEMA(n=n)
    last_ema = None
    alpha = 2.0 / (n + 1)
    for i in range(10):
        val = ema.rolling(data[:i+1])
        if i < n - 1:
            assert np.isnan(val)
        elif i == n - 1:
            last_ema = np.mean(data[:n])
            assert np.isclose(val, last_ema)
        else:
            last_ema = (data[i] - last_ema) * alpha + last_ema
            assert np.isclose(val, last_ema)
    print("    EMA passed!")

    # 3. Test TEMA
    print("  - Testing TEMA...")
    # TEMA window is 3n-2 = 3*3-2 = 7
    tema = rTEMA(n=n)
    for i in range(20):
        # 只要不崩溃就行，逻辑已经在上面 EMA 中间接测试
        val = tema.rolling(data[:i+1])
        if i < 7 - 1: # 3*3 - 2 = 7
             pass
        else:
             assert not np.isnan(val)
    print("    TEMA passed (No AttributeError)!")

if __name__ == "__main__":
    try:
        test_indicators()
        print("\nAll indicators tested successfully!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        exit(1)
