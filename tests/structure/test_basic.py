from pyta_dev.structure import Boll, ZigZag
from fintest.crypto.binance import api

def test_boll():
    df = api.get_future_klines_demo1(100, to_pandas=True)
    ub, mid, lb = Boll(df['close'], 20, 2)
    print(ub[20:28])
    print(mid[20:28])
    print(lb[20:28])

def test_zigzag():
    df = api.get_future_klines_demo1(100, to_pandas=True)
    zigzag = ZigZag(df['close'], 10, 0.01)
    print(zigzag[20:28])

if __name__ == '__main__':
    test_boll()
    test_zigzag()
