#import numpy as np


class MovingVector(object):
    """
    [@2021-07-21 23:38:59] cc
    q = NumPyVector(100) 
    """ 
    def __init__(self): 
        self._discard_end = 0 
        self._values = [] 
        self._notional_len = 0 

    def append(self, value): 
        self._notional_len += 1 
        self._values.append(value) 

    def rekeep_n(self, n):
        '''只保留最近的n个点的数据, 其余丢弃'''
        if len(self._values) > n:
            self._discard_end += len(self._values) - n
            self._values = self._values[-n:]

    def __getitem__(self, i):
        if i < 0:
            rv = self._values[i]
        elif i - self._discard_end >= 0:
            rv = self._values[i - self._discard_end]
        else:
            raise IndexError()
        return rv

    def __setitem__(self, i, value):
        if i < 0:
            self._values[i] = value
        elif i - self._discard_end >= 0:
            self._values[i - self._discard_end] = value
        else:
            raise IndexError

    @property
    def discard_end(self):
        return self._discard_end

    @property
    def notional_len(self):
        return self._notional_len

    @property
    def kept_len(self):
        return self._notional_len - self._discard_end

    @property
    def kept_values(self):
        return self._values

    def __repr__(self): 
        # self._notional_len - self._discard_end
        s = (f'MovingVector: notional_len={self.notional_len}'
             f' kept_len={self.kept_len}'
             f' kept_values={str(self.kept_values)}')
        return s


if __name__ == '__main__':
    if 1:
        import pytest as pt
        vec = MovingVector()
        vec.append(10)
        vec.append(11)
        vec.append(12)
        print(vec)
        assert vec.kept_values == [10, 11, 12]
        vec.rekeep_n(2)
        assert vec.kept_values == [11, 12]
        assert vec.kept_len == 2
        assert vec.notional_len == 3
        with pt.raises(IndexError):
            print(vec[0])
        print(vec[1])
        vec.append(3)
        vec.append(7)
        vec.append(8)
        vec.append(9)
        vec.append(10)
        assert vec.notional_len == 8
        assert vec.kept_len == 7

        assert vec[-1] == 10
        assert vec[-2] == 9
        assert vec[7] == 10
        assert vec[6] == 9
        vec.rekeep_n(3)
        assert vec[-1] == 10
        assert vec[-2] == 9
        assert vec[7] == 10
        assert vec[6] == 9
        assert vec[-3] == 8
        assert vec[5] == 8

        with pt.raises(IndexError):
            vec[-4]
        with pt.raises(IndexError):
            vec[4]
