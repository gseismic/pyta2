# -*- encoding: utf8 -*-
import math
import numpy as np


class NumPyVector(object):
    """
    [@2021-07-21 23:38:59] cc
    q = NumPyVector(100)
    """
    def __init__(self, capacity=100, dtype=np.float64, buffer_factor=1.5):
        assert(buffer_factor >= 1.1) 
        assert(capacity >= 1) 
        self._dtype = dtype 
        self._buffer_factor = buffer_factor 
        self._size = 0
        self._capacity = capacity
        self._data = np.empty((self._capacity,), dtype=dtype)

    def append(self, value):
        self.push(value)

    def push(self, value):
        if self._size >= self._capacity:
            # print('grow ...', self._capacity)
            self._capacity = int(math.ceil(self._buffer_factor*self._capacity))
            # print('to:', self._capacity)
            tmp = self._data[:self._size]
            # self._data = np.zeros((self._capacity,), dtype=self._dtype)
            self._data = np.empty((self._capacity,), dtype=self._dtype)
            self._data[:self._size] = tmp
        self._data[self._size] = value
        self._size += 1

    @property
    def values(self):
        return self._data[:self._size]

    def __getitem__(self, i):
        return self.values[i]

    def __setitem__(self, i, value):
        self.values[i] = value

    def __len__(self):
        return self._size

    @property
    def capacity(self):
        return self._capacity

    def __repr__(self):
        return f'NumPyVector({str(self.values.tolist())})'

NumpyVector = NumPyVector

if __name__ == "__main__":
    q = NumPyVector(3)
    print(q)
    q.push(1)
    print(q)
    q.push(2)
    print(q)
    q.push(3)
    print(q)