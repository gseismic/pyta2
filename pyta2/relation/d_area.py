import numpy as np
from pyta2.base import rIndicatorRelation2


class rDArea(rIndicatorRelation2):

    def __init__(self, n):
        super(rDArea, self).__init__(name="DArea", 
                                   window=n, 
                                   output_dim=1)
        self.n = n

    def rolling(self, values1, values2):
        if len(values1) < self.n:
            return np.nan
        return np.sum(values1[-self.n:] - values2[-self.n:])
