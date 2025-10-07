from ..base.indicator_ii import rIndicatorII

class IndicatorFunc1Call_XXX(rIndicatorII):

    def __init__(self, callback, n, *args, **kwargs):
        self.callback = callback
        self.n = n
        super(IndicatorFunc1Call_XXX, self).__init__(window=n, *args, **kwargs)
    
    def forward(self, values):
        return self.callback(values)
    