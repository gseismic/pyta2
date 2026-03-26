import numpy as np
from pyta2.base.indicator import rIndicator
from pyta2.utils.space.box import Box
from pyta2.stats.highlow import rHigh, rLow


class rFibDC(rIndicator):
    """
    Fibonacci Donchian Channels (FibDC)
    //@version=3
    study("Fibonacci Zone", overlay=true)

    per=input(21, "calculate for last ## bars")
    hl=highest(high,per) //High Line (Border)
    ll=lowest(low,per)   //Low Line  (Border)
    dist=hl-ll          //range of the channel    
    hf=hl-dist*0.236    //Highest Fibonacci line
    cfh=hl-dist*0.382    //Center High Fibonacci line
    cfl=hl-dist*0.618    //Center Low Fibonacci line
    lf=hl-dist*0.764     //Lowest Fibonacci line
    """
    name = "FibDC"

    def __init__(self, n=20, **kwargs):
        assert n >= 1, f'{self.name} window n must be at least 1, got {n}'
        self.n = n
        self._H = rHigh(n)
        self._L = rLow(n)
        
        super(rFibDC, self).__init__(
            window=n,
            schema=[
                ('ub', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('hf', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('chf', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('mid', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('clf', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('lf', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('lb', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        self._H.reset()
        self._L.reset()

    def forward(self, highs, lows):
        if len(highs) < self.n:
            return (np.nan,) * 7
            
        ub = self._H.rolling(highs)
        lb = self._L.rolling(lows)
        
        dist = ub - lb
        hf = ub - dist * 0.236
        chf = ub - dist * 0.382
        mid = ub - dist * 0.5
        clf = ub - dist * 0.618
        lf = ub - dist * 0.764
        
        return ub, hf, chf, mid, clf, lf, lb

    @property
    def full_name(self):
        return f'{self.name}({self.n})'
