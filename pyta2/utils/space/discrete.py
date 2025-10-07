import numpy as np
from .base import Space

class Discrete(Space):
    def __init__(self, n, seed=None):
        assert n > 0, "n must be positive"
        self.n = n
        super().__init__((), np.int64, seed)

    def sample(self, mask=None):
        if mask is not None:
            assert isinstance(mask, np.ndarray) and mask.dtype == np.int8
            assert mask.shape == (self.n,), f"Expected mask shape {(self.n,)} not {mask.shape}"
            valid_choices = np.where(mask == 0)[0]
            if len(valid_choices) == 0:
                raise ValueError("At least one choice must be unmasked")
            return self.np_random.choice(valid_choices)
        return self.np_random.integers(0, self.n)

    def contains(self, x):
        return 0 <= x < self.n

    def __repr__(self):
        return f"Discrete({self.n})"