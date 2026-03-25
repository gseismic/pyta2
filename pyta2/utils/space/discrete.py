import numpy as np

from .base import Space


class Discrete(Space):
    def __init__(self, n, seed=None):
        assert n > 0, "n must be positive"
        self.n = n
        super().__init__(dtype=np.int64, seed=seed)

    def sample(self, mask=None):
        if mask is not None:
            assert isinstance(mask, np.ndarray) and mask.dtype == np.int8
            assert mask.shape == (self.n,), (
                f"Expected mask shape {(self.n,)} not {mask.shape}"
            )
            valid_choices = np.where(mask == 1)[0]
            if len(valid_choices) == 0:
                raise ValueError("At least one valid choice must be provided")
            return self.np_random.choice(valid_choices)
        return self.np_random.integers(0, self.n)

    def contains(self, x):
        if not np.issubdtype(np.asarray(x).dtype, np.integer):
            return False
        x = int(x)
        return 0 <= x < self.n

    def __eq__(self, other):
        if not isinstance(other, Discrete):
            return False
        return self.n == other.n

    def __repr__(self):
        return f"Discrete({self.n})"

    def to_json(self):
        return {
            "type": "Discrete",
            "n": self.n,
            "dtype": str(self.dtype),
        }

    @classmethod
    def from_json(cls, data):
        return cls(n=data["n"])
