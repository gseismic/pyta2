from abc import ABC, abstractmethod
import numpy as np


class Space(ABC):

    def __init__(self, dtype=None, seed=None):
        self.dtype = dtype
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    @abstractmethod
    def sample(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def contains(self, x, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}()"