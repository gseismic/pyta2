from abc import ABC, abstractmethod

class Space(ABC):
    
    def __init__(self, dtype=None):
        self.dtype = dtype

    @abstractmethod
    def sample(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def contains(self, x, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}()"