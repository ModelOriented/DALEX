from abc import ABC, abstractmethod

class Explanation(ABC):
    result = None

    @abstractmethod
    def fit(self):
        pass
    @abstractmethod
    def plot(self):
        pass
