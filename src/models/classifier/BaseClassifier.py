from abc import ABC

class BaseClassifier(ABC):
    @abstractmethod
    def __predict__(self, X, y):
        
        pass