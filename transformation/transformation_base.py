from abc import ABC, abstractmethod
from typing import List, Any


class Transformation(ABC):
    """Base class for image transformations"""

    def __call__(self, imgs: List[Any]) -> List[Any]:
        return [self.transform(img) for img in imgs]

    @abstractmethod
    def transform(self, img: Any) -> Any:
        pass