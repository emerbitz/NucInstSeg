from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union

from torch import Tensor

from evaluation.utils import is_batched


class Postprocess(ABC):
    """
    Base class for postprocessing.
    """

    @abstractmethod
    def __call__(self, pred: Dict[str, Tensor]) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Define here the postprocessing pipeline.

        First, you might want to perform some operation on the batch level like applying an activation function.
        Finally, you will probably want to call the postprocess function.
        """
        pass

    @abstractmethod
    def postprocess_fn(self, *args) -> Tensor:
        """
        Define here the postprocessing function.
        """
        pass

    def postprocess(self, tensor_0: Tensor, tensor_1: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Iterates over the batched tensor(s) and applies the postprocess_fn function to them.
        """
        if not is_batched(tensor_0):
            raise ValueError(f"Tensor_0 should have shape (B, C, W, H). Got instead {tensor_0.shape}.")
        if tensor_1 is not None and not is_batched(tensor_1):
            raise ValueError(f"Tensor_1 should have shape (B, C, W, H). Got instead {tensor_1.shape}.")
        batch_size = tensor_0.shape[0]
        device = tensor_0.device

        if batch_size == 1:
            if tensor_1 is None:
                return self.postprocess_fn(tensor_0).to(device)
            else:
                return self.postprocess_fn(tensor_0, tensor_1).to(device)
        else:
            if tensor_1 is None:
                return tuple(self.postprocess_fn(t).to(device) for t in tensor_0)
            else:
                return tuple(self.postprocess_fn(t0, t1).to(device) for t0, t1 in zip(tensor_0, tensor_1))
