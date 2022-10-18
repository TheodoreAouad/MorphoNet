"""Base model with default optimizer, scheduler, and tracked metrics."""

from abc import ABCMeta, abstractmethod
from typing import Dict, Any, Callable, Tuple
from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl
import torch
from typing import Optional, List
import inspect


VAL_LOSS = "val_loss"
TRAIN_LOSS = "train_loss"


class BaseNetwork(pl.LightningModule, metaclass=ABCMeta):
    """Base network class with default settings."""

    def __init__(
        self, loss_function: Callable, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loss_function = loss_function

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=20, verbose=True, threshold=1e-4
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": VAL_LOSS,
                "frequency": 1,
            },
        }

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        *args: Any,
        **kwargs: Any
    ) -> STEP_OUTPUT:
        # pylint: disable=arguments-differ
        """Training step, calls `self.forward()`."""
        inputs, targets = batch

        predictions = self(inputs)
        loss = self.loss_function(predictions, targets)

        self.log(TRAIN_LOSS, loss)

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        *args: Any,
        **kwargs: Any
    ) -> STEP_OUTPUT:
        # pylint: disable=arguments-differ
        """Validation step, calls `self.forward()` without grad."""
        inputs, targets = batch

        predictions = self(inputs)
        loss = self.loss_function(predictions, targets)

        self.log(VAL_LOSS, loss)

        return loss

    @classmethod
    def select(cls, name: str, **kwargs: Any) -> Optional["BaseNetwork"]:
        """
        Class method iterating over all subclasses to instantiate the desired
        model.
        """
        if cls.__name__.lower() == name:
            return cls(**kwargs)

        for subclass in cls.__subclasses__():
            instance = subclass.select(name, **kwargs)
            if instance is not None:
                return instance

        return None

    @classmethod
    def listing(cls) -> List[str]:
        """List all the available models."""
        subclasses = set()
        if not inspect.isabstract(cls):
            subclasses = {cls.__name__.lower()}

        for subclass in cls.__subclasses__():
            subclasses = subclasses.union(subclass.listing())

        return list(subclasses)

    @abstractmethod
    def forward(
        self, batch: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        """"""
        