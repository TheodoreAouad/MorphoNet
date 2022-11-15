"""Loaders for the supported datasets."""

from abc import abstractmethod, ABCMeta
from typing import Any, List, Optional, Tuple, Type

import inspect
import pytorch_lightning as pl

import torchvision
import torch
from torch.utils.data import DataLoader

from misc.utils import PRECISIONS_NP, PRECISIONS_TORCH
from operations.base import Operation


NOISY_NAME = "NOISY_SRGB_010"
GT_NAME = "GT_SRGB_010"

# TODO add way to split data from args
# TODO with heavy data, dynamic loading needs to be implemented
# TODO ensure data is loaded as float and in [0,1] before targets computation


class Dataset(torch.utils.data.Dataset):
    """Implementation of torch.utils.Data.Dataset abstract class."""

    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index, :], self.targets[index, :]


class DataModule(pl.LightningDataModule, metaclass=ABCMeta):  # pylint: disable=too-many-instance-attributes
    """Base abstract class for datasets."""

    def __init__(
        self,
        batch_size: int,
        dataset_path: str,
        precision: str,
        operation: Operation,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.torch_precision = PRECISIONS_TORCH[precision]
        self.np_precision = PRECISIONS_NP[precision]
        self.num_workers = 8
        self.operation = operation

        self.train_dataset: Dataset
        self.val_dataset: Dataset

    def scale(self, tensor: torch.Tensor) -> torch.Tensor:
        return torchvision.transforms.ConvertImageDtype(self.torch_precision)(
            tensor
        )

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return torchvision.transforms.Lambda(
            lambda x: (x - torch.mean(x)) / torch.std(x)
        )(tensor)

    def remodel_data(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = self.operation(inputs, targets)
        return inputs.to(self.torch_precision), targets.to(self.torch_precision)

    @property
    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a sample of the validation dataset."""
        # TODO change method when shuffling to always have same data
        return self.val_dataset.inputs[:10], self.val_dataset.targets[:10]

    def prepare_data(self) -> None:
        pass

    @abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def _create_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(
            dataset=self.train_dataset,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader(
            dataset=self.val_dataset,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(Dataset(torch.empty(0), torch.empty(0)))

    @classmethod
    def select_(cls, name: str) -> Optional[Type["DataModule"]]:
        """
        Class method iterating over all subclasses to load the desired dataset.
        """
        if cls.__name__.lower() == name:
            return cls

        for subclass in cls.__subclasses__():
            selected = subclass.select_(name)
            if selected is not None:
                return selected

        return None

    @classmethod
    def select(cls, name: str, **kwargs: Any) -> "DataModule":
        """
        Class method iterating over all subclasses to instantiate the desired
        data module.
        """

        selected = cls.select_(name)
        if selected is None:
            raise ValueError("The selected dataset was not found.")

        return selected(**kwargs)

    @classmethod
    def listing(cls) -> List[str]:
        """List all the available dataset loaders."""
        subclasses = set()
        if not inspect.isabstract(cls):
            subclasses = {cls.__name__.lower()}

        for subclass in cls.__subclasses__():
            subclasses = subclasses.union(subclass.listing())

        return list(subclasses)
