"""DataModule for the MNIST dataset."""

from typing import Optional
import torchvision

from .base import DataModule, Dataset


class FashionMNIST(DataModule):
    """FasionMNIST DataModule."""

    def prepare_data(self) -> None:
        torchvision.datasets.FashionMNIST(self.dataset_path, train=True, download=True)
        torchvision.datasets.FashionMNIST(
            self.dataset_path, train=False, download=True
        )

    def setup(self, stage: Optional[str] = None) -> None:
        train_dataset = torchvision.datasets.FashionMNIST(
            self.dataset_path,
            train=True,
        )
        inputs_scaled = self.scale(train_dataset.data)
        inputs, targets = self.remodel_data(
            inputs_scaled, train_dataset.targets
        )

        self.train_dataset = Dataset(
            inputs=self.normalize(inputs),
            targets=targets,
        )

        val_dataset = torchvision.datasets.FashionMNIST(self.dataset_path, train=False)
        inputs_scaled = self.scale(val_dataset.data)
        inputs, targets = self.remodel_data(inputs_scaled, val_dataset.targets)

        self.val_dataset = Dataset(
            inputs=self.normalize(inputs),
            targets=targets,
        )
