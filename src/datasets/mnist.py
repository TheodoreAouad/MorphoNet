"""DataModule for the MNIST dataset."""

from typing import Optional
import torchvision

from .base import DataModule, Dataset


class MNIST(DataModule):
    """MNIST DataModule."""

    def prepare_data(self) -> None:
        torchvision.datasets.MNIST(self.dataset_path, train=True, download=True)
        torchvision.datasets.MNIST(
            self.dataset_path, train=False, download=True
        )

    def setup(self, stage: Optional[str] = None) -> None:
        train_dataset = torchvision.datasets.MNIST(
            self.dataset_path,
            train=True,
        )
        input_transformed = self.input_transform(train_dataset.data)
        self.train_dataset = Dataset(
            inputs=input_transformed,
            targets=self.target_transform(
                input_transformed, train_dataset.targets
            ),
        )

        val_dataset = torchvision.datasets.MNIST(self.dataset_path, train=False)
        input_transformed = self.input_transform(val_dataset.data)
        self.val_dataset = Dataset(
            inputs=input_transformed,
            targets=self.target_transform(
                input_transformed, val_dataset.targets
            ),
        )
