from .base import DataModule, Dataset
import torchvision
from typing import Optional


class MNIST(DataModule):
    def setup(self, stage: Optional[str] = None):
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
