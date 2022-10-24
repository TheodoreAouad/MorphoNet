"""Loaders for the supported datasets."""

from abc import abstractmethod, ABCMeta
from typing import Any, List, Optional, Tuple, Callable

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


class DataModule(pl.LightningDataModule, metaclass=ABCMeta):
    """Base abstract class for datasets."""

    def __init__(
        self,
        batch_size: int,
        dataset_path: str,
        precision: str,
        operation: Operation,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.torch_precision = PRECISIONS_TORCH[precision]
        self.np_precision = PRECISIONS_NP[precision]

        self.input_transform = self._input_transform()
        self.target_transform = self._target_transform(operation)

        self.train_dataset: Dataset
        self.val_dataset: Dataset

    def _input_transform(self) -> torchvision.transforms.Compose:
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.ConvertImageDtype(
                    self.torch_precision
                ),  # already scale image
                torchvision.transforms.Lambda(
                    lambda x: (x - torch.mean(x)) / torch.std(x)
                ),
                torchvision.transforms.Lambda(lambda x: x[:, None, :, :]),
            ]
        )

    def _target_transform(self, operation: Operation) -> Callable:
        return lambda inputs, targets: torch.from_numpy(
            operation(inputs.numpy(), targets.numpy())
        ).to(self.torch_precision)

    @property
    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a sample of the dataset."""
        # TODO change method when shuffling to always have same data
        return self.val_dataset.inputs[:10], self.val_dataset.targets[:10]

    def prepare_data(self) -> None:
        pass

    @abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def _create_dataloader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(
            self.train_dataset,
        )

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader(
            self.val_dataset,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(Dataset(torch.empty(0), torch.empty(0)))

    @classmethod
    def select(cls, name: str, **kwargs: Any) -> Optional["DataModule"]:
        """
        Class method iterating over all subclasses to load the desired dataset.
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
        """List all the available dataset loaders."""
        subclasses = set()
        if not inspect.isabstract(cls):
            subclasses = {cls.__name__.lower()}

        for subclass in cls.__subclasses__():
            subclasses = subclasses.union(subclass.listing())

        return list(subclasses)


# pylint: disable=all
"""
class FMNISTDataset(Dataset):
    def __init__(self, precision, dataset_path, train=False, **kwargs):
        dataset = torchvision.datasets.FashionMNIST(dataset_path, train=train)
        images = dataset.data.numpy().astype(PRECISIONS_NP[precision])
        self.targets = dataset.targets.numpy().astype(np.int8)

        images = images[:, np.newaxis, :, :].astype(PRECISIONS_NP[precision])
        images /= 255.0

        self.inputs = torch.Tensor(images).type(PRECISIONS_TORCH[precision])


class BIWTOHDataset(Dataset):
    def __init__(self, precision, dataset_path, train=False, **kwargs):
        dtype = PRECISIONS_NP[precision]

        if train:
            self.inputs = np.load(f"{dataset_path}/train-images.npy").astype(
                dtype
            )
            self.targets = np.load(f"{dataset_path}/train-labels.npy").astype(
                dtype
            )
        else:
            self.inputs = np.load(f"{dataset_path}/t10k-images.npy").astype(
                dtype
            )
            self.targets = np.load(f"{dataset_path}/t10k-labels.npy").astype(
                dtype
            )


class SIDDDataset(Dataset):
    # https://paperswithcode.com/dataset/sidd

    def __init__(
        self,
        precision,
        dataset_path,
        train=False,
        patch_size=None,
        smartphone_codes=None,
        iso_levels=None,
        shutter_speeds=None,
        illuminants=None,
        ibcs=None,
        **kwargs,
    ):
        patch_shape = (patch_size, patch_size)
        dataset_path = dataset_path[0]
        filters = {
            "smartphone_code": split_arg(smartphone_codes),
            "iso_level": split_arg(iso_levels, int),
            "shutter_speed": split_arg(shutter_speeds, int),
            "illuminant": split_arg(illuminants),
            "illuminant_brightness_code": split_arg(ibcs),
        }

        def filter_data(instance: Dict[str, Any]):
            nonlocal filters
            keep = True
            for key, values in filters.items():
                if values is not None:
                    keep = keep and instance[key] in values
            return keep

        instances = SIDDDataset._get_instances(dataset_path, filter_data)
        self.inputs = SIDDDataset._load(instances, dataset_path, NOISY_NAME)
        self.targets = SIDDDataset._load(instances, dataset_path, GT_NAME)

        print(f"Raw X: {len(self.targets)}\nRaw Y: {len(self.inputs)}")

        patches = []
        x_all = []
        y_all = []
        for x, y in zip(self.inputs, self.targets):
            p = make_patches(x.shape, patch_shape)
            patches.append(p)
            for pp in p:
                x_all.append(x[pp])
                y_all.append(y[pp])
        x_all = np.array(x_all).astype("float32")[:, np.newaxis, :, :] / 255.0
        y_all = np.array(y_all).astype("float32")[:, np.newaxis, :, :] / 255.0

        print(f"Cut into {len(x_all)} patches of shape {patch_shape}")

        x_train, x_valid = model_selection.train_test_split(
            x_all, test_size=0.33, random_state=42
        )

    def _get_instances(dataset_path, filter_predicate=lambda inst: True):
        f = open(f"{dataset_path}/Scene_Instances.txt")
        instances_str = list(
            filter(
                lambda l: len(l) != 0,
                map(lambda l: l.strip(), f.read().split("\n")),
            )
        )

        def annotate_instance(l):
            parts = l.split("_")
            return {
                "path": l,
                "scene_number": parts[1],
                "smartphone_code": parts[2],
                "iso_level": int(parts[3]),
                "shutter_speed": int(parts[4]),
                "illuminant": parts[5],
                "illuminant_brightness_code": parts[6],
            }

        return list(
            filter(filter_predicate, map(annotate_instance, instances_str))
        )

    def _load(
        instances: List[Dict[str, Any]], dataset_path: str, image_name: str
    ):
        image_path = f"{dataset_path}/Data/{{}}/{image_name}.PNG"

        def load_instance(instance: Dict[str, Any]):
            nonlocal image_path
            return io.imread(image_path.format(instance["path"]))

        return [load_instance(instance) for instance in instances]


# LOADERS = {
#     "sidd": load_sidd,
#     "mnist": load_mnist,
#     "fashion_mnist": load_fmnist,
#     "biwtoh": load_biwtoh,
#     "gwtoh": load_biwtoh,
# }
LOADERS = {
    "sidd": None,
    "mnist": None,
    "fashion_mnist": None,
    "biwtoh": None,
    "gwtoh": None,
}

"""
