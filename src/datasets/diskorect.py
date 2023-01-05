"""DataModule for the Diskorect dataset."""

from typing import Optional, Tuple, Dict

import numpy as np
from PIL import Image, ImageDraw
import random
from torch.utils.data import Dataset as torchDataset
import torch

from .base import DataModule
from operations.base import BinaryMorphologicalOperation
from .utils import (
    rand_shape_2d,
    invert_proba,
    get_rect_vertices,
    draw_poly,
    draw_ellipse,
    set_borders_to
)


def get_random_rotated_diskorect(
    size: Tuple = (50, 50), n_shapes: int = 20, max_shape: Tuple[int] = (20, 20), p_invert: float = 0.5,
    border=(0, 0), n_holes: int = 10, max_shape_holes: Tuple[int] = (10, 10), noise_proba=0.02,
    rng_float=np.random.rand, rng_int=np.random.randint, **kwargs
):
    diskorect = np.zeros(size)
    img = Image.fromarray(diskorect)
    draw = ImageDraw.Draw(img)

    def draw_shape(max_shape, fill_value):
        x = rng_int(0, size[0] - 2)
        y = rng_int(0, size[0] - 2)

        if rng_float() < .5:
            W = rng_int(1, max_shape[0])
            L = rng_int(1, max_shape[1])

            angle = rng_float() * 45
            draw_poly(draw, get_rect_vertices(x, y, W, L, angle), fill_value=fill_value)

        else:
            rx = rng_int(1, max_shape[0]//2)
            ry = rng_int(1, max_shape[1]//2)
            draw_ellipse(draw, np.array([x, y]), (rx, ry), fill_value=fill_value)

    for _ in range(n_shapes):
        draw_shape(max_shape=max_shape, fill_value=1)

    for _ in range(n_holes):
        draw_shape(max_shape=max_shape_holes, fill_value=0)

    diskorect = np.asarray(img) + 0
    diskorect[rand_shape_2d(diskorect.shape, rng_float=rng_float) < noise_proba] = 1  # bernoulli noise
    diskorect = invert_proba(diskorect, p_invert, rng_float=rng_float)  # random invert

    diskorect = set_borders_to(diskorect, border, value=0)

    return torch.tensor(diskorect)


class DiskorectDataset(torchDataset):
    def __init__(
        self,
        operation: BinaryMorphologicalOperation,
        torch_precision,
        random_gen_args: Dict = {},
        len_dataset: int = 100,
        seed: int = None,
        max_generation_nb: int = 0,
    ) -> None:
        self.random_gen_args = random_gen_args
        self.torch_precision = torch_precision
        self.len_dataset = len_dataset
        self.operation = operation
        self.max_generation_nb = max_generation_nb
        self.data = {}
        self.rng = np.random.default_rng(seed)

        self.create_input_target_samples(10)


    def create_input_target_samples(self, nb_samples: int) -> None:
        self.inputs = []
        self.targets = []
        for _ in range(nb_samples):
            input_, target = self.generate_input_target()
            self.inputs.append(input_.unsqueeze(0))
            self.targets.append(target.unsqueeze(0))

        self.inputs = torch.cat(self.inputs, axis=0)
        self.targets = torch.cat(self.targets, axis=0)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.max_generation_nb == 0:
            return self.generate_input_target()

        idx = idx % self.max_generation_nb

        if idx not in self.data.keys():
            self.data[idx] = self.generate_input_target()

        return self.data[idx]

    def generate_input_target(self) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ = get_random_rotated_diskorect(rng_float=self.rng.random, rng_int=self.rng.integers, **self.random_gen_args,)
        input_ = input_.unsqueeze(0)
        input_, target = self.operation(input_, None)
        # print(input_.shape, target.shape)
        input_, target = input_[0].to(self.torch_precision), target[0].to(self.torch_precision)
        # target = torch.tensor(target).float()
        # input_ = torch.tensor(input_).float()

        # if input_.ndim == 2:
        #     input_ = input_.unsqueeze(-1)  # Must have at least one channel

        # input_ = input_.permute(2, 0, 1)  # From numpy format (W, L, H) to torch format (H, W, L)
        # target = target.permute(2, 0, 1)  # From numpy format (W, L, H) to torch format (H, W, L)

        return input_, target

    def __len__(self) -> int:
        return self.len_dataset


class Diskorect(DataModule):
    """Diskorect DataModule."""

    def __init__(
        self,
        n_steps: int = 10000,
        random_gen_args: Dict = {},
        seed: int = None,
        max_generation_nb: int = 0,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.n_steps = n_steps
        self.random_gen_args = random_gen_args
        self.max_generation_nb = max_generation_nb

        if seed is None:
            seed = np.random.randint(2**32)
        self.seed = seed

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = DiskorectDataset(
            operation=self.operation,
            random_gen_args=self.random_gen_args,
            len_dataset=self.batch_size * self.n_steps,
            seed=self.train_seed,
            max_generation_nb=self.max_generation_nb,
            torch_precision=self.torch_precision,
        )

        self.val_dataset = DiskorectDataset(
            operation=self.operation,
            random_gen_args=self.random_gen_args,
            len_dataset=self.batch_size,
            seed=self.val_seed,
            max_generation_nb=self.max_generation_nb,
            torch_precision=self.torch_precision,
        )

    @property
    def val_seed(self):
        return self.seed + 1

    @property
    def train_seed(self):
        return self.seed