"""DataModule for the Diskorect dataset."""

from typing import Optional, Tuple, Dict

import numpy as np
from PIL import Image, ImageDraw
import random
import torch
from torch.utils.data import Dataset as torchDataset

from .base import DataModule
from .utils import (
    rand_shape_2d,
    invert_proba,
    get_rect_vertices,
    draw_poly,
    set_borders_to
)


def get_sticks_noised(
    size: Tuple=(70, 70), n_shapes: int = 30, lengths_lim: Tuple = (12, 15), widths_lim: Tuple = (0, 0), p_invert: float = 0,
    angles=[0, 45, 90], border: Tuple = (0, 0), noise_proba: float = 0.1,
    rng_float=np.random.rand, rng_int=np.random.randint, **kwargs
):
    sticks = np.zeros(size)
    img = Image.fromarray(sticks)
    draw = ImageDraw.Draw(img)

    def draw_shape():
        x = rng_int(0, size[0] - 2)
        y = rng_int(0, size[0] - 2)

        L = rng_int(lengths_lim[0], lengths_lim[1] + 1)
        W = rng_int(widths_lim[0], widths_lim[1] + 1)

        angle = angles[rng_int(0, len(angles))]
        draw_poly(draw, get_rect_vertices(x, y, W, L, angle), fill_value=1)


    for _ in range(n_shapes):
        draw_shape()

    sticks = np.asarray(img) + 0

    sticks_noisy = sticks + 0

    sticks_noisy[rand_shape_2d(sticks.shape, rng_float=rng_float) < noise_proba] = 1  # bernoulli noise
    sticks, sticks_noisy = invert_proba([sticks, sticks_noisy], p_invert, rng_float=rng_float)  # random invert

    sticks = set_borders_to(sticks, border, value=0)
    sticks_noisy = set_borders_to(sticks_noisy, border, value=0)

    return sticks, sticks_noisy



class SticksNoisedGeneratorDataset(torchDataset):
    def __init__(
        self,
        torch_precision,
        random_gen_args: Dict = {},
        len_dataset: int = 1000,
        seed: int = None,
        max_generation_nb: int = 0,
    ) -> None:
        self.random_gen_args = random_gen_args
        self.torch_precision = torch_precision
        self.len_dataset = len_dataset
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
        target, input_ = get_sticks_noised(rng_float=self.rng.random, rng_int=self.rng.integers, **self.random_gen_args,)
        input_ = torch.tensor(input_).unsqueeze(0).to(self.torch_precision)
        target = torch.tensor(target).unsqueeze(0).to(self.torch_precision)
        return input_, target

    def __len__(self) -> int:
        return self.len_dataset


class NoiSti(DataModule):
    """NoiSti DataModule."""

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
        self.train_dataset = SticksNoisedGeneratorDataset(
            random_gen_args=self.random_gen_args,
            len_dataset=self.batch_size * self.n_steps,
            seed=self.train_seed,
            max_generation_nb=self.max_generation_nb,
            torch_precision=self.torch_precision,
        )

        self.val_dataset = SticksNoisedGeneratorDataset(
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