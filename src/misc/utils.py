from torch.utils.data import DataLoader
import torch
import numpy as np


PRECISIONS_TORCH = {"f32": torch.float32, "f64": torch.float64}
PRECISIONS_NP = {"f32": "float32", "f64": "float64"}

# TODO do something with this file


def split_arg(value, mapper=lambda a: a):  # type: ignore
    if value is None:
        return None
    return [mapper(v) for v in value.split(",")]


def get_data(train_ds, valid_ds, bs):  # type: ignore
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


def pad_inputs(x, model_name, filter_padding, pad_value=0):  # type: ignore
    if "double" in model_name:
        filter_padding *= 2
    elif "five" in model_name:
        filter_padding *= 5
    elif "four" in model_name:
        filter_padding *= 4

    padded = np.pad(
        x,
        (
            (0, 0),
            (0, 0),
            (filter_padding, filter_padding),
            (filter_padding, filter_padding),
        ),
        mode="constant",
        constant_values=(pad_value,),
    )

    return padded


def make_patches(data_shape, patch_shape):  # type: ignore
    patch_shape = np.array(patch_shape)
    data_shape = np.array(data_shape)
    n_patches = np.ceil(data_shape / patch_shape)
    x, y = np.mgrid[
        0 : data_shape[0] - patch_shape[0] : n_patches[0] * 1j,
        0 : data_shape[1] - patch_shape[1] : n_patches[1] * 1j,
    ]
    translations = np.round(np.array([x, y]).T.reshape((-1, 2))).astype(int)
    return [
        (slice(t[0], t[0] + patch_shape[0]), slice(t[1], t[1] + patch_shape[1]))
        for t in translations
    ]


def reconstruct_patches(data_patches, patches, data_shape):  # type: ignore
    data = np.zeros(data_shape, dtype=data_patches.dtype)
    count = np.zeros(data_shape, dtype=data_patches.dtype)
    for data_patch, patch in zip(data_patches, patches):
        data[patch] += data_patch
        count[patch] += 1
    data = data / count
    return data
