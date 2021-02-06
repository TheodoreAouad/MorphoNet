import numpy as np


def make_patches(data_shape, patch_shape):
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


def reconstruct_patches(data_patches, patches, data_shape):
    data = np.zeros(data_shape, dtype=data_patches.dtype)
    count = np.zeros(data_shape, dtype=data_patches.dtype)
    for data_patch, patch in zip(data_patches, patches):
        data[patch] += data_patch
        count[patch] += 1
    data = data / count
    return data
