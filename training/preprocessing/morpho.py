import numpy as np


def draw_disk(radius, shape=None, dtype="float32"):
    from skimage import morphology

    res = morphology.disk(radius).astype(dtype)
    if shape is not None:
        return center_in(res, shape)
    return res


def draw_diamond(radius, shape=None, dtype="float32"):
    from skimage import morphology

    res = morphology.diamond(radius).astype(dtype)
    if shape is not None:
        return center_in(res, shape)
    return res


def center_in(arr, shape, dtype="float32"):
    res = np.zeros(shape=shape, dtype=dtype)
    pad_before = (shape[0] - arr.shape[0]) // 2, (shape[1] - arr.shape[1]) // 2
    pad_after = shape[0] - pad_before[0], shape[1] - pad_before[1]
    res[pad_before[0] : pad_after[0], pad_before[1] : pad_after[1]] = arr
    return res


def shape_aa(original, target_shape):
    from skimage import transform

    return transform.resize(
        original, target_shape, preserve_range=True, anti_aliasing=True
    )


def draw_disk_aa(radius, shape=None, dtype="float32"):
    from skimage import transform

    dim = radius * 2 + 1
    res = draw_disk(dim, dtype=dtype)
    res = shape_aa(res, (dim, dim))

    if shape is not None:
        return center_in(res, shape)
    return res


def draw_diamond_aa(radius, shape=None, dtype="float32"):
    dim = radius * 2 + 1
    res = draw_diamond(dim, dtype=dtype)
    res = shape_aa(res, (dim, dim))

    if shape is not None:
        return center_in(res, shape, dtype=dtype)
    return res


def draw_cross(size, shape=None, dtype="float32"):
    res = np.zeros((size, size), dtype=dtype)
    res[size // 2, :] = 1.0
    res[:, size // 2] = 1.0
    if shape is not None:
        return center_in(res, shape, dtype=dtype)
    return res


def draw_x(size, shape=None, dtype="float32"):
    x, y = np.mgrid[0:size, 0:size]
    res = np.zeros((size, size), dtype=dtype)
    res[x == y] = 1.0
    if shape is not None:
        return center_in(res, shape, dtype=dtype)
    return res

def draw_complex(shape=None, dtype="float32"):
    return draw_disk_aa(3, shape, dtype=dtype) \
           - draw_diamond_aa(2, shape, dtype=dtype)

def draw_bsquare(size, shape=None, dtype="float32"):
    res = np.ones((size, size), dtype=dtype)
    if shape is not None:
        return center_in(res, shape, dtype=dtype)
    return res

def draw_bdiamond(size, shape=None, dtype="float32"):
    l = np.arange(0, size * 2 + 1)
    x, y = np.meshgrid(l, l)
    res = np.array(np.abs(x - size) + np.abs(y - size) <= size, dtype=dtype)
    if shape is not None:
        return center_in(res, shape, dtype=dtype)
    return res

def dilation(img, fil):
    from scipy.ndimage import morphology as ndmorph

    crop_h, crop_w = fil.shape[0] // 2, fil.shape[1] // 2

    return ndmorph.grey_dilation(img, structure=fil)[
        crop_h : img.shape[0] - crop_h, crop_w : img.shape[1] - crop_w,
    ]

def erosion(img, fil):
    from scipy.ndimage import morphology as ndmorph

    crop_h, crop_w = fil.shape[0] // 2, fil.shape[1] // 2

    return ndmorph.grey_erosion(img, structure=fil)[
        crop_h : img.shape[0] - crop_h, crop_w : img.shape[1] - crop_w
    ]


def closing(img, fil):
    dilated = dilation(img, fil)
    eroded = erosion(dilated, fil)
    return eroded


def opening(img, fil):
    eroded = erosion(img, fil)
    dilated = dilation(eroded, fil)
    return dilated

