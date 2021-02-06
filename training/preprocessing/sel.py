from .morpho import (
    draw_disk,
    draw_disk_aa,
    draw_diamond,
    draw_diamond_aa,
    draw_cross,
    draw_x,
)
from numpy.random import default_rng
import numpy as np

rng = default_rng()

STRUCTURING_ELEMENTS = {
    "disk2": lambda filter_shape, dtype: draw_disk(2, filter_shape, dtype=dtype),
    "diskaa1": lambda filter_shape, dtype: draw_disk_aa(1, filter_shape, dtype=dtype),
    "diskaa2": lambda filter_shape, dtype: draw_disk_aa(2, filter_shape, dtype=dtype),
    "diskaa3": lambda filter_shape, dtype: draw_disk_aa(3, filter_shape, dtype=dtype),
    "diamondaa3": lambda filter_shape, dtype: draw_diamond_aa(
        3, filter_shape, dtype=dtype
    ),
    "complex": lambda filter_shape, dtype: draw_disk_aa(3, filter_shape, dtype=dtype)
    - draw_diamond_aa(2, filter_shape, dtype=dtype),
    "cross3": lambda filter_shape, dtype: draw_cross(3, filter_shape, dtype=dtype),
    "cross5": lambda filter_shape, dtype: draw_cross(5, filter_shape, dtype=dtype),
    "cross7": lambda filter_shape, dtype: draw_cross(7, filter_shape, dtype=dtype),
    "x3": lambda filter_shape, dtype: draw_x(3, filter_shape, dtype=dtype),
    "x5": lambda filter_shape, dtype: draw_x(5, filter_shape, dtype=dtype),
    "rand": lambda filter_shape, dtype: rng.normal(
        size=filter_shape, loc=1.5, scale=0.05
    ).astype(dtype),
}

