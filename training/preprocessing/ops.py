from .morpho import (
    dilation,
    erosion,
    opening,
    closing,
)


def preprocess_dilation(X, sel):
    import numpy as np

    return np.array([dilation(x.squeeze(), sel)[np.newaxis, ...] for x in X])


def preprocess_erosion(X, sel):
    import numpy as np

    return np.array([erosion(x.squeeze(), sel)[np.newaxis, ...] for x in X])


def preprocess_opening(X, sel):
    import numpy as np

    return np.array([opening(x.squeeze(), sel)[np.newaxis, ...] for x in X])


def preprocess_closing(X, sel):
    import numpy as np

    return np.array([closing(x.squeeze(), sel)[np.newaxis, ...] for x in X])


OPS = {
    "dilation": preprocess_dilation,
    "erosion": preprocess_erosion,
    "closing": preprocess_closing,
    "opening": preprocess_opening,
}
