from .morpho import (
    dilation,
    erosion,
    opening,
    closing,
)

from .noise import (
    salt_noise,
    pepper_noise,
    salt_pepper_noise
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

def preprocess_salt(X, percentage, filter_shape):
    import numpy as np

    return np.array([salt_noise(x.squeeze(), percentage)[np.newaxis, ...] for x in X])

def preprocess_pepper(X, percentage, filter_shape):
    import numpy as np

    return np.array([pepper_noise(x.squeeze(), percentage)[np.newaxis, ...] for x in X])

def preprocess_salt_pepper(X, percentage, filter_shape):
    import numpy as np

    return np.array([salt_pepper_noise(x.squeeze(), percentage)[np.newaxis, ...] for x in X])

OPS_MORPH = {
    "dilation": preprocess_dilation,
    "erosion": preprocess_erosion,
    "closing": preprocess_closing,
    "opening": preprocess_opening
}

OPS_NOISE = {
    "salt": preprocess_salt,
    "pepper": preprocess_pepper,
    "saltpepper": preprocess_salt_pepper
}

OPS = dict(OPS_MORPH, **OPS_NOISE)
