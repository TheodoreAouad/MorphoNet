from .morpho import (
    dilation,
    erosion,
    opening,
    closing,
    wtophat,
    bdilation,
    berosion,
    bclosing,
    bopening
)

from .noise import (
    salt_noise,
    pepper_noise,
    salt_pepper_noise
)

#####################
# Morpho Operations #
#####################

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

def preprocess_wtophat(X, sel):
    import numpy as np

    return np.array([wtophat(x.squeeze(), sel)[np.newaxis, ...] for x in X])

####################
# Noise Operations #
####################

def preprocess_salt(X, percentage, filter_shape):
    import numpy as np

    return np.array([salt_noise(x.squeeze(), percentage)[np.newaxis, ...] for x in X])

def preprocess_pepper(X, percentage, filter_shape):
    import numpy as np

    return np.array([pepper_noise(x.squeeze(), percentage)[np.newaxis, ...] for x in X])

def preprocess_salt_pepper(X, percentage, filter_shape):
    import numpy as np

    return np.array([salt_pepper_noise(x.squeeze(), percentage)[np.newaxis, ...] for x in X])

#####################
# Binary Operations #
#####################

def preprocess_bdilation(X, filter_shape):
    import numpy as np

    return np.array([bdilation(x.squeeze(), filter_shape)[np.newaxis, ...] for x in X])

def preprocess_berosion(X, filter_shape):
    import numpy as np

    return np.array([berosion(x.squeeze(), filter_shape)[np.newaxis, ...] for x in X])

def preprocess_bclosing(X, filter_shape):
    import numpy as np

    return np.array([bclosing(x.squeeze(), filter_shape)[np.newaxis, ...] for x in X])

def preprocess_bopening(X, filter_shape):
    import numpy as np

    return np.array([bopening(x.squeeze(), filter_shape)[np.newaxis, ...] for x in X])

OPS_MORPH = {
    "dilation": preprocess_dilation,
    "erosion": preprocess_erosion,
    "closing": preprocess_closing,
    "opening": preprocess_opening,
    "wtophat": preprocess_wtophat,
    "bdilation": preprocess_bdilation,
    "berosion": preprocess_berosion,
    "bclosing": preprocess_bclosing,
    "bopening": preprocess_bopening
}

OPS_NOISE = {
    "salt": preprocess_salt,
    "pepper": preprocess_pepper,
    "saltpepper": preprocess_salt_pepper
}

OPS = dict(OPS_MORPH, **OPS_NOISE)
