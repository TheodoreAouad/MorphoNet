def _load_ops():
    import os
    import importlib
    import torch

    spec = importlib.util.find_spec(f"{__package__}._C")
    torch.ops.load_library(spec.origin)


_load_ops()

from ._lmorph import LMorph
