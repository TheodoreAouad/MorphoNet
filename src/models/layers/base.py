"""Base layer essentially to declare plots functions."""

from typing import Optional, Tuple
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from mpl_toolkits.axes_grid1.axes_divider import AxesDivider


class BaseLayer(pl.LightningModule):
    """Base layer containing code shared by all layers."""

    def plot_(
        self,
        axis: Axes,
        cmap: str = "plasma",
        target: Optional[np.ndarray] = None,
        comments: str = "",
        divider: Optional[AxesDivider] = None,
    ) -> Axes:
        """
        Method specific to each layer that plots its visualization.
        """
        raise NotImplementedError

    def plot(  # pylint: disable=too-many-arguments
        self,
        figure: Optional[Tuple[Figure, Axes]] = None,
        cmap: str = "plasma",
        path: Optional[str] = None,
        target: Optional[np.ndarray] = None,
        comments: str = "",
        divider: Optional[AxesDivider] = None,
    ) -> None:
        """
        Function calling implemented `plot_` method while managing figure and
        plot saving.
        """
        if figure is None:
            fig, axis = plt.subplots(1, 1, figsize=(6, 6))
        else:
            fig, axis = figure

        self.plot_(axis, cmap, target, comments, divider)

        if path is not None:
            fig.savefig(path)
            plt.close(fig)
