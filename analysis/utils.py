"""Utility functions to display training results."""

import os
import pickle
import inspect
from typing import Any, Dict, List, Union, Callable, Optional, Type
from abc import ABCMeta, abstractmethod

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mlflow.entities import Run
from matplotlib.axes._axes import Axes


def get_true_path(path: str) -> str:
    """Get system path from path obtained via MLflow."""
    if "file://" in path:
        return path[7:]

    return path


def get_visfile_path(run: Run) -> str:
    """Get last saved visualisation file path."""
    outputs_path = get_true_path(run.info.artifact_uri) + "/outputs/"

    paths = list(filter(lambda p: p != "meta.pickle", os.listdir(outputs_path)))
    paths.sort()

    return outputs_path + paths[-1]


def get_keys(path: str) -> None:
    """Returns available keys in pickle file."""
    path = get_true_path(path)
    with open(path, "rb") as f:
        out = pickle.load(f)
        print(out.keys())


def plot_sel(
    name: str, **kwargs: Any
) -> None:  # pylint: disable=unused-argument
    """Plot desired structuring element."""
    raise Exception("Not implemented")


class Plot(metaclass=ABCMeta):
    """Base plot class."""

    @staticmethod
    @abstractmethod
    def plot(axis: Axes, data: Dict[str, Any], run: Run) -> Axes:
        """Plot the filter weights on a given axis."""

    @classmethod
    def select(cls, name: str) -> Type["Plot"]:
        """
        Class method iterating over all subclasses to return the desirect class.
        """
        selected = cls.select_(name)
        if selected is None:
            raise Exception("No layer found")

        return selected

    @classmethod
    def select_(cls, name: str) -> Optional[Type["Plot"]]:
        """
        Class method iterating over all subclasses to return the desirect class.
        """
        if cls.__name__ == name:
            return cls

        for subclass in cls.__subclasses__():
            instance = subclass.select(name)
            if instance is not None:
                return instance

        return None

    @classmethod
    def listing(cls) -> List[str]:
        """List all the available ploting models."""
        subclasses = set()
        if not inspect.isabstract(cls):
            subclasses = {cls.__name__}

        for subclass in cls.__subclasses__():
            subclasses = subclasses.union(subclass.listing())

        return list(subclasses)


class SMorph(Plot):
    """Class containing function to plot SMorph layer."""

    @staticmethod
    def plot(axis: Axes, data: Dict[str, Any], run: Run) -> Axes:
        alpha = data["alpha"].squeeze()
        cmap = "plasma" if alpha > 0 else "plasma_r"

        axis.invert_yaxis()
        axis.get_yaxis().set_ticks([])
        axis.get_xaxis().set_ticks([])
        axis.set_box_aspect(1)

        plot = axis.pcolormesh(data["filter"].squeeze(), cmap=cmap)
        divider = make_axes_locatable(axis)
        clb_ax = divider.append_axes("right", size="5%", pad=0.05)
        clb_ax.set_box_aspect(15)
        plt.colorbar(plot, cax=clb_ax)

        axis.set_title(r"$\alpha$: " + f"{alpha:.3f}", fontsize=20)
        axis.set_xlabel("RMSE: {}\nLoss: {}", fontsize=20)

        return axis


class SMorphTanh(Plot):
    """Class containing function to plot SMorphTanh layer."""

    @staticmethod
    def plot(axis: Axes, data: Dict[str, Any], run: Run) -> Axes:
        """Plot the filter weights on a given axis."""
        axis.invert_yaxis()
        axis.get_yaxis().set_ticks([])
        axis.get_xaxis().set_ticks([])
        axis.set_box_aspect(1)

        plot = axis.pcolormesh(data["filter"].squeeze(), cmap="plasma")
        divider = make_axes_locatable(axis)
        clb_ax = divider.append_axes("right", size="5%", pad=0.05)
        clb_ax.set_box_aspect(15)
        plt.colorbar(plot, cax=clb_ax)

        axis.set_title(
            r"$\alpha$: " + f"{data['alpha'].squeeze():.3f}", fontsize=20
        )
        axis.set_xlabel("RMSE: {}\nLoss: {}", fontsize=20)

        return axis


PLOTABLE_CLASSES: List[str] = Plot.listing()


def plot_selem_row(axes: Axes, structuring_elements: List[str]) -> Axes:
    """Plot the first row with structuring elements aspects."""
    base_path = "../tests/operations/structuring_elements/data/"
    for idx_s, structuring_element in enumerate(structuring_elements):
        path = f"{base_path}/{structuring_element}_7.npy"

        axis = axes[0, idx_s]
        axis.pcolormesh(np.load(path).squeeze(), cmap="plasma")
        axis.set_box_aspect(1)
        axis.invert_yaxis()
        axis.axis("off")
        axis.set_title(structuring_element, fontsize=20, y=-0.15)

        divider = make_axes_locatable(axis)
        clb_ax1 = divider.append_axes("right", size="40%", pad=0.05)
        clb_ax2 = divider.append_axes("left", size="25%", pad=0.05)
        clb_ax1.axis("off")
        clb_ax2.axis("off")

    return axes


def filter_runs(
    runs: List[Run],
    model: Union[List[str], str],
    structuring_element: Union[List[str], str],
    operation: Union[List[str], str],
    test: Callable,
) -> List[Run]:
    """Filter the runs to get only the desired ones."""
    filtered = list(
        filter(lambda run: test(run.data.params["model"], model), runs)
    )
    filtered = list(
        filter(
            lambda run: test(
                run.data.params["structuring_element"], structuring_element
            ),
            filtered,
        )
    )
    filtered = list(
        filter(
            lambda run: test(run.data.params["operation"], operation), filtered
        )
    )

    return filtered


def is_plotable(class_name: str, excluded: Optional[List[str]]) -> bool:
    """Check if the layer can be displayed."""
    if excluded is None:
        return class_name in PLOTABLE_CLASSES

    return class_name not in excluded and class_name in PLOTABLE_CLASSES


def ploting(  # pylint: disable=too-many-locals,too-many-arguments,unused-argument
    uri: str,
    experiment_name: str,
    models: List[str],
    structuring_elements: List[str],
    operations: List[str],
    iterations: List[int],
    stretch_x: int = 1,
    excluded: Optional[List[str]] = None,
    **kwargs: Any,
) -> None:
    """
    Plot an entire figure with the results for desired models, structuring
    elements and operations.
    """
    client = mlflow.tracking.MlflowClient(uri)
    experiment = client.get_experiment_by_name(experiment_name)

    runs = client.search_runs(
        experiment.experiment_id, order_by=["attribute.start_time ASC"]
    )
    runs = filter_runs(
        runs, models, structuring_elements, operations, lambda x, y: x in y
    )

    for iteration in iterations:
        y_len, x_len = len(models) * len(operations), len(structuring_elements)
        fig, axes = plt.subplots(
            1 + y_len, x_len, figsize=(stretch_x * 6 * x_len, 6 * (1 + y_len))
        )
        fig.tight_layout(h_pad=10.0, w_pad=5.0)
        if x_len == 1:
            axes = np.array([axes]).T

        axes = plot_selem_row(axes, structuring_elements)

        for idx_m, model in enumerate(models):
            for idx_o, operation in enumerate(operations):
                axis = axes[1 + idx_m * len(operations) + idx_o, 0]
                axis.set_ylabel(
                    f"{model}\n\n{operation}",
                    fontsize=30,
                    rotation=0,
                    labelpad=100,
                    va="center",
                )

                for idx_s, structuring_element in enumerate(
                    structuring_elements
                ):
                    filtered = filter_runs(
                        runs,
                        model,
                        structuring_element,
                        operation,
                        lambda x, y: x == y,
                    )
                    run = filtered[iteration]

                    with open(get_visfile_path(run), "rb") as f:
                        saved_data = pickle.load(f)
                        axis = axes[1 + idx_m * len(operations) + idx_o, idx_s]
                        divider = make_axes_locatable(axis)
                        first_layer = True
                        for (  # pylint: disable=unused-variable
                            layer_index,
                            class_name,
                            data,
                        ) in saved_data[
                            "layers_weights"
                        ]:
                            if not is_plotable(class_name, excluded):
                                continue

                            if not first_layer:
                                axis = divider.append_axes(
                                    "right", size="100%", pad=0.2
                                )

                            axis = Plot.select(class_name).plot(axis, data, run)

                            first_layer = False

        plt.show()
