"""Utility functions to display training results."""

import os
import pickle
from typing import Any, List, Union, Callable, Optional, Iterator, Tuple
from collections import OrderedDict

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mlflow.entities import Run, RunData
from matplotlib.axes._axes import Axes
import pytorch_lightning as pl
import torch

from models.base import BaseNetwork
from operations.structuring_elements.base import StructuringElement


def get_true_path(path: str) -> str:
    """Get system path from path obtained via MLflow."""
    if "file://" in path:
        return path[7:]

    return path


def get_visfile_path(run: Run) -> str:
    """Get last saved visualisation file path."""
    outputs_path = get_true_path(run.info.artifact_uri) + "/outputs/"

    paths = list(
        filter(
            lambda p: p != "meta.pickle"
            and not os.path.isdir(outputs_path + p),
            os.listdir(outputs_path),
        )
    )
    paths.sort()

    return outputs_path + paths[-1]


def get_metafile_path(run: Run) -> str:
    """Get meta file path."""
    return get_true_path(run.info.artifact_uri) + "/outputs/meta.pickle"


def get_keys(path: str) -> None:
    """Returns available keys in pickle file."""
    path = get_true_path(path)
    with open(path, "rb") as visfile:
        out = pickle.load(visfile)
        print(out.keys())


def plot_sel(
    name: str, **kwargs: Any
) -> None:  # pylint: disable=unused-argument
    """Plot desired structuring element."""
    raise Exception("Not implemented")


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


def recreate_target_selem(
    run: RunData, structuring_element: str
) -> Optional[np.ndarray]:
    """Recreate target selem used in given run."""
    structuring_element_instance = StructuringElement.select(
        structuring_element,
        filter_size=int(run.params["filter_size"]),
        precision=run.params["precision"],
    )

    try:
        return structuring_element_instance()
    except NotImplementedError:
        return None


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


def forward(pl_module: pl.LightningModule, inputs: torch.Tensor) -> OrderedDict:
    """
    Execute a forward pass with the given model to get layers' `plot_` methods.
    """
    index, modules, handles = 0, {}, []

    def add_hook(module_: torch.nn.Module) -> None:
        def forward_hook(  # pylint: disable=unused-argument
            module__: torch.nn.Module,
            module_input: torch.Tensor,
            module_output: torch.Tensor,
        ) -> None:
            nonlocal index

            if hasattr(module_, "plot_"):
                plot_method = getattr(module_, "plot_")
            else:
                plot_method = None

            modules[index] = (module_.__class__.__name__, plot_method)

            index += 1

        nonlocal handles
        handles.append(module_.register_forward_hook(forward_hook))

    pl_module.apply(add_hook)

    with torch.no_grad():
        pl_module.predict_step(inputs, -1)

    for handle in handles:
        handle.remove()

    return OrderedDict(sorted(modules.items()))


def iterate_over_axes(
    models: List[str],
    operations: List[str],
    structuring_elements: List[str],
    axes: Axes,
) -> Iterator[Tuple[int, str, int, str, int, str]]:
    """Iterate over the parameters."""
    for idx_m, model in enumerate(models):
        for idx_o, operation in enumerate(operations):
            axis = axes[1 + idx_m * len(operations) + idx_o, 0]
            axis.set_ylabel(
                f"{model}\n\n{operation}",
                fontsize=300 / len(model),
                rotation=0,
                labelpad=100,
                va="center",
            )

            for idx_s, structuring_element in enumerate(structuring_elements):
                yield idx_m, model, idx_o, operation, idx_s, structuring_element


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

        for (
            idx_m,
            model,
            idx_o,
            operation,
            idx_s,
            structuring_element,
        ) in iterate_over_axes(models, operations, structuring_elements, axes):
            filtered = filter_runs(
                runs,
                model,
                structuring_element,
                operation,
                lambda x, y: x == y,
            )
            if len(filtered) == 0:
                continue

            run = filtered[iteration]
            target_structuring_element = recreate_target_selem(
                run.data, structuring_element
            )

            model_class = BaseNetwork.select_(model)
            if model_class is None:
                continue

            path = get_visfile_path(run)
            pl_module = model_class.load_from_checkpoint(path)

            with open(get_metafile_path(run), "rb") as metafile:
                inputs = pickle.load(metafile)["inputs"]

                axis = axes[1 + idx_m * len(operations) + idx_o, idx_s]
                divider = make_axes_locatable(axis)
                plot_index = 0

                modules = forward(pl_module, inputs[0][None, :, :, :]).items()

                comments = ""
                if "val_loss" in run.data.metrics:
                    comments = f"Loss: {run.data.metrics['val_loss']:.3e}"

                for (
                    layer_index,  # pylint: disable=unused-variable
                    (class_name, plot_method),
                ) in modules:
                    if plot_method is None or (
                        excluded is not None and class_name in excluded
                    ):
                        continue

                    if plot_index > 0:
                        axis = divider.append_axes(
                            "right", size="100%", pad=0.3
                        )
                        comments = ""

                    try:
                        plot_method(
                            axis=axis,
                            target=target_structuring_element,
                            comments=comments,
                            divider=divider,
                        )
                        plot_index += 1
                    except NotImplementedError:
                        pass

        plt.show()
